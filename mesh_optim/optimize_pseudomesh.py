import argparse
from collections import defaultdict
import json
import logging
import logging.config
from pathlib import Path
import shutil
from time import time
import yaml

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
import wandb

from data import ImageCamDataset
from mesh_utils import prune_mesh, prune_optimizer
from ml_utils import dp_schedule, get_optimizer, Losser, get_scheduler
from models import PseudomeshRenderer

LOGGER = None

def _load_config(path):
    with open(path, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data

def setup_logger(config):
    with open("./log_config.yaml", "r") as file:
        _config = yaml.load(file, Loader=yaml.FullLoader)
    log_filepath = Path(
        config["experiments_dir"], config["experiment_name"], "logs.log"
    )
    log_filepath.parent.mkdir(exist_ok=True, parents=True)
    _config["handlers"]["file_handler"]["filename"] = str(log_filepath)
    logging.config.dictConfig(_config)
    
    global LOGGER
    LOGGER = logging.getLogger(__name__)


def iter_pass(model: PseudomeshRenderer, data: dict, config: dict, loss_obj: Losser, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler = None, test: bool = False):
    if not test:
        optimizer.zero_grad()
    
    # get bakcgorund
    bg_color = [1, 1, 1] if config["white_background"] else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    mvp_mat = data["mvp_matrix"]
    gt_image = data["img"]
    
    if test:
        start_time = time()
    rgb, alpha = model(
        mvp_mat,
        gt_image.shape[2],  # width
        gt_image.shape[1],  # height
        config["renderer"]["depth_steps"]
    )
    if test:
        end_time = time() - start_time
    
    # handle background
    final_color = rgb * alpha + background[..., :3] * (1 - alpha)
    if gt_image.shape[-1] == 4:
        gt_image = gt_image[... ,:3] * gt_image[..., 3:4] + background * (1 - gt_image[..., 3:4])
    else:
        gt_image = gt_image[..., :3]
    
    # calculate losses
    data_to_loss = {
        "img_pred": final_color,
        "img_gt": gt_image,
    }
    losses = loss_obj(data_to_loss)
    if test:
        losses["render_time"] = torch.tensor(end_time)
    
    # update net
    if not test:
        losses["loss"].backward()
        model.acc_grad()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    return losses, data_to_loss


def wandb_log(losses: dict, step: int, data_to_loss: dict = None, mode: str = "train", epoch: int = None) -> None:
    assert mode in ["train", "val"]
    if losses is not None:
        log_losses = {f"{mode}_{_k}": _v.item() if isinstance(_v, torch.Tensor) else _v for _k, _v in losses.items()}
        wandb.log(
            data=log_losses,
            step=step
        )
    if data_to_loss is not None:
        grid_data = []
        name = ""
        for data_name, data in data_to_loss.items():
            if "delta" in data_name:
                continue
            if len(data.shape) == 4:
                data = data.squeeze(0)
            if data.shape[0] == 1:
                data = data.repeat(3, 1, 1)
            grid_data.append(data)
            name += "" if not name else "   "
            name += data_name
        grid_data = torch.cat([x.permute(0, 3, 1, 2) for x in grid_data], dim=0)
        grid_img = torchvision.utils.make_grid(grid_data, nrow=grid_data.shape[0] // 2)
        grid_img = torch.clip(grid_img, 0., 1.)
        grid_img = F.interpolate(
            grid_img.unsqueeze(0), 
            size=(grid_img.shape[1]//4, grid_img.shape[2]//4), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255
        grid_img = grid_img.astype(np.uint8)
        img_grid = wandb.Image(grid_img, mode="RGB")
        wandb.log(
            data={name: img_grid},
            step=step
        )
        deltas = {f"{mode}_{_k}": torch.norm(_v, p=2, dim=1).mean().item() for _k, _v in data_to_loss.items() if "delta_xyz" in _k}
        wandb.log(
            data=deltas,
            step=step
        )


def _collect_imgs_for_logs(data_to_loss:dict, container: dict, _iter: int, step: int) -> None:
    if not _iter % step:
        for _img_name, _img in data_to_loss.items():
            if container.get(_img_name, None) is None:
                container[_img_name] = _img
            else:
                container[_img_name] = torch.cat(
                    [
                        container[_img_name],
                        _img
                    ],
                    dim=0
                )


def _save_ckpt(ckpt_dir: Path, epoch: int, model: PseudomeshRenderer, optimizer: torch.optim.Optimizer) -> None:
    out_path = Path(ckpt_dir, "best_model")
    LOGGER.info(f"Save checkpoint: {out_path}")
    
    ckpt = {
        "epoch": epoch,
        "model": model.pseudomesh.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(ckpt, out_path)


def batch_to_device(data, device):
    data["img"] = data["img"].detach().to(device)
    data["mvp_matrix"] = data["mvp_matrix"].detach().to(device)


def pruning(model, optimizer, alpha_eps, grad_eps, mode, _type):
    init_shape = model.pseudomesh.vertices.shape[0]
    if _type in ["alpha", "both"]:
        opacity_mask = (model.pseudomesh.vertex_colors[:, -1:] < eval(alpha_eps)).reshape(-1)
    else: 
        opacity_mask = torch.zeros_like(model.pseudomesh.vertex_colors[:, -1:]).reshape(-1).bool()
    if _type in ["gradient", "both"]:
        grad_mask = (model.pseudomesh.grad_acc / model.pseudomesh.grad_denom) < eval(grad_eps)
    else:
        grad_mask = torch.zeros_like(model.pseudomesh.vertex_colors[:, -1:]).reshape(-1).bool()
    grad_opac_mask = torch.logical_or(opacity_mask, grad_mask)
    new_verts, new_faces, new_vert_colors, mask = prune_mesh(
        model.pseudomesh.vertices,
        model.pseudomesh.faces,
        model.pseudomesh.vertex_colors,
        grad_opac_mask,
        mode
    )
    LOGGER.info(f"Pruned {init_shape - new_verts.shape[0]} vertices. Left vertices: {new_verts.shape[0]}, left faces: {new_faces.shape[0]}")
    optimized_params = prune_optimizer(optimizer, mask)
    
    new_verts = optimized_params.get("vertices", new_verts)
    new_vert_colors = optimized_params.get("vertex_colors", new_vert_colors)
    model.set_values(new_verts, new_faces, new_vert_colors)
    model.reset_acc_grad()


def train(pseudomesh: PseudomeshRenderer, dloader: DataLoader, config: dict) -> None:
    # create optimizer
    params = {
        param_name: getattr(pseudomesh.pseudomesh, param_name) for param_name in config["model"]["optimizable_params"] if param_name not in config["model"]["optim_epoch_start"]
    }
    optimizer = get_optimizer(config, params)
    scheduler = get_scheduler(optimizer, config["lr_scheduler"])
    # create loss function
    loss_obj = Losser(config["loss"])
    
    _iter = 0
    best_loss = float("inf")
    start_time = time()
    best_time = time()
    
    LOGGER.info("Start training")
    for epoch in range(config["training"]["epochs"]):
        LOGGER.info(f"Epoch {epoch}")
        # train
        _batch_iter = 0
        _img_iter = 0
        batch_loss = 0
        train_data_to_loss_epoch = {}
        for data in tqdm(dloader):
            batch_to_device(data, config["device"])
            losses, data_to_loss = iter_pass(
                pseudomesh, 
                data, 
                config, 
                loss_obj, 
                optimizer,
                scheduler
            )
            losses["depth_steps"] = config["renderer"]["depth_steps"]
            losses["color_lr"] = optimizer.param_groups[0]['lr']
            if config["wandb"]["use"]:
                wandb_log(
                    losses=losses,
                    step=_iter,
                    data_to_loss=None,
                    mode="train",
                    epoch=epoch
                )
                _collect_imgs_for_logs(
                    data_to_loss,
                    train_data_to_loss_epoch,
                    _img_iter,
                    len(dloader.dataset) // config["wandb"]["imgs_per_epoch"]
                )
            batch_loss += losses["loss"].item()
            _img_iter += data["img"].shape[0]
            _batch_iter += 1
            _iter += data["img"].shape[0]
        if config["wandb"]["use"]:
            wandb_log(
                losses=None,
                step=_iter,
                data_to_loss=train_data_to_loss_epoch,
                mode="train",
                epoch=epoch
            )
        batch_loss /= _batch_iter
        if batch_loss < best_loss:
            best_time = time()
            _save_ckpt(
                ckpt_dir=config["ckpt_dir"],
                epoch=epoch,
                model=pseudomesh,
                optimizer=optimizer
            )
            best_loss = batch_loss

        alpha_prun_cond = epoch >= config["alpha_pruning"]["start_epoch"]
        alpha_prun_cond = alpha_prun_cond and not (
            (epoch - config["alpha_pruning"]["start_epoch"]) % config["alpha_pruning"]["epoch_step"]
        )
        if alpha_prun_cond:
            LOGGER.info("Perform pruning")
            pruning(
                pseudomesh, 
                optimizer, 
                config["alpha_pruning"]["alpha_eps"], 
                config["alpha_pruning"]["grad_eps"], 
                config["alpha_pruning"]["mode"],
                config["alpha_pruning"]["type"]
            )
        # check if i should update optimizer params
        for param_name, epoch_start in config["model"]["optim_epoch_start"].items():
            if epoch_start == epoch:
                LOGGER.info(f"Add {param_name} to optimizer")
                optimizer.add_param_group({
                    "name": param_name,
                    "params": getattr(pseudomesh.pseudomesh, param_name),
                    "lr": float(config["optimizer"]["lrs"][param_name])
                })

        # do the depth peeling scheduling
        if config["renderer"]["dp_scheduler"]["perform"]:  # if scheduling should be applied
            new_dp_val = dp_schedule(
                config["renderer"]["dp_scheduler"]["type"],
                config["renderer"]["depth_steps"],
                config["renderer"]["dp_scheduler"]["init_depth_steps"],
                epoch,
                config["renderer"]["dp_scheduler"]["params"]
            )
            if config["renderer"]["depth_steps"] != new_dp_val:
                config["renderer"]["depth_steps"] = new_dp_val
                LOGGER.info(f"Changed depth steps to: {config['renderer']['depth_steps']}")

    return best_time - start_time

def test(pseudomesh: PseudomeshRenderer, dloader: DataLoader, config: dict) -> dict:
    LOGGER.info("Start test")
    
    loss_obj = Losser(config["test_loss"])
    metrics = defaultdict(lambda: 0)
    _iter = 0
    for data in tqdm(dloader):
        batch_to_device(data, config["device"])
        with torch.no_grad():
            losses, data_to_loss = iter_pass(
                pseudomesh, 
                data, 
                config, 
                loss_obj, 
                None,
                test=True
            )
        _iter += 1
        for metric_name, metric_val in losses.items():
            metrics[metric_name] += metric_val.item()
        # save imgs?
    for metric_name, metric_val in metrics.items():
        metrics[metric_name] /= _iter
    
    return metrics


def prepare_output_dir(config: dict, cfg_path: str) -> None:
    experiment_dir = Path(config["experiments_dir"], config["experiment_name"])
    ckpt_dir = Path(experiment_dir, "checkpoints")
    imgs_dir = Path(experiment_dir, "imgs")
    config["experiment_dir"] = experiment_dir
    config["ckpt_dir"] = ckpt_dir
    config["imgs_dir"] = imgs_dir
    
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    imgs_dir.mkdir(parents=True, exist_ok=True)

    out_config_path = Path(config["experiment_dir"], "config.yaml")
    out_config = config
    out_config["experiment_dir"] = str(out_config["experiment_dir"])
    out_config["ckpt_dir"] = str(out_config["ckpt_dir"])
    out_config["imgs_dir"] = str(out_config["imgs_dir"])
    with open(out_config_path, "w") as file:
        yaml.dump(out_config, file, default_flow_style=False)


def setup_wandb(config: dict) -> None:
    LOGGER.info("Setup wandb")
    with open(config["wandb"]["key_path"], "r") as file:
        wandb_key = json.load(file)
    wandb.login(
        key=wandb_key,
        relogin=True
    )
    wandb.init(
        project=config["wandb"]["project_name"],
        entity=config["wandb"]["entity"],
        config=config,
        name=config["experiment_name"]
    )


def _load_best_model(config: dict) -> PseudomeshRenderer:
    pseudomesh = PseudomeshRenderer.create_model(config).to(config["device"])
    ckpt_path = Path(config["experiments_dir"], config["experiment_name"], "checkpoints", "best_model")
    checkpoint = torch.load(ckpt_path)
    pseudomesh.set_values(
        checkpoint["model"]["vertices"], 
        checkpoint["model"]["faces"],                  
        checkpoint["model"]["vertex_colors"], 
        checkpoint["model"]["grad_acc"]
    )
    pseudomesh.pseudomesh.load_state_dict(checkpoint["model"])
    return pseudomesh


def _save_results(results: dict, config: dict) -> None:
    out_path = Path(config["experiments_dir"], config["experiment_name"], "test_results.json")
    with open(out_path, "w") as file:
        json.dump(results, file)
        

def main() -> None:
    _parser = argparse.ArgumentParser(
        description="Optimize pseudomesh."
    )
    _parser.add_argument(
        "--cfg_path", "-cp",
        type=str, 
        help="Path to config",
        default="./optimize_pseudo_config.yaml"
    )
    _parser.add_argument(
        "--exp_name",
        type=str, 
        help="Experiment name",
        default=""
    )
    _parser.add_argument(
        "--exp_dir",
        type=str, 
        help="Experiments dir",
        default=""
    )
    _parser.add_argument(
        "--pseudomesh_path",
        type=str, 
        help="Path to pseudomesh",
        default=""
    )
    _parser.add_argument(
        "--dset_path",
        type=str, 
        help="Path to dset",
        default=""
    )
    _parser.add_argument(
        "--white_background",
        action="store_true"
    )
    _parser.add_argument(
        "--res",
        type=int,
        default=-1
    )

    _args = _parser.parse_args()
    config = _load_config(_args.cfg_path)
    if _args.exp_name:
        config["experiment_name"] = _args.exp_name
    if _args.pseudomesh_path:
        config["pseudomesh_path"] = _args.pseudomesh_path
    if _args.dset_path:
        config["dataset"]["dataset_path"] = _args.dset_path
    if _args.exp_dir:
        config["experiments_dir"] = _args.exp_dir
    if _args.white_background:
        config["white_background"] = _args.white_background
    if _args.res != -1:
        config["dataset"]["res"] = _args.res
    prepare_output_dir(config, _args.cfg_path)
    setup_logger(config)
    if config["wandb"]["use"]:
        setup_wandb(config)
    
    # load init splat
    pseudomesh = PseudomeshRenderer.create_model(config).to(config["device"])
    LOGGER.info(f"Model starts with {pseudomesh.pseudomesh.vertices.shape[0]} vertices and {pseudomesh.pseudomesh.faces.shape[0]} faces.")
    dset = ImageCamDataset(
        **config["dataset"]
    )
    dloader = DataLoader(
        dset,
        **config["dloader"]
    )
    
    train_time_s = train( 
        pseudomesh=pseudomesh,
        dloader=dloader,
        config=config
    )
    
    del pseudomesh
    
    # perform tests
    torch.cuda.empty_cache()
    
    # load model
    best_model = _load_best_model(config)
    
    # save model as npz
    np.savez(
        Path(config["experiments_dir"], config["experiment_name"], "checkpoints", "best_model.npz"), 
        vertices=best_model.pseudomesh.vertices.detach().cpu().numpy(), 
        faces=best_model.pseudomesh.faces.detach().cpu().numpy(), 
        vertex_colors=best_model.pseudomesh.vertex_colors.detach().cpu().numpy()
    )
    
    final_train_losses = test(
        pseudomesh=best_model,
        dloader=dloader,
        config=config
    )
    
    test_dset = ImageCamDataset(
        **{
            "test": True,
            **config["dataset"],
        }
    )
    test_dloader = DataLoader(
        test_dset,
        **config["dloader"]
    )
    
    final_test_losses = test(
        pseudomesh=best_model,
        dloader=test_dloader,
        config=config
    )
    
    out_losses = {
        "train_time": train_time_s,
        **{f"train_{k}": v for k, v in final_train_losses.items()},
        **{f"test_{k}": v for k, v in final_test_losses.items()},
    }
    
    _save_results(out_losses, config)


if __name__ == "__main__":
    main()




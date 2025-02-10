import argparse
from pathlib import Path
import yaml

import torch
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm

from data import ImageCamDataset
from optimize_pseudomesh import _load_best_model


def _load_config(path):
    with open(path, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data


def render_img(model, mvp_mat, img_shape, white_background, depth_steps, device):
    bg_color = [1, 1, 1] if white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    rgb, alpha = model(
        mvp_mat,
        img_shape[0],  # width
        img_shape[1],  # height
        depth_steps
    )
    
    final_color = rgb * alpha + background[..., :3] * (1 - alpha)
    return final_color


def main(cfg_path, method, white_background):
    config = _load_config(cfg_path)
    
    method_dir_name = "ours_30000"
    if method == "2dgs":
        # method_dir_name = "ours_10000"  # for blender scenes
        method_dir_name = "ours_30000"  # for real scenes
    
    out_imgs_dir = Path(
        config["experiments_dir"],
        config["experiment_name"],
        "test",
        method_dir_name,
        "pseudomesh_renders"
    )
    out_imgs_dir.mkdir(parents=True, exist_ok=True)
    
    # setup dataloader without shuffle
    dset = ImageCamDataset(
        **{
            "test": True,
            **config["dataset"],
        }
    )
    dloader = DataLoader(
        dset,
        batch_size=1,
        shuffle=False,
        num_workers=10
    )
    # load model
    pseudomesh = _load_best_model(config)

    # render all images and save them in the experiment's directory
    for data in tqdm(dloader):
        out_path = Path(out_imgs_dir, data["name"][0]).with_suffix(".png")
        mvp_mat = data["mvp_matrix"].detach().to(config["device"])
        
        gt_img = data["img"].squeeze()
        height = gt_img.shape[0]
        width = gt_img.shape[1]
        
        with torch.no_grad():
            pred_img = render_img(
                pseudomesh,
                mvp_mat,
                [width, height],
                white_background,
                config["renderer"]["depth_steps"],
                config["device"]
            )
        
        torchvision.utils.save_image(pred_img.squeeze().permute(2, 0, 1), out_path)


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(
        description="Process a single string argument."
    )
    _parser.add_argument(
        "--cfg_path", "-cp",
        type=str, 
        help="Path to config",
        required=True
    )
    _parser.add_argument(
        "--method",
        type=str,
        choices=["3dgs", "games", "sugar", "2dgs"],
        required=True
    )
    _parser.add_argument(
        '--white_background', 
        action='store_true'
    )

    _args = _parser.parse_args()
    main(_args.cfg_path, _args.method, _args.white_background)

import math

import numpy as np
import torch
import torch.nn.functional as F

from lpipsPyTorch import get_lpips_model

def _exp_schedule(init_val, gamma, timestep):
    return init_val * gamma ** timestep


def _step_schedule(curr_val, steps, timestep):
    return steps.get(timestep, curr_val)


def dp_schedule(_type, curr_val, init_val, epoch, params):
    assert _type in ["exp", "step"]
    if _type == "exp":
        out_val = _exp_schedule(init_val, float(params["gamma"]), epoch)
    elif _type == "step":
        out_val = _step_schedule(curr_val, params["steps"], epoch)
    return max(1, int(out_val))


class ColorExponentialLR(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(self, optimizer, gamma, param_group_index=0, last_epoch=-1, verbose=False):
        self.param_group_index = param_group_index
        super().__init__(optimizer, gamma, last_epoch, verbose)
    
    def get_lr(self):        
        # Only update the learning rate for the specified parameter group
        lrs = []
        for i, _ in enumerate(self.optimizer.param_groups):
            if i == self.param_group_index:
                lrs.append(self.base_lrs[i] * self.gamma ** self.last_epoch)
            else:
                lrs.append(self.optimizer.param_groups[i]['lr'])
        return lrs


def _calc_bs_mul(method: str, init_bs: float, curr_bs: float) -> float:
    assert method in ["constant", "linear", "sqrt"]
    if method == "linear":
        return curr_bs / init_bs
    elif method == "sqrt":
        return np.sqrt(curr_bs / init_bs)
    else:
        return 1.


def get_optimizer(config: dict, params: dict) -> torch.optim.Optimizer:
    optimizer_config = config["optimizer"]
    init_bs = 1
    
    lr_mul = _calc_bs_mul(
        method=optimizer_config["batch_scal_method"],
        init_bs=init_bs,
        curr_bs=config["dloader"]["batch_size"]
    )
    
    _params = [{"params": val, "name": key, "lr": lr_mul * float(optimizer_config["lrs"][key])} for key, val in params.items()]
    if optimizer_config["name"] == "adam":
        return torch.optim.Adam(
            _params,
            lr=0.
        )
    else:
        raise NotImplementedError(f"{optimizer_config['name']} not yet implemented")


def get_scheduler(optimizer: torch.optim.Optimizer, params: dict) -> ColorExponentialLR:
    if not params["use"]:
        return None
    return ColorExponentialLR(
        optimizer,
        params["gamma"],
        0
    )


class Losser:
    def __init__(self, loss_cfg: dict) -> None:
        self.loss_cfg = loss_cfg
        self.pips = None
        if any(["pips" in x for x in self.loss_cfg.keys()]):
            self.pips = get_lpips_model(
                "cuda",
                "vgg"
            )
            
        self._method_lookup = {
            "ssim": {
                "method": self.ssim_wrap,
                "params": ["img_pred", "img_gt"]
            },
            "img_l1": {
                "method": self.img_l1_wrap,
                "params": ["img_pred", "img_gt"]
            },
            "pips": {
                "method": self.pips_wrap,
                "params": ["img_pred", "img_gt"]
            },
            "psnr": {
                "method": self.psnr_wrap,
                "params": ["img_pred", "img_gt"]
            },
            "dice": {
                "method": self.dice_wrap,
                "params": ["mask_pred", "mask_gt"]
            },
            "delta_xyz": {
                "method": self.delta_wrap,
                "params": ["delta_xyz"]
            },
            "delta_rots": {
                "method": self.delta_wrap,
                "params": ["delta_rots"]
            },
            "delta_scales": {
                "method": self.delta_wrap,
                "params": ["delta_scales"]
            },
            "mse": {
                "method": self.mse_wrap,
                "params": ["img_pred", "img_gt"]
            },
            "body_displ": {
                "method": self.vector_norm_mean,
                "params": ["body_displ_pred", "body_displ_gt"]
            },
            "pose_displ": {
                "method": self.vector_norm_mean,
                "params": ["pose_displ_pred", "pose_displ_gt"]
            },
            "final_splat": {
                "method": self.vector_norm_mean,
                "params": ["final_splats_pred", "final_splats_gt"]
            }
        }

    def vector_norm_mean(self, loss_data: dict, params: list) -> torch.Tensor:
        return torch.mean(torch.norm(
            loss_data[params[0]] - loss_data[params[1]],
            dim=1,
            keepdim=True
        ))
    
    def img_l1_wrap(self, loss_data: dict, params: list) -> torch.Tensor:
        return l1_loss(loss_data[params[0]], loss_data[params[1]])

    def ssim_wrap(self, loss_data: dict, params: list) -> torch.Tensor:
        return 1. - ssim(loss_data[params[0]], loss_data[params[1]])

    def mse_wrap(self, loss_data: dict, params: list) -> torch.Tensor:
        return mse_loss(loss_data[params[0]], loss_data[params[1]])

    def pips_wrap(self, loss_data: dict, params: list) -> torch.Tensor:
        inp_data_1, inp_data_2 = loss_data[params[0]], loss_data[params[1]]
        if inp_data_1.shape[1] != 3:
            inp_data_1 = inp_data_1.permute(0, 3, 1, 2)
        if inp_data_2.shape[1] != 3:
            inp_data_2 = inp_data_2.permute(0, 3, 1, 2)
        return self.pips(inp_data_1, inp_data_2)

    def dice_wrap(self, loss_data: dict, params: list) -> torch.Tensor:
        return dice_loss(loss_data[params[0]], loss_data[params[1]])

    def delta_wrap(self, loss_data: dict, params: list) -> torch.Tensor:
        return norm_loss(loss_data[params[0]], p=2)
    
    def psnr_wrap(self, loss_data: dict, params: list) -> torch.Tensor:
        pred_img = loss_data[params[0]]
        gt_img = loss_data[params[1]]
        return psnr(pred_img, gt_img)

    def __call__(self, loss_data: dict) -> dict:
        out_losses = {}
        for loss_name in self.loss_cfg.keys():
            loss_method_name = loss_name.split("-")[0]
            loss_method = self._method_lookup[loss_method_name]["method"]
            params = self._method_lookup[loss_method_name]["params"]
            out_losses[loss_name] = loss_method(loss_data, params)
        out_losses["loss"] = sum([out_losses[key] * float(weight) for key, weight in self.loss_cfg.items() if "debug" not in key])
        return out_losses


def psnr(pred_img: torch.Tensor, gt_img: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    assert pred_img.shape[0] == gt_img.shape[0]
    mse = (((pred_img - gt_img)) ** 2).view(pred_img.shape[0], -1).mean(1, keepdim=True).mean()
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def dice_loss(mask_pred: torch.Tensor, mask_gt: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    masks both are tensors which have values 0 and 1 (0 for bg, 1 for mask)
    """
    intersection = torch.sum(mask_pred * mask_gt)
    cardinality = torch.sum(mask_pred) + torch.sum(mask_gt)
    dice = (2 * intersection + smooth) / (cardinality + smooth)
    return 1 - dice


def norm_loss(data: torch.Tensor, p: int = 1, dim: int = 1) -> torch.Tensor:
    return torch.norm(data, p=p, dim=dim).mean()


def l1_loss(network_output: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.abs((network_output - gt)).mean()


def mse_loss(network_output: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.sum((network_output - gt) ** 2) / (np.prod(list(network_output.shape)))


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int) -> torch.Tensor:
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True) -> torch.Tensor:
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1: torch.Tensor, img2: torch.Tensor, window: int, window_size: int, channel: int, size_average: bool = True) -> torch.Tensor:
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
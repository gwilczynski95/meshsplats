#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from argparse import ArgumentParser
import json
import os
from pathlib import Path
from PIL import Image

import numpy as np
import torch
import torchvision.transforms.functional as tf
from tqdm import tqdm

from lpipsPyTorch import lpips
from ml_utils import psnr, ssim


def PILtoTorch(pil_image):
    resized_image = torch.from_numpy(np.array(pil_image)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def readImages(renders_dir, gt_dir, mesh_renders_dir):
    gs_renders = []
    mesh_renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        mesh_render = Image.open(mesh_renders_dir / fname)
        gs_render = Image.open(renders_dir / fname)
        gt_path = gt_dir / fname
        try:
            gt = Image.open(gt_path)
        except FileNotFoundError:
            try:
                gt = Image.open(gt_path.with_suffix(".jpg"))
            except FileNotFoundError:
                gt = Image.open(gt_path.with_suffix(".JPG"))
        if gt.width != gs_render.width or gt.height != gs_render.height:
            gt = gt.resize((int(gs_render.width), int(gs_render.height)))
        if mesh_render.width != gs_render.width or mesh_render.height != gs_render.height:
            mesh_render = mesh_render.resize((int(gs_render.width), int(gs_render.height)))
        gs_renders.append(PILtoTorch(gs_render).cuda())
        mesh_renders.append(PILtoTorch(mesh_render).cuda())
        gt_render = PILtoTorch(gt).cuda()
        if gt_render.shape[0] != 3:
            gt_render = gt_render[:3, ...]
            gt_mask = gt_render[3:4, ...]
            gt_render *= gt_mask.to(gt_render.device)
        gts.append(gt_render)
        image_names.append(fname)
    return gs_renders, mesh_renders, gts, image_names

def evaluate(model_path, dset_path, algorithm):
    if Path(dset_path, "transforms_train.json").exists():
        dset_type = "nerf"
    else:
        dset_type = "colmap"
    
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}

    full_dict[model_path] = {}
    per_view_dict[model_path] = {}
    full_dict_polytopeonly[model_path] = {}
    per_view_dict_polytopeonly[model_path] = {}

    test_dir = Path(model_path) / "test"

    for method in os.listdir(test_dir):
        print("Method:", method)

        full_dict[model_path][method] = {}
        per_view_dict[model_path][method] = {}
        full_dict_polytopeonly[model_path][method] = {}
        per_view_dict_polytopeonly[model_path][method] = {}

        method_dir = test_dir / method
        gt_dir = Path(dset_path, "test") if dset_type == "nerf" else Path(dset_path, "images")
        
        renders_dir_name = "renders"
        if algorithm == "games":
            renders_dir_name += "_gs_flat"
        
        gs_renders_dir = method_dir / renders_dir_name
        mesh_renders_dir = method_dir / "pseudomesh_renders"
        gs_renders, mesh_renders, gts, image_names = readImages(gs_renders_dir, gt_dir, mesh_renders_dir)

        gs_ssims = []
        gs_psnrs = []
        gs_lpipss = []

        mesh_ssims = []
        mesh_psnrs = []
        mesh_lpipss = []

        for idx in tqdm(range(len(gs_renders)), desc="Metric evaluation progress"):
            gs_ssims.append(ssim(gs_renders[idx], gts[idx]))
            gs_psnrs.append(psnr(gs_renders[idx], gts[idx]))
            gs_lpipss.append(lpips(gs_renders[idx], gts[idx], net_type='vgg'))
            mesh_ssims.append(ssim(mesh_renders[idx], gts[idx]))
            mesh_psnrs.append(psnr(mesh_renders[idx], gts[idx]))
            mesh_lpipss.append(lpips(mesh_renders[idx], gts[idx], net_type='vgg'))

        full_dict[model_path][method].update({
            "GS_SSIM": torch.tensor(gs_ssims).mean().item(),
            "GS_PSNR": torch.tensor(gs_psnrs).mean().item(),
            "GS_LPIPS": torch.tensor(gs_lpipss).mean().item(),
            "MESH_SSIM": torch.tensor(mesh_ssims).mean().item(),
            "MESH_PSNR": torch.tensor(mesh_psnrs).mean().item(),
            "MESH_LPIPS": torch.tensor(mesh_lpipss).mean().item(),
        })
        per_view_dict[model_path][method].update({
            "GS_SSIM": {name: ssim for ssim, name in zip(torch.tensor(gs_ssims).tolist(), image_names)},
            "GS_PSNR": {name: psnr for psnr, name in zip(torch.tensor(gs_psnrs).tolist(), image_names)},
            "GS_LPIPS": {name: lp for lp, name in zip(torch.tensor(gs_lpipss).tolist(), image_names)},
            "MESH_SSIM": {name: ssim for ssim, name in zip(torch.tensor(mesh_ssims).tolist(), image_names)},
            "MESH_PSNR": {name: psnr for psnr, name in zip(torch.tensor(mesh_psnrs).tolist(), image_names)},
            "MESH_LPIPS": {name: lp for lp, name in zip(torch.tensor(mesh_lpipss).tolist(), image_names)},
        })

    with open(model_path + "/results.json", 'w') as fp:
        json.dump(full_dict[model_path], fp, indent=True)
    with open(model_path + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[model_path], fp, indent=True)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_path', '-m', required=True, type=str)
    parser.add_argument('--set_path', '-s', required=True, type=str)
    parser.add_argument('--white_background', action="store_true")
    parser.add_argument(
        "--method",
        type=str,
        default="3dgs",
        choices=["3dgs", "games", "sugar", "2dgs"]
    )
    args = parser.parse_args()
    evaluate(args.model_path, args.set_path, args.method, args.white_background)

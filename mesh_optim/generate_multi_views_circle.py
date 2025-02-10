import argparse
import math
from pathlib import Path
import time

import torch
import torchvision
import imageio.v2 as imageio

from data import ImageCamDataset
from optimize_pseudomesh import _load_best_model
from render_pseudomesh import _load_config

def get_depth_map(model, mvp_mat, width, height, cam_pos, num_layers):
    optim_depth_map, optim_depth_map_alpha = model.get_depth_map(mvp_mat.unsqueeze(0), width, height, cam_pos, num_layers)
    optim_depth_map = (optim_depth_map - optim_depth_map.min()) / (optim_depth_map.max() - optim_depth_map.min())
    optim_depth_map_alpha = torch.clamp(optim_depth_map_alpha, 0.0, 1.0).squeeze(-1)
    
    optim_depth_map = torch.cat(
        [
            optim_depth_map[..., 0],
            torch.zeros_like(optim_depth_map[..., 0]),
            1 - optim_depth_map[..., 0],
        ],
        dim=0
    )
    background = torch.tensor([1, 0, 0], dtype=torch.float32, device=optim_depth_map.device)
    optim_depth_map = optim_depth_map * optim_depth_map_alpha + background[:3, None, None] * (1 - optim_depth_map_alpha)
    return optim_depth_map

def get_gray_map(model, mvp_mat, width, height, color_verts, num_layers):
    _color_verts = torch.cat([color_verts, model.pseudomesh.vertex_colors[..., -1:]], dim=1)
    rgb, alpha = model.get_gray_map(mvp_mat.unsqueeze(0), width, height, num_layers, _color_verts)
    return rgb.squeeze(0).permute(2, 0, 1)#, alpha.squeeze(0)

def get_normal_map(model, mvp_mat, width, height, num_layers):
    rgb, alpha = model.get_normal_map(mvp_mat.unsqueeze(0), width, height, num_layers)
    rgb = torch.clamp(rgb, 0.0, 1.0)
    return rgb.squeeze(0).permute(2, 0, 1)#, alpha.squeeze(0)

def get_all_maps(model, mvp_mat, width, height, cam_pos, num_layers, color_verts=None):
    results = model.render_all_maps(mvp_mat.unsqueeze(0), width, height, cam_pos, num_layers, color_verts)
    color = results["color"][0].squeeze(0).permute(2, 0, 1).clamp(0.0, 1.0)
    normal = results["normal"][0].squeeze(0).permute(2, 0, 1).clamp(0.0, 1.0)
    depth = results["depth"][0].squeeze(0).permute(2, 0, 1)
    depth_alpha = results["depth"][1].squeeze(0)
    gray = results["gray"][0].squeeze(0).permute(2, 0, 1).clamp(0.0, 1.0)
    
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth_alpha = torch.clamp(depth_alpha, 0.0, 1.0).squeeze(-1).unsqueeze(0)
    
    depth = torch.cat(
        [
            depth[0:1, ...],
            torch.zeros_like(depth[0:1, ...]),
            1 - depth[0:1, ...],
        ],
        dim=0
    )
    background = torch.tensor([1, 0, 0], dtype=torch.float32, device=depth.device)
    depth = depth * depth_alpha + background[:3, None, None] * (1 - depth_alpha)
    
    return color, normal, depth, gray

def normalize(v, eps=1e-6):
    """Return the normalized vector v."""
    norm = torch.norm(v)
    return v if norm < eps else v / norm

def look_at(eye, target, up):
    # Compute the forward vector (from the camera toward the target)
    forward = normalize(target - eye)
    
    # Compute the right vector
    right = normalize(torch.cross(forward, up))
    
    # Recompute the orthogonal up vector
    true_up = torch.cross(right, forward)
    # true_up = up
    
    # Build rotation matrix.
    # Note: We use -forward so that the camera looks along its -z axis.
    R = torch.stack([right, true_up, -forward], dim=1)  # Shape: (3,3)
    
    # Build the full 4x4 transformation matrix
    T = torch.eye(4)
    T[:3, :3] = R
    T[:3, 3] = eye
    return T

def generate_camera_matrices(target, radius, num_views=20, 
                             up=torch.tensor([0, -1, 0], dtype=torch.float32),
                             tilt_angle=0.0):
    target = target.float()
    cam_mats = []
    
    # Precompute the rotation matrix for tilting about the x-axis
    c = math.cos(tilt_angle)
    s = math.sin(tilt_angle)
    R_tilt = torch.tensor([
        [1,  0,  0],
        [0,  c, -s],
        [0,  s,  c]
    ], dtype=torch.float32)
    
    for i in range(num_views):
        angle = 2 * math.pi * i / num_views
        
        # Compute a point on the circle (initially in the XZ-plane)
        circle_point = torch.tensor([
            radius * math.cos(angle),
            0.0,
            radius * math.sin(angle)
        ], dtype=torch.float32)
        
        # Apply the tilt rotation to the circle point
        tilted_point = torch.matmul(R_tilt, circle_point)
        
        # Compute the camera position by adding the target (center of the circle)
        cam_pos = target + tilted_point
        
        # Create the camera-to-world transformation matrix for this camera
        cam_mat = look_at(cam_pos, target, up)
        cam_mats.append(cam_mat)
    
    return torch.stack(cam_mats)

def calculate_mvp(focal_x, focal_y, img_shape, view_mat, far, near):
    proj_matrix = torch.zeros(4, 4, device="cpu", dtype=torch.float32)
    proj_matrix[0, 0] = 2 * focal_x / img_shape[0]
    proj_matrix[1, 1] = -2 * focal_y / img_shape[1]
    proj_matrix[2, 2] = -(far + near) / (far - near)
    proj_matrix[2, 3] = -2.0 * far * near / (far - near)
    proj_matrix[3, 2] = -1.0
    
    mvp_matrix = proj_matrix @ view_mat
    return mvp_matrix


def create_gif(image_paths, output_path, fps):
    """Create a GIF from a list of image paths"""
    images = []
    # Sort paths numerically based on the frame number in the filename
    sorted_paths = sorted(image_paths, key=lambda x: int(x.stem.split('_')[0]))
    for path in sorted_paths:
        images.append(imageio.imread(path))
    imageio.mimsave(output_path, images, fps=fps)

def calculate_alphas(img1, img2, left_int, dist_interval):
    dist = (img1["view_matrix"][:3, 3] - img2["view_matrix"][:3, 3]).pow(2).sum().sqrt()
    alphas = []
    left_dist = dist
    curr_pos = -left_int
    while dist_interval < left_dist:
        curr_pos += dist_interval
        alphas.append(curr_pos / dist)
        left_dist = dist - curr_pos
    left_int = dist - curr_pos
    return torch.tensor(alphas, dtype=torch.float32), left_int

def main(cfg_path, no_sec, fps):
    config = _load_config(cfg_path)
    output_dir = Path(config["experiment_dir"]) / "multi_views"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dset = ImageCamDataset(
        **{
            **config["dataset"],
        }
    )
    optim_pseudomesh = _load_best_model(config)
    
    circle_center = torch.cat([x["transf_matrix"][:3, 3].unsqueeze(0) for x in dset.image_files], dim=0).mean(dim=0)
    radius = 4.
    all_frames = no_sec * fps
    
    tilt = math.radians(20)
    up = torch.tensor([0, -0.78, -0.22], dtype=torch.float32)
    up = up / torch.norm(up)
    camera_matrices = generate_camera_matrices(circle_center, radius, all_frames, up=up, tilt_angle=tilt)
    
    _range = 0.3
    color_verts = torch.rand(
        optim_pseudomesh.pseudomesh.vertices.shape[0],
        device=optim_pseudomesh.pseudomesh.vertices.device,
    )[:, None].repeat(1, 3) * _range + 0.5 - _range / 2.
    
    _iter = 0
    focal_x, focal_y = dset.image_files[0]["focal_x"], dset.image_files[0]["focal_y"]
    width, height = dset.image_files[0]["img"].shape[1], dset.image_files[0]["img"].shape[0]
    print("Generate views")
    for cam_mat in camera_matrices:
        start_time = time.time()
        view_mat = torch.inverse(cam_mat)
        mvp_mat = calculate_mvp(focal_x, focal_y, [width, height], view_mat, dset.far, dset.near).to(config["device"])
        with torch.no_grad():
            optim_img, optim_normal_map, optim_depth_map, optim_gray_map = get_all_maps(optim_pseudomesh, mvp_mat, width, height, cam_mat[:3, 3].to(config["device"]), config["renderer"]["depth_steps"], color_verts)

        optim_img_path = output_dir / f"{_iter:05d}_optim.png"
        depth_img_path = output_dir / f"{_iter:05d}_optim-depth.png"
        gray_img_path = output_dir / f"{_iter:05d}_optim-gray.png"
        normal_img_path = output_dir / f"{_iter:05d}_optim-normal.png"
        torchvision.utils.save_image(optim_img.squeeze().permute(2, 0, 1), optim_img_path)
        torchvision.utils.save_image(optim_depth_map, depth_img_path)
        torchvision.utils.save_image(optim_gray_map, gray_img_path)
        torchvision.utils.save_image(optim_normal_map, normal_img_path)
        print(f"Saved {_iter}, took: {round(time.time() - start_time, 3)} s")
        _iter += 1

    # Create GIFs after generating all images
    optim_images = list(output_dir.glob("*_optim.png"))
    gray_images = list(output_dir.glob("*_optim-gray.png"))
    depth_images = list(output_dir.glob("*_optim-depth.png"))
    normal_images = list(output_dir.glob("*_optim-normal.png"))
    
    create_gif(optim_images, output_dir / "optim.gif", fps)
    create_gif(gray_images, output_dir / "gray.gif", fps)
    create_gif(depth_images, output_dir / "depth.gif", fps)
    create_gif(normal_images, output_dir / "normal.gif", fps)


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(
        description="Process a single string argument."
    )
    _parser.add_argument(
        "--cfg_path", "-cp",
        type=str, 
        help="Path to config",
        default="/home/grzegos/experiments/mip_games_gs-flat_sh0_bicycle/config.yaml"
    )
    _parser.add_argument(
        "--sec",
        type=int,
        help="How long the video should be",
        default=10
    )
    _parser.add_argument(
        "--fps",
        type=int,
        help="Frames per second",
        default=30
    )

    _args = _parser.parse_args()
    main(_args.cfg_path, _args.sec, _args.fps)
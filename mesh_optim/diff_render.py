import json
from pathlib import Path

import torch
import numpy as np
import nvdiffrast.torch as dr
import matplotlib.pyplot as plt


def create_obj(device):
    vertices = torch.tensor([
        [-1, 1, -1],
        [1, 1, -1],
        [1, -1, -1],
        [-1, -1, -1],
        [-1, 1, 1],
        [1, 1, 1],
        [1, -1, 1],
        [-1, -1, 1],
    ], device=device, dtype=torch.float32, requires_grad=True)

    _verts_add = torch.tensor([
        [-1, 1, -1],
        [1, 1, -1],
        [1, -1, -1],
        [-1, -1, -1],
        [-1, 1, 1],
        [1, 1, 1],
        [1, -1, 1],
        [-1, -1, 1],
    ], device=device, dtype=torch.float32, requires_grad=True)
    z_change = torch.zeros_like(vertices)
    z_change[:, -1] = -1.1
    
    vertices = torch.cat(
        [
            vertices,
            _verts_add * 0.2 + z_change
        ],
        dim=0
    )

    # vertices = _verts_add * 0.2 + z_change
    
    colors = torch.tensor([
        [1, 0, 0, 0.6],
        [0, 1, 0, 0.6],
        [1, 0, 0, 0.6],
        [0, 0, 1, 0.6],
        [0, 0, 1, 1.],
        [1, 0, 0, 1.],
        [0, 0, 1, 1.],
        [0, 1, 0, 1.],
    ], device=device, dtype=torch.float32, requires_grad=True)
    
    _colors_add = torch.cat(
        [
            torch.zeros_like(colors[:, :3], requires_grad=True),
            torch.ones_like(colors[:, 3:], requires_grad=True),
        ],
        dim=-1
    )
    colors = torch.cat(
        [
            colors,
            _colors_add
        ],
        dim=0
    )
    # colors = _colors_add
    
    # Define faces (triangles)
    faces = torch.tensor([
        [0, 1, 4],
        [1, 5, 4],
        [1, 2, 5],
        [2, 6, 5],
        [2, 3, 6],
        [3, 7, 6],
        [3, 0, 7],
        [0, 4, 7],
        [0, 1, 3],
        [1, 2, 3],
        [4, 5, 7],
        [5, 6, 7],
    ], device=device, dtype=torch.int32).contiguous()
    _new_faces = faces + 8
    faces = torch.cat(
        [
            faces,
            _new_faces
        ],
        dim=0
    )
    
    return vertices, colors, faces


def load_pseudomesh(_path):
    mesh_data = np.load(_path)
    vertices = mesh_data["vertices"]
    faces = mesh_data["faces"]
    vert_colors = mesh_data["vertex_colors"]
    face_colors = mesh_data["face_colors"]
    return vertices, faces, vert_colors, face_colors


def create_proj_mats(camera_angle_x, img_size, transf_mat, far, near):
    focal = img_size / (2 * np.tan(camera_angle_x / 2))
    transf_matrix = torch.tensor(transf_mat, device="cpu", dtype=torch.float32)
    view_matrix = torch.inverse(transf_matrix)

    proj_matrix = torch.zeros(4, 4, device="cpu", dtype=torch.float32)
    proj_matrix[0, 0] = 2 * focal / img_size
    proj_matrix[1, 1] = 2 * focal / img_size
    proj_matrix[2, 2] = -(far + near) / (far - near)
    proj_matrix[2, 3] = -2.0 * far * near / (far - near)
    proj_matrix[3, 2] = -1.0
    
    mvp_matrix = proj_matrix @ view_matrix
    
    return {
        "transf_matrix": transf_matrix,
        "view_matrix": view_matrix,
        "proj_matrix": proj_matrix,
        "mvp_matrix": mvp_matrix
    }


def create_camera_mats(cam_data, img_size=512, near=0.01, far=100., device="cuda"):
    height, width = img_size, img_size
    camera_angle_x = cam_data["camera_angle_x"]
    focal = width / (2 * np.tan(camera_angle_x / 2))
    
    out = {}
    for cam in cam_data["frames"]:
        transf_matrix = cam["transform_matrix"]
        
        transf_matrix = torch.tensor(transf_matrix, device=device, dtype=torch.float32)
        
        # view matrix - transforms vertices from world space 
        # vertex coords to camera/view space
        
        view_matrix = torch.inverse(transf_matrix)
        
        # Convert focal length to OpenGL-style projection matrix
        proj_matrix = torch.zeros(4, 4, device=device, dtype=torch.float32)
        proj_matrix[0, 0] = 2 * focal / width
        proj_matrix[1, 1] = 2 * focal / height
        proj_matrix[2, 2] = -(far + near) / (far - near)
        proj_matrix[2, 3] = -2.0 * far * near / (far - near)
        proj_matrix[3, 2] = -1.0
        
        # mvp (model-view-projection) matrix - combines all transformations into one matrix
        # basically u apply it to verts and you have them on an image lol
        mvp_matrix = proj_matrix @ view_matrix
        
        out[cam["file_path"]] = {
            "transf_matrix": transf_matrix,
            "view_matrix": view_matrix,
            "proj_matrix": proj_matrix,
            "mvp_matrix": mvp_matrix
        }
        _out = create_proj_mats(
            camera_angle_x, 
            img_size,
            cam["transform_matrix"],
            far,
            near
        )
        out[cam["file_path"]] = {k: v.cuda() for k, v in _out.items()}
    return out


def render(glctx, verts, colors, faces, cam_data, img_size, device):
    # world vertices to image positions
    pos_clip = torch.matmul(
        torch.cat(
            [
                verts, 
                torch.ones_like(verts[:, :1])    
            ], dim=1
        ), 
        cam_data["mvp_matrix"].t()
    )[None, ...]
    
    with torch.no_grad():
        # Calculate face centers in clip space
        face_verts = pos_clip[0, faces.long()]
        face_centers = face_verts.mean(dim=1)
        # Sort faces by Z coordinate (depth) in descending order (back to front)
        face_z = face_centers[..., 2]
        face_order = torch.argsort(face_z, descending=True)
        faces = faces[face_order]
    
    rast_out, rast_db = dr.rasterize(
        glctx, 
        pos_clip, 
        faces, 
        (img_size, img_size),
        grad_db=True  # Enable gradients for all attributes
    )
    
    # Interpolate colors
    color_interp = dr.interpolate(colors[None, ...], rast_out, faces, rast_db=rast_db,diff_attrs='all')[0]
    
    # Background color
    background = torch.ones_like(color_interp)
    
    output = color_interp
    # Apply built-in transparency blending
    output = dr.antialias(
        color_interp,
        rast_out,
        pos_clip,
        faces,
    )  # Remove batch dimension
    
    # Blend with background
    alpha = output[..., 3:4]
    rgb = output[..., :3]
    final_color = rgb + background[..., :3] * (1 - alpha)
    
    output = torch.cat([final_color, alpha], dim=-1)
    
    output = torch.clamp(output, 0., 1.)
    return output


# def render_with_depth_peeling(glctx, verts, colors, faces, cam_data, img_size, num_layers=4, device="cuda"):
#     """
#     Render with nvdiffrast's DepthPeeler for order-independent transparency
#     """
#     # Transform vertices to clip space
#     pos_clip = torch.matmul(
#         torch.cat([verts, torch.ones_like(verts[:, :1])], dim=1),
#         cam_data["mvp_matrix"].t()
#     )[None, ...]
    
#     # Initialize DepthPeeler
#     background = torch.ones((1, img_size, img_size, 4), device=device)
#     final_color = torch.zeros_like(background)
    
#     # Iterate through layers
#     import time
#     start_time = time.time()
#     with dr.DepthPeeler(glctx, pos_clip, faces, (img_size, img_size)) as peeler:
#         for layer_idx in range(num_layers):
#             # Get rasterization output for current layer
#             rast_out, rast_db = peeler.rasterize_next_layer()
            
#             # Skip if no geometry in this layer
#             if rast_out is None:
#                 break
                
#             # Interpolate colors
#             color_layer = dr.interpolate(
#                 colors[None, ...],
#                 rast_out,
#                 faces,
#                 rast_db=rast_db,
#                 diff_attrs='all'
#             )[0]
            
#             # # Antialias the colors
#             # color_aa = dr.antialias(
#             #     color_layer,
#             #     pos_clip,
#             #     faces,
#             #     rast_out,
#             #     rast_db=rast_db
#             # )
            
#             # Front-to-back blending
#             alpha = color_layer[..., 3:4]
#             rgb = color_layer[..., :3]
#             final_color = final_color + (1.0 - final_color[..., 3:4]) * torch.cat([rgb * alpha, alpha], dim=-1)
#     print(f"Final time: {time.time() - start_time}s")
#     # Blend with background
#     alpha = final_color[..., 3:4]
#     rgb = final_color[..., :3]
#     final_color = rgb + background[..., :3] * (1 - alpha)
    
#     loss = final_color.mean()
#     loss.backward()
    
#     return torch.clamp(final_color, 0., 1.)


def to_torch(*args, device="cpu"):
    return [torch.from_numpy(x).float().to(device) for x in args]
    

def test_render():
    cam_info_path = Path("/home/grzegos/projects/phd/gs_raytracing/data/cam_test/transforms_train.json")
    with open(cam_info_path, "r") as file:
        cam_data = json.load(file)
    
    glctx = dr.RasterizeGLContext()
    img_size = 512
    near = 1e-2
    far = 1e2
    device = "cuda:0"
    
    verts, colors, faces = create_obj(device)
    
    cam_matrices = create_camera_mats(
        cam_data=cam_data,
        img_size=img_size,
        near=near,
        far=far,
        device=device
    )
    
    for cam_name, cam_mats in cam_matrices.items():
        out_img = render_with_depth_peeling(
            glctx=glctx,
            verts=verts,
            colors=colors,
            faces=faces,
            cam_data=cam_mats,
            img_size=img_size,
            num_layers=2,
            device=device
        )
        
        plt.imsave(f"02_{cam_name.split('/')[-1]}", out_img.detach().cpu().numpy()[0])
    

def render_pseudomesh():
    path_to_dataset = Path("/home/grzegos/datasets/games_set/hotdog")
    path_to_camera_data = Path(path_to_dataset, "transforms_train.json")
    path_to_pseudomesh = Path("/home/grzegos/projects/phd/games_nerf_output/hotdog_test_fff_sh0/pseudomeshes/scale_2.30_pts_8.npz")
    
    with open(path_to_camera_data, "r") as file:
        cam_data = json.load(file)
    
    glctx = dr.RasterizeGLContext()
    img_size = 800
    near = 1e-2
    far = 1e2
    device = "cuda:0"
    num_layers = 100
        
    cam_matrices = create_camera_mats(
        cam_data=cam_data,
        img_size=img_size,
        near=near,
        far=far,
        device=device
    )
    
    vertices, faces, vert_colors, face_colors = load_pseudomesh(path_to_pseudomesh)
    
    vertices, faces, vert_colors = to_torch(vertices, faces, vert_colors, device=device)
    vert_colors.requires_grad = True
    faces = faces.int()
    
    for cam_name, cam_mats in cam_matrices.items():
        out_img = render_with_depth_peeling(
            glctx=glctx,
            verts=vertices,
            vert_colors=vert_colors,
            faces=faces,
            cam_data=cam_mats,
            img_size=img_size,
            num_layers=num_layers,
            device=device
        )
    
        plt.imsave(f"{str(num_layers)}_{cam_name.split('/')[-1]}.png", out_img.detach().cpu().numpy()[0])
def render_with_depth_peeling(glctx, verts, vert_colors, faces, cam_data, img_size, num_layers=4, device="cuda"):
    """
    Render with nvdiffrast's DepthPeeler for order-independent transparency.
    Optimized for performance, especially regarding the vert_colors tensor.
    """
    # Transform vertices to clip space
    pos_clip = torch.matmul(
        torch.cat([verts, torch.ones_like(verts[:, :1])], dim=1),
        cam_data["mvp_matrix"].t()
    )[None, ...]
    
    # Initialize DepthPeeler
    background = torch.ones((1, img_size, img_size, 4), device=device)
    final_color = torch.zeros_like(background)
    
    # Iterate through layers
    with dr.DepthPeeler(glctx, pos_clip, faces, (img_size, img_size)) as peeler:
        for layer_idx in range(num_layers):
            # Get rasterization output for current layer
            rast_out, rast_db = peeler.rasterize_next_layer()
            
            # Skip if no geometry in this layer
            if rast_out is None:
                break
                
            # Interpolate colors
            # Use `interp1d` for more efficient color interpolation
            color_layer = dr.interpolate(
                vert_colors[None, ...],
                rast_out,
                faces,
                rast_db=rast_db,
                diff_attrs='all'
            )[0]
            
            # Front-to-back blending
            alpha = color_layer[..., 3:4]
            rgb = color_layer[..., :3]
            final_color = final_color + (1.0 - final_color[..., 3:4]) * torch.cat([rgb * alpha, alpha], dim=-1)
    
    # Blend with background
    alpha = final_color[..., 3:4]
    rgb = final_color[..., :3]
    final_color = rgb + background[..., :3] * (1 - alpha)
    
    loss = final_color.mean()
    loss.backward()
    
    return torch.clamp(final_color, 0., 1.)

if __name__ == "__main__":
    # main()
    render_pseudomesh()
    # Regular render
    # save_render(angle=np.pi/6)
    
    # Or run optimization
    # optimize_colors()

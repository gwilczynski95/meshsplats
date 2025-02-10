import torch
from torch import nn



def hard_prune_mesh(vertices, faces, vertex_color, mask):
    vertices = vertices[~mask]
    vertex_color = vertex_color[~mask]
    
    face_mask = torch.any(mask[faces], dim=1)
    faces = faces[~face_mask]
    
    vertices_to_new_idx = torch.zeros(mask.shape[0], dtype=torch.long, device=vertices.device)
    vertices_to_new_idx[~mask] = torch.arange(vertices.shape[0], device=vertices.device)
    faces = vertices_to_new_idx[faces]
    return vertices, faces, vertex_color, mask


def soft_prune_mesh(vertices, faces, vertex_color, mask):
    face_mask = torch.all(mask[faces], dim=1)
    faces = faces[~face_mask]
    valid_vertices = faces.unique().long()

    valid_verts_mask = torch.zeros_like(vertices[:, 0]).bool()
    valid_verts_mask[valid_vertices] = True

    vertices_to_new_idx = torch.zeros(valid_verts_mask.shape[0], dtype=torch.long, device=vertices.device)
    vertices_to_new_idx[valid_verts_mask] = torch.arange(valid_verts_mask.sum(), device=vertices.device)

    faces = vertices_to_new_idx[faces.long()]
    
    vertices = vertices[valid_verts_mask]
    vertex_color = vertex_color[valid_verts_mask]
    
    return vertices, faces, vertex_color, ~valid_verts_mask


def prune_mesh(vertices, faces, vertex_color, mask, mode):
    assert mode in ["soft", "hard"]
    method = hard_prune_mesh if mode == "hard" else soft_prune_mesh
    vertices, faces, vertex_color, mask = method(
        vertices,
        faces,
        vertex_color,
        mask
    )
    return vertices, faces.int(), vertex_color, mask

def prune_optimizer(optimizer, mask):
    optimizable_tensors = {}
    for group in optimizer.param_groups:
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][~mask]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][~mask]

            del optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter((group["params"][0][~mask].requires_grad_(True)))
            optimizer.state[group['params'][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]
        else:
            group["params"][0] = nn.Parameter(group["params"][0][~mask].requires_grad_(True))
            optimizable_tensors[group["name"]] = group["params"][0]
    return optimizable_tensors

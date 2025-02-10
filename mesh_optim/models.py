import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn as nn


class Pseudomesh(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vert_requires_grad = "vertices" in config["model"]["optimizable_params"]
        self.vert_col_requires_grad = "vertices" in config["model"]["optimizable_params"]
        
        self.register_buffer('faces', torch.tensor(0).int())
        self.vertices = nn.Parameter(torch.tensor(0., requires_grad=self.vert_requires_grad))
        self.vertex_colors = nn.Parameter(torch.tensor(0., requires_grad=self.vert_col_requires_grad))
        self.register_buffer('grad_acc', torch.tensor(0).float())
        self.register_buffer('grad_denom', torch.tensor(0).int())

class PseudomeshRenderer():
    def __init__(self, config):
        self.pseudomesh = Pseudomesh(config)
        self.glctx = dr.RasterizeGLContext()
    
    @classmethod
    def create_model(cls, config):
        mesh_data = np.load(config["pseudomesh_path"])
        obj = cls(config)
        obj.set_values(mesh_data["vertices"], mesh_data["faces"], mesh_data["vertex_colors"])
        return obj
    
    @property
    def vert_requires_grad(self):
        return self.pseudomesh.vert_requires_grad
    
    @property
    def vert_col_requires_grad(self):
        return self.pseudomesh.vert_col_requires_grad
    
    def set_values(self, vertices, faces, vertex_colors, grad_acc=None):
        if isinstance(vertices, np.ndarray) and not isinstance(vertices, torch.nn.parameter.Parameter):
            vertices = torch.tensor(vertices, dtype=torch.float32, requires_grad=self.vert_requires_grad)
        if isinstance(faces, np.ndarray):
            faces = torch.tensor(faces, dtype=torch.int32)
        if isinstance(vertex_colors, np.ndarray) and not isinstance(vertex_colors, torch.nn.parameter.Parameter):
            vertex_colors = torch.tensor(vertex_colors, dtype=torch.float32, requires_grad=self.vert_col_requires_grad)
        self.pseudomesh.faces.data = faces
        
        if isinstance(vertices, torch.nn.parameter.Parameter):
            self.pseudomesh.vertices = vertices
        else:
            self.pseudomesh.vertices = nn.Parameter(vertices)
        
        if isinstance(vertex_colors, torch.nn.parameter.Parameter):
            self.pseudomesh.vertex_colors = vertex_colors
        else:
            self.pseudomesh.vertex_colors = nn.Parameter(vertex_colors)
            
        if grad_acc is None:
            grad_acc = torch.zeros_like(self.pseudomesh.vertex_colors[:,0])
        self.pseudomesh.grad_acc.data = grad_acc
    
    def acc_grad(self):
        grad = torch.norm(self.pseudomesh.vertex_colors.grad, dim=-1)
        self.pseudomesh.grad_acc.data = self.pseudomesh.grad_acc.data + grad
        self.pseudomesh.grad_denom.data = self.pseudomesh.grad_denom.data + 1
    
    def reset_acc_grad(self):
        self.pseudomesh.grad_acc.data = torch.zeros_like(self.pseudomesh.vertex_colors[:,0])
        self.pseudomesh.grad_denom.data = torch.tensor(0).int()
    
    def to(self, device):
        self.pseudomesh = self.pseudomesh.to(device)
        return self
    
    def __call__(self, mvp_mat, width, height, num_layers):
        pos_clip =         torch.matmul(
            torch.cat(
                [
                    self.pseudomesh.vertices,
                    torch.ones_like(self.pseudomesh.vertices[:, :1])
                ],
                dim=1
            ),
            mvp_mat.permute(0, 2, 1)
        )
        
        final_color = torch.zeros((mvp_mat.shape[0], height, width, 4), device=self.pseudomesh.vertices.device, dtype=torch.float32)
        
        with dr.DepthPeeler(self.glctx, pos_clip, self.pseudomesh.faces, (height, width)) as peeler:
            for layer_idx in range(num_layers):
                rast_out, rast_db = peeler.rasterize_next_layer()
                
                if rast_out is None:
                    break
                
                color_layer = dr.interpolate(
                    self.pseudomesh.vertex_colors[None, ...],
                    rast_out,
                    self.pseudomesh.faces,
                    rast_db=rast_db,
                    diff_attrs='all'
                )[0]
                
                alpha = color_layer[..., 3:4]
                rgb = color_layer[..., :3]
                final_color = final_color + (1.0 - final_color[..., 3:4]) * torch.cat([rgb * alpha, alpha], dim=-1)
        
        alpha = final_color[..., 3:4]
        rgb = final_color[..., :3]
        return rgb, alpha

    def get_depth_map(self, mvp_mat, width, height, cam_pos, num_layers):
        vertices = self.pseudomesh.vertices  # shape: (N, 3)
        vertex_dist = torch.norm(vertices - cam_pos, dim=1, keepdim=True)  # shape: (N, 1)
        
        dmin = vertex_dist.min()
        dmax = vertex_dist.max()
        vertex_dist_norm = (vertex_dist - dmin) / (dmax - dmin + 1e-8)

        vertex_colors = torch.cat([
            vertex_dist_norm, 
            vertex_dist_norm, 
            vertex_dist_norm, 
            self.pseudomesh.vertex_colors[..., -1:]
        ], dim=1)  # shape: (N, 4)
        
        vertices_homo = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=1)
        pos_clip = torch.matmul(vertices_homo, mvp_mat.permute(0, 2, 1))
        
        final_color = torch.zeros((mvp_mat.shape[0], height, width, 4), device=self.pseudomesh.vertices.device, dtype=torch.float32)
        
        with dr.DepthPeeler(self.glctx, pos_clip, self.pseudomesh.faces, (height, width)) as peeler:
            for layer_idx in range(num_layers):
                rast_out, rast_db = peeler.rasterize_next_layer()
                
                if rast_out is None:
                    break
                
                color_layer = dr.interpolate(
                    vertex_colors[None, ...],
                    rast_out,
                    self.pseudomesh.faces,
                    rast_db=rast_db,
                    diff_attrs='all'
                )[0]
                
                alpha = color_layer[..., 3:4]
                rgb = color_layer[..., :3]
                final_color = final_color + (1.0 - final_color[..., 3:4]) * torch.cat([rgb * alpha, alpha], dim=-1)

        alpha = final_color[..., 3:4]
        rgb = final_color[..., :3]
        return rgb, alpha
    
    def get_gray_map(self, mvp_mat, width, height, num_layers, color_verts):
        pos_clip = torch.matmul(
            torch.cat(
                [
                    self.pseudomesh.vertices,
                    torch.ones_like(self.pseudomesh.vertices[:, :1])
                ],
                dim=1
            ),
            mvp_mat.permute(0, 2, 1)
        )
        
        final_color = torch.zeros((mvp_mat.shape[0], height, width, 4), device=self.pseudomesh.vertices.device, dtype=torch.float32)
        
        with dr.DepthPeeler(self.glctx, pos_clip, self.pseudomesh.faces, (height, width)) as peeler:
            for layer_idx in range(num_layers):
                rast_out, rast_db = peeler.rasterize_next_layer()
                
                if rast_out is None:
                    break
                
                color_layer = dr.interpolate(
                    color_verts[None, ...],
                    rast_out,
                    self.pseudomesh.faces,
                    rast_db=rast_db,
                    diff_attrs='all'
                )[0]
                alpha = color_layer[..., 3:4]
                rgb = color_layer[..., :3]
                final_color = final_color + (1.0 - final_color[..., 3:4]) * torch.cat([rgb * alpha, alpha], dim=-1)
        
        alpha = final_color[..., 3:4]
        rgb = final_color[..., :3]
        return rgb, alpha

    def get_normal_map(self, mvp_mat, width, height, num_layers):
        vertices = self.pseudomesh.vertices
        faces = self.pseudomesh.faces
        
        # Get face vertices
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        
        # Calculate face normals
        face_normals = torch.cross(v1 - v0, v2 - v0)
        face_normals = face_normals / (torch.norm(face_normals, dim=1, keepdim=True) + 1e-8)
        
        # Initialize vertex normals
        vertex_normals = torch.zeros_like(vertices)
        
        # Accumulate face normals to vertices
        for i in range(3):
            vertex_indices = faces[:, i]
            vertex_normals.index_add_(0, vertex_indices, face_normals)
        
        # Normalize vertex normals
        vertex_normals = vertex_normals / (torch.norm(vertex_normals, dim=1, keepdim=True) + 1e-8)
        
        # Rest of the rendering process remains the same
        pos_clip = torch.matmul(
            torch.cat(
                [
                    vertices,
                    torch.ones_like(vertices[:, :1])
                ],
                dim=1
            ),
            mvp_mat.permute(0, 2, 1)
        )
        
        vertex_attrs = torch.cat([
            vertex_normals,
            self.pseudomesh.vertex_colors[..., -1:]
        ], dim=1)
        
        final_color = torch.zeros((mvp_mat.shape[0], height, width, 4), 
                                device=vertices.device, 
                                dtype=torch.float32)
        
        with dr.DepthPeeler(self.glctx, pos_clip, faces, (height, width)) as peeler:
            for layer_idx in range(num_layers):
                rast_out, rast_db = peeler.rasterize_next_layer()
                
                if rast_out is None:
                    break
                
                color_layer = dr.interpolate(
                    vertex_attrs[None, ...],
                    rast_out,
                    faces,
                    rast_db=rast_db,
                    diff_attrs='all'
                )[0]
                
                alpha = color_layer[..., 3:4]
                normals = color_layer[..., :3]
                normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-8)
                final_color = final_color + (1.0 - final_color[..., 3:4]) * torch.cat([normals * alpha, alpha], dim=-1)
        
        alpha = final_color[..., 3:4]
        rgb = final_color[..., :3]
        return rgb, alpha

    def render_all_maps(self, mvp_mat, width, height, cam_pos, num_layers, gray_verts=None):
        vertices = self.pseudomesh.vertices
        faces = self.pseudomesh.faces
        
        # Calculate vertex normals
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        
        face_normals = torch.cross(v1 - v0, v2 - v0)
        face_normals = face_normals / (torch.norm(face_normals, dim=1, keepdim=True) + 1e-8)
        
        vertex_normals = torch.zeros_like(vertices)
        for i in range(3):
            vertex_indices = faces[:, i]
            vertex_normals.index_add_(0, vertex_indices, face_normals)
        vertex_normals = vertex_normals / (torch.norm(vertex_normals, dim=1, keepdim=True) + 1e-8)
        
        # Calculate depth values
        vertex_dist = torch.norm(vertices - cam_pos, dim=1, keepdim=True)
        dmin, dmax = vertex_dist.min(), vertex_dist.max()
        vertex_dist_norm = (vertex_dist - dmin) / (dmax - dmin + 1e-8)
        depth_colors = torch.cat([
            vertex_dist_norm.repeat(1, 3),
            self.pseudomesh.vertex_colors[..., -1:]
        ], dim=1)
        
        # Prepare normal colors
        normal_colors = torch.cat([
            vertex_normals,
            self.pseudomesh.vertex_colors[..., -1:]
        ], dim=1)
        
        # Initialize output tensors
        device = vertices.device
        shape = (mvp_mat.shape[0], height, width, 4)
        final_color = torch.zeros(shape, device=device, dtype=torch.float32)
        final_normal = torch.zeros(shape, device=device, dtype=torch.float32)
        final_depth = torch.zeros(shape, device=device, dtype=torch.float32)
        final_gray = torch.zeros(shape, device=device, dtype=torch.float32) if gray_verts is not None else None
        
        # Transform vertices to clip space
        pos_clip = torch.matmul(
            torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=1),
            mvp_mat.permute(0, 2, 1)
        )
        
        with dr.DepthPeeler(self.glctx, pos_clip, faces, (height, width)) as peeler:
            for _ in range(num_layers):
                rast_out, rast_db = peeler.rasterize_next_layer()
                
                if rast_out is None:
                    break
                
                # Interpolate all attributes at once
                color_layer = dr.interpolate(
                    self.pseudomesh.vertex_colors[None, ...],
                    rast_out, faces, rast_db=rast_db, diff_attrs='all'
                )[0]
                
                normal_layer = dr.interpolate(
                    normal_colors[None, ...],
                    rast_out, faces, rast_db=rast_db, diff_attrs='all'
                )[0]
                
                depth_layer = dr.interpolate(
                    depth_colors[None, ...],
                    rast_out, faces, rast_db=rast_db, diff_attrs='all'
                )[0]
                
                # Process color map
                alpha = color_layer[..., 3:4]
                rgb = color_layer[..., :3]
                final_color = final_color + (1.0 - final_color[..., 3:4]) * torch.cat([rgb * alpha, alpha], dim=-1)
                
                # Process normal map
                normals = normal_layer[..., :3]
                normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-8)
                final_normal = final_normal + (1.0 - final_normal[..., 3:4]) * torch.cat([normals * alpha, alpha], dim=-1)
                
                # Process depth map
                depth_rgb = depth_layer[..., :3]
                final_depth = final_depth + (1.0 - final_depth[..., 3:4]) * torch.cat([depth_rgb * alpha, alpha], dim=-1)
                
                # Process gray map if gray_verts provided
                if gray_verts is not None:
                    gray_layer = dr.interpolate(
                        gray_verts[None, ...],
                        rast_out, faces, rast_db=rast_db, diff_attrs='all'
                    )[0]
                    gray_rgb = gray_layer[..., :3]
                    final_gray = final_gray + (1.0 - final_gray[..., 3:4]) * torch.cat([gray_rgb * alpha, alpha], dim=-1)
        
        results = {
            'color': (final_color[..., :3], final_color[..., 3:4]),
            'normal': (final_normal[..., :3], final_normal[..., 3:4]),
            'depth': (final_depth[..., :3], final_depth[..., 3:4])
        }
        
        if gray_verts is not None:
            results['gray'] = (final_gray[..., :3], final_gray[..., 3:4])
        
        return results

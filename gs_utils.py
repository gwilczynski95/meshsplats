import numpy as np
from plyfile import PlyData
from scipy.special import expit as sigmoid
import torch

try:
    import sys
    sys.path.append("/path/to/GaMeS_repo")
    import games
except Exception as e:
    print(e)

C0 = 0.28209479177387814


def load_games_pt(path):
    params = torch.load(path)
    alpha = params['_alpha']
    scale = params['_scale']
    vertices, triangles, faces = None, None, None
    if 'vertices' in params:
        vertices = params['vertices']
    if 'triangles' in params:
        triangles = params['triangles']
    if 'faces' in params:
        faces = params['faces']
    return {
        "alpha": alpha,
        "scale": scale,
        "vertices": vertices,
        "triangles": triangles,
        "faces": faces
    }


def get_games_scales_and_rots(data, eps=1e-8):
    def dot(v, u):
        return (v * u).sum(dim=-1, keepdim=True)
    
    def proj(v, u):
        """
        projection of vector v onto subspace spanned by u

        vector u is assumed to be already normalized
        """
        coef = dot(v, u)
        return coef * u
    
    triangles = data["triangles"]
    normals = torch.linalg.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
        dim=1
    )
    v0 = normals / (torch.linalg.vector_norm(normals, dim=-1, keepdim=True) + eps)
    means = torch.mean(triangles, dim=1)
    v1 = triangles[:, 1] - means
    v1_norm = torch.linalg.vector_norm(v1, dim=-1, keepdim=True) + eps
    v1 = v1 / v1_norm
    v2_init = triangles[:, 2] - means
    v2 = v2_init - proj(v2_init, v0) - proj(v2_init, v1)  # Gram-Schmidt
    v2 = v2 / (torch.linalg.vector_norm(v2, dim=-1, keepdim=True) + eps)

    s1 = v1_norm / 2.
    s2 = dot(v2_init, v2) / 2.
    
    scales = torch.concat([s1, s2], dim=1).unsqueeze(dim=1)
    scales = scales.broadcast_to((*data["alpha"].shape[:2], 2))
    scales = torch.nn.functional.relu(data["scale"] * scales.flatten(start_dim=0, end_dim=1)) + eps
    # scales [N, 2]
    
    rots = torch.stack((v1, v2), dim=1).unsqueeze(dim=1)
    rots = rots.broadcast_to((*data["alpha"].shape[:2], 2, 3)).flatten(start_dim=0, end_dim=1)
    # rots [N, 2, 3]
    
    return scales, rots


def load_ply(path, max_sh_degree):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    features_dc = np.transpose(features_dc, [0, 2, 1])  # TODO check if this is ok
    features_rest = np.transpose(features_extra, [0, 2, 1])  # TODO check if this is ok

    return xyz, features_dc, features_rest, opacities, scales, rots


def normalize_rots(mat):
    return mat / np.linalg.norm(mat, axis=1, keepdims=True)

def build_euler_rotation(r):
    norm = np.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = np.zeros((q.shape[0], 3, 3), dtype=np.float64)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def get_scaling(scales: np.ndarray) -> np.ndarray:
    return np.exp(scales)

def get_opacity(opacities: np.ndarray) -> np.ndarray:
    return sigmoid(opacities)


def generate_pseudomesh_games(ckpt_data: dict, xyz: np.ndarray, features_dc: np.ndarray, opacities: np.ndarray, scales: np.ndarray, rots: np.ndarray, scale_muls: np.ndarray, no_of_points: int = 4):
    if ckpt_data is not None:
        scale, rotation = get_games_scales_and_rots(ckpt_data)
        scale = scale.cpu().detach().numpy()
        rotation = rotation.cpu().detach().numpy()
    else:
        scale = get_scaling(scales)[:, :2]
        rotation = np.transpose(build_euler_rotation(rots)[:, :, 1:], [0, 2, 1])
    init_colors = get_rgb_colors(features_dc)
    init_opacities = get_opacity(opacities)
    origin = xyz
    
    output = gen_2d_pseudomesh(scale_muls, no_of_points, init_colors, init_opacities, scale, rotation, origin)
    return output


def generate_pseudomesh_surfels(xyz: np.ndarray, features_dc: np.ndarray, opacities: np.ndarray, scales: np.ndarray, rots: np.ndarray, scale_muls: np.ndarray, no_of_points: int = 4):
    init_colors = get_rgb_colors(features_dc)
    init_opacities = get_opacity(opacities)
    scale = get_scaling(scales)[:, :2]
    rotation = np.transpose(build_euler_rotation(rots)[:, :, :2], [0, 2, 1])
    origin = xyz
    
    output = gen_2d_pseudomesh(scale_muls, no_of_points, init_colors, init_opacities, scale, rotation, origin)
    return output


def generate_pseudomesh_2dgs(xyz: np.ndarray, features_dc: np.ndarray, opacities: np.ndarray, scales: np.ndarray, rots: np.ndarray, scale_muls: np.ndarray, no_of_points: int = 4):
    init_colors = get_rgb_colors(features_dc)
    init_opacities = get_opacity(opacities)
    scale = get_scaling(scales)
    rotation = np.transpose(build_euler_rotation(rots)[:, :, :2], [0, 2, 1])
    origin = xyz
    
    output = gen_2d_pseudomesh(scale_muls, no_of_points, init_colors, init_opacities, scale, rotation, origin)
    return output

def generate_pseudomesh_sugar_2d(xyz: np.ndarray, features_dc: np.ndarray, opacities: np.ndarray, scales: np.ndarray, rots: np.ndarray, scale_muls: np.ndarray, no_of_points: int = 8):
    init_colors = get_rgb_colors(features_dc)
    init_opacities = get_opacity(opacities)
    scale = get_scaling(scales)
    rotation = np.transpose(build_euler_rotation(rots), [0, 2, 1])
    origin = xyz
    
    scale = scale[:, 1:]
    rotation = rotation[:, 1:, :]
    
    output = gen_2d_pseudomesh(scale_muls, no_of_points, init_colors, init_opacities, scale, rotation, origin)
    return output

def generate_pseudomesh_sugar_3d(xyz: np.ndarray, features_dc: np.ndarray, opacities: np.ndarray, scales: np.ndarray, rots: np.ndarray, scale_muls: np.ndarray, no_of_points: int = 8, opac_mul: float = .2):
    assert no_of_points == 8
    # handle background generation
    init_colors = get_rgb_colors(features_dc)
    init_opacities = get_opacity(opacities)
    scale = get_scaling(scales)
    rotation = np.transpose(build_euler_rotation(rots), [0, 2, 1])
    origin = xyz
    
    scales_1 = scale[:, [0, 1]]
    rots_1 = rotation[:, [0, 1], :]
    
    scales_2 = scale[:, [1, 2]]
    rots_2 = rotation[:, [1, 2], :]
    
    scales_3 = scale[:, [0, 2]]
    rots_3 = rotation[:, [0, 2], :]
    
    output = {}
    
    for scale_mul in scale_muls:
        new_points_1 = _get_vertices(origin, scales_1, rots_1, scale_mul, no_of_points)
        new_points_2 = _get_vertices(origin, scales_2, rots_2, scale_mul, no_of_points)
        new_points_3 = _get_vertices(origin, scales_3, rots_3, scale_mul, no_of_points)
        
        # remove redundant verts
        new_points_2 = new_points_2[:, [1, 2, 3, 5, 6, 7], :]
        new_points_3 = new_points_3[:, [1, 3, 5, 7], :]
        
        vertices = np.vstack([origin, *[new_points_1[:, _idx, :] for _idx in range(new_points_1.shape[1])]])
        vertices = np.vstack([vertices, *[new_points_2[:, _idx, :] for _idx in range(new_points_2.shape[1])]])
        vertices = np.vstack([vertices, *[new_points_3[:, _idx, :] for _idx in range(new_points_3.shape[1])]])
        
        origin_idxs = np.arange(0, origin.shape[0]).reshape(-1, 1)
        indexes_1 = np.hstack([
            np.arange(_idx * origin.shape[0], (_idx + 1) * origin.shape[0]).reshape(-1, 1) for _idx in range(1, 9)
        ])
        indexes_2 = np.hstack([
            np.arange(_idx * origin.shape[0], (_idx + 1) * origin.shape[0]).reshape(-1, 1) for _idx in [3, 9, 10, 11, 7, 12, 13, 14]
        ])
        indexes_3 = np.hstack([
            np.arange(_idx * origin.shape[0], (_idx + 1) * origin.shape[0]).reshape(-1, 1) for _idx in [1, 15, 10, 16, 5, 17, 13, 18]
        ])
        
        # create faces
        all_faces = []
        
        # faces 1
        for face_idx in range(no_of_points):
            last_idx = face_idx + 1 if face_idx + 1 < no_of_points else 0
            all_faces.append(
                np.hstack([origin_idxs, indexes_1[:, face_idx].reshape(-1, 1), indexes_1[:, last_idx].reshape(-1, 1)])
            )
        
        # faces 2
        for face_idx in range(no_of_points):
            last_idx = face_idx + 1 if face_idx + 1 < no_of_points else 0
            all_faces.append(
                np.hstack([origin_idxs, indexes_2[:, face_idx].reshape(-1, 1), indexes_2[:, last_idx].reshape(-1, 1)])
            )

        # faces 3
        for face_idx in range(no_of_points):
            last_idx = face_idx + 1 if face_idx + 1 < no_of_points else 0
            all_faces.append(
                np.hstack([origin_idxs, indexes_3[:, face_idx].reshape(-1, 1), indexes_3[:, last_idx].reshape(-1, 1)])
            )

        all_faces = np.vstack(all_faces)
        
        vertex_colors = np.vstack([init_colors] * 19)
        face_colors = np.vstack([init_colors] * 24)
        
        # vertex_opacities = np.vstack([init_opacities] * (no_of_points + 1))
        vertex_opacities = np.vstack([init_opacities, *[init_opacities * opac_mul] * 18])
        # face_opacities = np.vstack([init_opacities] * no_of_points)
        face_opacities = np.vstack([init_opacities] * 24)
        
        vertex_rgba = np.hstack([vertex_colors, vertex_opacities])
        face_rgba = np.hstack([face_colors, face_opacities])
        output[scale_mul] = dict(
            vertices=vertices,
            vertex_colors=vertex_rgba,
            faces=all_faces,
            face_colors=face_rgba
        )
    
    return output


def _get_vertices(origin, scales, rots, scale_mul, no_of_points):
    scales = scale_mul * np.repeat(np.repeat(scales[:, None, :, None], no_of_points, 1), 3, -1)
    rots = np.repeat(rots[:, None, :, :], no_of_points, 1)
    scaled_rots = scales * rots
    
    angle = np.linspace(-np.pi, np.pi, no_of_points + 1)[:-1]
    sins = np.repeat(np.repeat((np.sin(angle).reshape(-1, 1)[None, :, :]), origin.shape[0], 0), 3, -1)
    coss = np.repeat(np.repeat((np.cos(angle).reshape(-1, 1)[None, :, :]), origin.shape[0], 0), 3, -1)
    
    _origin = np.repeat(origin[:, None, :], no_of_points, 1)
    new_points = _origin + coss * scaled_rots[:, :, 0, :] + sins * scaled_rots[:, :, 1, :]
    return new_points

def gen_2d_pseudomesh(scale_muls, no_of_points, init_colors, init_opacities, scale, rotation, origin, opac_mul=.2):
    angle = np.linspace(-np.pi, np.pi, no_of_points + 1)[:-1]
    sins = np.repeat(np.repeat((np.sin(angle).reshape(-1, 1)[None, :, :]), origin.shape[0], 0), 3, -1)
    coss = np.repeat(np.repeat((np.cos(angle).reshape(-1, 1)[None, :, :]), origin.shape[0], 0), 3, -1)

    _origin = np.repeat(origin[:, None, :], no_of_points, 1)
    
    output = {}
    for scale_mul in scale_muls:
        scales = scale_mul * np.repeat(np.repeat(scale[:, None, :, None], no_of_points, 1), 3, -1)
        rots = np.repeat(rotation[:, None, :, :], no_of_points, 1)
        scaled_rots = scales * rots
        
        new_points = _origin + coss * scaled_rots[:, :, 0, :] + sins * scaled_rots[:, :, 1, :]
        
        vertices = np.vstack([origin, *[new_points[:, _idx, :] for _idx in range(no_of_points)]])
        origin_idxs = np.arange(0, origin.shape[0]).reshape(-1, 1)
        indexes = np.hstack([np.arange(_idx * origin.shape[0], (_idx + 1) * origin.shape[0]).reshape(-1, 1) for _idx in range(1, no_of_points + 1)])

        # create faces
        all_faces = []
        for face_idx in range(no_of_points):
            last_idx = face_idx + 1 if face_idx + 1 < no_of_points else 0
            all_faces.append(
                np.hstack([origin_idxs, indexes[:, face_idx].reshape(-1, 1), indexes[:, last_idx].reshape(-1, 1)])
            )
        all_faces = np.vstack(all_faces)
        
        vertex_colors = np.vstack([init_colors] * (no_of_points + 1))
        face_colors = np.vstack([init_colors] * no_of_points)
        
        vertex_opacities = np.vstack([init_opacities, *[init_opacities * opac_mul] * no_of_points])
        face_opacities = np.vstack([init_opacities] * no_of_points)
        
        vertex_rgba = np.hstack([vertex_colors, vertex_opacities])
        face_rgba = np.hstack([face_colors, face_opacities])
        
        output[scale_mul] = dict(
            vertices=vertices, 
            vertex_colors=vertex_rgba, 
            faces=all_faces, 
            face_colors=face_rgba
        )
        
    return output
    

def get_rgb_colors(color_features):
    colors = np.transpose(color_features, [0, 2, 1])
    result = C0 * colors[..., 0] + 0.5
    result = result.clip(min=0., max=1.)
    return result


def generate_3dgs_pseudomesh(xyz: np.ndarray, features_dc: np.ndarray, opacities: np.ndarray, scales: np.ndarray, rots: np.ndarray, scale_muls: np.ndarray, no_of_points: int = 8):
    assert no_of_points == 8
    
    init_colors = get_rgb_colors(features_dc)
    init_opacities = get_opacity(opacities)
    scale = get_scaling(scales)
    rotation = np.transpose(build_euler_rotation(rots), [0, 2, 1])
    origin = xyz
    
    scales_1 = scale[:, [0, 1]]
    rots_1 = rotation[:, [0, 1], :]
    
    scales_2 = scale[:, [1, 2]]
    rots_2 = rotation[:, [1, 2], :]
    
    scales_3 = scale[:, [0, 2]]
    rots_3 = rotation[:, [0, 2], :]
    
    output = {}
    for scale_mul in scale_muls:
        new_points_1 = _get_vertices(origin, scales_1, rots_1, scale_mul, no_of_points)
        new_points_2 = _get_vertices(origin, scales_2, rots_2, scale_mul, no_of_points)
        new_points_3 = _get_vertices(origin, scales_3, rots_3, scale_mul, no_of_points)
        
        # remove redundant verts
        new_points_2 = new_points_2[:, [1, 2, 3, 5, 6, 7], :]
        new_points_3 = new_points_3[:, [1, 3, 5, 7], :]
        
        vertices = np.vstack([origin, *[new_points_1[:, _idx, :] for _idx in range(new_points_1.shape[1])]])
        vertices = np.vstack([vertices, *[new_points_2[:, _idx, :] for _idx in range(new_points_2.shape[1])]])
        vertices = np.vstack([vertices, *[new_points_3[:, _idx, :] for _idx in range(new_points_3.shape[1])]])
        
        origin_idxs = np.arange(0, origin.shape[0]).reshape(-1, 1)
        indexes_1 = np.hstack([
            np.arange(_idx * origin.shape[0], (_idx + 1) * origin.shape[0]).reshape(-1, 1) for _idx in range(1, 9)
        ])
        indexes_2 = np.hstack([
            np.arange(_idx * origin.shape[0], (_idx + 1) * origin.shape[0]).reshape(-1, 1) for _idx in [3, 9, 10, 11, 7, 12, 13, 14]
        ])
        indexes_3 = np.hstack([
            np.arange(_idx * origin.shape[0], (_idx + 1) * origin.shape[0]).reshape(-1, 1) for _idx in [1, 15, 10, 16, 5, 17, 13, 18]
        ])
        
        # create faces
        all_faces = []
        
        # faces 1
        for face_idx in range(no_of_points):
            last_idx = face_idx + 1 if face_idx + 1 < no_of_points else 0
            all_faces.append(
                np.hstack([origin_idxs, indexes_1[:, face_idx].reshape(-1, 1), indexes_1[:, last_idx].reshape(-1, 1)])
            )
        
        # faces 2
        for face_idx in range(no_of_points):
            last_idx = face_idx + 1 if face_idx + 1 < no_of_points else 0
            all_faces.append(
                np.hstack([origin_idxs, indexes_2[:, face_idx].reshape(-1, 1), indexes_2[:, last_idx].reshape(-1, 1)])
            )

        # faces 3
        for face_idx in range(no_of_points):
            last_idx = face_idx + 1 if face_idx + 1 < no_of_points else 0
            all_faces.append(
                np.hstack([origin_idxs, indexes_3[:, face_idx].reshape(-1, 1), indexes_3[:, last_idx].reshape(-1, 1)])
            )

        all_faces = np.vstack(all_faces)
        
        vertex_colors = np.vstack([init_colors] * 19)
        face_colors = np.vstack([init_colors] * 24)
        
        vertex_opacities = np.vstack([init_opacities, *[init_opacities * 0.2] * 18])
        face_opacities = np.vstack([init_opacities] * 24)
        
        vertex_rgba = np.hstack([vertex_colors, vertex_opacities])
        face_rgba = np.hstack([face_colors, face_opacities])
        
        output[scale_mul] = dict(
            vertices=vertices,
            vertex_colors=vertex_rgba,
            faces=all_faces,
            face_colors=face_rgba
        )
    
    return output

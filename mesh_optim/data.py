import json
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
import numpy as np

from cam_utils import qvec2rotmat, read_extrinsics_binary, read_intrinsics_binary


def create_blend_proj_mats(camera_angle_x, img_shape, transf_mat, far, near):
    focal = img_shape[0] / (2 * np.tan(camera_angle_x / 2))
    transf_matrix = torch.tensor(transf_mat, device="cpu", dtype=torch.float32)
    view_matrix = torch.inverse(transf_matrix)

    proj_matrix = torch.zeros(4, 4, device="cpu", dtype=torch.float32)
    proj_matrix[0, 0] = 2 * focal / img_shape[0]
    proj_matrix[1, 1] = -2 * focal / img_shape[1]
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


def create_colmap_proj_mats(focal_x, focal_y, img_shape, transf_mat, far, near):  # TODO test it bruh
    transf_matrix = torch.tensor(transf_mat, device="cpu", dtype=torch.float32)
    view_matrix = torch.inverse(transf_matrix)
    
    proj_matrix = torch.zeros(4, 4, device="cpu", dtype=torch.float32)
    proj_matrix[0, 0] = 2 * focal_x / img_shape[0]
    proj_matrix[1, 1] = -2 * focal_y / img_shape[1]
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
    

class ImageCamDataset(Dataset):
    def __init__(self, dataset_path, near, far, imgs_in_ram=True, res=1, test=False):
        self.data_dir = Path(dataset_path)
        self.near = near
        self.far = far
        self._imgs_in_ram = imgs_in_ram
        self._res = res
        self.test = test
        self.mode = None
        self.shape = None
        # print(self.test)
        if Path(dataset_path, "transforms_train.json").exists():
            self.mode = "blender"
        else:
            self.mode = "colmap"
        self.load_cameras()
        # print(len(self))
        
    
    def _load_blender_cameras(self):
        if self.test:
            self._cam_info_path = Path(self.data_dir, "transforms_test.json")
        else:
            self._cam_info_path = Path(self.data_dir, "transforms_train.json")
        # Load camera parameters from JSON file
        with open(self._cam_info_path, 'r') as f:
            self.camera_data = json.load(f)
        
        self.image_files = []
        _cam_angle_x = self.camera_data["camera_angle_x"]
        for _idx, _cam in enumerate(self.camera_data["frames"]):
            img_path = Path(self.data_dir, _cam["file_path"]).with_suffix(".png")
            img = None
            if not _idx or self._imgs_in_ram:
                img, width, height = self.load_image(img_path, get_shape=True)
                self.shape = [width, height]
            trans_mat = np.array(_cam["transform_matrix"], dtype=np.float32)
            proj_mats = create_blend_proj_mats(
                _cam_angle_x,
                self.shape,
                trans_mat,
                self.far,
                self.near
            )
            
            self.image_files.append({
                "name": img_path.name,
                "img_path": str(img_path),
                "img": img,
                **proj_mats
            })
    
    def _load_colmap_cameras(self):
        print(self._res)
        try:
            cameras_extrinsic_file = Path(self.data_dir, "sparse/0", "images.bin")
            cameras_intrinsic_file = Path(self.data_dir, "sparse/0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
            raise NotImplementedError(".txt Colmap structure not supported")
        
        image_files = []
        for _idx, key in enumerate(cam_extrinsics):
            extr = cam_extrinsics[key]
            intr = cam_intrinsics[extr.camera_id]
            height = intr.height
            width = intr.width
            img_path = Path(self.data_dir, "images", extr.name)
            
            img = None
            if not _idx or self._imgs_in_ram:
                img, width, height = self.load_image(img_path, self._res, get_shape=True)
                self.shape = [width, height]
            elif self._res != 1:
                height = round(height / self._res)
                width = round(width / self._res)

            R = qvec2rotmat(extr.qvec)
            T = np.array(extr.tvec)
            transf_mat = np.eye(4, dtype=np.float32)
            transf_mat[:3, :3] = R
            transf_mat[:3, -1] = T
            
            transf_mat = np.linalg.inv(transf_mat)
            transf_mat[:3, 1:3] *= -1

            if intr.model=="SIMPLE_PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = focal_length_y
            elif intr.model=="PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
            else:
                assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
            
            # focal_x = focal_length_x / (2 * self._res)  # TODO FIX FOR TANDT
            focal_x = focal_length_x / self._res
            # focal_y = focal_length_y / (2 * self._res)
            focal_y = focal_length_y / self._res
            
            proj_mats = create_colmap_proj_mats(
                focal_x,
                focal_y,
                self.shape,
                transf_mat,
                self.far,
                self.near
            )
            
            image_files.append({
                "name": extr.name,
                "img_path": str(img_path),
                "img": img,
                "focal_x": focal_x,
                "focal_y": focal_y,
                **proj_mats
            })
        sorted_imgs = sorted(image_files, key=lambda x: x["name"].split(".")[0])
        if self.test:
            cond = lambda x: x % 8 == 0
        else:
            cond = lambda x: x % 8 != 0
        self.image_files = [c for idx, c in enumerate(sorted_imgs) if cond(idx)]
                
    
    def load_cameras(self):
        if self.mode == "blender":
            self._load_blender_cameras()
        else: 
            self._load_colmap_cameras()
    
    @staticmethod
    def load_image(path, res=None, get_shape=False):
        _img = Image.open(path)
        if res is not None and res != 1:
            _img.thumbnail(
                [
                    round(_img.width / res),
                    round(_img.height / res)
                ],
                Image.Resampling.LANCZOS
            )
        img = np.array(_img).astype(np.float32) / 255
        img = torch.from_numpy(img)
        if get_shape:
            return img, _img.width, _img.height
        return img
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Returns:
            dict: Contains image and camera parameters
        """
        if self.image_files[idx]["img"] is None:
            img = self.load_image(self.image_files[idx]["img_path"], self._res)
            self.image_files[idx]["img"] = img
        self.image_files[idx]["img"] = self.image_files[idx]["img"]
        return self.image_files[idx]

# # Example usage:
# if __name__ == "__main__":
    
#     # Create dataset instance
#     dataset = ImageCamDataset(
#         dataset_path="/path/to/dataset",
#         near=0.01,
#         far=100.,
#         imgs_in_ram=False
#     )
    
#     # Get a sample
#     sample = dataset[0]
#     print(f"Image shape: {sample['image'].shape}")
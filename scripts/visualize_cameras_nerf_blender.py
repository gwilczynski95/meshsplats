from pathlib import Path

import bpy
import bpy.ops
import json
from math import tan, atan
from mathutils import Matrix

import numpy as np


sensor_width = 36.0  # mm, typical of a full-frame camera
sensor_height = 24.0  # mm, typical of a full-frame camera


def _calculate_fov(focal, pixels):
    return 2 * np.arctan(pixels / (2 * focal))


def create_camera(name, position, rot_matrix, fovx, fovy, image_width, image_height):
    # Create the camera object
    bpy.ops.object.camera_add()
    camera_obj = bpy.context.object
    camera_obj.name = str(name)
    camera_obj.location = position
    
    # Set rotation from matrix
    rotation_matrix = Matrix(rot_matrix)
    camera_obj.rotation_euler = rotation_matrix.to_euler()
    
    # Assuming the sensor width is fixed and calculating the sensor height based on the aspect ratio
    sensor_width = 36.0  # Adjust as needed, standard full frame sensor width
    aspect_ratio = image_width / image_height
    sensor_height = sensor_width / aspect_ratio
    
    # Set camera data
    camera = camera_obj.data
    camera.sensor_width = sensor_width
    camera.sensor_height = sensor_height
    
    # Calculate focal length from FoV using formula: focal_length = sensor_width / (2 * tan(fov / 2))
    camera.lens = sensor_width / (2 * tan(fovx / 2))
    
    # Use FoVy to adjust the sensor height if needed
    calculated_fovy = 2 * atan((sensor_height / 2) / camera.lens)
    if calculated_fovy != fovy:
        camera.sensor_height = 2 * camera.lens * tan(fovy / 2)
    
    return camera_obj


def main():
    _cam_json_path = "cameras.json"
    
    with open(_cam_json_path, "r") as file:
        _camera_data = json.load(file)
    _camera_data = sorted(_camera_data, key=lambda x: x["id"])
    _camera_data = [{
        "id": x["id"],
        "img_name": x["img_name"],
        "width": x["width"],
        "height": x["height"],
        "position": np.array(x["position"]),
        "rotation": np.array(x["rotation"]),
        "fovy": x["fy"],
        "fovx": x["fx"]
    } for x in _camera_data]
    
    adjustment = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    
    for _new_cam in _camera_data:
        fovx = _calculate_fov(_new_cam["fovx"], _new_cam["width"])
        fovy = _calculate_fov(_new_cam["fovy"], _new_cam["height"])
        _new_cam_obj = create_camera(
            name=_new_cam["id"],
            position=_new_cam["position"],
            rot_matrix=_new_cam["rotation"] @ adjustment,
            fovx=fovx,
            fovy=fovy,
            image_width=_new_cam["width"],
            image_height=_new_cam["height"]
        )

if __name__ == "__main__":
    main()

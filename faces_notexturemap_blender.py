import json
from math import tan, atan
from pathlib import Path
import sys

import bpy
import numpy as np

from mathutils import Matrix

def load_npy(_path):
    _data = np.load(_path)
    return _data

def _calculate_fov(focal, pixels):
    return 2 * np.arctan(pixels / (2 * focal))

def setup_ambient_light():
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("World")

    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    nodes.clear()

    background_node = nodes.new(type='ShaderNodeBackground')
    background_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)  # White color for ambient light
    background_node.inputs['Strength'].default_value = 1.0  # Adjust the strength as needed

    world_output_node = nodes.new(type='ShaderNodeOutputWorld')

    links.new(background_node.outputs['Background'], world_output_node.inputs['Surface'])

    bpy.context.view_layer.update()


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


def gen_faces_from_texture_map(_path, _cam_json_path=None, _wanted_cam_idx=None):
    if _wanted_cam_idx is None or _cam_json_path is None:
        _wanted_cam_idx = []
    elif isinstance(_wanted_cam_idx, int):
        _wanted_cam_idx = [_wanted_cam_idx]
    assert isinstance(_wanted_cam_idx, list)
    
    if _cam_json_path is not None:
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
        
        if _wanted_cam_idx:
            _new_cameras = [_camera_data[i] for i in _wanted_cam_idx]
        else:
            _new_cameras = _camera_data

    if bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()
    
    mesh_data = load_npy(_path)
    vertices = mesh_data["vertices"]
    faces = mesh_data["faces"]
    vert_colors = mesh_data["vertex_colors"]
    
    mesh = bpy.data.meshes.new(name="MyMesh")
    splat_obj = bpy.data.objects.new(name="MyObject", object_data=mesh)

    # Link object to collection
    bpy.context.collection.objects.link(splat_obj)

    # Create mesh from vertices and faces
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()

    # Deselect all objects
    bpy.ops.object.select_all(action='DESELECT')

    # Select the newly created object
    splat_obj.select_set(True)

    # Set the newly created object as the active object
    bpy.context.view_layer.objects.active = splat_obj

    if not splat_obj.data.vertex_colors:
        splat_obj.data.vertex_colors.new()

    vertex_color_layer = splat_obj.data.vertex_colors.active

    for poly in splat_obj.data.polygons:
        for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
            loop = splat_obj.data.loops[loop_index]
            vertex_index = loop.vertex_index
            vertex_color = vert_colors[vertex_index].tolist()
            
            # Assign color and opacity (RGBA) to each vertex
            vertex_color_layer.data[loop_index].color = vertex_color

    # Create a new material
    material = bpy.data.materials.new(name="VertexColorMaterial")
    material.use_nodes = True

    # Enable transparency
    material.blend_method = 'HASHED'
    material.shadow_method = "NONE"
    material.use_backface_culling = False
    material.alpha_threshold = 0.01

    # Get the material's node tree
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    # Clear all nodes
    for node in nodes:
        nodes.remove(node)

    # Add a Vertex Color node
    vertex_color_node = nodes.new(type='ShaderNodeVertexColor')
    vertex_color_node.location = (-300, 300)
    vertex_color_node.layer_name = vertex_color_layer.name

    # Add a Principled BSDF node
    bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf_node.location = (100, 300)
    # Set metallic and specular to 0 for flat shading
    bsdf_node.inputs['Metallic'].default_value = 0.0
    bsdf_node.inputs['Specular'].default_value = 0.0
    bsdf_node.inputs['Roughness'].default_value = 1.0

    # Connect the vertex color to the BSDF node
    links.new(vertex_color_node.outputs['Color'], bsdf_node.inputs['Base Color'])
    links.new(vertex_color_node.outputs['Alpha'], bsdf_node.inputs['Alpha'])

    # Add an Output node
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    output_node.location = (300, 300)

    # Link the BSDF node to the output node
    links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])

    # Assign the material to the cube
    if splat_obj.type == 'MESH':
        if len(splat_obj.data.materials) == 0:
            splat_obj.data.materials.append(material)
        else:
            splat_obj.data.materials[0] = material

    # Update the mesh
    splat_obj.data.update()

    # Set the render engine to Eevee
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    
    # Configure Eevee settings for better quality
    bpy.context.scene.eevee.taa_render_samples = 64  # Increase samples for better quality
    bpy.context.scene.eevee.use_soft_shadows = True
    bpy.context.scene.eevee.use_ssr = True  # Screen Space Reflections
    bpy.context.scene.eevee.use_ssr_refraction = True
    bpy.context.scene.eevee.use_taa_reprojection = True
    
    # Enable transparency settings in Eevee
    bpy.context.scene.eevee.use_bloom = False
    bpy.context.scene.eevee.use_ssr_halfres = False
    bpy.context.scene.eevee.volumetric_tile_size = '2'

    setup_ambient_light()
    
    # Set render settings
    bpy.context.scene.cycles.samples = 128  # Set the number of samples for rendering

    _cameras = []
    
    adjustment = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    for _new_cam in _new_cameras:
        fovx = _calculate_fov(_new_cam["fovx"], _new_cam["width"])
        fovy = _calculate_fov(_new_cam["fovy"], _new_cam["height"])
        _new_cam_obj = create_camera(
            name=_new_cam["img_name"],
            position=_new_cam["position"],
            rot_matrix=_new_cam["rotation"] @ adjustment,
            fovx=fovx,
            fovy=fovy,
            image_width=_new_cam["width"],
            image_height=_new_cam["height"]
        )
        _cameras.append(_new_cam_obj)
    
    scale = Path(_path).stem.split("_")[1]
    out_dir = Path(Path(_path).parent, "images", scale)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # setup output img size
    bpy.context.scene.render.resolution_x = _new_cameras[0]["width"]
    bpy.context.scene.render.resolution_y = _new_cameras[0]["height"]
    
    for _cam_obj in _cameras:
        output_path = str(Path(out_dir, f"{_cam_obj.name}.png"))
        # Set camera as active camera
        bpy.context.scene.camera = _cam_obj

        # Set output path for rendered image
        bpy.context.scene.render.filepath = output_path

        # Render the image
        bpy.ops.render.render(write_still=True)
        print(f"Rendered image saved to {output_path}")

args = sys.argv

args_start_index = args.index("--") + 1
script_args = args[args_start_index:]

assert "--npz_path" in script_args
assert "--cam_path" in script_args

_npz_path = script_args[script_args.index("--npz_path") + 1]
_cam_path = script_args[script_args.index("--cam_path") + 1]

gen_faces_from_texture_map(
    _npz_path,
    _cam_path,
)

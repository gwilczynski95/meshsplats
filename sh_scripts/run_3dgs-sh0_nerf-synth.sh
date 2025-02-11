#!/bin/bash
DATA_DIR="/path/to/datasets/games_set"
EXP_DIR="/path/to/experiments"

GS_TRAIN_PY="/path/to/gaussian-splatting"
PYTHON_PATH="/path/to/python"

GS_PSEUDO_PATH="/path/to/gs_raytracing"
GS_OPTIM_PATH="/path/to/gs_raytracing/mesh_optim"

CONFIG_PATH="/path/to/gs_raytracing/sh_scripts/configs/3dgs_sh0_pseudo_config.yaml"

for data_path in $DATA_DIR/*; do
    dset_name=$(basename ${data_path})

    # Iterate over each element in the ELEMENTS_DIR
    exp_name="nerf-synth_3dgs_sh0_${dset_name}"
    exp_out_dir="${EXP_DIR}/${exp_name}"
    exp_cfg_path="${exp_out_dir}/config.yaml"
    ply_path="${exp_out_dir}/point_cloud/iteration_30000/point_cloud.ply"
    pseudomesh_path="${exp_out_dir}/pseudomeshes/scale_2.70_pts_8.npz"

    cd $GS_TRAIN_PY
    CUDA_VISIBLE_DEVICES=0 $PYTHON_PATH train.py --resolution 1 -s $data_path -m $exp_out_dir --iteration 30000 --sh_degree 0 --eval
    CUDA_VISIBLE_DEVICES=0 $PYTHON_PATH render.py --resolution 1 -s $data_path -m $exp_out_dir --iteration 30000 --sh_degree 0 --skip_train --eval

    cd $GS_PSEUDO_PATH
    $PYTHON_PATH generate_pseudomesh.py --ply $ply_path --algorithm 3dgs --scale_min 2.7 --scale_max 2.8 --scale_step 0.2 --no_points 8

    cd $GS_OPTIM_PATH
    CUDA_VISIBLE_DEVICES=0 $PYTHON_PATH optimize_pseudomesh.py --cfg_path $CONFIG_PATH --exp_name $exp_name --pseudomesh_path $pseudomesh_path --dset_path $data_path --res 1
    CUDA_VISIBLE_DEVICES=0 $PYTHON_PATH render_pseudomesh.py --cfg_path $exp_cfg_path --method 3dgs
    CUDA_VISIBLE_DEVICES=0 $PYTHON_PATH metrics.py -s $data_path -m $exp_out_dir --method 3dgs
done
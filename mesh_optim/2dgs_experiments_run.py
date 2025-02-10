import os
from pathlib import Path
import subprocess
import yaml


PYTHON_PATH = Path("/home/olaf/miniconda3/envs/splats310/bin/python")
PSEUDOMESH_PY = Path("/home/grzegos/projects/phd/gs_raytracing/generate_pseudomesh.py")
PSEUDO_CWD_PATH = PSEUDOMESH_PY.parent
PSEUDOMESH_OPTIM_PY = Path("/home/grzegos/projects/phd/gs_raytracing/mesh_optim/optimize_pseudomesh.py")
OPTIM_CWD_PATH = PSEUDOMESH_OPTIM_PY.parent

DSET_MAIN_DIR = Path("/home/grzegos/datasets/games_set")
TWODGS_EXP_DIR = Path("/home/grzegos/projects/phd/2dgs_nerf_output")
OUT_EXP_DIR = Path("/home/grzegos/projects/phd/2dgs_nvdiff_exps")

TMP_CONFIG_PATH = Path("/home/grzegos/projects/phd/gs_raytracing/mesh_optim/temp_config.yaml")

def main():
    with open("/home/grzegos/projects/phd/gs_raytracing/template_config.yaml", "r") as file:
        template_cfg = yaml.load(file, Loader=yaml.FullLoader)
    twodgs_exp_dirs = list(TWODGS_EXP_DIR.glob("*"))
    
    env = os.environ.copy()  # Copy current environment
    env['CUDA_VISIBLE_DEVICES'] = '1'
    
    for exp_dir in twodgs_exp_dirs:
        dset_path = Path(DSET_MAIN_DIR, exp_dir.stem)
        exp_name = exp_dir.stem
        
        # create pseudomesh
        ply_path = Path(exp_dir, "point_cloud", "iteration_10000", "point_cloud.ply")
        subprocess.run(
            [
                str(PYTHON_PATH),
                str(PSEUDOMESH_PY),
                "--ply", str(ply_path),
                "--algorithm", "2dgs",
                "--scale_min", "2.3",
                "--scale_max", "2.4",
                "--scale_step", "0.2",
                "--no_points", "8"
            ],
            cwd=str(PSEUDO_CWD_PATH)
        )
        
        # run optim
        pseudomesh_path = Path(exp_dir, "pseudomeshes", "scale_2.30_pts_8.npz")
        # change config
        template_cfg["experiment_name"] = exp_name
        template_cfg["experiments_dir"] = str(OUT_EXP_DIR)
        template_cfg["pseudomesh_path"] = str(pseudomesh_path)
        template_cfg["dataset"]["dataset_path"] = str(dset_path)
        # save cfg
        with open(TMP_CONFIG_PATH, 'w') as outfile:
            yaml.dump(template_cfg, outfile, default_flow_style=False)
        # pass cfg path to optimizer
        subprocess.run(
            [
                str(PYTHON_PATH),
                str(PSEUDOMESH_OPTIM_PY),
                "--cfg_path", str(TMP_CONFIG_PATH)
            ],
            cwd=str(OPTIM_CWD_PATH),
            env=env
        )


if __name__ == "__main__":
    main()

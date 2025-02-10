import argparse
from pathlib import Path

import numpy as np

import gs_utils

def main(ply_path: Path, algorithm: str, scale_muls: np.ndarray, no_of_points: int):
    xyz, feat_dc, feat_res, opac, scales, rots_quaternion = gs_utils.load_ply(
        ply_path, 0
    )
    if algorithm == "games":
        games_ckpt_path = Path(ply_path.parent, "model_params.pt")
        try:
            games_data = gs_utils.load_games_pt(games_ckpt_path)
        except FileNotFoundError:
            games_data = None

    if algorithm == "2dgs":
        pseudomeshes = gs_utils.generate_pseudomesh_2dgs(
            xyz=xyz,
            features_dc=feat_dc,
            opacities=opac,
            scales=scales,
            rots=rots_quaternion,
            scale_muls=scale_muls,
            no_of_points=no_of_points
        )
    elif algorithm == "surfels":
        pseudomeshes = gs_utils.generate_pseudomesh_surfels(
            xyz=xyz,
            features_dc=feat_dc,
            opacities=opac,
            scales=scales,
            rots=rots_quaternion,
            scale_muls=scale_muls,
            no_of_points=no_of_points
        )
    elif algorithm == "games":
        pseudomeshes = gs_utils.generate_pseudomesh_games(
            ckpt_data=games_data,
            xyz=xyz,
            features_dc=feat_dc,
            opacities=opac,
            scales=scales,
            rots=rots_quaternion,
            scale_muls=scale_muls,
            no_of_points=no_of_points
        )
    elif algorithm == "sugar2d":
        pseudomeshes = gs_utils.generate_pseudomesh_sugar_2d(
            xyz=xyz,
            features_dc=feat_dc,
            opacities=opac,
            scales=scales,
            rots=rots_quaternion,
            scale_muls=scale_muls,
            no_of_points=no_of_points
        )
    elif algorithm == "sugar3d":
        pseudomeshes = gs_utils.generate_pseudomesh_sugar_3d(
            xyz=xyz,
            features_dc=feat_dc,
            opacities=opac,
            scales=scales,
            rots=rots_quaternion,
            scale_muls=scale_muls,
            no_of_points=no_of_points
        )
    elif algorithm == "3dgs":
        pseudomeshes = gs_utils.generate_3dgs_pseudomesh(
            xyz=xyz,
            features_dc=feat_dc,
            opacities=opac,
            scales=scales,
            rots=rots_quaternion,
            scale_muls=scale_muls,
            no_of_points=no_of_points
        )
    else:
        raise NotImplementedError(f"Algorithm {algorithm} not yet implemented")
    
    # save data
    if "sugar" in algorithm:
        main_out_dir = ply_path.parent
    else:
        main_out_dir = ply_path.parent.parent.parent
    out_dir = Path(main_out_dir, "pseudomeshes")
    out_dir.mkdir(exist_ok=True, parents=True)
    
    for scale_val, pseudomesh_data in pseudomeshes.items():
        scale_val_str = "{:.2f}".format(scale_val)
        out_path = Path(out_dir, f"scale_{scale_val_str}_pts_{str(no_of_points)}.npz")
        np.savez(
            out_path,
            **pseudomesh_data
        )
        print(f"Saved {out_path}")
    
def read_args():
    parser = argparse.ArgumentParser(description="This script accepts ply file and outputs pseudomesh of it")
    parser.add_argument(
        "--ply",
        # required=True,
        # default="/home/grzegos/projects/phd/2dgs_nerf_output/hotdog/point_cloud/iteration_10000/point_cloud.ply",
        # default="/home/grzegos/projects/phd/2dgs_read_output/garden1/point_cloud/iteration_30000/point_cloud.ply",
        # default="/home/grzegos/projects/phd/games_nerf_output/hotdog_test_fff_sh0/point_cloud/iteration_30000/point_cloud.ply",
        default="/home/grzegos/projects/phd/pseudomesh_paper/experiments/3dgs_hotdog_test_sh0/point_cloud/iteration_30000/point_cloud.ply",
        # default="/home/grzegos/projects/phd/surfel_nerf_output/chair_test/point_cloud/iteration_50000/point_cloud.ply",
        # default="/home/grzegos/projects/phd/games_nerf_output/hotdog_new_test/point_cloud/iteration_1000/point_cloud.ply",
        # default="/home/grzegos/projects/phd/SuGaR/output/refined_ply/char_sh0/sugarfine_3Dgs7000_sdfestim02_sdfnorm02_level03_decim1000000_normalconsistency01_gaussperface1.ply",
        type=str,
        help="Path to the ply file"
    )
    parser.add_argument(
        "--algorithm",
        # default="games",
        default="3dgs",
        # default="surfels",
        # default="games",
        # default="sugar3d",
        type=str,
        choices=["2dgs", "games", "surfels", "sugar2d", "sugar3d", "3dgs"]
    )
    parser.add_argument(
        "--scale_min",
        default=2.3,
        type=float
    )
    parser.add_argument(
        "--scale_max",
        default=2.4,
        type=float
    )
    parser.add_argument(
        "--scale_step",
        default=0.2,
        type=float
    )
    parser.add_argument(
        "--no_points",
        default=8,
        type=int
    )
    args = parser.parse_args()
    
    return {
        "ply_path": Path(args.ply),
        "algorithm": args.algorithm,
        "scale_muls": np.arange(args.scale_min, args.scale_max, args.scale_step),
        "no_of_points": args.no_points
    }

if __name__ == "__main__":
    main(**read_args())

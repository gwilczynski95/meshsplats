# MeshSplats

This repository contains the code for the MeshSplats paper. With our work you can transform 3D or 2D Gaussians into a mesh and benefit from the advantages of both representations!


![Lego](https://github.com/user-attachments/assets/f8025d21-83f2-4723-8dd1-06fdce672d72)
https://github-production-user-asset-6210df.s3.amazonaws.com/47139865/411926772-4867f174-d66a-4f8e-a2ba-a7b32a105cb6.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250211%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250211T103409Z&X-Amz-Expires=300&X-Amz-Signature=e24200dc04b091ee968f4552ab6e332dbfd3328eabc0b02107b6d5b6b4591842&X-Amz-SignedHeaders=host
![Bicycle](https://github.com/user-attachments/assets/4867f174-d66a-4f8e-a2ba-a7b32a105cb6)

Note: You can find this videos in `demo` directory.


## Installation

```bash
pip install -r requirements.txt
```

Also this work depends on output of the following repositories:

- [Gaussian Mesh Splatting](https://github.com/waczjoan/gaussian-mesh-splatting)
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting)

Therefore, you need to install them first.

## Usage

You can find the scripts for running the experiments in the `sh_scripts` folder. Remember to change all paths to the correct ones in the scripts. We provided configs for the experiments in the `sh_scripts/configs` folder (once again remember to change the paths).

Each script is designed to run on a single GPU.

For example, to run the experiments for the DeepBlending dataset with the GaMeS algorithm, you can use the following command:

```bash
./sh_scripts/run_games_gs-flat_sh0_db.sh
```

## Datasets
This repository is prepared to work with the following datasets (as you can see in the `sh_scripts` scripts):

- NeRF-Synthetic
- Tanks and Temples
- MiP NeRF
- DeepBlending

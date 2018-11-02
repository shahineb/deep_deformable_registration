# Deep Learning for Deformable Fusion

Image registration is one of the most critical problems in radiology targeting to establish correspondences between images modalities of the same patient or longitudinal studies. This problem is traditionally casted as an optimization problem. In the advent of deep learning, the objective of this project will be to study recent advances for unsupervised deep learning deformable registration in the context of CT images for radiation oncology.

# Setup
## Machine access

1) Download `ressources/config` file and move it to local `~.ssh/` folder
2) Run `ssh reg`
3) Registration account password : `reg123`
4) Root password : `12345678`

## Machine specs

- OS : Ubuntu 16.04 (`/etc/lsb-release`)
- Memory : 32GB (`/proc/meminfo`)
- CPU : 32 Intel(R) Xeon(R) CPU E5-2630 v3 @ 2.40GHz (`/proc/cpuinfo`)
- GPU : 4 NVIDIA Tesla K40c (`/proc/driver/nvidia/gpus`)

## Environment Setup

### Install Anaconda
1) cd to `/workspace` directory
2) `wget https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh` (or latest version)
3) `sh` the downloaded file and follow instructions (installation takes few minutes)

### Install CUDA Tookit
1) Install linux headers (`apt-get install linux-headers-generic`)
2) Download runtime installer (`wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux`)
3) Make it executable (`chmod +x cuda_10.0.130_410.48_linux`)
4) Run it and follow the prompt (`./cuda_10.0.130_410.48_linux`). Do not have it run xconfig, say yes to all the other questions.
5) Path addition (`export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}` and `export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}`)

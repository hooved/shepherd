#!/usr/bin/env bash
# tested on RTX 5090, with NVIDIA Driver Version: 570.172.08     CUDA Version: 12.8
deactivate
rm -rf venv
uv venv venv --python=3.11
source venv/bin/activate
uv pip install jupyterlab jupyterlab-vim
#uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
#uv pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
uv pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
uv pip install -e .
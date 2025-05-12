# Step-by-step installation instructions

Following https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation.

Detailed package versions can be found in [requirements.txt](../requirements.txt).



**a. Create a conda virtual environment and activate it.**
```shell
conda create -n vad python=3.8 -y
conda activate vad
```

```shell
mamba create -n vad python=3.8 -y
mamba activate vad
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# Recommended torch>=1.9
```
```shell
mamba install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
mamba install -c "nvidia/label/cuda-11.8.0" cuda-toolkit        
```

Test pytorch with cuda:
```shell
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version (PyTorch Built with): {torch.version.cuda}'); print(f'Available CUDA Devices: {torch.cuda.device_count()}'); print(f'Current CUDA Device Index: {torch.cuda.current_device() if torch.cuda.is_available() else "N/A"}'); print(f'Current CUDA Device Name: {torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "N/A"}')"
```
The output should be like follows if there is no problem:
```shell
PyTorch Version: 2.1.0
CUDA Available: True
CUDA Version (PyTorch Built with): 11.8
Available CUDA Devices: 1
Current CUDA Device Index: 0
Current CUDA Device Name: NVIDIA GeForce RTX 3090
```


**c. Install gcc>=5 in conda env (optional, not tried this time).**
```shell
conda install -c omgarcia gcc-5 # gcc-6.2
```
Confirm c++ versions:
```shell
gcc --version

gcc (conda-forge gcc 10.4.0-19) 10.4.0
Copyright (C) 2020 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

g++ --version

g++ (conda-forge gcc 10.4.0-19) 10.4.0
Copyright (C) 2020 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```
Recommend to install ninja
```shell
mamba install ninja
```

**c. Install mmcv-full.**
```shell
pip install mmcv-full==1.4.0
#  pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

```shell
mamba install openmim
mim install mmcv==1.4.0
```

**d. Install mmdet and mmseg.**
```shell
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**e. Install timm.**
```shell
pip install timm
```

**f. Install mmdet3d.**
```shell
conda activate vad
git clone https://github.com/open-mmlab/mmdetection3d.git
cd /path/to/mmdetection3d
git checkout -f v0.17.1
python setup.py develop
```
Install scikit-image and numpy
```shell
mamba install scikit-image==0.18.3
mamba install numpy==1.21
mamba install networkx==2.2
```
Make modifications of mmdet3d ops:
t.b.c

**g. Install nuscenes-devkit.**
```shell
pip install nuscenes-devkit==1.1.9
pip install lyft-dataset-sdk
```

**h. Clone VAD.**
```shell
git clone https://github.com/hustvl/VAD.git
```

**i. Prepare pretrained models.**
```shell
cd /path/to/VAD
mkdir ckpts
cd ckpts 
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
```

Finish above, we may get warning
```shell
mmdet3d 0.17.1 requires numpy<1.20.0, but you have numpy 1.24.4 which is incompatible.
```
But please do not downgrade, or maybe you can refer for the following link for a solution.
https://github.com/open-mmlab/mmdetection/issues/9580
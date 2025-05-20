# Step-by-step installation instructions

Following https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation.

Detailed package versions can be found in [requirements.txt](../requirements.txt).



- **STEP 1. Create a conda virtual environment and activate it.**

    Here we use miniforge and mamba to management the environments.
    Please follow for https://github.com/conda-forge/miniforge for installation.
    After installing it, run the following commands to start a new environment.
    ```
    mamba create -n vad python=3.8 -y
    mamba activate vad
    ```

- **STEP 2: Install cudatoolkit**
    ```
    mamba install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
    ```
- **STEP 3: Install torch**
    ```
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
- **STEP 4: Set environment variables**
    ```
    # cuda 11.8 and GCC 9.4 is strongly recommended. Otherwise, it might encounter errors.
    export PATH=YOUR_GCC_PATH/bin:$PATH
    export CUDA_HOME=YOUR_CUDA_PATH/
    ```
    or
    ```
    mamba install -c conda-forge c-compiler cxx-compiler gcc_linux-64=9.4 gxx_linux-64=9.4 make -y

    ```
- **STEP 5: Install ninja and packaging**
    ```
    pip install ninja packaging
    ```
- **STEP 6: Install our repo**
    ```
    pip install -v -e .
    ```

- **STEP 7: Prepare pretrained weights.**
    create directory `ckpts`

    ```
    mkdir ckpts 
    ```
    Download `resnet50-19c8e357.pth` form [Hugging Face](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/resnet50-19c8e357.pth) or [Baidu Cloud](https://pan.baidu.com/s/1LlSrbYvghnv3lOlX1uLU5g?pwd=1234 ) or from Pytorch official website.
  
    Download `r101_dcn_fcos3d_pretrain.pth` form [Hugging Face](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/r101_dcn_fcos3d_pretrain.pth) or [Baidu Cloud](https://pan.baidu.com/s/1o7owaQ5G66xqq2S0TldwXQ?pwd=1234) or from BEVFormer official repo.
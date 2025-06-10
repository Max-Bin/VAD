# VAD-TensorRT
VAD: Vectorized Scene Representation for Efficient Autonomous Driving([VAD](https://github.com/hustvl/VAD.git)) is an end-to-end vectorized paradigm for autonomous driving.

In this demo, we will use `VAD-Tiny` as our deployment target.

| Method   | Backbone<br>precision | Head<br>precision | Framework | avg. L2 | Latency(ms, t.b.c)   |
| :---:    | :---:  | :---:     | :--:      | :---:   | :---: |
| VAD-Tiny (official) | fp32   | fp32      | Pytorch  | 0.78  |  |
| VAD-Tiny (official) | fp16   | fp32      | TensorRT  | 0.78  |  |
| VAD-Tiny (ours) | fp32   | fp32      | Pytorch  | 0.640  |  |
| VAD-Tiny (ours) | fp16   | fp32      | TensorRT  | 0.647  |  |

The reason why our benchmark is higher than the original VAD maybe we fixed the following bugs of the original script.
* https://github.com/hustvl/VAD/issues/18
* https://github.com/hustvl/VAD/blob/36047b6b5985e01832d8a2ecb0355d7f3c753ee1/projects/mmdet3d_plugin/VAD/modules/encoder.py#L186
* https://github.com/hustvl/VAD/pull/89

## Environment setup
Please follow xxx to setup the vad python evironment firstly.
Then you can have the checkpoint trained by yourself or just download the one we provide from xxx.

You may verify your installation with
```bash
cd /workspace/VAD
CUDA_VISIBLE_DEVICES=0 python tools/test.py /workspace/VAD/projects/configs/VAD/VAD_tiny_stage_2.py /workspace/VAD/ckpts/VAD_tiny.pth --launcher none --eval bbox --tmpdir tmp
```
This command line is expected to output the benchmark results. 


## Export to ONNX
To setup the deployment environment, you may run the following commands.
```bash
cd export_eval
ln -s ../../data data # create a soft-link to the data folder
```
As `VAD` is a temporal model, the inference behavior is different between the first frame and the subsequent frames.
When one frame has its previous frame, it will first do a temporal warp then concat the warped feature map to the current one.
To deploy the ONNX of the first frame, you may run
```bash
python export_no_prev.py ../../configs/VAD/VAD_tiny_stage_2.py ../../ckpts/VAD_tiny.pth --launcher none --eval bbox --tmpdir tmp
```
To deploy the ONNX of the subsequent frames, you may run
```bash
python export_prev.py ../../configs/VAD/VAD_tiny_stage_2.py ../../ckpts/VAD_tiny.pth --launcher none --eval bbox --tmpdir tmp
```
After these two command lines, you are expected to see `vadv1.extract_img_feat`, `vadv1.pts_bbox_head.forward`, `vadv1_prev.pts_bbox_head.forward` under `/workspace/DL4AGX/AV-Solutions/vad-trt/export_eval/scratch`. Each folder contains dumped input and output tensors in binary format, and an ONNX file begin with `sim_`.

## Benchmark and Evaluation with TensorRT on x86
We provide `test_tensorrt.py` to run benchmark with TensorRT. It will produce similar result as the original benchmark with pytorch.

1. To prepare dependencies for benchmark:
Install TensorRT following the instructions of https://developer.nvidia.com/tensorrt/download/10x. Please note that a version compatible with cuda11.8 is needed.
```bash
export TRT_ROOT=/opt/TensorRT-10.11.0.33
export PATH=${TRT_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${TRT_ROOT}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
```
Then install the python wheel by:
```bash
pip install <TRT_ROOT>/python/tensorrt-<version>-cp38-none-linux_aarch64.whl
```
2. Build plugins for benchmark
```bash
cd plugins/
mkdir build && cd build
cmake .. 
make
```

3. Then you need to build tensorrt engine.
```bash
cd export_eval
# build image encoder
trtexec --onnx=scratch/vadv1.extract_img_feat/sim_vadv1.extract_img_feat.onnx \
        --staticPlugins=../plugins/build/libplugins.so \
        --profilingVerbosity=detailed --dumpProfile \
        --separateProfileRun --useSpinWait --useManagedMemory \
        --fp16 \
        --saveEngine=scratch/vadv1.extract_img_feat/sim_vadv1.extract_img_feat_fp16.engine

# build heads
trtexec --onnx=scratch/vadv1.pts_bbox_head.forward/sim_vadv1.pts_bbox_head.forward.onnx \
        --staticPlugins=../plugins/build/libplugins.so \
        --profilingVerbosity=detailed --dumpProfile \
        --separateProfileRun --useSpinWait --useManagedMemory \
        --saveEngine=scratch/vadv1.pts_bbox_head.forward/sim_vadv1.pts_bbox_head.forward.engine

# build heads with prev_bev
trtexec --onnx=scratch/vadv1_prev.pts_bbox_head.forward/sim_vadv1_prev.pts_bbox_head.forward.onnx \
        --staticPlugins=../plugins/build/libplugins.so \
        --profilingVerbosity=detailed --dumpProfile \
        --separateProfileRun --useSpinWait --useManagedMemory \
        --saveEngine=scratch/vadv1_prev.pts_bbox_head.forward/sim_vadv1_prev.pts_bbox_head.forward.engine

4. Run benchmark with tensorrt

Before run the benchmark, we need to make sure the input order is exactly same as the onnx file. If not, please modify the corresponding parts in `test_tensorrt.py`.

```bash
python test_tensorrt.py ../../configs/VAD/VAD_tiny_stage_2.py ../../ckpts/VAD_tiny.pth --launcher none --eval bbox --tmpdir tmp
```
As we replace the backend from pytorch to tensorrt while keeping other parts like data loading and evaluation unchanged, you are expected to see outputs similar to the pytorch benchmark.


## License
- VAD and it's related code was licensed under [Apache-2.0](https://github.com/hustvl/VAD/blob/main/LICENSE)
- cuOSD and it's related code was licensed under [MIT](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/blob/master/LICENSE.md)
- trt and it's related code was licensed under [Apache-2.0](https://github.com/hustvl/VAD/blob/main/LICENSE)

## Reference <a name="ref"></a>
- [VAD official Repo](https://github.com/hustvl/VAD)
- [NVIDIA TensorRT Github](https://github.com/NVIDIA/TensorRT)
- [NVIDIA Drive](https://developer.nvidia.com/drive)
- [stb](https://github.com/nothings/stb/tree/master)
- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
- [CUDA-BEVFusion Repository](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-BEVFusion)
- [cuOSD Repository](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/libraries/cuOSD)
- [DL4AGX Repository](https://github.com/NVIDIA/DL4AGX)

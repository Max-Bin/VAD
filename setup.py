"""Setup script for MMCV.

This setup.py is simplified as most configuration has been moved to pyproject.toml.
It mainly handles the compilation of C++/CUDA extensions.
"""

import glob
import os
import platform
from packaging.version import parse as parse_version
from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


def get_extensions():
    """Get extension modules for compilation."""
    extensions = []
    
    # Extension name
    ext_name = 'mmcv._ext'
    
    # Set environment variables
    os.environ['MMCV_WITH_OPS'] = '1'
    if torch.cuda.is_available():
        os.environ['FORCE_CUDA'] = '1'
    
    # Limit build resources
    try:
        import psutil
        cpu_use = min(4, psutil.cpu_count())
    except (ModuleNotFoundError, AttributeError):
        cpu_use = 4
    os.environ.setdefault('MAX_JOBS', str(cpu_use))
    
    # Define macros
    define_macros = []
    
    # Compiler arguments
    extra_compile_args = {'cxx': []}
    if platform.system() != 'Windows':
        if parse_version(torch.__version__) <= parse_version('1.12.1'):
            extra_compile_args['cxx'] = ['-std=c++14']
        else:
            extra_compile_args['cxx'] = ['-std=c++17']
    else:
        if parse_version(torch.__version__) <= parse_version('1.12.1'):
            extra_compile_args['cxx'] = ['/std:c++14']
        else:
            extra_compile_args['cxx'] = ['/std:c++17']
    
    include_dirs = []
    
    # Build with CUDA if available
    if torch.cuda.is_available():
        print(f'Building {ext_name} with CUDA support')
        define_macros += [('MMCV_WITH_CUDA', None)]
        cuda_args = os.getenv('MMCV_CUDA_ARGS')
        extra_compile_args['nvcc'] = [cuda_args] if cuda_args else []
        
        # Add CUDA C++ standard
        if 'nvcc' in extra_compile_args and platform.system() != 'Windows':
            if parse_version(torch.__version__) <= parse_version('1.12.1'):
                extra_compile_args['nvcc'] += ['-std=c++14']
            else:
                extra_compile_args['nvcc'] += ['-std=c++17']
        
        # Source files
        op_files = (glob.glob('./mmcv/ops/csrc/pytorch/*.cpp') + 
                   glob.glob('./mmcv/ops/csrc/pytorch/cpu/*.cpp') + 
                   glob.glob('./mmcv/ops/csrc/pytorch/cuda/*.cu') + 
                   glob.glob('./mmcv/ops/csrc/pytorch/cuda/*.cpp'))
        extension = CUDAExtension
        include_dirs.extend([
            os.path.abspath('./mmcv/ops/csrc/common'),
            os.path.abspath('./mmcv/ops/csrc/common/cuda')
        ])
    else:
        print(f'Building {ext_name} without CUDA')
        op_files = (glob.glob('./mmcv/ops/csrc/pytorch/*.cpp') + 
                   glob.glob('./mmcv/ops/csrc/pytorch/cpu/*.cpp'))
        extension = CppExtension
        include_dirs.append(os.path.abspath('./mmcv/ops/csrc/common'))
    
    # Create extension if source files exist
    if op_files:
        ext_ops = extension(
            name=ext_name,
            sources=op_files,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args)
        extensions.append(ext_ops)
    else:
        print(f'Warning: No source files found for {ext_name}')
    
    return extensions


def make_cuda_ext(name, module, sources, sources_cuda=None, 
                  extra_args=None, extra_include_path=None):
    """Create CUDA extension."""
    if sources_cuda is None:
        sources_cuda = []
    if extra_args is None:
        extra_args = []
    if extra_include_path is None:
        extra_include_path = []
    
    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}
    
    if torch.cuda.is_available():
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print(f'Compiling {name} without CUDA')
        extension = CppExtension
    
    return extension(
        name=f'{module}.{name}',
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)


def get_additional_extensions():
    """Get additional CUDA extensions."""
    extensions = []
    
    # IOU3D extension
    iou3d_ext = make_cuda_ext(
        name='iou3d_cuda',
        module='mmcv.ops.iou3d_det',
        sources=['src/iou3d.cpp'],
        sources_cuda=['src/iou3d_kernel.cu'])
    
    # ROIAware extension
    roiaware_ext = make_cuda_ext(
        name='roiaware_pool3d_ext',
        module='mmcv.ops.roiaware_pool3d',
        sources=[
            'src/roiaware_pool3d.cpp',
            'src/points_in_boxes_cpu.cpp',
        ],
        sources_cuda=[
            'src/roiaware_pool3d_kernel.cu',
            'src/points_in_boxes_cuda.cu',
        ])
    
    if iou3d_ext is not None:
        extensions.append(iou3d_ext)
    if roiaware_ext is not None:
        extensions.append(roiaware_ext)
    
    return extensions


if __name__ == '__main__':
    # Most configuration is in pyproject.toml
    # This setup() only handles the extensions
    setup(
        ext_modules=get_extensions() + get_additional_extensions(),
        cmdclass={'build_ext': BuildExtension},
    )
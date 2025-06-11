import glob
import os
import platform
import re
from packaging.version import parse as parse_version
from setuptools import find_packages, setup
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

EXT_TYPE = 'pytorch'

def make_cuda_ext(name,
                  module,
                  sources,
                  sources_cuda=[],
                  extra_args=[],
                  extra_include_path=[]):

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
        print('Compiling {} without CUDA'.format(name))
        extension = CppExtension

    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)

def parse_requirements(fname='requirements.txt', with_version=True):
    """解析requirements文件"""
    import sys
    from os.path import exists
    require_fpath = fname

    def parse_line(line):
        """解析requirements文件中的每一行"""
        if line.startswith('-r '):
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages

def get_extensions():
    """获取扩展模块"""
    extensions = []
    
    if EXT_TYPE == 'pytorch':
        ext_name = 'mmcv._ext'
        
        # 强制设置环境变量
        os.environ['MMCV_WITH_OPS'] = '1'
        os.environ['FORCE_CUDA'] = '1'
        
        # 限制ninja使用的资源
        try:
            import psutil
            cpu_use = min(4, psutil.cpu_count())
        except (ModuleNotFoundError, AttributeError):
            cpu_use = 4

        os.environ.setdefault('MAX_JOBS', str(cpu_use))
        define_macros = []

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

        if torch.cuda.is_available():
            print(f'Building {ext_name} with CUDA support')
            define_macros += [('MMCV_WITH_CUDA', None)]
            cuda_args = os.getenv('MMCV_CUDA_ARGS')
            extra_compile_args['nvcc'] = [cuda_args] if cuda_args else []
            op_files = glob.glob('./mmcv/ops/csrc/pytorch/*.cpp') + \
                glob.glob('./mmcv/ops/csrc/pytorch/cpu/*.cpp') + \
                glob.glob('./mmcv/ops/csrc/pytorch/cuda/*.cu') + \
                glob.glob('./mmcv/ops/csrc/pytorch/cuda/*.cpp')
            extension = CUDAExtension
            include_dirs.append(os.path.abspath('./mmcv/ops/csrc/common'))
            include_dirs.append(os.path.abspath('./mmcv/ops/csrc/common/cuda'))
        else:
            print(f'Building {ext_name} without CUDA')
            op_files = glob.glob('./mmcv/ops/csrc/pytorch/*.cpp') + \
                glob.glob('./mmcv/ops/csrc/pytorch/cpu/*.cpp')
            extension = CppExtension
            include_dirs.append(os.path.abspath('./mmcv/ops/csrc/common'))

        if 'nvcc' in extra_compile_args and platform.system() != 'Windows':
            if parse_version(torch.__version__) <= parse_version('1.12.1'):
                extra_compile_args['nvcc'] += ['-std=c++14']
            else:
                extra_compile_args['nvcc'] += ['-std=c++17']

        # 确保有源文件可以编译
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

def get_additional_extensions():
    """获取额外的CUDA扩展"""
    extensions = []
    
    # iou3d扩展
    iou3d_ext = make_cuda_ext(
        name='iou3d_cuda',
        module='mmcv.ops.iou3d_det',
        sources=[
            'src/iou3d.cpp',
            'src/iou3d_kernel.cu',
        ])
    
    # roiaware扩展
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
    # 确保环境变量设置
    os.environ['MMCV_WITH_OPS'] = '1'
    os.environ['FORCE_CUDA'] = '1'
    
    setup(
        name='mmcv',
        version='0.0.1',
        description='OpenMMLab Computer Vision Foundation',
        keywords='computer vision',
        packages=[
            *find_packages(include=('mmcv', "mmcv.*")), 
            *find_packages(include=('adzoo', "adzoo.*")), 
        ],
        include_package_data=True,
        classifiers=[
            'Development Status :: 4 - Beta',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Topic :: Utilities',
        ],
        url='https://github.com/open-mmlab/mmcv',
        author='MMCV Contributors',
        author_email='openmmlab@gmail.com',
        install_requires=parse_requirements(),
        ext_modules=get_extensions() + get_additional_extensions(),
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)

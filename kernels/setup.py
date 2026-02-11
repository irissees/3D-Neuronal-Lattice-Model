"""
Setup script for building CUDA kernels.

Usage:
    python setup.py build_ext --inplace
    
Or install as package:
    pip install -e .
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get the directory containing this file
here = os.path.dirname(os.path.abspath(__file__))

setup(
    name='pulse_propagation_kernels',
    version='1.0.0',
    description='CUDA kernels for 3D cell pulse propagation',
    author='Cross-Dimensional Architecture',
    ext_modules=[
        CUDAExtension(
            name='pulse_propagation_cuda',
            sources=[os.path.join(here, 'cuda_kernels.cu')],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-gencode', 'arch=compute_70,code=sm_70',  # V100
                    '-gencode', 'arch=compute_75,code=sm_75',  # T4
                    '-gencode', 'arch=compute_80,code=sm_80',  # A100
                    '-gencode', 'arch=compute_86,code=sm_86',  # RTX 30xx
                    '-gencode', 'arch=compute_89,code=sm_89',  # RTX 40xx
                    '-gencode', 'arch=compute_90,code=sm_90',  # H100
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
    ],
)

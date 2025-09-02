"""
Modern setup.py for MIDI-GPT Python 3.9+ compatibility
Removes bundled dependencies and uses system packages
"""

import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import pybind11


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build extensions")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DCMAKE_BUILD_TYPE=Release',
            '-DBUILD_SHARED_LIBS=OFF',
        ]

        # Handle different Python versions
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        cmake_args.append(f'-DPYTHON_VERSION={python_version}')

        # Platform specific arguments
        if sys.platform.startswith("darwin"):
            cmake_args += ['-DMAC_OS=ON']
        
        # Handle PyTorch
        try:
            import torch
            print(f"Found PyTorch {torch.__version__}")
            
            # Get PyTorch cmake prefix path
            torch_path = torch.utils.cmake_prefix_path
            if torch_path:
                cmake_args.append(f'-DCMAKE_PREFIX_PATH={torch_path}')
            
            # Check if CUDA is available
            if torch.cuda.is_available():
                cmake_args.append('-DCUDA_AVAILABLE=ON')
                
        except ImportError:
            print("PyTorch not found, building without torch support")
            cmake_args.append('-DNO_TORCH=ON')

        # Handle protobuf
        try:
            import google.protobuf
            print(f"Found protobuf {google.protobuf.__version__}")
        except ImportError:
            raise RuntimeError("protobuf is required but not installed")

        build_args = ['--config', 'Release']
        
        # Parallel build
        if hasattr(self, 'parallel') and self.parallel:
            build_args += [f'-j{self.parallel}']
        else:
            build_args += ['-j4']

        env = os.environ.copy()
        env['CXXFLAGS'] = f'{env.get("CXXFLAGS", "")} -DVERSION_INFO=\\"{self.distribution.get_version()}\\"'
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


def get_version():
    """Get version from git or default"""
    try:
        result = subprocess.run(['git', 'describe', '--tags', '--always'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "0.1.0-dev"


# Check Python version
if sys.version_info < (3, 9):
    raise RuntimeError("Python 3.9 or higher is required")

setup(
    name="midigpt",
    version=get_version(),
    author="Jeff Ens, Rafael Arias", 
    author_email="raa60@sfu.ca",
    description="MIDI-GPT: Machine learning library for MIDI generation",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    
    ext_modules=[CMakeExtension('midigpt')],
    cmdclass={'build_ext': CMakeBuild},
    
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0,<2.0",  # Temporary constraint
        "protobuf>=4.0.0",
        "pybind11>=2.12.0",
        "transformers>=4.30.0",
        "tqdm",
    ],
    
    extras_require={
        'dev': [
            'pytest>=6.0',
            'black',
            'isort',
            'mypy',
        ],
        'cuda': ['torch[cuda]>=2.0.0'],
        'cpu': ['torch[cpu]>=2.0.0'],
    },
    
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers", 
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
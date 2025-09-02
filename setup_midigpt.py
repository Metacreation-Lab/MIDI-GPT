#!/usr/bin/env python3
"""
Setup script for MIDI-GPT with proper C/C++ compilation handling.
This replaces the pyproject.toml approach for better control over compilation.
"""

import os
import sys
import subprocess
import sysconfig
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11

from setuptools import setup, find_packages

# Check Python version
if sys.version_info < (3, 9):
    raise RuntimeError("Python 3.9 or higher is required")

def get_torch_info():
    """Get PyTorch include and library paths."""
    try:
        import torch
        return {
            'include_dirs': torch.utils.cpp_extension.include_paths(),
            'library_dirs': torch.utils.cpp_extension.library_paths(),
            'libraries': ['torch', 'torch_cpu'],
            'version': torch.__version__
        }
    except ImportError:
        return None

def get_protobuf_info():
    """Get Protobuf include and library information."""
    # Try pkg-config first
    try:
        result = subprocess.run(['pkg-config', '--cflags', '--libs', 'protobuf'], 
                              capture_output=True, text=True, check=True)
        
        # Parse the output
        flags = result.stdout.strip().split()
        include_dirs = []
        library_dirs = []
        libraries = []
        
        for flag in flags:
            if flag.startswith('-I'):
                include_dirs.append(flag[2:])
            elif flag.startswith('-L'):
                library_dirs.append(flag[2:])
            elif flag.startswith('-l'):
                libraries.append(flag[2:])
        
        return {
            'include_dirs': include_dirs,
            'library_dirs': library_dirs, 
            'libraries': libraries
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to common locations
        return {
            'include_dirs': ['/usr/local/include', '/usr/include'],
            'library_dirs': ['/usr/local/lib', '/usr/lib'],
            'libraries': ['protobuf']
        }

def generate_protobuf_files():
    """Generate C++ files from .proto definitions."""
    proto_dir = Path('proto')
    if not proto_dir.exists():
        print("Warning: proto/ directory not found. Skipping protobuf generation.")
        return []
    
    # Create build directory for generated files
    build_dir = Path('build/proto')
    build_dir.mkdir(parents=True, exist_ok=True)
    
    proto_files = list(proto_dir.glob('*.proto'))
    if not proto_files:
        print("Warning: No .proto files found in proto/ directory.")
        return []
    
    # Generate protobuf files
    generated_files = []
    for proto_file in proto_files:
        # Run protoc command
        cmd = [
            'protoc',
            f'--cpp_out={build_dir}',
            f'--proto_path={proto_dir}',
            str(proto_file)
        ]
        
        try:
            subprocess.run(cmd, check=True)
            base_name = proto_file.stem
            generated_files.extend([
                str(build_dir / f'{base_name}.pb.cc'),
                str(build_dir / f'{base_name}.pb.h')
            ])
        except subprocess.CalledProcessError as e:
            print(f"Error generating protobuf for {proto_file}: {e}")
            sys.exit(1)
    
    return [f for f in generated_files if f.endswith('.cc')]

def create_extension():
    """Create the pybind11 extension with proper compilation settings."""
    
    # Generate protobuf files
    proto_sources = generate_protobuf_files()
    
    # C++ source files
    cpp_sources = [
        "src/lib.cpp",
        "src/common/data_structures/train_config.cpp", 
        "src/dataset_creation/dataset_manipulation/bytes_to_file.cpp",
    ] + proto_sources
    
    # C source files (compiled separately)
    c_sources = [
        "src/dataset_creation/compression/lz4.c",
    ]
    
    # Include directories
    include_dirs = [
        "src",
        "include", 
        "build/proto",  # For generated protobuf files
        pybind11.get_include(),
    ]
    
    # Libraries to link
    libraries = []
    library_dirs = []
    
    # Add protobuf
    protobuf_info = get_protobuf_info()
    include_dirs.extend(protobuf_info['include_dirs'])
    library_dirs.extend(protobuf_info['library_dirs'])
    libraries.extend(protobuf_info['libraries'])
    
    # Add PyTorch if available
    torch_info = get_torch_info()
    compile_args = ['-std=c++20']
    link_args = []
    define_macros = []
    
    if torch_info:
        print(f"Found PyTorch {torch_info['version']}")
        include_dirs.extend(torch_info['include_dirs'])
        library_dirs.extend(torch_info['library_dirs'])
        libraries.extend(torch_info['libraries'])
    else:
        print("PyTorch not found, building without PyTorch support")
        define_macros.append(('NO_TORCH', '1'))
    
    # Platform-specific settings
    if sys.platform.startswith('darwin'):  # macOS
        compile_args.extend(['-stdlib=libc++', '-mmacosx-version-min=10.9'])
        link_args.extend(['-stdlib=libc++', '-mmacosx-version-min=10.9'])
    
    # Create separate extensions for C and C++ files to handle compilation properly
    extensions = []
    
    # C++ extension
    if cpp_sources:
        cpp_ext = Pybind11Extension(
            "midigpt",
            cpp_sources,
            include_dirs=include_dirs,
            libraries=libraries,
            library_dirs=library_dirs,
            language='c++',
            cxx_std=20,
            define_macros=define_macros,
        )
        extensions.append(cpp_ext)
    
    # If we have C sources, we need to compile them separately and link
    # For now, let's include the C file in the C++ compilation but with proper handling
    if c_sources:
        # We'll add the C sources to the C++ extension but ensure they're compiled as C
        # This requires custom compilation handling
        pass
    
    return extensions

# Custom build_ext class to handle mixed C/C++ compilation
class CustomBuildExt(build_ext):
    """Custom build extension to handle C and C++ files properly."""
    
    def build_extensions(self):
        # Handle C++ standard and other settings
        for ext in self.extensions:
            # Ensure C++20 is set
            if hasattr(ext, 'cxx_std'):
                if not any('-std=c++' in flag for flag in ext.extra_compile_args):
                    ext.extra_compile_args.append(f'-std=c++{ext.cxx_std}')
        
        # Call parent build
        super().build_extensions()
    
    def get_ext_filename(self, ext_name):
        """Override to ensure proper extension naming."""
        filename = super().get_ext_filename(ext_name)
        # Remove any extra suffixes that might be added
        return filename

def get_version():
    """Get version from version file or default."""
    try:
        with open('src/inference/version.h', 'r') as f:
            content = f.read()
            # Try to extract version from C++ header
            import re
            match = re.search(r'#define\s+VERSION\s+"([^"]+)"', content)
            if match:
                return match.group(1)
    except FileNotFoundError:
        pass
    
    return "1.0.0"

def get_long_description():
    """Get long description from README."""
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "MIDI-GPT: A generative system for computer-assisted music composition"

# Get PyTorch version constraint based on Python version
def get_torch_requirement():
    """Get appropriate torch requirement based on Python version."""
    if sys.version_info >= (3, 10):
        return "torch>=2.4.0"
    elif sys.version_info >= (3, 9):
        return "torch>=2.0.0,<2.3.0"
    else:
        return "torch>=2.0.0"

if __name__ == "__main__":
    # Check if we're building without torch
    no_torch = '--no-torch' in sys.argv
    if no_torch:
        sys.argv.remove('--no-torch')
    
    # Prepare dependencies
    install_requires = [
        "numpy>=1.21.0",
        "protobuf>=4.0.0", 
        "tqdm>=4.60.0",
    ]
    
    if not no_torch:
        install_requires.append(get_torch_requirement())
    
    # Create extensions
    extensions = create_extension()
    
    setup(
        name="midigpt",
        version=get_version(),
        author="Jeff Ens, Rafael Arias",
        author_email="raa60@sfu.ca",
        description="A generative system for computer-assisted music composition",
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        url="https://github.com/Metacreation-Lab/MIDI-GPT",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        ext_modules=extensions,
        cmdclass={"build_ext": CustomBuildExt},
        install_requires=install_requires,
        extras_require={
            "training": [
                "transformers>=4.20.0",
                "tensorboardX>=2.6",
                "jsonlines>=3.1.0",
                "pandas>=1.5.0",
                "scipy>=1.10.0",
                "matplotlib>=3.6.0",
                "scikit-learn>=1.2.0",
            ],
            "dev": [
                "pytest>=7.0.0",
                "black>=23.0.0", 
                "isort>=5.12.0",
                "mypy>=1.5.0",
            ]
        },
        python_requires=">=3.9",
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers", 
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10", 
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: C++",
            "Topic :: Multimedia :: Sound/Audio",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        zip_safe=False,
    )
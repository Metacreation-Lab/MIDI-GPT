#!/usr/bin/env python3.9
"""
MIDI-GPT setup script - Python 3.9 compatible version
Replacement for create_python_library.sh
"""

import os
import sys
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path

def get_torch_info():
    """Get PyTorch information with compatibility handling"""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        torch_info = {
            'version': torch.__version__,
            'include_dirs': [],
            'library_dirs': [],
            'libraries': ['torch', 'torch_cpu']
        }
        
        # Handle PyTorch include paths across versions
        try:
            # Try newer PyTorch API first
            torch_info['include_dirs'] = torch.utils.cpp_extension.include_paths()
        except AttributeError:
            # Fallback for newer versions where API changed
            try:
                import torch.utils.cpp_extension as cpp_ext
                torch_info['include_dirs'] = [
                    torch.utils.cpp_extension.CUDA_HOME + "/include" if torch.utils.cpp_extension.CUDA_HOME else "",
                    str(Path(torch.__file__).parent / "include")
                ]
                torch_info['include_dirs'] = [p for p in torch_info['include_dirs'] if p and os.path.exists(p)]
            except:
                # Manual fallback
                torch_path = Path(torch.__file__).parent
                include_path = torch_path / "include"
                if include_path.exists():
                    torch_info['include_dirs'] = [str(include_path)]
                else:
                    print("Warning: Could not find PyTorch include directory")
                    torch_info['include_dirs'] = []
        
        # Get library paths
        try:
            torch_info['library_dirs'] = torch.utils.cpp_extension.library_paths()
        except AttributeError:
            torch_path = Path(torch.__file__).parent
            lib_path = torch_path / "lib"
            if lib_path.exists():
                torch_info['library_dirs'] = [str(lib_path)]
            
        print(f"Found PyTorch {torch.__version__}")
        return torch_info
        
    except ImportError as e:
        print(f"PyTorch not available: {e}")
        return None
    except Exception as e:
        print(f"Error getting PyTorch info: {e}")
        return None

def check_dependencies():
    """Check and install required dependencies"""
    required_packages = [
        ("numpy", "numpy<2.0"),  # NumPy compatibility constraint
        ("torch", "torch>=2.0.0"),
        ("pybind11", "pybind11>=2.12.0"),
        ("protobuf", "protobuf>=4.0.0"),
    ]
    
    missing = []
    for package, pip_spec in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(pip_spec)
    
    if missing:
        print(f"Installing missing dependencies: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)

def setup_proto_directory():
    """Set up protobuf directory structure"""
    proto_dir = Path("proto")
    proto_dir.mkdir(exist_ok=True)
    
    # Try to copy proto files from old location
    old_proto_path = Path("libraries/protobuf/src")
    if old_proto_path.exists():
        for proto_file in old_proto_path.glob("*.proto"):
            shutil.copy(proto_file, proto_dir)
            print(f"Copied {proto_file.name} to proto/")
    
    # Check if we have proto files
    proto_files = list(proto_dir.glob("*.proto"))
    if not proto_files:
        print("Warning: No proto files found. Protobuf generation will be skipped.")
        return False
    
    return True

def generate_protobuf_files():
    """Generate protobuf C++ files"""
    proto_dir = Path("proto")
    if not proto_dir.exists():
        print("Warning: proto/ directory not found. Skipping protobuf generation.")
        return
    
    proto_files = list(proto_dir.glob("*.proto"))
    if not proto_files:
        print("Warning: No proto files found. Skipping protobuf generation.")
        return
    
    build_dir = Path("build")
    build_dir.mkdir(exist_ok=True)
    
    # Generate protobuf files
    for proto_file in proto_files:
        cmd = [
            "protoc",
            f"--cpp_out={build_dir}",
            f"--proto_path={proto_dir}",
            str(proto_file)
        ]
        try:
            subprocess.check_call(cmd)
            print(f"Generated protobuf files for {proto_file.name}")
        except subprocess.CalledProcessError:
            print(f"Warning: Failed to generate protobuf for {proto_file.name}")
        except FileNotFoundError:
            print("Warning: protoc not found. Install protobuf compiler.")

def create_extension(no_torch=False, mac_os=False, dev=False):
    """Create the C++ extension"""
    from setuptools import Extension
    import pybind11
    
    # Get torch info if not disabled
    torch_info = None
    if not no_torch:
        torch_info = get_torch_info()
        if not torch_info:
            print("PyTorch not found, building without torch support")
            no_torch = True
    
    # Base source files
    sources = [
        "src/lib.cpp",
        "src/common/data_structures/train_config.cpp", 
        "src/dataset_creation/compression/lz4.c",
        "src/dataset_creation/dataset_manipulation/bytes_to_file.cpp",
    ]
    
    # Add generated protobuf sources
    build_dir = Path("build")
    for pb_file in build_dir.glob("*.pb.cc"):
        sources.append(str(pb_file))
    
    # Include directories
    include_dirs = [
        pybind11.get_include(),
        "src",
        "include", 
        str(build_dir),  # For generated protobuf headers
        "proto",
    ]
    
    # Add torch includes if available
    if torch_info and torch_info['include_dirs']:
        include_dirs.extend(torch_info['include_dirs'])
    
    # Libraries to link
    libraries = []
    library_dirs = []
    
    if torch_info:
        libraries.extend(torch_info['libraries'])
        if torch_info['library_dirs']:
            library_dirs.extend(torch_info['library_dirs'])
    
    # Add protobuf
    libraries.append('protobuf')
    
    # Handle midifile - check if bundled version exists
    midifile_path = Path("libraries/midifile")
    if midifile_path.exists():
        include_dirs.append(str(midifile_path / "include"))
        # Will be built by CMake
    
    # Compiler flags
    extra_compile_args = ['-std=c++17']
    extra_link_args = []
    
    if mac_os:
        extra_compile_args.extend([
            '-stdlib=libc++',
            '-mmacosx-version-min=10.14'
        ])
    
    if no_torch:
        extra_compile_args.append('-DNO_TORCH')
    
    if dev:
        extra_compile_args.extend(['-g', '-O0'])
    else:
        extra_compile_args.extend(['-O3', '-DNDEBUG'])
    
    extension = Extension(
        'midigpt',
        sources=sources,
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++'
    )
    
    return [extension]

def main():
    parser = argparse.ArgumentParser(description="MIDI-GPT setup script")
    parser.add_argument("--dev", action="store_true", help="Development build")
    parser.add_argument("--test", action="store_true", help="Test build")
    parser.add_argument("--no-torch", action="store_true", help="Build without PyTorch")
    parser.add_argument("--mac-os", action="store_true", help="Build for macOS")
    parser.add_argument("--compute-canada", action="store_true", help="Build for Compute Canada")
    parser.add_argument("--clean", action="store_true", help="Clean build directory")
    
    args = parser.parse_args()
    
    if args.clean:
        shutil.rmtree("build", ignore_errors=True)
        shutil.rmtree("python_lib", ignore_errors=True)
        print("Cleaned build directories")
        return
    
    print("=== MIDI-GPT Python 3.9 Setup ===")
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("Error: Python 3.9 or higher required")
        sys.exit(1)
    
    print(f"Using Python {sys.version}")
    
    # Check and install dependencies
    print("Checking dependencies...")
    check_dependencies()
    
    # Setup protobuf
    print("Setting up protobuf...")
    has_proto = setup_proto_directory()
    if has_proto:
        generate_protobuf_files()
    
    # Create extensions
    print("Creating C++ extensions...")
    extensions = create_extension(
        no_torch=args.no_torch,
        mac_os=args.mac_os,
        dev=args.dev
    )
    
    # Build the extension
    from setuptools import setup
    from setuptools.command.build_ext import build_ext
    
    class CustomBuildExt(build_ext):
        def build_extensions(self):
            # Ensure build directory exists
            os.makedirs("python_lib", exist_ok=True)
            
            # Set compiler options
            for ext in self.extensions:
                ext.extra_compile_args = ext.extra_compile_args or []
                ext.extra_compile_args.extend([
                    '-Wall', '-Wextra', '-Wpedantic',
                    '-fPIC', '-fvisibility=hidden'
                ])
            
            build_ext.build_extensions(self)
    
    setup(
        name="midigpt",
        version="0.1.0",
        ext_modules=extensions,
        cmdclass={'build_ext': CustomBuildExt},
        zip_safe=False,
    )
    
    if args.test:
        print("Testing import...")
        try:
            import midigpt
            print("✅ midigpt import successful")
            
            # Test basic functionality
            if hasattr(midigpt, 'getEncoderType'):
                print("✅ Core functions available")
            else:
                print("⚠️  Some functions may be missing")
                
        except ImportError as e:
            print(f"❌ Import failed: {e}")
            sys.exit(1)
    
    print("=== Setup completed successfully ===")

if __name__ == "__main__":
    main()
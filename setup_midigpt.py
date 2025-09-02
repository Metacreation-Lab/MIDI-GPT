#!/usr/bin/env python3

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import pybind11
    from setuptools import setup, Extension
    from setuptools.command.build_ext import build_ext
except ImportError as e:
    print(f"Missing required dependency: {e}")
    sys.exit(1)

def get_pytorch_info():
    """Get PyTorch compilation information"""
    try:
        import torch
        from torch.utils.cpp_extension import include_paths, library_paths
        
        return {
            'available': True,
            'version': torch.__version__,
            'include_dirs': include_paths(),
            'library_dirs': library_paths(),
            'libraries': ['torch', 'torch_cpu']
        }
    except ImportError:
        return {'available': False}

def check_and_setup_midifile():
    """Check for midifile library and clone if needed"""
    midifile_path = Path("libraries/midifile")
    
    if not midifile_path.exists() or not any(midifile_path.iterdir()):
        print("midifile library not found, downloading...")
        libraries_dir = Path("libraries")
        libraries_dir.mkdir(exist_ok=True)
        
        try:
            # Clone midifile as the original script did
            subprocess.run([
                "git", "clone", 
                "https://github.com/craigsapp/midifile", 
                str(midifile_path)
            ], check=True, capture_output=True)
            
            # Reset to the specific commit used by the original project
            subprocess.run([
                "git", "reset", "--hard", "838c62c"
            ], cwd=midifile_path, check=True, capture_output=True)
            
            print("Successfully downloaded and configured midifile library")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to download midifile library: {e}")
            return False
    else:
        print("Found existing midifile library")
        return True

def setup_proto_directory():
    """Set up protobuf directory structure and copy proto files if needed"""
    proto_dir = Path("proto")
    proto_dir.mkdir(exist_ok=True)
    
    # Check if proto files already exist
    existing_proto_files = list(proto_dir.glob("*.proto"))
    if existing_proto_files:
        print(f"Found {len(existing_proto_files)} proto files in proto/")
        return True
    
    # Try to copy proto files from old location
    old_proto_path = Path("libraries/protobuf/src")
    if old_proto_path.exists():
        copied_files = 0
        for proto_file in old_proto_path.glob("*.proto"):
            shutil.copy(proto_file, proto_dir)
            print(f"Copied {proto_file.name} to proto/")
            copied_files += 1
        
        if copied_files > 0:
            return True
    
    print("Warning: No proto files found. Protobuf generation will be skipped.")
    return False

def generate_protobuf_files():
    """Generate protobuf C++ files and create legacy directory structure"""
    proto_dir = Path("proto")
    proto_files = list(proto_dir.glob("*.proto"))
    if not proto_files:
        print("Warning: No proto files found. Skipping protobuf generation.")
        return
    
    # Create build directory for new generated files
    build_dir = Path("build")
    build_dir.mkdir(exist_ok=True)
    
    # Create legacy directory structure that source code expects
    legacy_build_dir = Path("libraries/protobuf/build")
    legacy_build_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating protobuf C++ files...")
    
    success = False
    for proto_file in proto_files:
        cmd = [
            "protoc",
            f"--cpp_out={build_dir}",
            f"--proto_path={proto_dir}",
            str(proto_file)
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"  Generated protobuf files for {proto_file.name}")
            success = True
        except subprocess.CalledProcessError as e:
            print(f"  Warning: Failed to generate protobuf for {proto_file.name}: {e}")
        except FileNotFoundError:
            print("  Error: protoc not found. Please install protobuf compiler:")
            print("  macOS: brew install protobuf")
            print("  Ubuntu: sudo apt-get install protobuf-compiler")
            return
    
    if success:
        # Copy generated files to legacy location for compatibility
        print("Creating legacy protobuf directory structure...")
        for pb_file in build_dir.glob("*.pb.h"):
            legacy_file = legacy_build_dir / pb_file.name
            shutil.copy2(pb_file, legacy_file)
            print(f"  Copied {pb_file.name} to legacy location")
        
        for pb_file in build_dir.glob("*.pb.cc"):
            legacy_file = legacy_build_dir / pb_file.name
            shutil.copy2(pb_file, legacy_file)
    
    print("Protobuf generation complete!")

def main():
    parser = argparse.ArgumentParser(description="MIDI-GPT setup script")
    parser.add_argument("--dev", action="store_true", help="Development build")
    parser.add_argument("--test", action="store_true", help="Test build")
    parser.add_argument("--no-torch", action="store_true", help="Build without PyTorch")
    parser.add_argument("--mac-os", action="store_true", help="Build for macOS")
    parser.add_argument("--clean", action="store_true", help="Clean build directory")
    
    args = parser.parse_args()
    
    if args.clean:
        for dir_name in ["build", "python_lib", "*.egg-info", "libraries/protobuf/build"]:
            if Path(dir_name).exists():
                if dir_name.endswith('*'):
                    for path in Path('.').glob(dir_name):
                        if path.is_dir():
                            shutil.rmtree(path)
                else:
                    shutil.rmtree(dir_name, ignore_errors=True)
        print("Cleaned build directories")
        return
    
    print("=== MIDI-GPT Python 3.9 Setup ===")
    print(f"Using Python {sys.version}")
    
    # Create directories
    os.makedirs("python_lib", exist_ok=True)
    
    # Setup midifile library
    print("Checking midifile library...")
    midifile_ok = check_and_setup_midifile()
    if not midifile_ok:
        print("Warning: midifile library setup failed")
    
    # Setup protobuf - this must happen before building
    print("Setting up protobuf...")
    has_proto = setup_proto_directory()
    if has_proto:
        generate_protobuf_files()
    else:
        print("Continuing without protobuf support...")
    
    # Add midifile include paths
    additional_includes = []
    if Path("libraries/midifile/include").exists():
        additional_includes.append("libraries/midifile/include")
        print("Added midifile include directory")
    
    # Get PyTorch info
    torch_info = get_pytorch_info()
    no_torch = args.no_torch or not torch_info['available']
    
    if torch_info['available'] and not args.no_torch:
        print(f"Found PyTorch {torch_info['version']}")
    else:
        print("Building without PyTorch support")
        no_torch = True
    
    # Source files  
    sources = [
        "src/lib.cpp",
        "src/common/data_structures/train_config.cpp",
        "src/dataset_creation/compression/lz4.cpp",
        "src/dataset_creation/dataset_manipulation/bytes_to_file.cpp",
    ]
    
    # Include directories
    include_dirs = [
        pybind11.get_include(),
        "src",
        "include",
        "build", 
        "proto",
    ] + additional_includes
    
    if torch_info['available'] and not no_torch:
        include_dirs.extend(torch_info.get('include_dirs', []))
    
    # Libraries - add midifile if available
    libraries = ['protobuf']
    library_dirs = []
    
    if torch_info['available'] and not no_torch:
        libraries.extend(torch_info.get('libraries', []))
        library_dirs.extend(torch_info.get('library_dirs', []))
    
    # Compiler flags - we need C++17 for modern protobuf
    extra_compile_args = ['-std=c++17', '-fPIC', '-O3', '-DNDEBUG']
    
    if args.mac_os or sys.platform.startswith("darwin"):
        extra_compile_args.extend(['-stdlib=libc++', '-mmacosx-version-min=10.14'])
    
    if no_torch:
        extra_compile_args.append('-DNO_TORCH')
    
    if args.dev:
        extra_compile_args = [f for f in extra_compile_args if f not in ['-O3', '-DNDEBUG']]
        extra_compile_args.extend(['-g', '-O0'])
    
    print(f"Compiler flags: {extra_compile_args}")
    
    # Create extension
    extension = Extension(
        'midigpt',
        sources=sources,
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        extra_compile_args=extra_compile_args,
        language='c++'  # This tells setuptools to use C++ compiler for linking
    )
    
    # Custom build command to handle mixed C/C++ compilation
    class CustomBuildExt(build_ext):
        def build_extensions(self):
            os.makedirs("python_lib", exist_ok=True)
            build_ext.build_extensions(self)
            
        def build_extension(self, ext):
            # Handle C and C++ files differently
            original_sources = ext.sources[:]
            original_extra_compile_args = ext.extra_compile_args[:]
            
            # Separate C and C++ sources
            c_sources = [s for s in ext.sources if s.endswith('.c')]
            cpp_sources = [s for s in ext.sources if not s.endswith('.c')]
            
            # Compile C files without C++ flags
            if c_sources:
                print(f"Compiling C files: {c_sources}")
                c_flags = [f for f in original_extra_compile_args 
                          if f not in ['-std=c++17', '-stdlib=libc++']]
                ext.sources = c_sources
                ext.extra_compile_args = c_flags
                super().build_extension(ext)
            
            # Compile C++ files with full flags
            if cpp_sources:
                print(f"Compiling C++ files: {cpp_sources}")
                ext.sources = cpp_sources
                ext.extra_compile_args = original_extra_compile_args
                super().build_extension(ext)
            
            # Restore original values
            ext.sources = original_sources
            ext.extra_compile_args = original_extra_compile_args
            
        def get_ext_fullpath(self, ext_name):
            # Force output to python_lib directory
            filename = self.get_ext_filename(ext_name)
            return os.path.join("python_lib", filename)
    
    # Build - force the correct setuptools command
    print("Starting build process...")
    
    # Run setup with explicit commands to avoid argument conflicts
    sys.argv = ['setup.py', 'build_ext', '--inplace']
    
    setup(
        name="midigpt",
        version="0.1.0",
        ext_modules=[extension],
        cmdclass={'build_ext': CustomBuildExt},
        zip_safe=False,
    )
    
    # Test if requested
    if args.test:
        print("Testing import...")
        test_env = os.environ.copy()
        test_env['PYTHONPATH'] = f"{os.getcwd()}/python_lib:{test_env.get('PYTHONPATH', '')}"
        
        result = subprocess.run([
            sys.executable, "-c", "import midigpt; print('✅ Import successful')"
        ], env=test_env, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout.strip())
            print("=== Build completed successfully ===")
        else:
            print(f"❌ Import test failed: {result.stderr.strip()}")
            sys.exit(1)
    else:
        print("=== Build completed successfully ===")
        print("To test: export PYTHONPATH=$PWD/python_lib:$PYTHONPATH && python -c 'import midigpt'")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3.9
"""
MIDI-GPT setup script with automatic virtual environment management
Replacement for create_python_library.sh that handles venv properly
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path

def create_or_activate_venv():
    """Create or activate virtual environment"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("Found existing virtual environment")
        # Check if it's the right Python version
        try:
            result = subprocess.run([
                str(venv_path / "bin" / "python"), "--version"
            ], capture_output=True, text=True, check=True)
            
            if "3.9" not in result.stdout:
                print("Virtual environment has wrong Python version, recreating...")
                shutil.rmtree(venv_path)
            else:
                print("Virtual environment Python version OK")
        except subprocess.CalledProcessError:
            print("Virtual environment corrupted, recreating...")
            shutil.rmtree(venv_path)
    
    if not venv_path.exists():
        print("Creating new virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_path)])
    
    # Return paths
    if os.name == 'nt':  # Windows
        python_exe = venv_path / "Scripts" / "python.exe"
        pip_exe = venv_path / "Scripts" / "pip.exe"
    else:  # Unix-like
        python_exe = venv_path / "bin" / "python"
        pip_exe = venv_path / "bin" / "pip"
    
    return str(python_exe), str(pip_exe)

def install_dependencies(pip_exe, mac_os=False):
    """Install dependencies in virtual environment"""
    print("Installing dependencies in virtual environment...")
    
    # Upgrade pip first
    subprocess.check_call([pip_exe, "install", "--upgrade", "pip"])
    
    # Install in specific order to avoid conflicts
    packages = [
        "numpy<2.0",  # NumPy compatibility constraint
        "torch>=2.0.0",
        "pybind11>=2.12.0", 
        "protobuf>=4.0.0",
        "transformers>=4.30.0",
        "tqdm",
    ]
    
    if mac_os:
        # For macOS, might need specific torch version
        packages[1] = "torch>=2.0.0"
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([pip_exe, "install", package])
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to install {package}: {e}")

def setup_proto_directory():
    """Set up protobuf directory structure"""
    proto_dir = Path("proto")
    proto_dir.mkdir(exist_ok=True)
    
    # Try to copy proto files from old location
    old_proto_locations = [
        Path("libraries/protobuf/src"),
        Path("libraries/protobuf/build"), 
        Path("src/proto"),
    ]
    
    found_protos = False
    for old_path in old_proto_locations:
        if old_path.exists():
            for proto_file in old_path.glob("*.proto"):
                shutil.copy(proto_file, proto_dir)
                print(f"Copied {proto_file.name} to proto/")
                found_protos = True
    
    if not found_protos:
        print("No existing proto files found - you may need to create them")
    
    return found_protos

def run_in_venv(python_exe, script_args):
    """Run the actual setup in the virtual environment"""
    print("Running setup in virtual environment...")
    
    # Create a setup script that will run in the venv
    setup_script = '''
import os
import sys
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import pybind11

def get_torch_info():
    """Get PyTorch information"""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        # Get include paths
        include_dirs = []
        try:
            include_dirs = torch.utils.cpp_extension.include_paths()
        except AttributeError:
            # Fallback for newer PyTorch versions
            torch_path = Path(torch.__file__).parent
            include_path = torch_path / "include"
            if include_path.exists():
                include_dirs = [str(include_path)]
        
        return {
            'version': torch.__version__,
            'include_dirs': include_dirs,
            'libraries': ['torch', 'torch_cpu'],
            'available': True
        }
    except ImportError:
        return {'available': False}

# Get command line arguments
dev = "--dev" in sys.argv
test = "--test" in sys.argv
no_torch = "--no-torch" in sys.argv
mac_os = "--mac-os" in sys.argv

# Get torch info
torch_info = get_torch_info() if not no_torch else {'available': False}
if not torch_info['available'] and not no_torch:
    print("PyTorch not available, building without torch support")
    no_torch = True

# Source files
sources = [
    "src/lib.cpp",
    "src/common/data_structures/train_config.cpp",
    "src/dataset_creation/compression/lz4.c", 
    "src/dataset_creation/dataset_manipulation/bytes_to_file.cpp",
]

# Include directories
include_dirs = [
    pybind11.get_include(),
    "src",
    "include",
    "build",
    "proto",
]

if torch_info['available']:
    include_dirs.extend(torch_info.get('include_dirs', []))

# Libraries
libraries = ['protobuf']
if torch_info['available']:
    libraries.extend(torch_info.get('libraries', []))

# Compiler flags
extra_compile_args = ['-std=c++17', '-fPIC']
if no_torch:
    extra_compile_args.append('-DNO_TORCH')
if mac_os:
    extra_compile_args.extend(['-stdlib=libc++', '-mmacosx-version-min=10.14'])
if dev:
    extra_compile_args.extend(['-g', '-O0'])
else:
    extra_compile_args.extend(['-O3', '-DNDEBUG'])

# Create extension
extension = Extension(
    'midigpt',
    sources=sources,
    include_dirs=include_dirs,
    libraries=libraries,
    extra_compile_args=extra_compile_args,
    language='c++'
)

# Custom build command
class CustomBuildExt(build_ext):
    def build_extensions(self):
        # Ensure output directory exists
        os.makedirs("python_lib", exist_ok=True)
        build_ext.build_extensions(self)

# Run setup
setup(
    name="midigpt",
    version="0.1.0",
    ext_modules=[extension],
    cmdclass={'build_ext': CustomBuildExt},
    zip_safe=False,
)

print("Build completed successfully!")
'''
    
    # Write the setup script
    setup_script_path = Path("temp_setup_in_venv.py")
    setup_script_path.write_text(setup_script)
    
    try:
        # Run the setup in the virtual environment
        cmd = [python_exe, str(setup_script_path), "build_ext", "--inplace"] + script_args
        subprocess.check_call(cmd)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Setup failed: {e}")
        return False
    finally:
        # Clean up temp script
        if setup_script_path.exists():
            setup_script_path.unlink()

def test_import(python_exe):
    """Test the import in the virtual environment"""
    print("Testing import...")
    
    test_script = '''
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), "python_lib"))

try:
    import midigpt
    print("✅ midigpt import successful")
    attrs = [attr for attr in dir(midigpt) if not attr.startswith("_")]
    print(f"Available functions: {attrs[:10]}{'...' if len(attrs) > 10 else ''}")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
'''
    
    result = subprocess.run([python_exe, "-c", test_script], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
        return True
    else:
        print(result.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="MIDI-GPT setup with virtual environment")
    parser.add_argument("--dev", action="store_true", help="Development build")
    parser.add_argument("--test", action="store_true", help="Test build")
    parser.add_argument("--no-torch", action="store_true", help="Build without PyTorch")
    parser.add_argument("--mac-os", action="store_true", help="Build for macOS")
    parser.add_argument("--clean", action="store_true", help="Clean build and venv")
    
    args = parser.parse_args()
    
    if args.clean:
        for dir_name in ["build", "python_lib", "venv", "*.egg-info"]:
            shutil.rmtree(dir_name, ignore_errors=True)
        print("Cleaned all directories including virtual environment")
        return
    
    print("=== MIDI-GPT Setup with Virtual Environment ===")
    
    # Check base Python version
    if sys.version_info < (3, 9):
        print("Error: Python 3.9 or higher required")
        sys.exit(1)
    
    # Create/activate virtual environment
    python_exe, pip_exe = create_or_activate_venv()
    print(f"Using virtual environment Python: {python_exe}")
    
    # Install dependencies
    install_dependencies(pip_exe, args.mac_os)
    
    # Setup protobuf
    print("Setting up protobuf...")
    setup_proto_directory()
    
    # Build the extension
    script_args = []
    if args.dev:
        script_args.append("--dev")
    if args.no_torch:
        script_args.append("--no-torch")
    if args.mac_os:
        script_args.append("--mac-os")
    
    success = run_in_venv(python_exe, script_args)
    
    if not success:
        print("❌ Build failed")
        sys.exit(1)
    
    # Test if requested
    if args.test:
        if not test_import(python_exe):
            print("❌ Import test failed")
            sys.exit(1)
    
    print("=== Setup completed successfully ===")
    print(f"Virtual environment created at: venv/")
    print(f"Library built in: python_lib/")
    print(f"To use the library:")
    print(f"  source venv/bin/activate")
    print(f"  export PYTHONPATH=$PWD/python_lib:$PYTHONPATH")
    print(f"  python -c 'import midigpt'")

if __name__ == "__main__":
    main()
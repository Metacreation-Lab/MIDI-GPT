#!/usr/bin/env python3
"""
MIDI-GPT Environment Setup Script
Quick setup for virtual environment and core dependencies
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

PYTHON_VERSION = "3.9"
VENV_NAME = "venv"

def run_command(cmd, check=True, capture_output=False):
    """Run a shell command with error handling"""
    try:
        result = subprocess.run(cmd, shell=True, check=check, 
                              capture_output=capture_output, text=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {cmd}")
        print(f"Error: {e}")
        if capture_output:
            print(f"Output: {e.stdout}")
            print(f"Error: {e.stderr}")
        sys.exit(1)

def check_python():
    """Check if Python 3.9+ is available"""
    print("Checking Python version...")
    
    # Try python3.9 first
    try:
        result = run_command("python3.9 --version", capture_output=True)
        version = result.stdout.strip()
        print(f"✅ Python 3.9 found: {version}")
        return "python3.9"
    except:
        pass
    
    # Try python3
    try:
        result = run_command("python3 --version", capture_output=True)
        version = result.stdout.strip()
        if "3.9" in version or "3.10" in version or "3.11" in version or "3.12" in version:
            print(f"✅ Compatible Python found: {version}")
            return "python3"
    except:
        pass
    
    print("❌ Python 3.9+ not found. Please install Python 3.9+")
    print("   macOS: brew install python@3.9")
    print("   Ubuntu: sudo apt install python3.9 python3.9-venv")
    sys.exit(1)

def check_cuda():
    """Check if CUDA is available"""
    try:
        run_command("nvidia-smi", capture_output=True)
        print("CUDA detected")
        return True
    except:
        print("No CUDA detected - will install CPU-only PyTorch")
        return False

def setup_venv(python_cmd):
    """Create and setup virtual environment"""
    venv_path = Path(VENV_NAME)
    
    if venv_path.exists():
        print(f"Virtual environment already exists: {VENV_NAME}")
        response = input("Remove and recreate? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            shutil.rmtree(venv_path)
            print("Removed existing virtual environment")
        else:
            print("Keeping existing virtual environment")
            return
    
    print("Creating virtual environment...")
    run_command(f"{python_cmd} -m venv {VENV_NAME}")
    print(f"Virtual environment created: {VENV_NAME}")

def install_dependencies(has_cuda=False):
    """Install core dependencies"""
    # Determine pip command
    if sys.platform == "win32":
        pip_cmd = f"{VENV_NAME}\\Scripts\\pip"
    else:
        pip_cmd = f"{VENV_NAME}/bin/pip"
    
    print("Upgrading pip...")
    run_command(f"{pip_cmd} install --upgrade pip setuptools wheel")
    
    print("Installing core dependencies...")
    
    # Install PyTorch - FIXED: Use proper shell escaping
    if has_cuda:
        print("Installing PyTorch with CUDA support...")
        run_command(f'{pip_cmd} install "torch>=2.0.0" "torchvision" "torchaudio" '
                   f'--index-url https://download.pytorch.org/whl/cu118')
    else:
        print("Installing CPU-only PyTorch...")
        run_command(f'{pip_cmd} install "torch>=2.0.0" "torchvision" "torchaudio" '
                   f'--index-url https://download.pytorch.org/whl/cpu')
    
    # Install other dependencies
    deps = [
        'numpy>=1.21.0',
        'protobuf>=4.0.0', 
        'pybind11[global]>=2.12.0',
        'transformers>=4.30.0',
        'tqdm'
    ]
    
    for dep in deps:
        print(f"Installing {dep}...")
        run_command(f'{pip_cmd} install "{dep}"')

def print_next_steps():
    """Print next steps for the user"""
    if sys.platform == "win32":
        activate_cmd = f"{VENV_NAME}\\Scripts\\activate"
    else:
        activate_cmd = f"source {VENV_NAME}/bin/activate"
    
    print("\n✅ Environment setup complete!")
    print("\nNext steps:")
    print(f"1. Activate the environment: {activate_cmd}")
    print("2. Build MIDI-GPT:")
    print("   python setup_midigpt.py --test")
    if sys.platform == "darwin":  # macOS
        print("   (or: python setup_midigpt.py --mac-os --test)")
    
    print("\n📋 Quick verification:")
    print(f"   {activate_cmd}")
    print("   python -c \"import torch; print(f'PyTorch: {torch.__version__}')\"")
    print("   python -c \"import numpy; print(f'NumPy: {numpy.__version__}')\"")

def main():
    print("MIDI-GPT Environment Setup")
    print("==========================")
    
    # Check Python
    python_cmd = check_python()
    
    # Check CUDA
    has_cuda = check_cuda()
    
    # Setup virtual environment
    setup_venv(python_cmd)
    
    # Install dependencies
    install_dependencies(has_cuda)
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()
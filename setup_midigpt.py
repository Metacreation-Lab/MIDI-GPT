#!/usr/bin/env python3

import argparse
import os
import shutil
import subprocess
import sys
import platform
from pathlib import Path

try:
    import pybind11
    from setuptools import setup, Extension
    from setuptools.command.build_ext import build_ext
except ImportError as e:
    print(f"Missing required dependency: {e}")
    sys.exit(1)

def fix_macos_library_paths():
    """Fix macOS library path issues after build"""
    if platform.system() != "Darwin":
        return
    
    so_files = list(Path("python_lib").glob("*.so"))
    if not so_files:
        print("Warning: No .so files found to fix")
        return
    
    for so_file in so_files:
        print(f"Fixing library paths for {so_file}")
        
        try:
            # Get current library dependencies
            result = subprocess.run(['otool', '-L', str(so_file)], 
                                  capture_output=True, text=True, check=True)
            
            # Fix PyTorch library paths
            torch_lib_paths = []
            try:
                import torch
                torch_lib_path = Path(torch.__file__).parent / "lib"
                if torch_lib_path.exists():
                    torch_lib_paths.append(str(torch_lib_path))
                    print(f"Found PyTorch libraries at: {torch_lib_path}")
            except ImportError:
                print("PyTorch not available for path detection")
            
            # Fix any @rpath/libtorch.dylib references to absolute paths
            for line in result.stdout.split('\n'):
                if '@rpath/libtorch' in line:
                    old_path = line.strip().split()[0]
                    lib_name = old_path.replace('@rpath/', '')
                    
                    # Try to find the actual library
                    for torch_path in torch_lib_paths:
                        candidate = Path(torch_path) / lib_name
                        if candidate.exists():
                            print(f"Fixing PyTorch path: {old_path} -> {candidate}")
                            subprocess.run([
                                'install_name_tool', '-change',
                                old_path, str(candidate), str(so_file)
                            ], check=True)
                            break
            
            # Fix protobuf paths (existing logic)
            if '/usr/local/opt/protobuf/lib/libprotobuf' in result.stdout:
                for line in result.stdout.split('\n'):
                    if 'libprotobuf' in line and '/usr/local/opt/protobuf/lib/' in line:
                        old_path = line.strip().split()[0]
                        lib_name = Path(old_path).name
                        new_path = f"/usr/local/lib/{lib_name}"
                        
                        print(f"Fixing protobuf path: {old_path} -> {new_path}")
                        subprocess.run([
                            'install_name_tool', '-change', 
                            old_path, new_path, str(so_file)
                        ], check=True)
            
            # Add rpaths for library locations
            rpaths_to_add = [
                "/usr/local/lib",
                "/opt/homebrew/lib"
            ]
            
            # Add PyTorch lib paths as rpaths
            rpaths_to_add.extend(torch_lib_paths)
            
            for rpath in rpaths_to_add:
                if Path(rpath).exists():
                    try:
                        subprocess.run([
                            'install_name_tool', '-add_rpath', rpath, str(so_file)
                        ], check=True, capture_output=True)
                        print(f"Added rpath: {rpath}")
                    except subprocess.CalledProcessError:
                        # Rpath might already exist, ignore
                        pass
                        
        except subprocess.CalledProcessError as e:
            print(f"Warning: Could not fix library paths for {so_file}: {e}")

def test_import():
    """Test if the built module can be imported"""
    python_lib = Path("python_lib")
    if not python_lib.exists():
        print("Error: python_lib directory not found")
        return False
    
    # Test import
    try:
        original_path = sys.path[:]
        sys.path.insert(0, str(python_lib.absolute()))
        
        print("Testing midigpt import...")
        import midigpt
        print("✅ SUCCESS: midigpt imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        
        # Try to give helpful error messages
        if "symbol not found" in str(e):
            print("\nThis appears to be a library linking issue.")
            print("Try running: brew install protobuf")
            
        elif "Library not loaded" in str(e):
            print("\nThis appears to be a library path issue.")
            print("The fix_macos_library_paths() function should have resolved this.")
            
        return False
        
    finally:
        sys.path[:] = original_path

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
            subprocess.run([
                "git", "clone", 
                "https://github.com/craigsapp/midifile", 
                str(midifile_path)
            ], check=True, capture_output=True)
            
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
    
    existing_proto_files = list(proto_dir.glob("*.proto"))
    if existing_proto_files:
        print(f"Found {len(existing_proto_files)} proto files in proto/")
        return True
    
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
    
    build_dir = Path("build")
    build_dir.mkdir(exist_ok=True)
    
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
            return
    
    if success:
        print("Creating legacy protobuf directory structure...")
        for pb_file in build_dir.glob("*.pb.h"):
            legacy_file = legacy_build_dir / pb_file.name
            shutil.copy2(pb_file, legacy_file)
            print(f"  Copied {pb_file.name} to legacy location")
        
        for pb_file in build_dir.glob("*.pb.cc"):
            legacy_file = legacy_build_dir / pb_file.name
            shutil.copy2(pb_file, legacy_file)
    
    print("Protobuf generation complete!")

class BuildExtCommand(build_ext):
    def build_extensions(self):
        # Add compiler flags for macOS
        if platform.system() == "Darwin":
            for ext in self.extensions:
                ext.extra_compile_args.extend([
                    '-std=c++17', '-fPIC', '-O3', '-DNDEBUG', 
                    '-stdlib=libc++', '-mmacosx-version-min=10.14'
                ])
                ext.extra_link_args.extend([
                    '-stdlib=libc++', '-mmacosx-version-min=10.14'
                ])
        
        super().build_extensions()

def install_to_environment():
    """Install the built library to the current Python environment"""
    import site
    import glob
    
    # Find the built .so file
    so_files = glob.glob("python_lib/midigpt*.so")
    if not so_files:
        print("❌ No built library found. Run build first.")
        return False
    
    so_file = so_files[0]
    print(f"Found built library: {so_file}")
    
    # Get site-packages directory
    try:
        # If we're in a virtual environment, use that
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            import sysconfig
            site_packages = sysconfig.get_paths()['purelib']
        else:
            # Try to get user site-packages first
            if hasattr(site, 'getusersitepackages'):
                site_packages = site.getusersitepackages()
            else:
                site_packages = site.getsitepackages()[0]
        
        print(f"Installing to: {site_packages}")
        
        # Create site-packages if it doesn't exist
        os.makedirs(site_packages, exist_ok=True)
        
        # Copy the .so file
        dest_file = os.path.join(site_packages, os.path.basename(so_file))
        shutil.copy2(so_file, dest_file)
        
        print(f"✅ Successfully installed {os.path.basename(so_file)} to {site_packages}")
        
        # Test the installation
        try:
            import midigpt
            print("✅ Installation verified - midigpt can be imported")
            return True
        except ImportError as e:
            print(f"❌ Installation failed - cannot import midigpt: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Installation failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="MIDI-GPT setup script")
    parser.add_argument("--dev", action="store_true", help="Development build")
    parser.add_argument("--test", action="store_true", help="Test build and import")
    parser.add_argument("--no-torch", action="store_true", help="Build without PyTorch")
    parser.add_argument("--mac-os", action="store_true", help="Build for macOS")
    parser.add_argument("--clean", action="store_true", help="Clean build directory")
    parser.add_argument("--install", action="store_true", help="Install to current Python environment")
    
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
    
    # Setup protobuf
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
        "src/dataset_creation/dataset_manipulation/bytes_to_file.cpp"
    ]
    
    # Add generated protobuf source files
    build_dir = Path("build")
    if build_dir.exists():
        pb_cc_files = list(build_dir.glob("*.pb.cc"))
        for pb_file in pb_cc_files:
            sources.append(str(pb_file))
            print(f"Added protobuf source: {pb_file}")
    
    # Add midifile source files directly
    midifile_src_dir = Path("libraries/midifile/src")
    if midifile_src_dir.exists():
        midifile_sources = list(midifile_src_dir.glob("*.cpp"))
        if midifile_sources:
            for midifile_src in midifile_sources:
                sources.append(str(midifile_src))
                print(f"Added midifile source: {midifile_src}")
        else:
            # Try .c files if .cpp doesn't exist
            midifile_sources = list(midifile_src_dir.glob("*.c"))
            for midifile_src in midifile_sources:
                sources.append(str(midifile_src))
                print(f"Added midifile source: {midifile_src}")
    else:
        print("Warning: midifile source directory not found")
    
    # Include directories
    include_dirs = [
        "src", "include", "build", "proto"
    ] + additional_includes
    
    if torch_info['available'] and not no_torch:
        include_dirs.extend(torch_info['include_dirs'])
    
    # Library directories and libraries
    library_dirs = []
    libraries = ["protobuf"]
    
    # Add midifile library
    midifile_lib_path = Path("libraries/midifile/lib")
    if midifile_lib_path.exists():
        library_dirs.append(str(midifile_lib_path))
        libraries.append("midifile")
        print(f"Added midifile library: {midifile_lib_path}")
    
    if torch_info['available'] and not no_torch:
        library_dirs.extend(torch_info['library_dirs'])
        libraries.extend(torch_info['libraries'])
    
    # Compiler flags
    compiler_flags = ['-std=c++17', '-fPIC', '-O3', '-DNDEBUG']
    if args.mac_os or platform.system() == "Darwin":
        compiler_flags.extend(['-stdlib=libc++', '-mmacosx-version-min=10.14'])
    
    print(f"Compiler flags: {compiler_flags}")
    
    # Create extension
    ext = Extension(
        name="midigpt",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=compiler_flags,
        language="c++"
    )
    
    print("Starting build process...")
    print(f"Compiling C++ files: {sources}")
    
    # Override sys.argv to avoid passing our custom args to setup()
    original_argv = sys.argv[:]
    sys.argv = ['setup.py', 'build_ext', '--inplace', '--build-lib=python_lib']
    
    try:
        setup(
            name="midigpt",
            ext_modules=[ext],
            cmdclass={'build_ext': BuildExtCommand},
            zip_safe=False
        )
        
        # Fix macOS library paths after successful build
        fix_macos_library_paths()
        
        print("Build completed successfully!")

        # Install if requested
        if args.install:
            print("\nInstalling to current Python environment...")
            install_success = install_to_environment()
            if not install_success:
                print("⚠️  Installation failed, but build was successful.")
                print("You can still use the library by adding python_lib/ to your PYTHONPATH")
                sys.exit(1)
        
        if args.test:
            # If we installed, test from the installed location
            if args.install:
                print("\nTesting installed library...")
                try:
                    import midigpt
                    print("✅ Installed library imports successfully")
                    print(f"Library version: {midigpt.version()}")
                except Exception as e:
                    print(f"❌ Installed library test failed: {e}")
                    sys.exit(1)
            else:
                # Keep your existing test logic here
                success = test_import()
                if success:
                    print("\n🎉 MIDI-GPT Python 3.9 refactoring completed successfully!")
                    print("The library is ready to use.")
                else:
                    print("\n⚠️  Build completed but import test failed.")
                    print("You may need to install system dependencies:")
                    print("  brew install protobuf")
                    sys.exit(1)
                
    except Exception as e:
        print(f"Build failed: {e}")
        sys.exit(1)
    finally:
        sys.argv[:] = original_argv

if __name__ == "__main__":
    main()
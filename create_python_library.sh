#!/bin/bash

compute_canada=false
mac_os=false
cpu=false
no_torch=false
niagara=false
env_name="venv"
parent_dir="midigpt_workspace" # default parent directory name
cuda=false
trace=false


# Parse arguments
for ((i=1;i<=$#;i++)); do
  case ${!i} in
    --trace)
      trace=true
      ;;
    --compute_canada)
      compute_canada=true
      ;;
    --cpu)
      cpu=true
      ;;
    --mac_os)
      mac_os=true
      ;;
    --no_torch)
      no_torch=true
      ;;
    --niagara)
      niagara=true
      ;;
    --env_name)
      i=$((i+1))
      env_name=${!i}
      ;;
    -n=*|--name=*) # new parent directory name option
      parent_dir="${!i#*=}"
      shift
      ;;
    --cuda)
      if $no_torch; then
        echo "Cannot use --cuda and --no_torch at the same time."
		exit 1
	  fi
      cuda=true
	  ;;
	*)
	  echo "Unknown option ${!i}"
	  exit 1
	  ;;
  esac
done


# Get the current directory name
dir_name=$(basename `pwd`)

# Get the parent directory name
parent_dir_name=$(basename $(dirname `pwd`))

if [ "$parent_dir_name" != "$parent_dir" ]; then

    # Go to the parent directory
    cd ..

    # Create the new parent directory
    mkdir $parent_dir

    # Move the old directory into the new parent directory
    mv $dir_name $parent_dir/

    # Change to the old directory, which is now inside the parent directory
    cd $parent_dir/$dir_name
fi

# Load modules if we are in compute_canada and niagara
if $compute_canada; then
    if $niagara; then
        module load CCEnv arch/avx512
    fi
    module load StdEnv/2020
    module load cmake/3.23.1
    module load gcc/11.3.0
    module load protobuf/3.12.3
    module load python/3.8.2

    mkdir ../CCLOG
fi

# Environment creation
if [[ -n ../$env_name ]]; then
    if [[ -d ../$env_name ]]; then
        echo "Environment $env_name already exists, activating it..."
    else
        echo "Environment $env_name does not exist, creating it..."
        if $compute_canada; then
            virtualenv ../$env_name
        else
            python3 -m venv ../$env_name
        fi
    fi
fi

source ../$env_name/bin/activate

# Install requirements
pip install -r pip_requirements/common_requirements.txt

if $compute_canada; then
  pip install -r pip_requirements/create_dataset_requirements.txt
fi
if $mac_os; then
    pip install -r pip_requirements/inference_requirements.txt
fi

if $compute_canada && ! $niagara; then # anf if no torch
    pip install -r pip_requirements/train_requirements.txt
fi

#deactivate

# Set CMake flags based on command line arguments
cmake_flags=""
if $compute_canada; then
  cmake_flags="$cmake_flags -Dcompute_canada=ON"
fi

if $no_torch; then 
    cmake_flags="$cmake_flags -Dno_torch=ON"
fi

if $trace; then
    cmake_flags="$cmake_flags -Dtrace=ON"
fi

# Code to check if libtorch and pybind11 are already downloaded
if ! $no_torch; then
    libtorch_path="libraries/libtorch"
    libtorch_url="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip"
    if $cuda; then 
	  libtorch_url="https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcu118.zip"
	fi
fi

pybind11_path="libraries/pybind11"
midifile_path="libraries/midifile"


pybind11_url="https://github.com/pybind/pybind11.git"
midifile_url="https://github.com/craigsapp/midifile"

if ! $no_torch; then
    if $mac_os; then
      libtorch_url="https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.0.1.zip"
    fi

    if $cpu; then
      libtorch_url="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip"
    fi

    # Check if libtorch folder exists and is not empty
    if [ ! -d "$libtorch_path" ] || [ -z "$(ls -A "$libtorch_path")" ]; then
        echo "libtorch folder does not exist or is empty. Downloading and extracting..."
        mkdir -p "$libtorch_path"
        curl -L "$libtorch_url" -o libtorch.zip
        unzip -q libtorch.zip -d libraries/
        rm libtorch.zip
        echo "libtorch downloaded and extracted."
    else
        echo "libtorch folder exists and is not empty. No need to download."
    fi
fi

# Check if pybind11 folder exists and is not empty
if [ ! -d "$pybind11_path" ] || [ -z "$(ls -A "$pybind11_path")" ]; then
    echo "pybind11 folder does not exist or is empty. Cloning the repository..."
    mkdir -p libraries
    git clone "$pybind11_url" "$pybind11_path"
    echo "pybind11 downloaded."
    cd libraries/pybind11
    git reset --hard 5ccb9e4
    cd ../../
    echo "pybind11 reset to working build"
else
    echo "pybind11 folder exists and is not empty. No need to download."
fi

# Check if midifile folder exists and is not empty
if [ ! -d "$midifile_path" ] || [ -z "$(ls -A "$midifile_path")" ]; then
	echo "midifile folder does not exist or is empty. Cloning the repository..."
	mkdir -p libraries
	git clone "$midifile_url" "$midifile_path"
	echo "midifile downloaded."
 	cd libraries/midifile
  	git reset --hard 838c62c
   	cd ../../
    	echo "midifile reset to working build"
else
	echo "midifile folder exists and is not empty. No need to download."
fi

# Middle section of the script to build the python library
rm -rf ./python_lib
mkdir ./python_lib
rm -rf ./libraries/protobuf/build
mkdir ./libraries/protobuf/build

cd ./libraries/protobuf/src
protoc --cpp_out ../build *.proto
cd ../../..

cd ./python_lib

if $mac_os; then
  cmake $cmake_flags .. -Dmac_os=ON -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)')
else
  cmake $cmake_flags ..
fi
make
python3 -c "import midigpt; print('midigpt python library built successfully')"

cd ..
if $compute_canada; then
  dos2unix create_dataset_compute_canada.sh
  dos2unix train_dataset.sh
fi
cd ./python_lib

cd ..

#!/bin/bash

clone=no
replace=no
test_train=no
inference=no
mac=no
python_path=/Library/Frameworks/Python.framework/Versions/3.8/bin
og=no

usage="
$(basename "$0") [-h] [-n] [-c] [-i] [-m] [-r] [-d 
directory] 
-- Script for setting up and testing the MIDI-GPT repository

where:
    -h  Show this help text
    -n  Test the training script imports
    -c  Clone the github MIDI-GPT repository
	-i  If you wish to setup repository for inference
    -d  Provide directory name where repo is/will be cloned
    -r  Replace directory if already exists
    -m  If on MacOS CPU
	"


OPTSTRING=":cnhiomrd:k:"

while getopts ${OPTSTRING} opt; do
case ${opt} in
h)
echo "${usage}"
exit 0
;;
i)
inference=yes
;;
n)
test_train=yes
;;
m)
mac=yes
;;
r)
replace=yes
;;
c)
clone=yes
;;
d)
repo=${OPTARG}
;;
:)
echo "Option -${OPTARG} requires an argument"
exit 1
;;
?)
echo "Invalid option: -${OPTARG}"
exit 1
;;
esac
done

if test "${clone}" = "yes" 
then
echo "Cloning MIDI-GPT"
fi

echo "In directory: ${repo}"
	
if test "${replace}" = "yes"
then
if [[ -d ${repo} ]]
then
echo "Directory ${repo} already exists, removing it"
rm -rf ${repo}
fi
fi
 
mkdir -p ${repo}
cd ${repo}

echo "Loading modules"

if test "${clone}" = "yes"
then
if [[ -d MIDI-GPT ]] || [[ -d ENV ]] 
then
	echo "MIDI-GPT or ENV directories already exist"
	exit 1
fi 
if test "${og}" = "yes"
then
{
	git clone https://www.github.com/Metacreation-Lab/MIDI-GPT.git

} || {
	echo "Cloning failed"
	exit 1
}
else
{
	git clone https://www.github.com/Metacreation-Lab/MIDI-GPT.git

} || {
	echo "Cloning failed"
	exit 1
}
fi

${python_path}/virtualenv --no-download ./ENV

else
if ! [[ -d MIDI-GPT ]] 
then
	echo "MIDI-GPT doesn't exist, try cloning the repository with the -c option"
	exit 1
fi
fi

{
	source ./ENV/bin/activate
} || {
	echo "ENV virtual environment doesn't exist"
	exit 1
}

echo "pip installs"

pip install --no-index --upgrade pip
pip install torch==1.13.0
pip install transformers==4.26.1

cd MIDI-GPT
if test "${og}" = "no"
then
git checkout main
fi

echo "Starting python library build"

{	if test "${inference}" = "yes"
	then
	echo "Building for inference"
    if test "${mac}" = "yes"
    then
    echo "On MacOS CPU"
    bash create_python_library.sh --mac_os 
    else
    echo "On Compute Canada"
	bash create_python_library.sh --compute_canada
    fi
	else
	echo "Building for training only"
	bash create_python_library.sh --test_build --compute_canada --no_torch
	fi
} || {
	echo "Build failed"
	exit 1
}

if test "${test_train}" = "yes"
then

cd ../

deactivate

echo "Activating environment"

source $PWD/venv/bin/activate
cd $PWD/MIDI-GPT/python_scripts

echo "Testing training script"

python3 -c "import train"

echo "Import tests done" 

fi

echo "Finished"

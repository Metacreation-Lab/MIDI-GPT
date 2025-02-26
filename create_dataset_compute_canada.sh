#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00

root_dir="" # Model directory to replace MODEL_NAME
data_dir=""  # Data directory to replace DATA_DIR
encoding="" # Encoding to replace ENCODING
output="" # Output to replace OUTPUT
metadata=""
test="no"
zip="no"
res="12"
max="-1"

# Parse arguments
for arg in "$@"
do
    case $arg in
        --root_dir=*)
        root_dir="${arg#*=}"
        shift # Remove --root_dir= from processing
        ;;
        --metadata=*)
        metadata="${arg#*=}"
        shift # Remove --metadata= from processing
        ;;
        --res=*)
        res="${arg#*=}"
        shift # Remove --metadata= from processing
        ;;
        --test=*)
        test="${arg#*=}"
        shift # Remove --metadata= from processing
        ;;
        --data_dir=*)
        data_dir="${arg#*=}"
        shift # Remove --data_dir= from processing
        ;;
        --encoding=*)
        encoding="${arg#*=}"
        shift # Remove --encoding= from processing
        ;;
        --output=*)
        output="${arg#*=}"
        shift # Remove --output= from processing
        ;;
        --zip=*)
        zip="${arg#*=}"
        shift # Remove --output= from processing
        ;;
        --max=*)
        max="${arg#*=}"
        shift # Remove --output= from processing
        ;;
    esac
done

module load CCEnv arch/avx512
module load StdEnv/2020
module load cmake/3.23.1
module load gcc/11.3.0
module load protobuf/3.12.3
module load python/3.8.2

mkdir -p $root_dir/CCLOG
source $root_dir/venv/bin/activate

cp $root_dir/MIDI-GPT/python_lib/midigpt.cpython-38-x86_64-linux-gnu.so $root_dir/MIDI-GPT/python_scripts

python3 $root_dir/MIDI-GPT/python_scripts/create_dataset.py --nthreads 40 --max_size $max --data_dir $data_dir --encoding $encoding --output $output --metadata $metadata --test $test --expressive --resolution $res

if [[ "$zip" == "yes" ]]
then
    cd $output
    cd ../
    zip -r EXPRESSIVE_GIGAMIDI_24_1920.zip $output
fi
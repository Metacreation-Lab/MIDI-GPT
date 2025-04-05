[![N|Solid](https://drive.google.com/uc?export=view&id=1u4xiWN3s0PAii8zn3-qxJ7wn35tBOypY)](https://metacreation.net/category/projects/)

# MIDI-GPT Guide

This is the repository for MIDI-GPT, a generative system based on the Transformer architecture that is designed for computer-assisted music composition workflows. This work was presented at the 39th Annual AAAI Conference in Philadelphia, USA in this [paper](https://arxiv.org/abs/2501.17011)

# Using MIDI-GPT 

The model provided is trained on GigaMIDI, and includes the following controls: midi instrument, density (1-10), note duration, and number of polyphony voices. It also allows you to turn on/off velocity and micro-timing (to capture interpretation).

## Installation

To successfully install the midigpt python library, use the script ```midigpt_setup_helper.sh```. You may first download this script on its own and run it, which will clone the repository and build the library. Below is an example of the usage:

```sh
bash midigpt_setup_helper.sh -i -c -d midigpt_dir
```

>**Note:** Python 3.8 is required for the library
>**Note:** If you're building on mac, use the ```-m```argument.

## Inference

Once downloaded, MIDI-GPT is ready to use. ```python_scripts_for_testing/pythoninferencetest.py``` is an example of using MIDI-GPT. In summary, three objects need to be created before sampling:
- Piece: Load the MIDI file into a JSON representation of the MIDI piece
- Status: This dict indicates the sampling process that is desired (on which tracks, continuation/resampling/infilling, etc.) as well as attribute control values
- Param: This dict indicates sampling parameters such as temperature or number of generated bars per step

You must provide an input MIDI file, the checkpoint model file and an optional output MIDI file. Our model is provided in the ```models/model.zip``` file.

Then, using the ```midigpt``` Python API, call the sample function with these objects as arguments. After sampling, the result can then be converted and saved into a MIDI file.

# Training MIDI-GPT

Training the model was done on computing clusters on Compute Canada, therefore the training scripts are tailored to this platform but may easily be adapted to similar platforms. Training was done using the GigaMIDI dataset, first serialzed into a compressed file using ```create_dataset_compute_canada.sh``` and ```python_scripts/create_dataset.py```. The training was executed using the ```python_scripts/train.py```. Finally, the model weights file is converted from the training checkpoint using ```convert.py```.

If you're unfamiliar with Compute Canada, make sure to check the introductory .md [here]().

## Installation - Cedar and Niagara
0. You might want to allocate an interactive session with salloc:

>**Note:** You DON'T need to do this in Niagara.

```sh
salloc --time=3:0:0 --nodes 1 --cpus-per-task 32 --mem=128000 --account=user
```

1. First, make sure to clone the MMM_API into a folder in your CC machine:
```sh
https://github.com/Metacreation-Lab/MIDI-GPT
```
2. Then we must load the standard environments and some dependencies:

>**Note:** If you're building in Niagara, load this first:
```sh
module load CCEnv arch/avx512
```
Then proceed to load the rest (If you're in Cedar, start from here):
```sh
module load StdEnv/2020
module load cmake/3.23.1
module load gcc/11.3.0
module load protobuf/3.12.3
module load python/3.8.2
```
3. Then we must create an environment and activate it:
```sh
virtualenv --no-download ./ENV                # ENV is the name of the environment
source ./ENV/bin/activate
pip install --no-index --upgrade pip

# For training only
pip install torch==1.13.0
pip install transformers==4.26.1         
```
4. Finally, just call the bash script with the correct argument:
```sh
bash create_python_library.sh --test_build --compute_canada
```
Or if you are planning to just train the model, add the argument excluding to torch library required only for inference:
```sh
bash create_python_library.sh --no_torch --compute_canada
```
5. To test the library imports for training, run the train.py script by importing it:
```sh
cd python_scripts
python3 -c "import train"
```
> **Note:** A helper script ```midigpt_setup_helper.sh``` does all these steps autmoatically (for training or inference). Download it individually and run it where you wish to clone the repository.
> **Note:** If you run the code without the --test_build flag, it will still compile and create the python library but it won't test it with the current model in production.
> **Note:** The other flag (--compute_canada) is necesary to build the code properly.

That's it!

Everything should get installed correctly in your python environment! If you log out and back in to CC make sure to activate the environment in which you installed the API.

## Training

### Dataset Building

In order to train a new model, you must first build a dataset. You can upload the files you need using Globus (check the CC [guide]()).

> **Note**: Remember that to copy from the shared folder to your own folders you must use absolute paths.

The data should be organized in a way where all midi files are contained within three folders ```train```, ```test```, and ```valid```. Further directories can be used to organize the midi files as long as they are within these three directories.

If your dataset is a single folder containing all the midi files, we provide a helper script that automatically slits the dataset to 80%-10%-10%. Simply modify ```data_split.sh``` to match your cas and run.

Once you have the folder with the data, run the following command
```sh
sh create_dataset_compute_canada.sh --root_dir=<root_dir> --encoding=<encoding> --data_dir=<data_dir> --output=<output>
```
where:
- ```<root_dir>``` is the root folder where the midigpt repository folder is located
- ```<encoding>``` is the conder to use. We suggest using ```EXPRESSIVE_ENCODER```
- ```<data_dir>``` is the dataset folder containing the three ```train```, ```test```, and ```valid``` folders.
- ```<output>``` is the location of the ouptt ```.arr``` file. The resulting file while be ```<output>_NUM_BARS=<num_bars>_RESOLUTION_<resolution>.arr```
>**Note:** If you are on Compute Canada, we suggest you run these commands through an sbatch job as they can take some time.

### Training a Model

To train a model, run the train.py file. Different lab members have managed to set the paths differently. What works for me is to use global paths. An example would be:
```sh
python train.py --arch gpt2 --config /home/user/scratch/TRAINING-master/config/gpt2_tiny.json --encoding EXPRESSIVE_ENCODER --ngpu 4 --dataset /home/user/scratch/test_NUM_BARS=4_OPZ_False.arr --batch_size 32 --label DELETE_ME
```

### Running Jobs

To read the CC documentation, cick [here](https://docs.alliancecan.ca/wiki/Running_jobs). You can run small snippets of code to test things out without allocating any resources. However, to train a model or perform any time/resource consuming task, you must schedule a job. A list of different types of job scheduling will be added here.

#### Interactive Jobs
You can start an interactive session on a compute node with salloc.
```sh
salloc --time=3:0:0 --nodes 1 --cpus-per-task 32 --mem=128000 --account=user
```

#### Scheduled jobs (use this for training)
For time-expensive tasks it is better to create a bash file and submit a job with sbatch:
```sh
sbatch simple_job.sh
```

Here is an example of the contents of a bash file to submit a midigpt training job:
```sh
#!/bin/bash
#SBATCH --gres=gpu:v100l:4
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=2-23:00
#SBATCH --account=user
#SBATCH --mail-user USERNAME@domain.org  <---- MAKE SURE TO PUT YOUR EMAIL
#SBATCH --mail-type ALL
#SBATCH --output=CCLOG/FILENAME.out  <---- MAKE SURE TO CHANGE THE NAME OF THE FILE

source $SCRATCH/PY_3610/bin/activate   <---- THIS IS THE DIRECTORY TO THE ENV WHERE YOU HAVE THE midigpt_api INSTALLED
cd $SCRATCH/MMM_TRAINING-master
module load StdEnv/2020 protobuf python/3.6.10
source $SCRATCH/PY_3610/bin/activate  <---- SAME HERE, MAKE SURE THE DIRECTORY IS PLACED CORRECTLY
python train.py --arch reformer --config /home/user/scratch/MMM_TRAINING-master/config/reformer.json --encoding EXPRESSIVE_ENCODER --ngpu 4 --dataset /home/user/scratch/dataset_NUM_BARS=4.arr --batch_size 32 --label DELETE_ME
```

In this case we are using 4 v1001 GPUs (**gres** argument) and we're asking for 2 days and 23 hours of time to run the job (**time** argument).

#### Check jobs and eliminate session
To show all the users
```sh
who -u
```

To kill all the sessions
```sh
pkill -u username
```



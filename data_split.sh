#!/bin/bash

root=$SCRATCH/datasets/GigaMIDI_Cleaned/Cleaned_Ver_EP_Class-GigaMIDI/Cleaned_GigaMIDI
new_root=$SCRATCH/datasets/GigaMIDI_Cleaned/Cleaned_Ver_EP_Class-GigaMIDI/Cleaned_GigaMIDI_Split
parent=$SCRATCH/workspace_train/parent_dir

mkdir -p $new_root
cd $new_root
mkdir -p train
mkdir -p test
mkdir -p valid

cd $SCRATCH

source $parent/venv/bin/activate
python $parent/MIDI-GPT/python_scripts/data_split.py $root $new_root
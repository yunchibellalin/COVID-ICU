#!/bin/bash

# Navigate to the project directory
cd /data/user/ycl2/covid_xray_icu/update/

# Set paths and job configurations
jobs_dir="/data/user/ycl2/covid_xray_icu/update/job_nyriseg"
history_dir="/scratch/ycl2/covid_history/history_nyriseg_CheSS"
job_name="test_nyriseg"

# Preprocessing parameters
# sharpen - Apply image sharpening (options: 'True', 'False')
# histeq - Apply histogram equalization with CLAHE (options: 'False', 'True_.04_8')
#          In 'True_.04_8': 
#            - .04 is the clipLimit (cl) multiplied by gs*gs
#            - 8 is the grid size (gs) for tileGridSize in CLAHE: cv2.createCLAHE(clipLimit=cl*gs*gs, tileGridSize=(gs, gs))
sharpen='False'
histeq='False'

# Model training parameters
# pretrained - Pretrained model to use (options: 'TorchX', 'ImageNet')
# opt - Optimization algorithm (options: 'Adam', 'SGD')
# lr - Initial learning rate
# bs - Batch size
# ep - Number of training epochs
pretrained='TorchX'
opt="Adam"
lr=0.00001
bs=32
ep=350

# Run the Python training script with the specified parameters
python /path/to/covid_icu.py $sharpen $histeq $pretrained $opt $lr $bs $ep $history_dir $job_name

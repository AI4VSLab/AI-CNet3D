#!/bin/sh
conda activate DL

# train
CONFIG_FILE="./configs/train.yaml"

python train.py --config $CONFIG_FILE 
#!/bin/sh
conda activate DL

# train
CONFIG_FILE="./configs/ssl_finetune.yaml"

python ssl_finetune.py --config $CONFIG_FILE 
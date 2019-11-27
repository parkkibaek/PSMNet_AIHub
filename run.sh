#!/bin/bash

python main.py --maxdisp 192 \
               --model stackhourglass \
               --datapath ../../03_datasets/NIA/ \
               --epochs 200 \
               --batchsz 12 \
               --nworker 20 \
               --savemodel ./trained/final/

               #--loadmodel ./trained/checkpoint_4.tar \


#python finetune.py --maxdisp 192 \
#                   --model stackhourglass \
#                   --datatype 2015 \
#                   --datapath dataset/data_scene_flow_2015/training/ \
#                   --epochs 300 \
#                   --loadmodel ./trained/checkpoint_10.tar \
#                   --savemodel ./trained/


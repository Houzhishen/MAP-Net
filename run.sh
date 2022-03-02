#!/bin/bash


python train.py --dataset cub --backbone conv4 --nExemplars 1 --alpha 0.2 --miu 1 --drop_rate 0.5 --model_name map
python test.py --dataset cub --backbone conv4 --nExemplars 1 --alpha 0.2 --miu 1 --drop_rate 0.5 --model_name map

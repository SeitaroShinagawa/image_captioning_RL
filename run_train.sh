#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python train.py -sn vanilla_model --gpu 0 -bs 512 

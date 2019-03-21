#!/bin/sh
# bash train.sh input_data_folder_path model_path pretrained_embeddings_path evaluation_txt_file_path evaluation_txt_td_file_path

python3 train.py $1 $2 $3 $4 $5

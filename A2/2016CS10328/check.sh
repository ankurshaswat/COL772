#!/bin/sh
# $eval_data=$1
# $eval_data_td=$2
# $predicted='output.txt'
python3 mrr.py $1 $2 'output.txt'

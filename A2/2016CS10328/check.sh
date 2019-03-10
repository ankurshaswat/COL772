#!/bin/sh
$eval_data=$1
$eval_data_td=$2
$predicted='output.txt'
python3 mrr.py $eval_data $eval_data_td $predicted

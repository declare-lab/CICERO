#!/bin/sh

FINETUNED_MODEL_FOLDER="experiments/finetune"
FINETUNED_MODEL=$1
DATA_VERSION=$2
DATA_FOLDER="data/cicero_${DATA_VERSION}/preprocessed"

CUDA_VISIBLE_DEVICES=0 python3 src/evaluate.py \
--checkpoint "${FINETUNED_MODEL_FOLDER}/${FINETUNED_MODEL}" \
--test_file "${DATA_FOLDER}/test_multi.json"

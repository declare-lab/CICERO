#!/bin/sh

if [ $# -eq 0 ]; then
  DATA_VERSION="v1"
fi

python src/get_pretrain_objectives.py --data_version $DATA_VERSION --do_ablation

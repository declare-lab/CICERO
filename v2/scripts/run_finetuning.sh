#!/bin/sh

if [ ! $# -eq 3 ]; then
  echo "Please specify (1) backbone model; (2) checkpoint steps; (3) random seed; in the commandline. "
  echo "E.g.: run_finetuning.sh t5-base 25000 42"
  exit
#  PRETRAINED_MODEL="t5-base"
#  CHECKPOINT="checkpoint-25000"
#  RANDOM_SEED=100
fi

PRETRAINED_MODEL=$1
CHECKPOINT="checkpoint-${2}"
RANDOM_SEED=$3

DATA_FOLDER="data/cicero_v2/preprocessed"

PRETRAINED_MODEL_FOLDER="experiments/pretrain/${PRETRAINED_MODEL}/${CHECKPOINT}"
OUTPUT_FOLDER="experiments/finetune"

CUDA_VISIBLE_DEVICES=1 python src/run_seq2seq.py --learning_rate=3e-5 --adafactor --num_train_epochs 5 \
--train_file=$DATA_FOLDER"/train_multi.json" --validation_file=$DATA_FOLDER"/val_multi.json" \
--text_column "input" --summary_column "output" --source_prefix="" \
--output_dir=$OUTPUT_FOLDER"/mcq_${PRETRAINED_MODEL}_${CHECKPOINT}_${RANDOM_SEED}" --model_name_or_path=$PRETRAINED_MODEL_FOLDER \
--max_source_length 768 --resize_position_embeddings True \
--per_device_train_batch_size=8 --per_device_eval_batch_size=4 --weight_decay=0.005 \
--do_train True --do_eval True --evaluation_strategy="epoch" --save_strategy="epoch" \
--report_to "wandb" --run_name "Mcq "$PRETRAINED_MODEL --save_total_limit=1 --overwrite_output_dir --seed $RANDOM_SEED
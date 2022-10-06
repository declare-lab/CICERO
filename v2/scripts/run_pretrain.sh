#!/bin/sh

if [ $# -eq 0 ]; then
  PRETRAINING_MODEL="t5-base"
else
  PRETRAINING_MODEL=$1
fi

DATA_FOLDER="data/cicero_v1/pretraining"
OUTPUT_FOLDER="experiments/pretrain"

# python -m torch.distributed.launch --nproc_per_node=4 --node_rank=0 src/run_seq2seq.py \
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/run_seq2seq.py \
--learning_rate=1e-5 --adafactor --num_train_epochs 6 \
--train_file=$DATA_FOLDER"/train_pretrain.json" --validation_file=$DATA_FOLDER"/val_pretrain.json" \
--text_column "input" --summary_column "output" --source_prefix="" \
--output_dir=$OUTPUT_FOLDER"/"$PRETRAINING_MODEL --model_name_or_path=$PRETRAINING_MODEL \
--max_source_length 768 --resize_position_embeddings True \
--per_device_train_batch_size=8 --per_device_eval_batch_size=8 --weight_decay=0.005 \
--do_train True --do_eval True --evaluation_strategy="steps" --eval_steps 5000 --save_strategy="steps" --save_steps 5000 \
--logging_steps 50 \
--report_to "wandb" --run_name "$PRETRAINING_MODEL pretrain" --save_total_limit=15 \
--seed 200



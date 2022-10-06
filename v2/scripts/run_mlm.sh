python src/run_t5_mlm_flax.py \
	--output_dir="./cicero-t5-base" \
	--model_type="t5" \
	--config_name="./t5-base" \
	--tokenizer_name="./t5-base" \
	--dataset_name="oscar" \
	--dataset_config_name="unshuffled_deduplicated_no" \
	--max_seq_length="512" \
	--per_device_train_batch_size="32" \
	--per_device_eval_batch_size="32" \
	--adafactor \
	--learning_rate="0.005" \
	--weight_decay="0.001" \
	--warmup_steps="2000" \
	--overwrite_output_dir \
	--logging_steps="500" \
	--save_steps="10000" \
	--eval_steps="2500" \
	--push_to_hub

#  --output_dir="./cicero-t5-base" \
#  --model_type="t5" \
#  --config_name="./t5-base" \
#  --tokenizer_name="./t5-base" \
#  --train_file="./data/cicero_v1/mlm/train.txt" \
#  --validation_file="./data/cicero_v1/mlm/val.txt" \
#  --max_seq_length="512" \
#  --per_device_train_batch_size="8" \
#  --per_device_eval_batch_size="8" \
#  --adafactor \
#  --learning_rate="0.005" \
#  --weight_decay="0.001" \
#  --warmup_steps="2000" \
#  --overwrite_output_dir \
#  --logging_steps="500" \
#  --save_steps="10000" \
#  --eval_steps="2500"


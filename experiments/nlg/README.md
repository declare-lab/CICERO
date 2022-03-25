## Prepare Dataset

The following script will prepare the dataset in appropriate format for the inference generation tasks. Note that, we use only consider the human written answers for this task.
```
python prepare.py
```

## Experiments

We train and evaluate different models for each of the sub-tasks. There are total of eight sub-tasks (1.1.1 - 1.1.5, 1.2.1 - 1.2.3) as mentioned in Section 3.1 of the paper. 

## Training and Evaluation

The T5 Large model for sub-task 1.1.1 (Cause) can be trained as follows:
```
CUDA_VISIBLE_DEVICES=0 python run_seq2seq.py --learning_rate=3e-5 --adafactor --num_train_epochs 5 \
--train_file="data/train_cause.json" --validation_file="data/val_cause.json" \
--text_column "input" --summary_column "output" --source_prefix="" \
--output_dir="saved/t5_cause" --model_name_or_path="t5-large" \
--max_source_length 768 --resize_position_embeddings True \
--per_device_train_batch_size=4 --per_device_eval_batch_size=4 --weight_decay=0.005 \
--do_train True --do_eval True --evaluation_strategy="epoch" --save_strategy="epoch" \
--report_to "wandb" --run_name "T5 Cause" --save_total_limit=5 --overwrite_output_dir
```

Then, evaluation can be perfomed by passing the trained model path as follows:
```
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "saved/t5_cause/checkpoint-12985" \
--input_path "data/test_cause.json" --output_path "results/test_cause.json"
```

This command will save the generated output in `--output_path` and will calculate the automatic evaluation metrics.


Change `--train_file`, `--validation_file` (and `--output_dir`) appropriately to train the models for the other sub-tasks with the `run_seq2seq.py` script. Also make sure to run `evaluate.py` with proper changes to obtain the sub-task specific generated outputs and metrics.

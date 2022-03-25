## Prepare Dataset

The following script will prepare the dataset in appropriate format for the single answer selection (with RoBERTa and ELECTRA), single answer generation (T5 and UnifiedQA), and all answer(s) generation (T5 and UnifiedQA) tasks.
```
python prepare_data.py
```

## Training
The single answer selection RoBERTa Large model can be trained as follows:
```
CUDA_VISIBLE_DEVICES=0 python run_mcq.py --learning_rate=3e-6 --num_train_epochs 5 \
--train_file="data/selection/train_single.json" --validation_file="data/selection/val_single.json" \
--output_dir="saved/roberta_selection" --model_name_or_path="roberta-large" \
--per_device_train_batch_size=4 --per_device_eval_batch_size=4 --weight_decay=0.005 \
--do_train True --do_eval True --evaluation_strategy="epoch" --save_strategy="epoch" \
--report_to "wandb" --run_name "Roberta Selection" --save_total_limit=5 --overwrite_output_dir
```

Use `--model_name_or_path google/electra-large-discriminator` to train the ELECTRA Large model. Change the `--run_name` to change the wandb project name. You can also change the `--train_file` and `--validation_file` appropriately to train the zero-shot models.


The single answer generation T5 Large model can be trained as follows:
```
CUDA_VISIBLE_DEVICES=0 python run_seq2seq.py --learning_rate=3e-5 --adafactor --num_train_epochs 5 \
--train_file="data/generation/train_single.json" --validation_file="data/generation/val_single.json" \
--text_column "input" --summary_column "output" --source_prefix="" \
--output_dir="saved/t5_single" --model_name_or_path="t5-large" \
--max_source_length 768 --resize_position_embeddings True \
--per_device_train_batch_size=4 --per_device_eval_batch_size=4 --weight_decay=0.005 \
--do_train True --do_eval True --evaluation_strategy="epoch" --save_strategy="epoch" \
--report_to "wandb" --run_name "T5 Single" --save_total_limit=5 --overwrite_output_dir
```

Use `--model_name_or_path allenai/unifiedqa-t5-large` to train the UnifiedQA Large model. 

The all answer(s) generation model can be trained using the same script above by changing `--train_file` and `--validation_file` to `data/generation/*_all.json` .


## Evaluation
For evaluation of single answer selection, use the test json files and saved model checkpoints in the `run_mcq.py` script:
```
CUDA_VISIBLE_DEVICES=0 python run_mcq.py --validation_file="data/selection/test_single.json" \
--output_dir="tmp/" --model_name_or_path="saved/roberta_selection/checkpoint-34035" \
--per_device_eval_batch_size=4 --do_train False --do_eval True --evaluation_strategy="epoch" \
--report_to "wandb" --run_name "Roberta Selection"
```


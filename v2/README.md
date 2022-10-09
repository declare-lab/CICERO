## Multiview Contextual Commonsense Inference: A New Dataset and Task
![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg?style=plastic)
![Pytorch 1.11.0](https://img.shields.io/badge/pytorch-1.11.0-green.svg?style=plastic)
![CUDA 11.5](https://img.shields.io/badge/cuda-11.5-green.svg?style=plastic)
![Transformers 4.17.0](https://img.shields.io/badge/Transformers-4.17.0-green.svg?style=plastic)
![License MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=plastic)


This repository contains the implementation of the following [paper](https://arxiv.org/abs/2210.02890):

> **Multiview Contextual Commonsense Inference: A New Dataset and Task**<br>
> Authors <br>
> Deepanway Ghosal, Siqi Shen, Navonil Majumder, Henry Lim, Rada Mihalcea, Soujanya Poria
>
> **Abstract:** 
> <br> *Contextual commonsense inference is the task of generating various types of explanations around the events in a dyadic dialogue, including cause, motivation, emotional reaction, and others. Producing a coherent and non-trivial explanation requires awareness of the dialogue's structure and of how an event is grounded in the context.*
> <br> <br> *In this work, we create CICERO-v2, a dataset consisting of 8,351 instances from 2,379 dialogues, containing multiple human-written answers for each contextual commonsense inference question, representing a type of explanation on cause, subsequent event, motivation, and emotional reaction. We show that the inferences in CICERO-v2 are more semantically diverse than other contextual commonsense inference datasets. To solve the inference task, we propose a collection of pre-training objectives, including concept denoising and utterance sorting to prepare a pre-trained model for the downstream contextual commonsense inference task. Our results show that the proposed pre-training objectives are effective at adapting the pre-trained T5-Large model for the contextual commonsense inference task.*


<img src="https://drive.google.com/uc?export=download&id=14RIbxgXhREdu5xZiKn5D-UUzaQLDNLqf" alt="CICERO_v2 Inferences" width="800"/>
<br>


## Resources

Resources related to this work. 

- Paper: https://arxiv.org/abs/2210.02890
- Dataset: https://github.com/declare-lab/CICERO/releases/download/v2.0.0/data.zip

## Data format

The CICEROv2 dataset can be found in the [data](https://github.com/declare-lab/CICERO/releases/download/v2.0.0/data.zip) directory. Each line of the files is a json object indicating a single instance. The json objects have the following key-value pairs:

| Key 	    | Value 	|
|:----------:| :-----:|
| ID 	    | Dialogue ID with dataset indicator. 	|
| Dialogue 	| Utterances of the dialogue in a list.	|
| Target 	| Target utterance. 	|
| Question 	| One of the five questions (inference types). 	|
| Choices   | Five possible answer choices in a list. One of the answers is<br>human written. The other four answers are machine generated<br>and selected through the Adversarial Filtering (AF) algorithm. |
| Human Written Answer | Index of the human written answer in a<br>single element list. Index starts from 0. |
| Correct Answers | List of all correct answers indicated as plausible<br>or speculatively correct by the human annotators.<br>Includes the index of the human written answer. |
---------------------------------------------------------------------------

An example of the data is shown below.

```
{
  "ID": "daily-dialogue-0404",
  "Dialogue": [
    "A: Dad , why are you taping the windows ?",
    "B: Honey , a typhoon is coming .",
    "A: Really ? Wow , I don't have to go to school tomorrow .",
    "B: Jenny , come and help , we need to prepare more food .",
    "A: OK . Dad ! I'm coming ."
  ],
  "Target": "Jenny , come and help , we need to prepare more food .",
  "Question": "What subsequent event happens or could happen following the target?",
  "Choices": [
    "Jenny and her father stockpile food for the coming days.",
    "The speaker and the listener go outside to purchase more food material for precaution.",
    "Jenny and her father give away all their food.",
    "Jenny and her father eat all the food in their refrigerator."
  ],
  "Correct Answers": [
    0,
    1
  ]
}
 ```

## System requirements

Refer to [requirement.txt](https://github.com/$account/$repo/blob/master/requirements.txt) for more details. Run
``` 
pip install -r requirements.txt
```
The code is checked for the following settings.
* Python 3.9 64-bit. 
* Pytorch v1.11.0 with CUDA 11.5 GPU support.
* Transformers v4.17.0.
* NVIDIA GPUs with at least 24GB of DRAM, NVIDIA A40/3090 is used in the experiments.

## Documentation
### Data Preparation
**[Data of v2 is not online yet]** Download CICERO v1/v2 dataset, separate samples multiview commonsense inference questions. 
```
sh script/download_dataset.sh
```

To prepare the pretraining objectives, including data for ablation study.  
```
sh script/get_pretrain_objectives.sh $data_version
```
* `--data_version`: Select from `v1` or `v2`, prepare the pretraining objective on which dataset, `v1` by default. 


### Run Pretraining
Train the language model in a seq2seq manner with proposed objectives built in the early steps. 
```
sh script/run_pretrain.sh $model
```
* `--model`: The model to do the pretraining on. Will use `t5-base` model if not specified. 

We provide the checkpoint based on `t5-large` [here](https://huggingface.co/shensq0814/CICEROv2)
### Run Finetuning and Evaluation
Finetune the model on multiview commonsense inference task, and measure its performance of exact match accuracy and F1 score. 

```
sh script/run_finetuning.sh $backbone_model $checkpoint_steps $random_seed
```

E.g. a valid set of parameter can be `t5-base 25000 42`

Evaluate the finetuned model by running
```
sh script/evaluate.sh $finetuned_model $data_version
```
<br>e.g. `sh script/evaluate.sh mcq_t5-base_checkpoint-15000_42 v1`
* `$finetuned_model`: the folder containing the checkpoint of finetuned model located in `experiments/finetune` folder.
* `$data_version`: `v1` or `v2`. 
## References

Please cite this repository using the following reference:

```
@inproceedings{shen-et-al-cicerov2-2022,
    title = "Multiview Contextual Commonsense Inference: A New Dataset and Task",
    author = "Shen, Siqi  and 
      Ghosal, Deepanway  and
      Majumder, Navonil  and
      Lim, Henry and
      Mihalcea, Rada  and
      Poria, Soujanya",
    journal = "arxiv"
}
```

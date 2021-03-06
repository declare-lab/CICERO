# CICERO: A Dataset for Contextualized Commonsense Inference in Dialogues

<img src="https://declare-lab.net/assets/images/resources/cicero.png" alt="CICERO Inferences" width="800"/>

We introduce CICERO, a new dataset for dialogue reasoning with contextualized commonsense inference. It contains 53K inferences for five commonsense dimensions – cause, subsequent event, prerequisite, motivation, and emotional reaction collected from 5.6K dialogues. We design several generative and multi-choice answer selection tasks to show the usefulness of CICERO in dialogue reasoning.

[Paper in ACL Anthology](https://aclanthology.org/2022.acl-long.344/)

[Paper in arXiv](https://arxiv.org/abs/2203.13926)

[Download the dataset](https://github.com/declare-lab/CICERO/releases/download/v1.0.0/data.zip)

The dataset can also be accessed with the huggingface `datasets` library

```
from datasets import load_dataset
cicero = load_dataset("declare-lab/cicero")
```

## Data Format

The CICERO dataset can be found in the [data](https://github.com/declare-lab/CICERO/releases/download/v1.0.0/data.zip) directory. Each line of the files is a json object indicating a single instance. The json objects have the following key-value pairs:

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
    "ID": "daily-dialogue-1291",
    "Dialogue": [
        "A: Hello , is there anything I can do for you ?",
        "B: Yes . I would like to check in .",
        "A: Have you made a reservation ?",
        "B: Yes . I am Belen .",
        "A: So your room number is 201 . Are you a member of our hotel ?",
        "B: No , what's the difference ?",
        "A: Well , we offer a 10 % charge for our members ."
    ],
    "Target": "Well , we offer a 10 % charge for our members .",
    "Question": "What subsequent event happens or could happen following the target?",
    "Choices": [
        "For future discounts at the hotel, the listener takes a credit card at the hotel.",
        "The listener is not enrolled in a hotel membership.",
        "For future discounts at the airport, the listener takes a membership at the airport.",
        "For future discounts at the hotel, the listener takes a membership at the hotel.",
        "The listener doesn't have a membership to the hotel."
    ],
    "Human Written Answer": [
        3
    ],
    "Correct Answers": [
        3
    ]
}
 ```

## Experiments

The details of the answer selection (MCQ) experiments can be found [here](https://github.com/declare-lab/CICERO/tree/main/experiments/mcq).
The details of the answer generation (NLG) experiments can be found [here](https://github.com/declare-lab/CICERO/tree/main/experiments/nlg).

<img src="https://declare-lab.net/assets/images/resources/MCQ-cider2-new5.png" alt="CICERO Tasks" width="800"/>

## Citation

```
CICERO: A Dataset for Contextualized Commonsense Inference in Dialogues. Deepanway Ghosal and Siqi Shen and Navonil Majumder and Rada Mihalcea and Soujanya Poria. ACL 2022.
```

### BibTeX
```
@inproceedings{ghosal-etal-2022-cicero,
    title = "{CICERO}: A Dataset for Contextualized Commonsense Inference in Dialogues",
    author = "Ghosal, Deepanway  and
      Shen, Siqi  and
      Majumder, Navonil  and
      Mihalcea, Rada  and
      Poria, Soujanya",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.344",
    pages = "5010--5028",
}
```

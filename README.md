# CICERO: A Dataset for Contextualized Commonsense Inference in Dialogues

![image-center](https://declare-lab.net/assets/images/resources/cicero.png)


Introducing CICERO, a new dataset for dialogue reasoning with contextualized commonsense inference. It contains 53K inferences for five commonsense dimensions – cause, subsequent event, prerequisite, motivation, and emotional reaction collected from  5.6K dialogues. To show the usefulness of CICERO for dialogue reasoning, we design several challenging generative and multichoice answer selection tasks for state-of-the-art NLP models to solve.

[Read the paper]()

## Data Format

The CICERO dataset can be found in the [data](https://github.com/declare-lab/CICERO/tree/main/data) directory. Each line of the files is a json object indicating a single instance. The json objects have the following key-value pairs:

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

## Experiments

The details of the answer selection (MCQ) experiments can be found [here](https://github.com/declare-lab/CICERO/tree/main/experiments/mcq).
The details of the answer generation (NLG) experiments can be found [here](https://github.com/declare-lab/CICERO/tree/main/experiments/nlg).

## Citation

CICERO: A Dataset for Contextualized Commonsense Inference in Dialogues. Deepanway Ghosal and Siqi Shen and Navonil Majumder and Rada Mihalcea and Soujanya Poria. ACL 2022.

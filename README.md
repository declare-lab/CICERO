# Contextualized Commonsense Inference in Dialogues (CICERO)

The purpose of this repository is to introduce new dialogue-level commonsense inference datasets and tasks. We chose dialogues as the data source because dialogues are known to be complex and rich in commonsense. At present, we have released two versions of the dataset:

## CICERO-v1

CICERO-v1 can be found here: https://github.com/declare-lab/CICERO/tree/main/v1.
In this dataset, each training instance is associated with only one human-written inference. There are two tasks pertaining to this dataset: 1) generative commonsense inference in dialogues, and 2) multiple choice answer selection.

## CICERO-v2

Depending on a situation, multiple different reasonings are possible each leading to various unique inferences. In constructing CICERO-v2, we asked annotators to write more than one plausible inference for each dialogue context. We call this task --- Multiview Contextual Commonsense Inference, a highly challenging task for large language models. CICERO-v2 is available here: https://github.com/declare-lab/CICERO/tree/main/v2.

## Citation

If these datasets are useful in your research, please cite the following papers:

### CICERO-v1

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

### CICERO-v2

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

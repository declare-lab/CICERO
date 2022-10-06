# Contextualized Commonsense Inference in Dialogues (CICERO)

The purpose of this repository is to introduce new dialogue-level commonsense inference datasets and tasks. We chose dialogues as the data source because dialogues are known to be complex and rich in commonsense. At present, we have released two versions of the dataset:

## CICERO-v1

CICERO-v1 can be found here: https://github.com/declare-lab/CICERO/tree/main/v1.
In this dataset, each training instance is associated with only one human-written inference. There are two tasks pertaining to this dataset: 1) generative commonsense inference in dialogues, and 2) multiple choice answer selection.

## CICERO-v2

Depending on a situation, multiple different reasonings are possible each leading to various unique inferences. In constructing CICERO-v2, we asked annotators to write more than one plausible inference for dialogue contexts. We call this task --- Multiview Contextual Commonsense Inference, a highly challenging task for large language models.

| \textbf{Description}                  | \textbf{\dataset{}} | \textbf{CICERO}   |
|---------------------------------------|---------------------|-------------------|
| \bf \# Dialogues / \# Instances       |                     |                   |
| $\quad$ DailyDialog                   | 1,025 / 3,422       | 2,113 / 4,344     |
| $\quad$ MuTual                        | 989 / 3,293         | 929 / 1,715       |
| $\quad$ DREAM                         | 243 / 946           | 516 / 1,386       |
| $\quad$ \bf Total                     | 2,257 / 7,661       | 3,558 / 7,445     |
| \# \bf Dialogues with \# Instances    |                     |                   |
| $\quad$ $<$ 4                         | 1,352               | 3,057             |
| $\quad$ 4 $\leq * \leq$ 8             | 839                 | 493               |
| $\quad$ $>$ 8                         | 66                  | 8                 |
| % \bf Avg. \# Inferences per Dialogue |                     | --                |
| % \bf \# Total Answers                |                     |                   |
| % $\quad$ $=$ 4                       | 4541                |                   |
| % $\quad$ $=$ 5                       | 2456                |                   |
| % $\quad$ $>$ 5                       | 52                  |                   |
| \bf Avg. \# of Correct Answers        | 2.38                | 2.49              |
| \bf Instances with \# Correct Answers |                     |                   |
| $\quad$ $=$ 2                         | 4,768               | 4,985             |
| $\quad$ $=$ 3                         | 2,869               | 1,552             |
| $\quad$ $>$ 3                         | 24                  | 908               |
| \bf Question Types in                 |
| $\quad$ Cause                         | 927 / 189 / 243     | 1,301 / 381 / 514 |
| $\quad$ Subsequent Event              | 1,999 / 618 / 793   | 1,193 / 568 / 759 |
| $\quad$ Motivation                    | 1,343 / 330 / 480   | 455 / 163 / 194   |
| $\quad$ Reaction                      | 482 / 116 / 141     | 234 / 105 / 116   |
| $\quad$ Prerequisite                  | -                   | 1,010 / 201 / 251 |
| % $\quad$ \bf Total                   |                     |                   |

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

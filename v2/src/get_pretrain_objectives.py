import argparse
import json
import random
import re
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import spacy
from fuzzywuzzy import fuzz
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm

mapper = {
    "What is or could be the cause of target?": "cause",
    "What is or could be the prerequisite of target?": "prerequisite",
    "What is the possible emotional reaction of the listener in response to target?": "reaction",
    "What is or could be the motivation of target?": "motivation",
    "What subsequent event happens or could happen following the target?": "subsequent event"
}


def convert_to_dataframe(data):
    id_, tid_, dialogue, target, speaker, relation = [], [], [], [], [], []
    utts, index, question, correct, incorrect, written = [], [], [], [], [], []

    for instance in tqdm(data):
        t = instance["Target"]
        utt_list = [u[3:] for u in instance["Dialogue"]]
        if t in utt_list:
            index.append(utt_list.index(t))
        else:
            index.append(np.argmax([fuzz.token_set_ratio(u, t) for u in utt_list]))

        id_.append(instance["ID"])
        tid_.append(instance["ID"] + "-utt-" + str(index[-1]))

        dialogue.append(" <utt> ".join(instance["Dialogue"]))
        utts.append(instance["Dialogue"])

        target.append(instance["Target"])
        speaker.append(instance["Dialogue"][index[-1]][0])
        relation.append(mapper[instance["Question"]])
        question.append(instance["Question"])

        choices = instance["Choices"]
        written.append(choices[instance["Human Written Answer"][0]])
        correct.append([choices[j] for j in instance["Correct Answers"]])
        incorrect.append([choices[j] for j in list(set(range(len(choices))) - set(instance["Correct Answers"]))])

    # ID: Dialogue ID
    # TID: Target utterance ID
    # Dialogue: The dialogue string with utterances concatenated with separator
    # Utts: Utterances of the dialogue in a list
    # Target: Target utterance
    # Index: Index of the target utterance in the "Utts" utterances list
    # Speaker: Speaker of the target utterance - A or B
    # Question: One of the five questions
    # Relation: Which of the five question types
    # Written: Human written answer
    # Correct: Correct answer(s) including the human written answer
    # Incorrect: Incorrect answer(s), could be empty

    df = pd.DataFrame({
        "ID": id_, "TID": tid_, "Dialogue": dialogue, "Utts": utts, "Target": target,
        "Index": index, "Speaker": speaker, "Question": question, "Relation": relation,
        "Written": written, "Correct": correct, "Incorrect": incorrect
    })

    return df


def downsample_data(data, downsample_ratio=5):
    obj_data = {}
    print("Down-sampling")
    for d in data:
        if d[2] not in obj_data:
            obj_data[d[2]] = []
        obj_data[d[2]].append(d)

    for k, v in obj_data.items():
        print(k, len(v))
        obj_data[k] = random.sample(v, k=len(v) // downsample_ratio)
        print(len(obj_data[k]))

    data_processed = []

    for k, v in obj_data.items():
        data_processed.extend(v)

    return data_processed


## The dataframe would be helpful for creating the data for objectives O7, O10 ->
## We can select all the rows corresponding to the particualr target utterance ID.
## This will help to obatin all the answers for that target ufterance across various relations.

if __name__ == "__main__":

    sep = " \\n "

    pos_set = ["NOUN", "VERB"]

    parser = argparse.ArgumentParser(description='Get pretraining objectives')
    parser.add_argument("--data_version", type=str, choices=['v1', 'v2'], default="v1")
    # parser.add_argument("--do_pretrain", action="store_true")
    parser.add_argument("--do_ablation", action="store_true")
    args = parser.parse_args()

    Path("data/cicero_{}/pretraining/".format(args.data_version)).mkdir(parents=True, exist_ok=True)

    nlp = spacy.load("en_core_web_md")


    def o1a(df):
        data = []
        for k in tqdm(range(len(df))):
            instance = df.iloc[k]
            inp = sep.join(
                [instance["Question"], "target: " + instance["Target"], "context: " + instance["Dialogue"]])
            out = instance["Written"]
            data.append([inp, out, "1a"])
        return data


    def o1b(df):
        data = []
        for k in range(len(df)):
            instance = df.iloc[k]
            q = "For which utterance in the context the {} is the following: {}".format(
                instance["Relation"], instance["Written"]
            )
            inp = sep.join([q, "context: " + instance["Dialogue"]])
            out = instance["Target"]
            data.append([inp, out, "1b"])
        return data


    ## add the other objectives
    def o2a(df):
        """ Generate the Utterance given Concepts from the Answer
        """
        data = []
        for k in tqdm(range(len(df))):
            instance = df.iloc[k]

            doc = nlp(instance["Written"])
            concepts = [d.lemma_ for d in doc if d.pos_ in pos_set and d.lemma_ not in STOP_WORDS]
            question_text = "For which utterance in the context the {} is related to the following concepts: {}".format(
                instance["Relation"], ", ".join(concepts))
            context_text = "context: " + instance["Dialogue"]

            inp = sep.join([question_text, context_text])
            out = instance["Target"]
            data.append([inp, out, "2a"])

        return data


    def o2b(df):
        """ Generate the Utterance given Answer, Concepts from The Utterance
        """
        data = []
        for k in tqdm(range(len(df))):
            instance = df.iloc[k]

            # get concepts form the target
            doc = nlp(instance["Target"])
            concepts = [d.lemma_ for d in doc if d.pos_ in pos_set and d.lemma_ not in STOP_WORDS]

            question_text = "For which utterance in the context the {} is the following: {}".format(
                instance["Relation"], instance["Written"])
            concept_text = "concept: " + ", ".join(concepts)
            context_text = "context: " + instance["Dialogue"]

            inp = sep.join([question_text, concept_text, context_text])
            out = instance["Target"]
            data.append([inp, out, "2b"])

        return data


    def o3a(df):
        """ Generate the Answer given Concepts from the Utterance
        """
        data = []
        for k in tqdm(range(len(df))):
            instance = df.iloc[k]

            doc = nlp(instance["Target"])
            concepts = [d.lemma_ for d in doc if d.pos_ in pos_set and d.lemma_ not in STOP_WORDS]

            question_text = instance["Question"]
            concept_text = "concepts in the target: " + ", ".join(concepts)
            context_text = "context: " + instance["Dialogue"]

            inp = sep.join([question_text, concept_text, context_text])
            out = instance["Written"]
            data.append([inp, out, "3a"])

        return data


    def o3b(df):
        """ Generate the Answer given Utterance, Concepts from The Answer
        """
        data = []
        for k in tqdm(range(len(df))):
            instance = df.iloc[k]

            doc = nlp(instance["Written"])
            concepts = [d.lemma_ for d in doc if d.pos_ in pos_set and d.lemma_ not in STOP_WORDS]

            question_text = instance["Question"]
            target_text = "target: " + instance["Target"]

            concept_text = "concepts in the answer: " + ", ".join(concepts)
            context_text = "context: " + instance["Dialogue"]

            inp = sep.join([question_text, target_text, concept_text, context_text])
            out = instance["Written"]
            data.append([inp, out, "3b"])

        return data


    def o4(df):
        def build_options(correct, incorrect):
            options = correct + incorrect
            random.shuffle(options)
            option_text = ""
            for i in range(len(options)):
                option_text += "({}) {}".format(i, options[i])
            return option_text

        data = []
        for k in range(len(df)):
            instance = df.iloc[k]
            option_text = build_options(instance["Correct"], instance["Incorrect"])
            inp = sep.join([instance["Question"], "target: " + instance["Target"], option_text,
                            "context: " + instance["Dialogue"]])

            out = sep.join(instance["Correct"])
            data.append([inp, out, "4"])
        return data


    def o5(df):
        data = []
        for k in range(len(df)):
            instance = df.iloc[k]
            inp = sep.join(
                ["answer: " + instance["Written"], "target: " + instance["Target"],
                 "context: " + instance["Dialogue"]])
            out = instance["Relation"]
            data.append([inp, out, "5"])
        return data


    def o6(df, seed=42):
        """
        Choose Correct Utterance given Relation, Answer.
        input: x, a_i^j, R^j, u_i*
        output: u_i
        """
        random.seed(seed)

        def get_incorrect_answers(sample, k=3):
            """ Prepare the options of 3 incorrect answers and a correct one.
            """
            utterances = sample["Utts"]
            incorrect = random.choices([utterances[i] for i in range(len(utterances)) if i != sample["Index"]],
                                       k=3)  # not choosing the correct one.
            incorrect = [re.sub(r"[AB]: ", "", u) for u in incorrect]  # removing the speaker tag
            options = incorrect + [sample["Target"]]
            random.shuffle(options)
            return options

        data = []
        for k in range(len(df)):
            instance = df.iloc[k]
            options = get_incorrect_answers(instance)
            options_text = "target options: " + " <utt> ".join(options)
            context_text = "context: " + instance["Dialogue"]
            answer_text = "The {} of the target: ".format(instance["Relation"]) + instance["Written"]

            inp = sep.join([answer_text, options_text, context_text])
            out = instance["Target"]
            data.append([inp, out, "6"])
        return data


    def o7(df):
        """ Given an illustration of question-answer pair of a relation, answer another relation.
        The amount of utts with i relations, [0, 1586, 3237, 863, 61, 1]
        """

        def get_sample_dict(df):
            # find samples with the same TID
            sample_dict = {}
            for k in range(len(df)):
                instance = df.iloc[k]
                TID = instance["TID"]
                if TID not in sample_dict:
                    sample_dict[TID] = [instance]
                else:
                    sample_dict[TID].append(instance)
            return sample_dict

        sample_dict = get_sample_dict(df)

        data = []
        for TID, samples in sample_dict.items():
            if len(samples) < 2:  # ignore utterances with less than 2 annotated relations.
                continue
            # for each TID create i*(i-1) samples.
            for i in range(len(samples)):
                for j in range(len(samples)):
                    if j == i: continue
                    s1 = samples[i]
                    s2 = samples[j]
                    # assert s1["Target"]==s2["Target"] # this doesn't pass
                    target = s1["Target"] if s1["Target"] > s2["Target"] else s2["Target"]  # use the longer target
                    target_text = "target: " + target
                    context_text = "context: " + s1["Dialogue"]
                    answer1_text = "The {} of the target: ".format(s1["Relation"]) + s1["Written"]
                    answer2_text = "What is the {} of the target?".format(s2["Relation"])

                    inp = sep.join([target_text, answer1_text, answer2_text, context_text])
                    out = s2["Written"]
                    data.append([inp, out, "7"])

        return data


    def o8(df, corruptions=("shuffle", "drop", "replace")):
        """ Corrupt Concept Detection in Answer.
        Input: context, target, corrupted concepts from answers
        Output: Correct Concepts.
        """
        data = []
        for k in range(len(df)):
            instance = df.iloc[k]

            # corrupt the concepts.
            doc = nlp(instance["Written"])
            concepts = [d.lemma_ for d in doc if d.pos_ in pos_set and d.lemma_ not in STOP_WORDS]
            if not len(concepts):
                continue
            out = ", ".join(concepts)
            # changing orders, drop a few, replacing concepts.
            for cor in corruptions:
                if cor == "shuffle":
                    random.shuffle(concepts)
                if cor == "drop" and len(concepts):
                    concepts.pop(random.randrange(len(concepts)))
                if cor == "replace":
                    pass  # NotImplemented

            corrupted_concepts_context = "corrupted concepts: " + ", ".join(concepts)
            concepts_text = "concepts in the answer: "

            target_text = "target: " + instance["Target"]

            context_text = "context: " + instance["Dialogue"]

            inp = sep.join([target_text, corrupted_concepts_context, context_text, concepts_text])
            data.append([inp, out, "8"])

        return data


    def o9(df, corruptions=("shuffle", "drop", "replace")):
        """ Corrupt Concept Detection in Utterance.
        Input: context, answer, corrupted concepts from target
        Output: Correct Concepts from target
        """
        data = []
        for k in range(len(df)):
            instance = df.iloc[k]

            # corrupt the concepts.
            doc = nlp(instance["Target"])
            concepts = [d.lemma_ for d in doc if d.pos_ in pos_set and d.lemma_ not in STOP_WORDS]
            if not len(concepts):
                continue
            out = ", ".join(concepts)
            # changing orders, drop a few, replacing concepts.
            for cor in corruptions:
                if cor == "shuffle":
                    random.shuffle(concepts)
                if cor == "drop" and len(concepts):
                    concepts.pop(random.randrange(len(concepts)))
                if cor == "replace":
                    pass  # NotImplemented

            corrupted_concepts_context = "corrupted concepts: " + ", ".join(concepts)
            concepts_text = "concepts in the target: "

            answer_text = "answer: " + instance["Written"]

            context_text = "context: " + instance["Dialogue"]

            inp = sep.join([answer_text, corrupted_concepts_context, context_text, concepts_text])
            data.append([inp, out, "9"])

        return data


    def o10(df, seed=42):
        """ Answer Sorting
        """

        # find the set of samples for the same dialogue
        def get_samples_ID(df):
            # find samples with the same ID
            sample_dict = {}
            for k in range(len(df)):
                instance = df.iloc[k]
                ID = instance["ID"]
                if ID not in sample_dict:
                    sample_dict[ID] = [instance]
                else:
                    sample_dict[ID].append(instance)
            return sample_dict

        sample_dict = get_samples_ID(df)

        data = []
        rel_order = ['cause', 'prerequisite', 'motivation', 'subsequent event', 'reaction']
        rel_order = dict(zip(rel_order, list(range(len(rel_order)))))

        for ID, samples in tqdm(sample_dict.items()):
            if len(samples) < 2:  # ignore dialogue with less than 2 annotated relations.
                continue
            # for each ID create 1 samples.
            answer_loc_rel = []
            for s in samples:
                answer_loc_rel.append((s["Written"], s["Index"], rel_order[s["Relation"]]))

            answer_loc_rel.sort(key=lambda x: 5 * x[1] + x[2])  # 5 is the maximum number of relations
            idx_answer = [(i, answer_loc_rel[i][0]) for i in range(len(answer_loc_rel))]
            random.shuffle(idx_answer)
            idx, answers = list(zip(*idx_answer))
            idx = [str(x) for x in idx]

            answer_text = " <utt> ".join(answers)

            inp = sep.join([answer_text])
            out = " ".join(idx)
            data.append([inp, out, "10"])

        return data


    def o11(df, seed=42):
        """ Utterance Sorting
        """
        data = []
        for k in range(len(df)):
            instance = df.iloc[k]

            utterances = instance["Dialogue"].split(" <utt> ")
            idx_utt = [(i, utterances[i]) for i in range(len(utterances))]
            random.shuffle(idx_utt)
            idx, utterances = list(zip(*idx_utt))
            idx = [str(x) for x in idx]
            utterance_text = " <utt> ".join(utterances)

            # context_text = "context: " + instance["Dialogue"]

            inp = sep.join([utterance_text])
            out = " ".join(idx)
            data.append([inp, out, "11"])

        return data


    data_folder = "data/cicero_{}".format(args.data_version)
    if args.do_ablation:
        # Group the objectives for ablation study
        ablation_mapping: dict[str, list[str]] = {"generate": ["1a", "1b", "2a", "2b", "3a", "3b", "7"],
                                                  "choose": ["4", "5", "6"],
                                                  "corruption": ["8", "9"],
                                                  "sorting": ["10", "11"],
                                                  "concept": ["2a", "2b", "3a", "3b", "8", "9"]
                                                  }
        ablation_folder: str = "{}/pretraining/ablation".format(data_folder)

        try:
            os.mkdir(ablation_folder)
        except OSError as error:
            print(error)

    for split in ["train", "val", "test"]:

        data = open("{}/{}.json".format(data_folder, split)).readlines()
        data = [json.loads(line) for line in data]

        df = convert_to_dataframe(data)
        data = []

        for objective in [o1a, ]:
            print(objective)
            data += objective(df)

        ## add data for the other objectives

        with open("{}/pretraining/{}_pretrain.json".format(data_folder, split), "w") as f:
            if split == "val":
                data = downsample_data(data, downsample_ratio=5)

            for content in data:
                # line = {"input": content[0], "output": content[1], "objective": content[2]}
                line = {"input": content[0], "output": content[1]}
                f.write(json.dumps(line) + "\n")

        if args.do_ablation:
            # path = Path("{}/{}_v1.json".format(ablation_folder, split))
            # if not path.is_file():
            #     print(path)
            #     continue
            # data = [json.loads(line) for line in open("{}/{}_v1.json".format(ablation_folder, split))]
            for ablation, objectives in ablation_mapping.items():
                data_ablation = [x.copy() for x in data if x[2] not in objectives]
                with open("{}/{}_ablate_{}.json".format(ablation_folder, split, ablation), "w") as f:
                    for content in data_ablation:
                        line = {"input": content[0], "output": content[1]}
                        f.write(json.dumps(line) + "\n")

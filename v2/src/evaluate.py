import re
import json
import pickle
import random
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
#from transformers import AutoTokenizer, T5ForConditionalGeneration

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    T5ForConditionalGeneration
)

# From Unified QA
def score_string_similarity(str1, str2):
    if str1 == str2:
        return 3.0  # Better than perfect token match
    str1 = fix_buggy_characters(replace_punctuation(str1))
    str2 = fix_buggy_characters(replace_punctuation(str2))
    if str1 == str2:
        return 2.0
    if " " in str1 or " " in str2:
        str1_split = str1.split(" ")
        str2_split = str2.split(" ")
        overlap = list(set(str1_split) & set(str2_split))
        return len(overlap) / max(len(str1_split), len(str2_split))
    else:
        if str1 == str2:
            return 1.0
        else:
            return 0.0


def replace_punctuation(str):
    return str.replace("\"", "").replace("'", "")


# Temporary fix for bug where {}^<\` characters roundtrip into \u2047 (??) character
def fix_buggy_characters(str):
    return re.sub("[{}^\\\\`\u2047<]", " ", str)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def generate(sentences, model, tokenizer, mode="beam", num_seq=2, bs=8, max_len=None):
    """Generate for text using beam search/top-k sampling."""
    device = str(model.device)

    outputs = []

    for k in tqdm(range(0, len(sentences), bs)):
        text = sentences[k: min(k + bs, len(sentences))]
        if mode == "beam":
            beams = 5
            batch = tokenizer(text, return_tensors="pt", padding=True)
            if max_len is not None:
                generated = model.generate(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    max_length=25, num_beams=beams, early_stopping=True, num_return_sequences=num_seq,
                )
            else:
                generated = model.generate(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    num_beams=beams, num_return_sequences=num_seq,
                    max_length=100, early_stopping=True,
                )

        elif mode == "greedy":
            batch = tokenizer(text, return_tensors="pt", padding=True)
            if max_len is not None:
                generated = model.generate(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device), max_length=25
                )
            else:
                generated = model.generate(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device)
                )

        out = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated]
        if mode == "greedy":
            outputs += out
        else:
            outputs += list(chunks(out, num_seq))
    if mode == "greedy":
        outputs = [[item] for item in outputs]

    return outputs


def main():
    parser = argparse.ArgumentParser(description='Evaluate options')
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--data_folder', type=str, default="data/generation/")
    parser.add_argument('--test_file', type=str, default="/home/poria/cicero-lm/data/v1_marker/combined/test_combined_v1.json")
    parser.add_argument('--evaluation_mode', type=str, choices=['single', 'multi', 'all'], default="multi")

    args = parser.parse_args()
    relations = [
        "What is or could be the cause of target?",
        "What subsequent event happens or could happen following the target?",
        "What is or could be the prerequisite of target?",
        "What is or could be the motivation of target?",
        "What is the possible emotional reaction of the listener in response to target?"
    ]

    rel = ["Cause", "SubEv", "Prere", "Motiv", "React"]

    sindices = [[], [], [], [], []]  # relation indices for single answer instances
    mindices = [[], [], [], [], []]  # relation indices for multi answer instances
    aindices = [[], [], [], [], []]  # relation indices for all instances

    ## all the files can be found in the zip
    ## there might be a single different line between these files and the ones in github
    ## we changed answer of an instance to match the illustrations in the paper after the experiments

    x = open(args.test_file).readlines()
    k = 0
    for _, line in enumerate(x):
        content = json.loads(line)
        qtype = relations.index(content["input"].split(" \\n ")[0])
        if len(content['output'].split()) == 1:
            sindices[qtype].append(k)
            k += 1

    x = open(args.test_file).readlines()
    k = 0
    for _, line in enumerate(x):
        content = json.loads(line)
        qtype = relations.index(content["input"].split(" \\n ")[0])
        if len(content['output'].split()) > 1:
            mindices[qtype].append(k)
            k += 1


    # Accuracy for multi answer instances
    def m_accuracy(generated, x, indices):
        all_scores, predictions = [], []
        for m in range(len(generated[0][:1])):
            gen = [item[m] for item in generated]
            scores, best = [], []
            precision = []
            recall = []

            for k in range(len(gen[:])):
                content = x.iloc[k]
                #choices = [content[i] for i in range(5)]
                label_sep = " \\n "
                label = sorted([int(item) for item in content["output"].split(label_sep)])
                # print(label)
                all_preds = []
                # print(gen[k])
                all_outs = gen[k].split(" n ")
                # for out in all_outs:
                #     similarities = [score_string_similarity(out, c) for c in choices]
                #     pred = np.argmax(similarities)
                #     all_preds.append(pred)
                all_preds = [int(a) for a in all_outs if a.isdigit()]

                all_preds = sorted(all_preds)
                # print(all_preds)
                if all_preds == label:
                    scores.append(1)
                else:
                    scores.append(0)

                tp = sum([x in label for x in all_preds])
                # fp = len(all_preds) - tp
                # fn = len(label) - tp
                if len(all_preds) == 0:
                    all_preds = [1]
                precision.append([tp, len(all_preds)])
                recall.append([tp, len(label)])

            micro_p = sum([x[0] for x in precision]) / sum([x[1] for x in precision])
            micro_r = sum([x[0] for x in recall]) / sum([x[1] for x in recall])
            micro_f = 2 * (micro_p * micro_r) / (micro_p + micro_r)

            macro_p = sum([x[0] / x[1] for x in precision]) / len(precision)
            macro_r = sum([x[0] / x[1] for x in recall]) / len(recall)
            macro_f = 2 * (macro_p * macro_r) / (macro_p + macro_r)

            print("Micro p: {} \t r:{} f1:{}".format(micro_p, micro_r, micro_f))
            print("Macro p: {} \t r:{} f1:{}".format(macro_p, macro_r, macro_f))

            scores = np.array(scores)
            predictions.append(scores)

            print("Beam" + str(m + 1) + ":")
            precision_rel = []
            recall_rel = []

            metrics_logging = {"model":args.checkpoint}
            for j, r in enumerate(rel):
                precision_rel.append([np.sum(scores[indices[j]]), indices[j]])

                print(r, round(np.mean(scores[indices[j]]), 4))
                metrics_logging[r] = round(np.mean(scores[indices[j]]), 4)
            print("Avg:", round(np.mean(scores), 4))
            metrics_logging["Avg"] = round(np.mean(scores), 4)
            all_scores.append(round(np.mean(scores), 4))
            print("")

            metrics_logging["micro_prf"] = [micro_p, micro_r, micro_f]
            metrics_logging["macro_prf"] = [macro_p, macro_r, macro_f]

            with open("aggregate_result.json", 'a') as f:
                f.write(json.dumps(metrics_logging)+'\n')

        # print (len(scores), np.sum(scores))
        return all_scores

    # This dataframe can be easily created from the json files
    # Shared in the zip file for ease of use
    # Label: Human written answer index
    # New Label: All correct answers index
    df = pd.read_json(args.test_file, lines=True)
    single = df[df["output"].apply(lambda x: len(x) == 1)].reset_index()
    multiple = df[df["output"].apply(lambda x: len(x) > 1)].reset_index()

    path = "/home/siqi/disk1/PycharmProjects/CICERO/experiments/mcq/saved/t5_pretrain_mcq/checkpoint-5245"
    tokenizer = AutoTokenizer.from_pretrained("t5-large")
    if args.checkpoint:
        model = T5ForConditionalGeneration.from_pretrained(args.checkpoint).to("cuda")
        config = AutoConfig.from_pretrained(args.checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    else:
        model = T5ForConditionalGeneration.from_pretrained(path).to("cuda")
    model.config.n_positions = 768

    # updated parameters
    if args.evaluation_mode == "multi":
        inputs5 = [json.loads(item)["input"] for item in open(args.test_file).readlines() if len(json.loads(item)["output"].split())>1]
        gen5 = generate(inputs5[:], model, tokenizer, "beam")
        #pickle.dump(gen5, open("results/generation/t5_multiple_marker_v1_multi.pkl", "wb"))
        #gen5 = pickle.load(open("results/generation/t5_multiple_marker_v1_multi.pkl","rb"))
        scores5 = m_accuracy(gen5, multiple, indices=mindices)
    elif args.evaluation_mode == "single":
        inputs5 = [json.loads(item)["input"] for item in open(args.test_file).readlines() if len(json.loads(item)["output"].split())==1]
        gen5 = generate(inputs5, model, tokenizer, "beam")
        #pickle.dump(gen5, open("results/generation/t5_multiple_marker_v1_single.pkl", "wb"))
        #gen5 = pickle.load(open("results/generation/t5_multiple_marker_v1_single.pkl","rb"))
        scores5 = m_accuracy(gen5, single, indices=sindices)
    elif args.evaluation_mode == "all":
        raise NotImplementedError
    else:
        raise NotImplementedError

    print(scores5)


if __name__ == "__main__":
    main()

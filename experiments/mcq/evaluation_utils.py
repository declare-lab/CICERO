import re
import numpy as np
from tqdm import tqdm

# From Unified QA
def score_string_similarity(str1, str2):
    if str1 == str2:
        return 3.0   # Better than perfect token match
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

def generate(sentences, model, tokenizer, mode="beam", bs=8, max_len=25, num_seq=2):
    """Generate for text using beam search/top-k sampling."""
    device, outputs = str(model.device), []
    
    for k in tqdm(range(0, len(sentences), bs)):
        text = sentences[k: min(k+bs, len(sentences))]
        if mode == "beam":
            beams = 5
            batch = tokenizer(text, return_tensors="pt", padding=True)
            generated = model.generate(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                num_beams=beams, num_return_sequences=num_seq,
                max_length=max_len, early_stopping=True,
            )
        
        elif mode == "greedy":
            batch = tokenizer(text, return_tensors="pt", padding=True)
            generated = model.generate(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device), max_length=max_len
            )

        out = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated]
        
        if mode == "greedy":
            outputs += out
        else:
            outputs += list(chunks(out, num_seq))
    if mode == "greedy":
        outputs = [[item] for item in outputs]
    
    return outputs

rel = ["Cause", "SubEv", "Prere", "Motiv", "React"]

def single_accuracy(generated, indices, x):
    all_scores = []
    gen = [item[0] for item in generated]
    scores, best = [], []
    for k in range(len(gen)):
        out = gen[k]
        content = x.iloc[k]
        choices = [content["Choice" + str(i)] for i in range(5)]
        label = [int(item) for item in content["Label"].split()]       
        similarities = [score_string_similarity(out, c) for c in choices]

        pred = np.argmax(similarities)
        best.append(round(max(similarities), 4))
        if pred == label:
            scores.append(1)
        else:
            scores.append(0)
    scores = np.array(scores)
    
    for j, r in enumerate(rel):
        print (r, round(np.mean(scores[indices[j]]), 4))
    print ("Avg:", round(np.mean(scores), 4))
    
    
def multiple_accuracy(generated, indices, x):
    all_scores = []
    gen = [item[0] for item in generated]
    scores, best = [], []
    for k in range(len(gen)):
        content = x.iloc[k]
        choices = [content["Choice" + str(i)] for i in range(5)]
        label = sorted([int(item) for item in content["Label"].split()])   
        all_preds = []

        all_outs = gen[k].split(" n ")
        for out in all_outs:
            similarities = [score_string_similarity(out, c) for c in choices]
            pred = np.argmax(similarities)
            all_preds.append(pred)

        all_preds = sorted(all_preds)

        if all_preds == label:
            scores.append(1)
        else:
            scores.append(0)
    scores = np.array(scores)

    for j, r in enumerate(rel):
        print (r, round(np.mean(scores[indices[j]]), 4))
    print ("Avg:", round(np.mean(scores), 4))
    

def all_accuracy(generated, sindices, mindices, aindices, x):
    i1 = np.array(x[x["Label"].apply(lambda y: len(y)==1)].index)
    i2 = np.array(x[x["Label"].apply(lambda y: len(y)>1)].index)

    all_scores = []
    gen = [item[0] for item in generated]
    scores, best = [], []
    for k in range(len(gen)):
        content = x.iloc[k]
        choices = [content["Choice" + str(i)] for i in range(5)]
        label = sorted([int(item) for item in content["Label"].split()])   
        all_preds = []

        all_outs = gen[k].split(" n ")
        for out in all_outs:
            similarities = [score_string_similarity(out, c) for c in choices]
            pred = np.argmax(similarities)
            all_preds.append(pred)

        all_preds = sorted(all_preds)

        if all_preds == label:
            scores.append(1)
        else:
            scores.append(0)
    scores = np.array(scores)
    
    print ("Overall (combined single and multiple):")
    for j, r in enumerate(rel):
        print (r, round(np.mean(scores[aindices[j]]), 4))
    print ("Avg:", round(np.mean(scores), 4))
    
    print ("\nSingle:")
    for j, r in enumerate(rel):
        print (r, round(np.mean(scores[i1][sindices[j]]), 4))

    print ("\nMultiple:")
    for j, r in enumerate(rel):
        print (r, round(np.mean(scores[i2][mindices[j]]), 4))

    print ("Avg Single: {}; Multiple: {}".format(round(np.mean(scores[i1]), 4), round(np.mean(scores[i2]), 4)))
    print ("---------------")

    
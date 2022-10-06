import numpy as np
import pandas as pd
import json, argparse
from tqdm import tqdm
from pathlib import Path
from nlgeval.pycocoevalcap.bleu.bleu import Bleu
from nlgeval.pycocoevalcap.rouge.rouge import Rouge
from nlgeval.pycocoevalcap.cider.cider import Cider
from nlgeval.pycocoevalcap.meteor.meteor import Meteor
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def chunks(lst, n):
    """
    Yield successive n-sized chunks from lst.
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def generate(text, model, tokenizer, bs=16):
    """
    Generate output using beam search.
    """
    beams, outputs, device = 5, [], str(model.device)
    print ("Generating outputs.")
    for j in tqdm(range(0, len(text), bs)):
        batch = text[j: min(j+bs, len(text))]
        batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        generated = model.generate(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            min_length=6,
            max_length=20,
            num_beams=beams,
            early_stopping=True,
            num_return_sequences=5,
            no_repeat_ngram_size=2
        )
        out = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated]
        segments = list(chunks(out, beams))
        outputs += segments
    return outputs

def metrics(gold, generated, scorers, simcse):
    """
    Compute metrics.
    """
    refs, hyps, task_scores = {}, {}, []
    for j in range(len(gold)):
        refs[j] = [gold[j]]
        hyps[j] = [generated[j]]

    for scorer, method in scorers:
        score, scores = scorer.compute_score(refs, hyps)
        if type(score) == list:
            for m, s in zip(method, score):
                task_scores.append(round(s, 4))
        else:
            task_scores.append(round(score, 4))

    embeddings1 = simcse.encode(gold, convert_to_tensor=True)
    embeddings2 = simcse.encode(generated, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    similarity = round(np.mean(cosine_scores.diag().cpu().numpy()), 4)
    task_scores.append(similarity)
    return task_scores

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation.")
    parser.add_argument("--input_path", type=str, help="Input json file.")
    parser.add_argument("--output_path", type=str, default="", help="Json file to save generated outputs.")
    parser.add_argument("--model_name_or_path", type=str, default="", help="Model to use for generation.")
    args = parser.parse_args()
    
    ## Read input json and specify output path ##
    data = [json.loads(line) for line in open(args.input_path).readlines()]
    inputs = [instance["input"] for instance in data]
    gold = [instance["output"] for instance in data]
    
    output_path = args.output_path
    if output_path == "":
        output_path = "results/" + args.input_path.split("/")[1]
        
    directory = "/".join(output_path.split("/")[:-1])
    Path(directory).mkdir(parents=True, exist_ok=True)
    output_metrics_path = output_path.split(".")[0] + ".txt"
    
    ## Load model ##
    model_name_or_path = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).cuda()
    
    ## Generate and save outputs ##
    generated = generate(inputs, model, tokenizer, args.batch_size)
    
    with open(output_path, "w") as f:
        for j in range(len(generated)):
            instance = data[j]
            ## Separate beam outputs with " && " ##
            instance["generated"] = " && ".join(generated[j]) 
            f.write(json.dumps(instance) + "\n")
        
    ## Compute metrics with the best beam search output ## 
    best = [out[0] for out in generated]
    
    ## Scorers and Sentence emebedding model ##
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    simcse = SentenceTransformer("princeton-nlp/sup-simcse-roberta-large").cuda()
    
    scores = metrics(gold, best, scorers, simcse)
    
    results = []
    metric_name = ["BLEU1", "BLEU2", "BLEU3", "BLEU4", "METEOR", "ROUGE_L", "CIDER", "Sem-Sim"]
    for m, s in zip(metric_name, scores):
        results.append(m + ": " + str(s))
    print ("Metrics:")
    print ("\n".join(results))

    with open(output_metrics_path, "a") as f:
        f.write("\n".join(results) + "\n\n")
    
    print ("Saved results in {} and {}".format(output_path, output_metrics_path))
    
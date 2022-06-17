import json
import pandas as pd
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from evaluation_utils import generate, single_accuracy, multiple_accuracy, all_accuracy

relations = [
    "What is or could be the cause of target?",
    "What subsequent event happens or could happen following the target?",
    "What is or could be the prerequisite of target?",
    "What is or could be the motivation of target?",
    "What is the possible emotional reaction of the listener in response to target?"
]

def relation_indices(f):
    indices = [[], [], [], [], []]
    x = open(f).readlines()[:41]
    for k, line in enumerate(x):
        content = json.loads(line)
        qtype = relations.index(content["input"].split(" \\n ")[0])
        indices[qtype].append(k)
    return indices

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--path", default="", help="Trained model path.")
    parser.add_argument("--mode", default="", help="Evaluate for which mode: single|multiple|all")
    parser.add_argument("--search", default="beam", help="Search algorithm for generation: beam|greedy")
    parser.add_argument("--max-len", type=int, default=25, help="Max length for generation. Use a higher value for multiple and all mode.")
    parser.add_argument("--bs", type=int, default=8, help="Batch size for evaluation.")
    
    
    global args
    args = parser.parse_args()
    print(args)
    
    path = args.path
    mode = args.mode
    search = args.search
    max_len = args.max_len
    bs = args.bs
    
    sindices = relation_indices("data/generation/test_single.json") # single answer instances
    mindices = relation_indices("data/generation/test_multiple.json") # multi answer instances
    aindices = relation_indices("data/generation/test_all.json") # all instances
    
    data = [json.loads(line) for line in open("../../data/test.json").readlines()]
    df = pd.DataFrame({
        "Choice0": [item["Choices"][0] for item in data], "Choice1": [item["Choices"][1] for item in data],
        "Choice2": [item["Choices"][2] for item in data], "Choice3": [item["Choices"][3] for item in data],
        "Choice4": [item["Choices"][4] for item in data],
        "Label": [" ".join([str(l) for l in item["Correct Answers"]]) for item in data]
    })

    single_df = df[df["Label"].apply(lambda x: len(x)==1)].reset_index()
    multiple_df = df[df["Label"].apply(lambda x: len(x)>1)].reset_index()
    
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path).cuda()
    model.config.n_positions = 768
    
    if mode == "single":
        inputs = [json.loads(item)["input"] for item in open("data/generation/test_single.json").readlines()]
        output = generate(inputs, model, tokenizer, search, bs, max_len)
        single_accuracy(output, sindices, single_df)
        
    elif mode == "multiple":
        inputs = [json.loads(item)["input"] for item in open("data/generation/test_multiple.json").readlines()]
        output = generate(inputs, model, tokenizer, search, bs, max_len)
        multiple_accuracy(output, mindices, multiple_df)
        
    elif mode == "all":
        inputs = [json.loads(item)["input"] for item in open("data/generation/test_all.json").readlines()]
        output = generate(inputs, model, tokenizer, search, bs, max_len)
        all_accuracy(output, sindices, mindices, aindices, df)

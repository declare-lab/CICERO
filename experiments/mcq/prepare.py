import json
from pathlib import Path

q1 = [
    "What is or could be the cause of target?",
    "What is or could be the prerequisite of target?",
    "What is the possible emotional reaction of the listener in response to target?"
]

q2 = [
    "What is or could be the motivation of target?",
    "What subsequent event happens or could happen following the target?"
]

def single_selection(split, zero_shot=False):
    """
    Prepares data for the single answer selection task for the RoBERTa and ELECTRA models.
    """
    data = [json.loads(line) for line in open("../../data/" + split + ".json").readlines()]
    sep = " \\n "
    
    if zero_shot:
        f = open("data/selection/" + split + "_zs_single.json", "w")
        if split == "test":
            question_set = q2
        else:
            question_set = q1
    else:
        f = open("data/selection/" + split + "_single.json", "w")
        question_set = q1 + q2
    
    for instance in data:
        if len(instance["Correct Answers"]) == 1 and instance["Question"] in question_set:
            choices = instance["Choices"]
            context = sep.join([instance["Question"], "target: " + instance["Target"], "context: " + " <utt> ".join(instance["Dialogue"])])
            line = {
                "ID": instance["ID"], "context": context, "choice0": choices[0], "choice1": choices[1], 
                "choice2": choices[2], "choice3": choices[3], "choice4": choices[4], 
                "label": instance["Correct Answers"][0]
            }
            f.write(json.dumps(line) + "\n")
    f.close()
    
    
def single_generation(split, zero_shot=False):
    """
    Prepares data for the single answer generation task for the T5 and UnifiedQA models.
    """
    data = [json.loads(line) for line in open("../../data/" + split + ".json").readlines()]
    sep = " \\n "
    
    if zero_shot:
        f = open("data/generation/" + split + "_zs_single.json", "w")
        if split == "test":
            question_set = q2
        else:
            question_set = q1
    else:
        f = open("data/generation/" + split + "_single.json", "w")
        question_set = q1 + q2
    
    for instance in data:
        if len(instance["Correct Answers"]) == 1 and instance["Question"] in question_set:
            choices, choice_str = instance["Choices"], ""
            for k, num in enumerate(["(0)", "(1)", "(2)", "(3)", "(4)"]):
                choice_str += num + " " + choices[k] + " "
            choice_str = choice_str[:-1]
            
            context = sep.join([instance["Question"], "target: " + instance["Target"], choice_str,
                                "context: " + " <utt> ".join(instance["Dialogue"])])
            correct_choice = choices[instance["Correct Answers"][0]]
            
            line = {"input": context, "output": correct_choice}
            f.write(json.dumps(line) + "\n")
    f.close()
    
    
def multiple_generation(split, zero_shot=False):
    """
    Prepares data for the only multiple-answers generation task for the T5 and UnifiedQA models.
    """
    data = [json.loads(line) for line in open("../../data/" + split + ".json").readlines()]
    sep = " \\n "
    
    if zero_shot:
        f = open("data/generation/" + split + "_zs_multiple.json", "w")
        if split == "test":
            question_set = q2
        else:
            question_set = q1
    else:
        f = open("data/generation/" + split + "_multiple.json", "w")
        question_set = q1 + q2
    
    for instance in data:
        if len(instance["Correct Answers"]) > 1 and instance["Question"] in question_set:
            choices, choice_str = instance["Choices"], ""
            for k, num in enumerate(["(0)", "(1)", "(2)", "(3)", "(4)"]):
                choice_str += num + " " + choices[k] + " "
            choice_str = choice_str[:-1]
            
            context = sep.join([instance["Question"], "target: " + instance["Target"], choice_str,
                                "context: " + " <utt> ".join(instance["Dialogue"])])
            
            correct_choices = sep.join([choices[index] for index in instance["Correct Answers"]])
            
            line = {"input": context, "output": correct_choices}
            f.write(json.dumps(line) + "\n")
    f.close()
    
    
def all_generation(split, zero_shot=False):
    """
    Prepares data for the all answer(s) generation task for the T5 and UnifiedQA models.
    """
    data = [json.loads(line) for line in open("../../data/" + split + ".json").readlines()]
    sep = " \\n "
    
    if zero_shot:
        f = open("data/generation/" + split + "_zs_all.json", "w")
        if split == "test":
            question_set = q2
        else:
            question_set = q1
    else:
        f = open("data/generation/" + split + "_all.json", "w")
        question_set = q1 + q2
    
    for instance in data:
        if instance["Question"] in question_set:
            choices, choice_str = instance["Choices"], ""
            for k, num in enumerate(["(0)", "(1)", "(2)", "(3)", "(4)"]):
                choice_str += num + " " + choices[k] + " "
            choice_str = choice_str[:-1]
            
            context = sep.join([instance["Question"], "target: " + instance["Target"], choice_str,
                                "context: " + " <utt> ".join(instance["Dialogue"])])
            
            correct_choices = sep.join([choices[index] for index in instance["Correct Answers"]])
            
            line = {"input": context, "output": correct_choices}
            f.write(json.dumps(line) + "\n")
    f.close()

    
if __name__ == "__main__":
    
    Path("data/selection/").mkdir(parents=True, exist_ok=True)
    Path("data/generation/").mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        single_selection(split)
        single_selection(split, zero_shot=True)
        single_generation(split)
        single_generation(split, zero_shot=True)
        multiple_generation(split)
        multiple_generation(split, zero_shot=True)
        all_generation(split)
        all_generation(split, zero_shot=True)
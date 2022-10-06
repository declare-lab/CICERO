import json
import numpy as np
import pandas as pd
from pathlib import Path
from fuzzywuzzy import fuzz

questions = {
    "cause": "What is or could be the cause of target?",
    "prerequisite": "What is or could be the prerequisite of target?",
    "reaction": "What is the possible emotional reaction of the listener in response to target?",
    "motivation": "What is or could be the motivation of target?",
    "subseq_event": "What subsequent event happens or could happen following the target?",
    "subseq_event_clipped": "What subsequent event happens or could happen following the target?"
}

cause_subseq_event_questions = [
    "What is or could be the cause of target?",
    "What subsequent event happens or could happen following the target?"
]

def generation(split, relation):
    """
    Prepares data for the different answer generation tasks.
    """
    data = [json.loads(line) for line in open("../../data/" + split + ".json").readlines()]
    sep = " \\n "
    f = open("data/" + split + "_" + relation + ".json", "w")
    for instance in data:
        if questions[relation] == instance["Question"]:
            if relation != "subseq_event_clipped":
                c = " <utt> ".join(instance["Dialogue"])
            else:
                t = instance["Target"]
                utts = [u[3:] for u in instance["Dialogue"]]
                if t in utts:
                    index = utts.index(t)
                else:
                    index = np.argmax([fuzz.token_set_ratio(u, t) for u in utts])
                    
                c = " <utt> ".join(instance["Dialogue"][:index+1])
                
            context = sep.join([instance["Question"], "target: " + instance["Target"], 
                                "context: " + c])
            written_answer = instance["Choices"][instance["Human Written Answer"][0]]
            line = {"input": context, "output": written_answer}
            f.write(json.dumps(line) + "\n")
    f.close()
    
    
def create_dataframe(split):
    """
    Creates a pandas dataframe from the json files for the chained cause and subsequent event generation tasks.
    Dataframe operations are later used to extract target utterances which have both cause and subsequent event annotated.
    """
    data = [json.loads(line) for line in open("../../data/" + split + ".json").readlines()]
    ids, dialogues, targets, locations, questions, answers = [], [], [], [], [], []
    for instance in data:
        if instance["Question"] in cause_subseq_event_questions:
            ids.append(instance["ID"])
            dialogues.append(instance["Dialogue"])

            t = instance["Target"]
            targets.append(t)
            utts = [u[3:] for u in instance["Dialogue"]]
            if t in utts:
                locations.append(utts.index(t))
            else:
                locations.append(np.argmax([fuzz.token_set_ratio(u, t) for u in utts]))

            questions.append(instance["Question"])
            answers.append(instance["Choices"][instance["Human Written Answer"][0]])

    df = pd.DataFrame({
        "ID": ids, "Dialogue": dialogues, "Target": targets, "Location": locations, 
        "Question": questions, "Answer": answers
    })
    return df


def chained_generation(split):
    """
    Prepares data for the chained cause and subsequent event generation tasks.
    """
    df = create_dataframe(split)
    sep = " \\n "
    f1 = open("data/" + split + "_chained_cause.json", "w")
    f2 = open("data/" + split + "_chained_subseq_event.json", "w")

    for ids in list(set(df["ID"])):
        idf = df[df["ID"]==ids]
        for target in list(set(idf["Target"])):
            tdf = idf[idf["Target"]==target].sort_values("Question")
            if len(tdf) == 2:
                cause_answer = list(tdf["Answer"])[0]
                subseq_event_answer = list(tdf["Answer"])[1]
                dialogue = list(tdf["Dialogue"])[0]
                c = " <utt> ".join(dialogue)

                # Generate cause with subsequent event in input # 
                cause_context = sep.join([questions["cause"], "target: " + target, 
                                          "subsequent event: " + subseq_event_answer, "context: " + c])
                line1 = {"input": cause_context, "output": cause_answer}
                f1.write(json.dumps(line1) + "\n")

                # Generate subsequent event with cause in input # 
                subseq_event_context = sep.join([questions["subseq_event"], "target: " + target, 
                                                 "cause: " + cause_answer, "context: " + c])
                line2 = {"input": subseq_event_context, "output": subseq_event_answer}
                f2.write(json.dumps(line2) + "\n")

    f1.close()
    f2.close()

    
if __name__ == "__main__":
    
    Path("data/").mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        for q in questions.keys():
            generation(split, q)
        chained_generation(split)
    
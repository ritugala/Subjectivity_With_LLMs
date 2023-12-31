from openai import OpenAI
import pandas as pd
import numpy as np
import json
from utils_fns import *
import numpy as np
client = OpenAI()

system_level_prompt = """Your objective is to capture all the possible interpretations for a given
response. You will be given responses from witness testimonials
in U.S. congressional hearings. You will play the role of a judge where you have to predict the ALL possible intent of the response given the question, response pair. 
The intents labels range from ["direct_answer", "over_answer", "shift_dodge", "shift_correct", "cant_answer_lying", "cant_answer_sincere"]
Six-Seven annotators have already labeled this task, which represent the ground truth. You are provided with the sentiments of these annotators, to serve as a proxy for annotator bias. This may help you capture subjectivity. Respond with a tuple, with the range of intents, and their explanations. 

Consider the following example:
Question: "So do you adjust your algorithms to prevent individuals interested in violence from being connected with like-minded individuals?"
Response: "Sorry. Could you repeat that?"
Number of Annotators: 7
Sentiments: ('somewhatNegative', 'somewhatNegative', 'somewhatNegative', 'somewhatPositive', 'neutral', 'somewhatPositive', 'neutral')


The range of intents and their explanations are 
(["cant_answer_lying", "cant_answer_sincere"], "A cynical annotator interprets the clarification question as lying in order to stall vs. an optimistic annotator might view it as being honest")

---
 """

user_level_act_prompt = """ Now do this example:
Question: "{question}"
Response: "{response}"
Number of Annotators: {num_annotators}
Sentiments: {sentiments}

The range of acts and their explanations are: 
"""






if __name__ == '__main__':
    df = pd.DataFrame(columns=["Index","Question", "Response", "Labels", "Predicted_Labels", "Predicted_Intent", "Predicted_Intent_Explanation"])
    data = pd.read_csv("data/dev.tsv", delimiter="\t")

    data = data[:100]
    target_labels_fine = []
    predicted_labels_fine = []
    target_labels_coarse = []
    predicted_labels_coarse = []
    for idx,example in data.iterrows():
        try:
            print("Currently processing.. ", idx)
            question = example["q_text_last_question"]
            response = example["r_text"]
            labels = example["gold_labels_binary"]
            sentiments = example["gold_sentiments"]
            num_annotators = len(sentiments)
            messages=[
                {"role": "system", "content": system_level_prompt},
                {"role": "user", "content": user_level_act_prompt.format(question=question, response=response, num_annotators=num_annotators, sentiments=sentiments )}
            ]
            completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages, 
            temperature=0
            )
            response = completion.choices[0].message.content
            intents, intent_explanations = eval(response)

            predicted_labels = create_predicted_labels(intents, is_hier=False)
            labels = int_to_str_label(labels)

            df= df.append({
                "Index": example["qa_index_digits"],
                "Question": question,
                "Response": example["r_text"],
                "Labels":labels, 
                "Predicted_Labels": predicted_labels,
                "Predicted_Intent":intents, 
                "Predicted_Intent_Explanation":intent_explanations
            }, ignore_index=True)

            target_labels_fine.append(labels)
            predicted_labels_fine.append(predicted_labels)

            target_labels_coarse.append(create_acts_labels(labels))
            predicted_labels_coarse.append(create_acts_labels(predicted_labels))
        except Exception as e:
            print("LLM probably hallucinated.. ", e)
            print(response)
            continue

    df.to_csv("predictions/llm_predictions_non_hier.csv")

    metrics_fine = get_metrics(target_labels_fine, predicted_labels_fine)
    with open('metrics/non_hier_metrics_fine.json', 'w') as file:
        json.dump(metrics_fine, file, indent=4)

    metrics_coarse = get_metrics(target_labels_coarse, predicted_labels_coarse)
    with open('metrics/non_hier_metrics_coarse.json', 'w') as file:
        json.dump(metrics_coarse, file, indent=4)




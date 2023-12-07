import openai
import pandas as pd
import numpy as np
import json
from utils_fns import *
import numpy as np
import requests

system_level_prompt = """Your objective is to capture all the possible interpretations for a given
response. You will be given responses from witness testimonials
in U.S. congressional hearings. You will play the role of a judge where you have to predict the ALL possible intent of the response given the question, response pair. 
The intents labels range from ["direct_answer", "over_answer", "shift_dodge", "shift_correct", "cant_answer_lying", "cant_answer_sincere"]. You can only choose from these intent labels.

Six-Seven annotators have already labeled this task, which represent the ground truth. You are provided with the sentiments of these annotators, to serve as a proxy for annotator bias. This may help you capture subjectivity. 
Respond with a tuple, with the range of intents, and their explanations.

Consider the following example:
Question: "So do you adjust your algorithms to prevent individuals interested in violence from being connected with like-minded individuals?"
Response: "Sorry. Could you repeat that?"
Number of Annotators: 7
Sentiments: ('somewhatNegative', 'somewhatNegative', 'somewhatNegative', 'somewhatPositive', 'neutral', 'somewhatPositive', 'neutral')


For this example, the range of intents and their explanations are 
(["cant_answer_lying", "cant_answer_sincere"], "A cynical annotator interprets the clarification question as lying in order to stall vs. an optimistic annotator might view it as being honest")

---
 """

user_level_act_prompt = """ Now do this example:
Question: "{question}"
Response: "{response}"
Number of Annotators: {num_annotators}
Sentiments: {sentiments}

For this example, the range of intents and their explanations are: 
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
                {"role": "user", "content": system_level_prompt},
                {"role": "assistant", "content": user_level_act_prompt.format(question=question, response=response, num_annotators=num_annotators, sentiments=sentiments )}
            ]
            # completion = client.chat.completions.create(
            # model="gpt-3.5-turbo",
            # messages=messages, 
            # temperature=0
            # )

            completion = requests.post("http://0.0.0.0:8000/v1/chat/completions",
                json = {
                    "model": "mistralai/Mistral-7B-Instruct-v0.1",
                    "messages": messages
                }
            )


            completions_response = completion.json()['choices'][0]['message']['content'].split('\n')
            completions_response = completions_response[0] if completions_response[0].strip() != "" else completions_response[1]
            
            intents, intent_explanations = eval(completions_response)

            predicted_labels = create_predicted_labels(intents, is_hier=False)
            labels = int_to_str_label(labels)

            new_row = pd.DataFrame({
                "Index": example["qa_index_digits"],
                "Question": question,
                "Response": example["r_text"],
                "Labels":labels, 
                "Predicted_Labels": predicted_labels,
                "Predicted_Intent":intents, 
                "Predicted_Intent_Explanation":intent_explanations
            })

            df = pd.concat([df, new_row], ignore_index=True)


            target_labels_fine.append(labels)
            predicted_labels_fine.append(predicted_labels)

            target_labels_coarse.append(create_acts_labels(labels))
            predicted_labels_coarse.append(create_acts_labels(predicted_labels))
        except Exception as e:
            # print("LLM probably hallucinated.. ", e)
            print("Error: ", e)
            print("Response: ", completions_response)
            continue

    df.to_csv("predictions/mistral_predictions_non_hier.csv")

    metrics_fine = get_metrics(target_labels_fine, predicted_labels_fine)
    with open('metrics/mistral_non_hier_metrics_fine.json', 'w') as file:
        json.dump(metrics_fine, file, indent=4)

    metrics_coarse = get_metrics(target_labels_coarse, predicted_labels_coarse)
    with open('metrics/mistral_non_hier_metrics_coarse.json', 'w') as file:
        json.dump(metrics_coarse, file, indent=4)




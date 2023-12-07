import openai
import pandas as pd
import numpy as np
import json
from utils_fns import *
import numpy as np
import requests

system_level_prompt = """Your objective is to capture all the possible interpretations for a given
response. You will be given responses from witness testimonials
in U.S. congressional hearings. You will play the role of a judge where you have to predict the intent of a given response given a question. To make this simpler, we first predict the entire range conversation acts possible, which ranges from ("answer", "can't answer", "shift"). Six-Seven annotators have already labeled this task, which represent the ground truth. You are provided with the sentiments of these annotators, to serve as a proxy for annotator bias. This may help you capture subjectivity. Respond with a tuple, with the range of acts, and their explanations. 

Consider the following example:
Question: "So do you adjust your algorithms to prevent individuals interested in violence from being connected with like-minded individuals?"
Response: "Sorry. Could you repeat that?"
Number of Annotators: 7
Sentiments: ('somewhatNegative', 'somewhatNegative', 'somewhatNegative', 'somewhatPositive', 'neutral', 'somewhatPositive', 'neutral')


The range of acts and their explanations are:  
(["cant answer"], "There is no ambiguity that the witness is signaling he is unable to answer the question")

---
 """

user_level_act_prompt = """ Now do this example:
Question: "{question}"
Response: "{response}"
Number of Annotators: {num_annotators}
Sentiments: {sentiments}

For this example, the range of acts and their explanations are 
"""

system_level_intent_prompt = """ \n
We now condition on this conversation act, and predict the intent. 
"answer" act can have two intentions, "direct" and "overanswer". 
"can't answer" act can have two intentions, "honest" and "lying". 
"shift" act can have two intentions, "dodge" and "correct".  
Respond with a tuple, with the range of intents, and their explanations. 

Consider the following example:
Question: "So do you adjust your algorithms to prevent individuals interested in violence from being connected with like-minded individuals?"
Response: "Sorry. Could you repeat that?"
Act: "Can't Answer"

The range of intents and their explanations are 
(["honest", "lying"], "A cynical annotator interprets the clarification question as lying in order to stall vs. an optimistic annotator might view it as being honest")

---
"""

user_level_intent_prompt = """Now do this example:

Question: "{question}"
Response: "{response}"
Act: "{act}"

For this example, the range of intents and their explanations are 
"""





if __name__ == '__main__':
    df = pd.DataFrame(columns=["Index","Question", "Response", "Labels", "Predicted_Labels", "Predicted_Acts", "Predicted_Act_Explanation", "Predicted_Intent", "Predicted_Intent_Explanation"])
    # TODO: combine train and dev since we are doing 0 shot anyway
    data = pd.read_csv("data/dev.tsv", delimiter="\t")

    data = data[:100]
    target_labels_fine = []
    predicted_labels_fine = []
    target_labels_coarse = []
    predicted_labels_coarse = []
    completions_response = ""

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


            completion = requests.post("http://0.0.0.0:8000/v1/chat/completions",
                json = {
                    "model": "mistralai/Mistral-7B-Instruct-v0.1",
                    "messages": messages
                }
            )


            completions_response = completion.json()['choices'][0]['message']['content']

            completions_response = completion.json()['choices'][0]['message']['content'].split('\n')
            completions_response = completions_response[0] if completions_response[0].strip() != "" else completions_response[1]
            # print("Response: ", completions_response)
            # breakpoint()
            
            acts, acts_explanation = eval(completions_response)
            predicted_intents_all = []
            intent_explanations = ""
            for act in acts:
                
                messages.extend([
                    {"role": "user", "content": system_level_intent_prompt},
                    {"role": "assistant", "content": user_level_intent_prompt.format(question=question, response=response, act=act)}
                ])

                completion = requests.post("http://0.0.0.0:8000/v1/chat/completions",
                    json = {
                        "model": "mistralai/Mistral-7B-Instruct-v0.1",
                        "messages": messages
                    }
                )

                completions_response = completion.json()['choices'][0]['message']['content'].split('\n')
                completions_response = completions_response[0] if completions_response[0].strip() != "" else completions_response[1]
                # print("Intent Response: ", completions_response)

                predicted_intents, intent_explanation = eval(completions_response)
                predicted_intents_all += predicted_intents
                intent_explanations += "; "+intent_explanation
            
            predicted_labels = create_predicted_labels(predicted_intents_all)
            labels = int_to_str_label(labels)
            
            new_row = pd.DataFrame({
                "Index": example["qa_index_digits"],
                "Question": question,
                "Response": example["r_text"],
                "Labels":[labels], 
                "Predicted_Labels": [predicted_labels],
                "Predicted_Acts": [acts], 
                "Predicted_Act_Explanation":[acts_explanation], 
                "Predicted_Intent":[predicted_intents_all], 
                "Predicted_Intent_Explanation":[intent_explanations]
            })

            df = pd.concat([df, new_row], ignore_index=True)

            target_labels_fine.append(labels)
            predicted_labels_fine.append(predicted_labels)

            target_labels_coarse.append(create_acts_labels(labels))
            predicted_labels_coarse.append(create_acts_labels(predicted_labels))
        
        except Exception as e:
            print(f"[ERROR] {e}")
            print(f"[RESP] {completions_response}")
            continue

    df.to_csv("predictions/mistral_predictions_hier.csv")

    metrics_fine = get_metrics(target_labels_fine, predicted_labels_fine)
    with open('metrics/mistral_hier_metrics_fine.json', 'w') as file:
        json.dump(metrics_fine, file, indent=4)

    metrics_coarse = get_metrics(target_labels_coarse, predicted_labels_coarse)
    with open('metrics/mistral_hier_metrics_coarse.json', 'w') as file:
        json.dump(metrics_coarse, file, indent=4)




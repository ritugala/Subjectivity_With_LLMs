from openai import OpenAI
import pandas as pd
import numpy as np
import json
import sys
sys.path.append('C:\\Users\\Ritu\\Downloads\\gold_cv_dev_data\\CongressionalHearing\\llms_subjective')
from utils_fns import create_predicted_label_leading_q, int_to_str_label, create_acts_labels, get_metrics
import numpy as np
client = OpenAI()

system_level_prompt = """Your role is to play that of a judge. However, you are to consider all possible interpretations. The following is question-response pairs from witness testimonials in U.S. congressional hearings."""

user_level_act_prompt = """ 
Question: "{question}"
Response: "{response}"
 
"""

COT= "\nProvide analysis for this question/answer pair to determine the intent of the answer. Consider ALL possible reasonable interpretations possible.\n"

leading_questions = """

Now provide "yes"/"no" answers to the following questions considering the various interpretations. Return yes if any of those interpretations satisfy a below question. Return a json dict with yes/no for the same. eg output : 
{
    "Does the answer satisfy the question and is it direct?": "no",
    " Does the answer satisfy the question but is the respondent overanswering?": "no",
    "Is the respondent attempting to shift and dodge the question, thus showing insincere intent?: "yes",
    "Is the respondent attempting to shift the question but correct it, thus showing sincere intent?": "no",
    "Is the respondent unable to answer the question and  If so does it seem like they are lying?": "no",
    "Is the respondent unable to answer the question and  If so does it seem like they are sincere?": "yes"
}
1. Does the answer satisfy the question and is it direct?
2. Does the answer satisfy the question but is the respondent overanswering?
3. Is the respondent attempting to shift and dodge the question, thus showing insincere intent?
4. Is the respondent attempting to shift the question but correct it, thus showing sincere intent?
5. Is the respondent unable to answer the question and  If so does it seem like they are lying?
6. Is the respondent unable to answer the question and  If so does it seem like they are sincere?

"""


def build_input(question, response,has_cot=True):
    fin = ""
    fin += user_level_act_prompt.format(question=question, response=response)
    if has_cot:
        fin += COT
    fin += leading_questions
    return fin
    
def parse_response(response):
    ind = response.index('{')
    expln = response[:ind]
    intents = eval(response[ind:])
    return intents, expln


if __name__ == '__main__':
    df = pd.DataFrame(columns=["Index","Question", "Response", "Labels", "Predicted_Labels", "Predicted_Intent", "Predicted_Intent_Explanation"])
    # TODO: combine train and dev since we are doing 0 shot anyway
    data = pd.read_csv("data/train.tsv", delimiter="\t")

    data = data[:5]
    target_labels_fine = []
    predicted_labels_fine = []
    target_labels_coarse = []
    predicted_labels_coarse = []
    for idx,example in data.iterrows():
        try:
            print("Currently processing.. ", idx)
            question = example["q_text"]
            answer = example["r_text"]
            labels = example["gold_labels_binary"]

            
            messages=[
                {"role": "system", "content": system_level_prompt},
                {"role": "user", "content": build_input(question, answer)}
            ]
            completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages, 
            temperature=0
            )
            response = completion.choices[0].message.content
            intents, intent_explanations = parse_response(response)

            predicted_labels = create_predicted_label_leading_q(intents)
            labels = int_to_str_label(labels)

            df= df.append({
                "Index": example["qa_index_digits"],
                "Question": question,
                "Response": answer,
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

    df.to_csv("predictions/llm_predictions_leading_questions.csv")

    metrics_fine = get_metrics(target_labels_fine, predicted_labels_fine)
    with open('metrics/leading_questions_metrics_fine.json', 'w') as file:
        json.dump(metrics_fine, file, indent=4)

    metrics_coarse = get_metrics(target_labels_coarse, predicted_labels_coarse)
    with open('metrics/leading_questions_metrics_coarse.json', 'w') as file:
        json.dump(metrics_coarse, file, indent=4)




# from openai import OpenAI
import openai
import pandas as pd
import numpy as np
import json
import sys
import ast
import requests
# LOL sorry i dont know how to fix this better
# sys.path.append('/Users/meghanaarajeev/Documents/ANLP/Subjectivity_With_LLMs')

sys.path.append('/home/ubuntu/Subjectivity_With_LLMs/')
from utils_fns import create_predicted_label_leading_q, int_to_str_label, create_acts_labels, get_metrics
import numpy as np
# client = OpenAI()

#@meghana: This is code from leading_questions.py I dont personally think you should mimic the "consider all interpreations part", considering we want these to be as objective as possible (since these features are not meant to be subjective). However use your best judgement. Hehe. 
system_level_prompt = """You will be given question-response pairs from witness testimonials in U.S. congressional hearings. Your job is to pick up on cues of the response which will later be used for determining intent of the response. To aid with this, you will be asked a list of questions. Please answer this as yes/no. """

user_level_act_prompt = """ Analyze the following response and question thoroughly. 
Question: "{question}"
Response: "{response}"
  
"""

feature_extraction_prompt = """
First provide explanation of this response. Then answer the following questions based on your objective interpretation of the response to the question.

Questions:
1. Does the text contain filler words (e.g., "um", "uh", "you know")? 
2. Does the text imply stuttering?
3. Is the text concise?
4. Is there repetition between the question and the answer?
5. Does the text contain rambling?
6. Does the text reference external sources or authorities?
7. Does the text use complicated terms/jargon?
8. Does the text use sarcasm? 
9. Does the text contain rhetorical questions?
10. Is the response logically structured and coherent without contradictions?
11. Does the response use hedging language(e.g. 'it seems', 'possibly', 'might be', 'could be', or similar expressions that indicate uncertainty or non-commitment.)?
12. Does the prompt use assertive language like (but not limited to) "definitely," "certainly," or "absolutely" to express confidence or conviction?

 

The answers to the questions should be in the following JSON dict format with yes or no for each question. eg output: 
{
 "Filler Words": "yes",
 "Stuttering": "no",
 "Concise": "no",
 "Repetition": "no",
 "Rambling": "no",
 "External References": "yes",
 "Complicated Terms": "yes",
 "Sarcasm": "yes",
 "Rhetorical Questions": "yes",
 "Logical Coherence": "yes",
 "Hedging Language": "no",
 "Assertive Language": "yes",
}.
The explanations and the answer_json is:
"""

def build_input(question, response):
    fin = ""
    fin += user_level_act_prompt.format(question=question, response=response)
    fin += feature_extraction_prompt
    return fin
    
def parse_response(response):

    start_ind = response.index('{')
    end_ind = response.index('}') + 1
    expln = response[:start_ind]
    features = eval(response[start_ind:end_ind])
    return features, expln


if __name__ == '__main__':
    df = pd.DataFrame(columns=["Index", "Question", "Response", "Labels", "Filler Words", "Stuttering", "Concise", "Repetition", "Rambling", "External References", "Complicated Terms", "Sarcasm", "Rhetorical Questions", "Logical Coherence", "Hedging Language", "Assertive Language", "Explanation"])
    # TODO: combine train and dev since we are doing 0 shot anyway
    data = pd.read_csv("data/train.tsv", delimiter="\t")

    target_labels_fine = []
    predicted_labels_fine = []
    target_labels_coarse = []
    predicted_labels_coarse = []
    completions_response = ""
    results = {'success': 0, 'fail': 0}
    num_retries = 3
    for idx,example in data.iterrows():
        curr_attempts = 0
        while curr_attempts < num_retries:
            try:
                print("Currently processing.. ", idx)
                question = example["q_text"]
                answer = example["r_text"]
                labels = example["gold_labels_binary"]
                
                messages=[
                    {"role": "user", "content": system_level_prompt},
                    {"role": "assistant", "content": build_input(question, answer)}
                ]

                completion = requests.post("http://0.0.0.0:8000/v1/chat/completions",
                    json = {
                        "model": "mistralai/Mistral-7B-Instruct-v0.1",
                        "messages": messages,
                        "temperature":0,
                        "max_tokens": 1000
                    }
                )

                completions_response = completion.json()['choices'][0]['message']['content']
                features, expln = parse_response(completions_response)
                
                
                # predicted_labels = create_predicted_label_leading_q(intents)
                labels = int_to_str_label(labels)

        
                new_row = {
                    "Index": example["qa_index_digits"], 
                    "Question": question, 
                    "Response": answer, 
                    "Labels": labels,
                    "Explanation": expln
                }
                for feature, value in features.items():
                    # print(feature, value)
                    new_row[feature] = value
                new_row_df = pd.DataFrame([new_row])

                df = pd.concat([df, new_row_df], ignore_index=True)
            

                print("Success!")
                results['success'] += 1
            except Exception as e:
                print("Error ", e)
                print("hallucination..")
                print(completions_response)
                
                curr_attempts +=1 
        if curr_attempts==3: 
            results['fail'] += 1
            print("Result still failed..")

        df.to_csv("predictions/mistral_llm_features_v2.csv")
    print(results)

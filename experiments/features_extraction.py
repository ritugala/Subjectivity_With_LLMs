from openai import OpenAI
import pandas as pd
import numpy as np
import json
import sys
import ast
# LOL sorry i dont know how to fix this better
sys.path.append('/Users/meghanaarajeev/Documents/ANLP/Subjectivity_With_LLMs')
from utils_fns import create_predicted_label_leading_q, int_to_str_label, create_acts_labels, get_metrics
import numpy as np
client = OpenAI()

#@meghana: This is code from leading_questions.py I dont personally think you should mimic the "consider all interpreations part", considering we want these to be as objective as possible (since these features are not meant to be subjective). However use your best judgement. Hehe. 
system_level_prompt = """You will be given question-response pairs from witness testimonials in U.S. congressional hearings. Your job is to pick up on cues of the response which will later be used for determining intent of the response. To aid with this, you will be asked a list of questions. Please answer this as yes/no. """

user_level_act_prompt = """ Analyze the following response and question thoroughly. 
Question: "{question}"
Response: "{response}"
  
"""

feature_extraction_prompt = """
Now answer the following questions based on your objective interpretation of the response to the question.

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
10. Does the response directly address and answer the question asked, or does it divert from the topic?
11. Is the response logically structured and coherent without contradictions?
12. Does the response use hedging language(e.g. 'it seems', 'possibly', 'might be', 'could be', or similar expressions that indicate uncertainty or non-commitment.)?
13. Does the prompt use assertive language like (but not limited to) "definitely," "certainly," or "absolutely" to express confidence or conviction?
14. Does the prompt use evasive language with terms like (but not limited to) to the best of my knowledge" or "as far as I am aware" to avoid direct commitment?
 

Provide answers in the following JSON dict format with yes or no for each question. eg output: 
{
 "Filler Words": "yes/no",
 "Stuttering": "yes/no",
 "Concise": "yes/no",
 "Repetition": "yes/no",
 "Rambling": "yes/no",
 "External References": "yes/no",
 "Complicated Terms": "yes/no",
 "Sarcasm": "yes/no",
 "Rhetorical Questions": "yes/no",
 "Direct Answer": "yes/no",
 "Logical Coherence": "yes/no",
 "Hedging Language": "yes/no",
 "Assertive Language": "yes/no",
 "Evasive Language": "yes/no"
}
"""

def build_input(question, response):
    fin = ""
    fin += user_level_act_prompt.format(question=question, response=response)
    fin += feature_extraction_prompt
    return fin
    
def parse_response(response):
    ind = response.index('{')
    # expln = response[:ind]
    features = eval(response[ind:])
    return features

if __name__ == '__main__':
    df = pd.DataFrame(columns=["Index", "Question", "Response", "Labels", "Predicted_Labels", "Features", "Filler Words", "Stuttering", "Concise", "Repetition", "Rambling", "External References", "Complicated Terms", "Sarcasm", "Rhetorical Questions", "Direct Answer", "Logical Coherence", "Hedging Language", "Assertive Language", "Evasive Language"])
    # TODO: combine train and dev since we are doing 0 shot anyway
    data = pd.read_csv("data/train.tsv", delimiter="\t")


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
            features = parse_response(response)
            
            
            # predicted_labels = create_predicted_label_leading_q(intents)
            labels = int_to_str_label(labels)

       
            new_row = {
                "Index": example["qa_index_digits"], 
                "Question": question, 
                "Response": answer, 
                "Labels": labels, 
                "Features": features
            }
            for feature, value in features.items():
                print(feature, value)
                new_row[feature] = value

           
            df = df.append(new_row, ignore_index=True)

            # target_labels_fine.append(labels)
            # predicted_labels_fine.append(predicted_labels)

            # target_labels_coarse.append(create_acts_labels(labels))
            # predicted_labels_coarse.append(create_acts_labels(predicted_labels))
        except Exception as e:
            print("Error ", e)
            print(response)
            continue

    df.to_csv("predictions/llm_features.csv")

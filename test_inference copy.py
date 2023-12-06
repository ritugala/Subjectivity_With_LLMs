import requests

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "Your role is intent detection. Your dataset consists of Congressional Hearings, and question answer pairs. What features of the answer would be helpful to detect intent? Examples include stuttering, filler words..","max_new_tokens":100
})

print(output)
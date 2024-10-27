import requests
from datasets import load_dataset
import json

# Load XSum dataset
dataset = load_dataset("xsum", split="train", trust_remote_code=True)

# API endpoint for text generation
LLAMACPP_URL = "http://localhost:8080/completion"

HEADERS = {
    "Content-Type": "application/json"
}

def summarize(article):
    data = {
        "prompt": "Write a short summary of this article:\n\n" + article,
        "n_predict": 64
    }
    
    response = requests.post(LLAMACPP_URL, headers=HEADERS, data=json.dumps(data))
    
    if response.status_code == 200:
        return response.json().get("content")
    else:
        raise Exception(f"Failed to generate summary: {response.text}")

pairs = []

for example in dataset:
    article = example['document']
    human_summary = example['summary']
    try:
        ai_summary = summarize(article)
        pairs.append({
            'article': article,
            'ai_summary': ai_summary,
            'human_summary': human_summary
        })
    except Exception as e:
        print(f"Error generating summary for article: {e}")

output_file = "xsum_summaries.json"
with open(output_file, 'w') as f:
    json.dump(pairs, f, indent=4)

print(f"Summaries saved to {output_file}")

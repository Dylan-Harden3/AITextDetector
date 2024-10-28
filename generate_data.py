from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch
import argparse
from dotenv import load_dotenv
import os

def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.getenv("HF_TOKEN"), cache_dir=os.getenv("HF_CACHE_DIR"))
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto", token=os.getenv("HF_TOKEN"), cache_dir=os.getenv("HF_CACHE_DIR"))

    return model, tokenizer


def summarize(article, model, tokenizer, max_length=64):
    prompt = f"Write a short summary of this article:\n\n{article}"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"], max_new_tokens=max_length, num_return_sequences=1
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = summary.replace(prompt, "").strip()
    return summary


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    dataset = load_dataset(
        "xsum",
        split="train",
        cache_dir=os.getenv("HF_CACHE_DIR"),
        trust_remote_code=True
        )

    model, tokenizer = load_model(args.model_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)
    pairs = []

    for example in dataset.select(range(2)):
        article = example["document"]
        human_summary = example["summary"]

        try:
            ai_summary = summarize(article, model, tokenizer)

            pairs.append(
                {
                    "article": article,
                    "ai_summary": ai_summary,
                    "human_summary": human_summary,
                }
            )
        except Exception as e:
            pass

    with open(args.output, "w") as f:
        json.dump(pairs, f, indent=4)

    print(f"Summaries saved to {args.output}")
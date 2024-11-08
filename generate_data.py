from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import torch
import argparse
from dotenv import load_dotenv
import os

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, required=True)
    args = parser.parse_args()
    dataset = load_dataset(
        "xsum",
        split="train",
        cache_dir="/scratch/user/dylanharden3/AITextDetector/dataset",
        trust_remote_code=True,
    )
    pairs = []
    summarizer = pipeline(task="text-generation", model=args.model_id, device=0)
    for example in dataset:
        article = example["document"]
        human_summary = example["summary"]
        prompt = f"Write a super short summary of this article:\n\n{article}"
        try:
            ai_summary = summarizer(
                [{"role": "user", "content": prompt}],
                do_sample=False,
                temperature=None,
                top_p=None,
                max_new_tokens=32,
                pad_token_id=summarizer.tokenizer.eos_token_id,
            )[0]["generated_text"][1]["content"]
            pairs.append(
                {
                    "article": article,
                    "ai_summary": ai_summary,
                    "human_summary": human_summary,
                }
            )
        except Exception as e:
            print(e)
    outname = args.model_id + ".json"
    if "/" in outname:
        outname = outname[outname.index("/") + 1 :]
    with open(outname, "w") as f:
        json.dump(pairs, f, indent=4)
    print(f"Wrote {len(dataset)} summaries to {outname}")

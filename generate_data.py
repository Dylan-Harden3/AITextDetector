from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import torch
import argparse
from dotenv import load_dotenv
import os
import numpy as np

def generate_data(dataset, model, args):
    dataset = load_dataset(
        dataset,
        split="train",
        cache_dir=args.cache,
        trust_remote_code=True
    )["document"]
    
    # remove newlines and strip whitespace
    dataset = [x.strip() for x in dataset]
    dataset = [' '.join(x.split()) for x in dataset]
    
    dataset = dataset[:args.n_samples]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    token = os.getenv("HF_TOKEN")
    
    tokenizer = AutoTokenizer.from_pretrained(model, padding_side="right", cache_dir=args.cache, token=token)
    model = AutoModelForCausalLM.from_pretrained(model, cache_dir=args.cache, token=token).to(device)
    model.eval()

    sampling_args = {
            "top_p": args.top_p,
            "temperature": args.temperature,
            "top_k": args.top_k
            }

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    data = []

    for batch in range(len(dataset) // args.batch_size):
        print(f"Batch {batch} of {len(dataset) // args.batch_size}")
        human_text = dataset[batch*args.batch_size:(batch+1)*args.batch_size]

        tokenized = tokenizer(human_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)
        # use the first 30 tokens to generate an article
        tokenized = {k: v[:, :30] for k, v in tokenized.items()}

        outputs = model.generate(**tokenized, max_length=200, do_sample=True, **sampling_args, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)

        ai_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for human, ai in zip(human_text, ai_text):
            min_len = min(len(human), len(ai))
            human, ai = human[:min_len], ai[:min_len]
            data.append((human, ai))

    output_file = args.output_file if args.output_file else f"{args.model[args.model.index('/'):]}_{args.dataset}.json"

    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)
        print(f"Saved dataset to {output_file}")


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--cache", type=str, default=".cache")
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    
    generate_data(args.dataset, args.model, args)

import json
import torch
import argparse
from dotenv import load_dotenv
import os
from utils import log_likelihood, rank, log_rank, load_model, roc


def detect(args):
    model, tokenizer = load_model(args.model, args.cache, os.getenv("HF_TOKEN"))
    with open(args.dataset_file, "r") as f:
        dataset = json.load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    method_funcs = {"ll": log_likelihood, "rank": rank, "logrank": log_rank}
    methods = ["ll", "rank", "logrank"]

    results = {}
    results["model"] = args.model
    results["dataset"] = args.dataset_file

    for method in methods:
        print(f"starting {method}")
        human_text_preds = []
        ai_text_preds = []

        criterion = method_funcs[method]

        for human_text, ai_text in dataset:
            tokens = tokenizer(
                human_text,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
            ).to(device)
            token_ids = tokens.input_ids[:, 1:]

            with torch.no_grad():
                logits = model(**tokens).logits[:, :-1]
                pred_human = criterion(token_ids, logits)

            tokens = tokenizer(
                ai_text, return_tensors="pt", padding=True, return_token_type_ids=False
            ).to(device)
            token_ids = tokens.input_ids[:, 1:]

            with torch.no_grad():
                logits = model(**tokens).logits[:, :-1]
                pred_ai = criterion(token_ids, logits)

            human_text_preds.append(pred_human)
            ai_text_preds.append(pred_ai)

        fpr, tpr, roc_auc = roc(human_text_preds, ai_text_preds)
        print(f"{method} {roc_auc}")
        results[method] = {
            "predictions": {"human": human_text_preds, "ai": ai_text_preds},
            "results": [fpr, tpr, roc_auc],
        }

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--cache", type=str, default=".cache")
    args = parser.parse_args()

    detect(args)

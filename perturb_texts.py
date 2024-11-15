from dotenv import load_dotenv
import argparse
from utils import load_mask_model
import os
import json
from math import ceil
import random
import re


def add_masks(args, text):
    tokens = text.split(
        " "
    )  # in detectgpt code they replace words instead of actual tokens
    n_masks = ceil(args.percent_perturb * len(tokens))

    assert n_masks < len(tokens)
    masked_idxs = set()
    for i in range(n_masks):
        mask_ix = random.randint(0, len(tokens) - 1)
        while mask_ix in masked_idxs:
            mask_ix = random.randint(0, len(tokens) - 1)

        tokens[mask_ix] = "<<<mask>>>"

    mask_num = 0
    for i in range(len(tokens)):
        if tokens[i] == "<<<mask>>>":
            tokens[i] = f"<extra_id_{mask_num}>"
            mask_num += 1

    return " ".join(tokens)


def find_largest_extra_id(texts):
    return [
        len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts
    ]


def replace_masks(args, mask_model, mask_tokenizer, texts):
    # generate the filler texts with mask model
    max_extra_id = max(find_largest_extra_id(texts))
    stop_id = mask_tokenizer.encode(f"<extra_id_{max_extra_id}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(
        next(mask_model.parameters()).device
    )
    outputs = mask_model.generate(
        **tokens,
        max_length=150,
        num_return_sequences=1,
        do_sample=True,
        eos_token_id=stop_id,
    )

    filler_texts = mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)
    # remove <pad> and </s>
    filler_texts = [
        text.replace("<pad>", "").replace("</s>", "").strip() for text in filler_texts
    ]
    # get the replacement text for each extra_id token
    pattern = re.compile(r"<extra_id_\d+>")
    fills = [pattern.split(text)[1:-1] for text in filler_texts]
    fills = [[y.strip() for y in x] for x in fills]
    # replace extra_id tokens with fills
    tokens = [x.split(" ") for x in texts]

    masks = find_largest_extra_id(texts)
    for i, (text, fill, n_masks) in enumerate(zip(tokens, fills, masks)):
        if len(fill) < n_masks:
            tokens[i] = []
        else:
            for fill_ix in range(n_masks):
                text[text.index(f"<extra_id_{fill_ix}>")] = fill[fill_ix]
    texts = [" ".join(x) for x in tokens]
    return texts


def perturb_texts(args, mask_model, mask_tokenizer, texts):
    chunk_size = 10
    res = []
    for i in range(0, len(texts), chunk_size):
        masked = [add_masks(args, x) for x in texts[i : i + chunk_size]]
        perturbed = replace_masks(args, mask_model, mask_tokenizer, masked)
        tries = 1
        while "" in perturbed:
            idxs = [i for i, x in enumerate(perturbed) if x == ""]
            print(f"{len(idxs)} texts have missing fills, tries {tries}")
            new_masked = [add_masks(args, x) for i, x in enumerate(texts) if i in idxs]
            new_perturbed = replace_masks(args, mask_model, mask_tokenizer, new_masked)
            for i, x in zip(idxs, new_perturbed):
                perturbed[i] = x
            tries += 1
        res.extend(perturbed)
    return res


def generate_perturbed_texts(args):
    perturb_filename = (
        f"{args.dataset_file[:args.dataset_file.index('.')]}_{args.n_perturb_seqs}.json"
    )
    print("loading model")
    mask_model, mask_tokenizer = load_mask_model(
        args.perturb_model, args.cache, os.getenv("HF_TOKEN")
    )
    mask_model.eval()

    with open(args.dataset_file, "r") as f:
        dataset = json.load(f)

    print("perturbing")
    perturbed_texts = []
    sample_num = 0
    for human_text, ai_text in dataset:
        print(f"on sample {sample_num}")
        human_perturbed = perturb_texts(
            args, mask_model, mask_tokenizer, [human_text] * args.n_perturb_seqs
        )
        ai_perturbed = perturb_texts(
            args, mask_model, mask_tokenizer, [ai_text] * args.n_perturb_seqs
        )
        perturbed_texts.append(
            {
                "human": human_text,
                "human_perturbed": human_perturbed,
                "ai": ai_text,
                "ai_perturbed": ai_perturbed,
            }
        )
        sample_num += 1
    with open(perturb_filename, "w") as f:
        json.dump(perturbed_texts, f, indent=4)


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--perturb_model", type=str, default="google-t5/t5-large")
    parser.add_argument("--n_perturb_seqs", type=int, default=10)
    parser.add_argument("--percent_perturb", type=float, default=0.15)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--cache", type=str)
    args = parser.parse_args()

    generate_perturbed_texts(args)

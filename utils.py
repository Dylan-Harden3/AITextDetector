import torch
import torch.nn.functional as F

def log_likelihood(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    logits = logits.view(-1, logits.shape[-1])[:-1] # chop off the next token prediction
    input_ids = input_ids.view(-1)[1:] # chop off bos token from labels
    N = input_ids.shape[0]

    probs = F.softmax(logits, dim=-1)

    actual_token_probs = probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

    return actual_token_probs.prod().item() ** (-1/N)


def average_rank(model, tokenizer, text, log=False):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    logits = logits.view(-1, logits.shape[-1])[:-1] # chop off the next token prediction
    input_ids = input_ids.view(-1)[1:] # chop off bos token from labels
    
    sorted_indices = torch.argsort(logits, dim=-1, descending=True)
    
    ranks = []
    for i in range(len(input_ids)):
        target_id = input_ids[i]
        rank = (sorted_indices[i] == target_id).nonzero().item() + 1
        ranks.append(rank)
    
    ranks = torch.tensor(ranks)

    if log:
        ranks = torch.log(ranks.float())
    average_rank = ranks.float().mean().item()
    
    return average_rank
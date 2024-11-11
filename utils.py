import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def log_likelihood(token_ids, logits):
    logits = logits.view(-1, logits.shape[-1])
    token_ids = token_ids.view(-1)

    log_probs = F.log_softmax(logits, dim=-1)
    actual_token_probs = log_probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
    return actual_token_probs.mean().item()

def rank(token_ids, logits):
    sorted_indices = torch.argsort(logits, dim=-1, descending=True)
    
    matches = (sorted_indices == token_ids.unsqueeze(-1)).nonzero()
    ranks = matches[:, -1]
    ranks = ranks.float() + 1

    return -ranks.mean().item()

def log_rank(token_ids, logits):
    sorted_indices = torch.argsort(logits, dim=-1, descending=True)

    matches = (sorted_indices == token_ids.unsqueeze(-1)).nonzero()
    ranks = matches[:, -1]

    log_ranks = torch.log(ranks.float() + 1)

    return -log_ranks.mean().item()

def load_model(model_id, cache=None, token=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache, token=token).to(device)

    return model, tokenizer

def roc(human_preds, ai_preds):
    fpr, tpr, _ = roc_curve([0] * len(human_preds) + [1] * len(ai_preds), human_preds + ai_preds)
    auc_roc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(auc_roc)

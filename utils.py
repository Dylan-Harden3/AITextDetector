import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from transformers import AutoTokenizer, AutoModelForCausalLM, GemmaTokenizerFast


def log_likelihood(token_ids, logits):
    logits = logits.view(-1, logits.shape[-1])
    token_ids = token_ids.view(-1)

    log_probs = F.log_softmax(logits, dim=-1)
    actual_token_probs = log_probs.gather(
        dim=-1, index=token_ids.unsqueeze(-1)
    ).squeeze(-1)
    return actual_token_probs.mean().item()


def rank(token_ids, logits):
    matches = (logits.argsort(-1, descending=True) == token_ids.unsqueeze(-1)).nonzero()
    ranks = matches[:, -1]
    ranks = ranks.float() + 1

    return -ranks.mean().item()


def log_rank(token_ids, logits):
    matches = (logits.argsort(-1, descending=True) == token_ids.unsqueeze(-1)).nonzero()
    ranks = matches[:, -1]

    log_ranks = torch.log(ranks.float() + 1)

    return -log_ranks.mean().item()


def load_model(model_id, cache=None, token=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # for some reason gemma toknizer was throwing an error, not sure why but this fixed it
    if model_id != "google/gemma-2-9b":
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, {"padding_side": "right"}, cache_dir=cache, token=token
        )
    else:
        tokenizer = GemmaTokenizerFast(
            vocab_file="/scratch/user/dylanharden3/AITextDetector/dataset/models--google--gemma-2-9b/snapshots/33c193028431c2fde6c6e51f29e6f17b60cbfac6/tokenizer.model",
            tokenizer_file="/scratch/user/dylanharden3/AITextDetector/dataset/models--google--gemma-2-9b/snapshots/33c193028431c2fde6c6e51f29e6f17b60cbfac6/tokenizer.json",
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=cache, token=token
    ).to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def roc(human_preds, ai_preds):
    fpr, tpr, _ = roc_curve(
        [0] * len(human_preds) + [1] * len(ai_preds), human_preds + ai_preds
    )
    auc_roc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(auc_roc)

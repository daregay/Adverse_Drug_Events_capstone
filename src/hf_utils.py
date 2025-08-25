from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, Trainer

# ---------------------------
# Token-length diagnostics
# ---------------------------

def token_length_stats(tokenizer, texts, quantiles=(0.95, 0.99)) -> Dict[str, Any]:
    lens = []
    for s in texts:
        lens.append(len(tokenizer.encode(str(s), add_special_tokens=True)))
    arr = np.asarray(lens)
    stats = {"max": int(arr.max()), "mean": float(arr.mean())}
    for q in quantiles:
        stats[f"q{int(q*100)}"] = int(np.quantile(arr, q))
    return stats

# ---------------------------
# Dataset & collator
# ---------------------------

class BinaryTextDataset(Dataset):
    def __init__(self, df, text_col: str, tokenizer, max_len: int):
        self.texts = df[text_col].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, i: int):
        enc = self.tokenizer(
            self.texts[i],
            truncation=True,
            padding=False,
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item

def make_hf_datasets(train_df, val_df, test_df, text_col: str, tokenizer, max_len: int):
    ds_train = BinaryTextDataset(train_df, text_col, tokenizer, max_len)
    ds_val   = BinaryTextDataset(val_df,   text_col, tokenizer, max_len)
    ds_test  = BinaryTextDataset(test_df,  text_col, tokenizer, max_len)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return ds_train, ds_val, ds_test, collator

# ---------------------------
# Weighted Trainer (CE loss)
# ---------------------------

class WeightedTrainer(Trainer):
    """
    CrossEntropyLoss with class weights for binary labels.
    Pass class_weights=[w0, w1] (order matches label ids).
    """
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is None:
            self.class_weights = None
        else:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        import torch.nn as nn
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# ---------------------------
# Logits → probs + metrics
# ---------------------------

from sklearn.metrics import average_precision_score

def proba_from_logits(logits: np.ndarray) -> np.ndarray:
    """
    logits shape: [N, 2] -> probability of class 1
    """
    logits = np.asarray(logits)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    return p[:, 1]

def compute_metrics_pr_auc(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    probs = proba_from_logits(np.asarray(logits))
    ap = average_precision_score(labels, probs)
    return {"pr_auc": float(ap)}

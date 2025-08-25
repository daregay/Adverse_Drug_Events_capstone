from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple, Dict, Any, Optional, Sequence
import json, random, re
import numpy as np
from sklearn.metrics import precision_recall_curve
# =========================
# Basics & I/O
# =========================

def set_seed(seed: int = 42) -> None:
    """Set seeds for random, numpy, torch, and transformers (if available)."""
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    try:
        from transformers import set_seed as hf_set_seed
        hf_set_seed(seed)
    except Exception:
        pass

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(obj: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default

def safe_read_csv(path: Path):
    import pandas as pd
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def df_expect_cols(df, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"DataFrame missing columns: {missing}")

# =========================
# Text cleaning & labeling
# =========================

def light_clean(s: str) -> str:
    s = str(s).replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def norm_for_dedup(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def label_balance(df) -> Dict[str, Any]:
    vc = df["label"].value_counts().reindex([0, 1]).fillna(0).astype(int)
    tot = int(vc.sum())
    ade_ratio = float((vc.loc[1] / tot) if tot else 0.0)
    return {
        "count_not_ade": int(vc.loc[0]),
        "count_ade": int(vc.loc[1]),
        "total": tot,
        "ade_ratio": ade_ratio,
    }

def choose_text_col(df, prefer: str = "text_clean") -> str:
    return prefer if prefer in df.columns else "text"

# =========================
# Class weights (train only)
# =========================

def compute_class_weights(labels: Iterable[int]) -> np.ndarray:
    """
    Inverse-frequency weights in label order [0, 1].
    """
    labels = np.asarray(list(labels), dtype=int)
    w = np.zeros(2, dtype=float)
    for c in (0, 1):
        cnt = np.sum(labels == c)
        w[c] = 0.0 if cnt == 0 else 1.0 / cnt
    # normalize to mean 1.0 (nice for torch losses)
    w = w * (2.0 / np.sum(w)) if np.sum(w) > 0 else np.array([1.0, 1.0])
    return w

# =========================
# TF-IDF for baselines
# =========================
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

EXTRA_STOP = {
    "patient","patients","mg","mcg","day","daily","days","tablet","tablets",
    "dose","doses","po","iv","bid","tid","qhs"
}
NEG_KEEP = {"no","not","never","without"}
STOP = sorted(list((ENGLISH_STOP_WORDS - NEG_KEEP).union(EXTRA_STOP)))

def build_tfidf(ngram=(1, 2), max_features=50_000, min_df=2, max_df=0.95) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=ngram,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        stop_words=STOP,
        lowercase=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w+\b",
        sublinear_tf=True,
    )

# =========================
# Metrics & thresholds
# =========================
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score,
    f1_score, precision_score, recall_score, confusion_matrix
)

def eval_binary(y_true: np.ndarray, y_score: np.ndarray, thresh: float = 0.5) -> Dict[str, float]:
    y_pred = (y_score >= thresh).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "threshold": float(thresh),
    }

def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, Dict[str, float]]:
    p, r, t = precision_recall_curve(y_true, y_score)
    f1 = 2 * p * r / (p + r + 1e-12)
    i = int(np.argmax(f1))
    thr = 0.5 if i == 0 else float(t[i - 1])
    stats = {"f1": float(f1[i]), "precision": float(p[i]), "recall": float(r[i])}
    return thr, stats

def pick_thr_with_recall_floor(
    y_true: np.ndarray,
    y_score: np.ndarray,
    floor: float = 0.90
) -> Tuple[float, Dict[str, float]]:
    """
    Choose the threshold that maximizes PRECISION subject to RECALL >= floor.
    If multiple thresholds have the same precision, break ties with higher F1.
    If no threshold can meet the recall floor, fall back to the global best-F1 point.

    Returns:
        (thr, stats) where stats = eval_binary(y_true, y_score, thr) + {"recall_floor": floor}
    """
    p, r, t = precision_recall_curve(y_true, y_score)  # len(t) = len(p) - 1
    f1 = 2 * p * r / (p + r + 1e-12)

    # candidates meeting the recall floor
    idxs = np.where(r >= float(floor))[0]

    if len(idxs) > 0:
        # precision-first, tie-break by F1
        best_i = None
        best_prec = -1.0
        best_f1 = -1.0
        for i in idxs:
            pi, f1i = float(p[i]), float(f1[i])
            if (pi > best_prec) or (pi == best_prec and f1i > best_f1):
                best_prec, best_f1, best_i = pi, f1i, int(i)
        i = best_i
    else:
        # fallback: global best-F1
        i = int(np.argmax(f1))

    thr = 0.5 if i == 0 else float(t[i - 1])

    stats = eval_binary(y_true, y_score, thr)
    stats.update({"recall_floor": float(floor)})
    return float(thr), stats
    

def threshold_sweep(
    y_true: np.ndarray,
    y_score: np.ndarray,
    grid: Optional[Iterable[float]] = None
):
    """Return a dict of metrics per threshold across a grid (useful for plots)."""
    import pandas as pd
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    rows = []
    for t in grid:
        m = eval_binary(y_true, y_score, float(t))
        rows.append(m)
    return pd.DataFrame(rows)

# =========================
# Simple ensembling
# =========================

def ensemble_mean(*probs: Iterable[np.ndarray]) -> np.ndarray:
    return np.mean(np.vstack(probs), axis=0)

def _logits(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log(1 - p)

def ensemble_logit_avg(*probs: Iterable[np.ndarray]) -> np.ndarray:
    ls = np.vstack([_logits(np.asarray(p)) for p in probs])
    z = np.mean(ls, axis=0)
    return 1.0 / (1.0 + np.exp(-z))

# =========================
# CSV helpers
# =========================

def save_probs_csv(
    ids: Iterable,
    y_true: Iterable,
    y_score: Iterable,
    path: Path,
    text: Optional[Iterable[str]] = None,
    col_name: str = "prob"
) -> None:
    import pandas as pd
    ensure_dir(path.parent)
    data = {"id": list(ids), "label": list(y_true), col_name: list(y_score)}
    if text is not None:
        data["text"] = list(text)
    pd.DataFrame(data).to_csv(path, index=False)

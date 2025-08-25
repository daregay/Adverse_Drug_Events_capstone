# src/plots.py

from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np

def plot_pr_roc(y_true, y_score, title: str, out_png: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
    p, r, _ = precision_recall_curve(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(r, p, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"{title} — PR")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png.with_name(out_png.stem + "_PR.png"), bbox_inches="tight")
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"{title} — ROC")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png.with_name(out_png.stem + "_ROC.png"), bbox_inches="tight")
    plt.show()
    plt.close()

def plot_calibration(y_true, y_prob, out_png: Path, n_bins: int = 10, title: Optional[str] = None) -> None:
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot([0,1],[0,1], linestyle="--", linewidth=1)
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Fraction of positives")
    if title: plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_confusion(y_true, y_pred, out_png: Path, title: str = "Confusion Matrix") -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not ADE", "ADE"])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def plot_threshold_sweep(df_metrics, out_png: Path, title: str = "Threshold Sweep") -> None:
    """
    df_metrics: output of utilities.threshold_sweep (columns: threshold, f1, precision, recall, pr_auc, roc_auc)
    """
    import matplotlib.pyplot as plt
    out_png.parent.mkdir(parents=True, exist_ok=True)

    x = df_metrics["threshold"].values
    plt.figure()
    plt.plot(x, df_metrics["f1"], label="F1")
    plt.plot(x, df_metrics["precision"], label="Precision")
    plt.plot(x, df_metrics["recall"], label="Recall")
    plt.xlabel("Threshold"); plt.ylabel("Score")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.show()
    plt.close()

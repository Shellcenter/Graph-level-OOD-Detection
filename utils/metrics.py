import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def compute_metrics(labels, scores):
    """Compute AUROC and FPR95 for OOD detection."""
    auc = roc_auc_score(labels, scores)

    # FPR95 is the false-positive rate at 95% true-positive rate.
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.where(tpr >= 0.95)[0]
    fpr95 = fpr[idx[0]] if len(idx) > 0 else 1.0

    return auc, fpr95
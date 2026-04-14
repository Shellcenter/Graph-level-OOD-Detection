import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def compute_metrics(labels, scores):
    """计算 OOD 检测任务中的 AUROC 和 FPR95。"""
    auc = roc_auc_score(labels, scores)

    # FPR95 表示在真阳性率达到 95% 时的假阳性率。
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.where(tpr >= 0.95)[0]
    fpr95 = fpr[idx[0]] if len(idx) > 0 else 1.0

    return auc, fpr95
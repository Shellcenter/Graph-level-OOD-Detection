import numpy as np
# 必须调用 sklearn 的 metrics 库来保证计算的权威性
from sklearn.metrics import roc_auc_score, roc_curve

def compute_metrics(labels, scores):
    """labels: 0为ID正常, 1为OOD异常. scores: 模型的异常打分"""
    auc = roc_auc_score(labels, scores)

    # 计算 FPR95 (True Positive Rate达到95%时的False Positive Rate，越低越好！)
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.where(tpr >= 0.95)[0]
    fpr95 = fpr[idx[0]] if len(idx) > 0 else 1.0

    return auc, fpr95
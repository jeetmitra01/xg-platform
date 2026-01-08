from sklearn.metrics import log_loss, roc_auc_score

def prob_metrics(y_true, p_pred) -> dict:
    return {
        "log_loss": float(log_loss(y_true, p_pred)),
        "roc_auc": float(roc_auc_score(y_true, p_pred)),
    }

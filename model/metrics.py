from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import time


def classification_metrics(y_true, y_pred, y_prob=None):

    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1_score"] = f1_score(y_true, y_pred, zero_division=0)

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except:
            metrics["roc_auc"] = None

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    return metrics


def measure_inference_time(model, X):

    start = time.time()
    model.predict(X)
    end = time.time()

    return end - start
"""
Funciones de evaluación para problemas de clasificación desbalanceada.
"""

def calcular_metricas(y_true, y_pred, y_proba=None):
    """Calcula Precision, Recall, F1-Score y opcionalmente ROC-AUC y PR-AUC."""
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score
    )

    resultados = {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }

    if y_proba is not None:
        resultados["roc_auc"] = roc_auc_score(y_true, y_proba)
        resultados["pr_auc"] = average_precision_score(y_true, y_proba)

    return resultados

"""
Compute regression metrics: MSE, RMSE, MAE, R², Pearson and Spearman correlations.
"""
import numpy as np
from scipy.stats import spearmanr

def compute_metrics(predictions: list, targets: list) -> dict:
    """
    Compute regression metrics.

    Args:
        predictions: List of predicted values
        targets: List of ground truth values

    Returns:
        Dictionary of metrics
    """
    predictions = np.array(predictions)
    targets = np.array(targets)

    # MSE and RMSE
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)

    # MAE
    mae = np.mean(np.abs(predictions - targets))

    # R² (coefficient of determination)
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Pearson correlation
    if len(predictions) > 1:
        pearson = np.corrcoef(predictions, targets)[0, 1]
    else:
        pearson = 0.0

    # Spearman correlation
    if len(predictions) > 1:
        spearman, _ = spearmanr(predictions, targets)
    else:
        spearman = 0.0

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "pearson": float(pearson),
        "spearman": float(spearman),
    }

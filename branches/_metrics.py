"""Shared regression metrics, so every branch reports the same simple set.

A single source of truth for the four numbers we actually care about -- RMSE,
MAE, MAPE and directional accuracy -- keeps branch11, branch32 and branch33
reporting identically (and matches pipeline/predict_stock.py). No R2, no
persistence/trailing-mean baselines, no per-stock breakdowns.
"""
import numpy as np


def regression_metrics(actual, predicted, previous):
    """Return {rmse, mae, mape, directional} in the target's own units.

    ``previous`` is the prior day's actual value; it is used only to score
    whether the predicted direction of change matches the realized one.
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    previous = np.asarray(previous, dtype=np.float64)
    return {
        "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "mae": float(np.mean(np.abs(actual - predicted))),
        "mape": float(np.mean(np.abs((actual - predicted) / actual)) * 100),
        "directional": float(
            np.mean(np.sign(actual - previous) == np.sign(predicted - previous)) * 100
        ),
    }


def format_metrics(metrics):
    """Render a metrics dict as the standard four-line block."""
    return (
        f"  RMSE              : {metrics['rmse']:.4f}\n"
        f"  MAE               : {metrics['mae']:.4f}\n"
        f"  MAPE              : {metrics['mape']:.2f}%\n"
        f"  Directional acc.  : {metrics['directional']:.2f}%"
    )

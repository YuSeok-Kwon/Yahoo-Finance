"""
Evaluation Metrics Module
"""

import numpy as np
from scipy.stats import spearmanr
from typing import Dict


def evaluate_predictions(
    actuals: np.ndarray,
    prophet_preds: np.ndarray,
    hybrid_preds: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate predictions using multiple metrics
    
    Parameters:
    -----------
    actuals : np.ndarray
        Actual returns
    prophet_preds : np.ndarray
        Prophet predictions
    hybrid_preds : np.ndarray
        Hybrid predictions
    
    Returns:
    --------
    Dict[str, float] : metrics
        - prophet_mape, hybrid_mape
        - prophet_rmse, hybrid_rmse
        - prophet_spearman, hybrid_spearman
    """
    # MAPE (Mean Absolute Percentage Error)
    prophet_mape = np.mean(np.abs((actuals - prophet_preds) / actuals)) * 100
    hybrid_mape = np.mean(np.abs((actuals - hybrid_preds) / actuals)) * 100
    
    # RMSE (Root Mean Squared Error)
    prophet_rmse = np.sqrt(np.mean((actuals - prophet_preds) ** 2))
    hybrid_rmse = np.sqrt(np.mean((actuals - hybrid_preds) ** 2))
    
    # Spearman Correlation
    prophet_spearman = spearmanr(actuals, prophet_preds)[0]
    hybrid_spearman = spearmanr(actuals, hybrid_preds)[0]
    
    return {
        'prophet_mape': prophet_mape,
        'hybrid_mape': hybrid_mape,
        'prophet_rmse': prophet_rmse,
        'hybrid_rmse': hybrid_rmse,
        'prophet_spearman': prophet_spearman,
        'hybrid_spearman': hybrid_spearman
    }


def print_metrics(metrics: Dict[str, float], year: int):
    """
    Print metrics in formatted table
    
    Parameters:
    -----------
    metrics : Dict[str, float]
        Metrics from evaluate_predictions()
    year : int
        Year of evaluation
    """
    print("\n" + "="*80)
    print(f"{year} Performance")
    print("="*80)
    print(f"{'Metric':<20s} | {'Prophet':>10s} | {'Hybrid':>10s} | {'Improve':>10s}")
    print("-"*80)
    print(f"{'MAPE (%)':<20s} | {metrics['prophet_mape']:>10.2f} | {metrics['hybrid_mape']:>10.2f} | {metrics['prophet_mape']-metrics['hybrid_mape']:>10.2f}")
    print(f"{'RMSE':<20s} | {metrics['prophet_rmse']:>10.4f} | {metrics['hybrid_rmse']:>10.4f} | {metrics['prophet_rmse']-metrics['hybrid_rmse']:>10.4f}")
    print(f"{'Spearman':<20s} | {metrics['prophet_spearman']:>10.4f} | {metrics['hybrid_spearman']:>10.4f} | {metrics['hybrid_spearman']-metrics['prophet_spearman']:>10.4f}")
    print("="*80)

"""
Backtesting Module
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from .hybrid_model import HybridModel


def run_backtest(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_year: int,
    sectors: List[str],
    alpha: float = 0.5
) -> Dict:
    """
    Run backtest for a single year
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data (before test_year)
    test_df : pd.DataFrame
        Test data (test_year only)
    test_year : int
        Test year
    sectors : List[str]
        List of sectors
    alpha : float
        Hybrid weight
    
    Returns:
    --------
    Dict : results with prophet_preds, hybrid_preds, actuals
    """
    results = {
        'year': test_year,
        'prophet_preds': {},
        'hybrid_preds': {},
        'actuals': {}
    }
    
    print(f"\nBacktesting {test_year}...")
    
    for sector in sectors:
        sector_train = train_df[train_df['Sector'] == sector].copy()
        sector_test = test_df[test_df['Sector'] == sector].copy()
        
        if len(sector_train) < 30 or len(sector_test) == 0:
            continue
        
        try:
            # Train hybrid model
            model = HybridModel(alpha=alpha)
            model.train(sector_train)
            
            # Predict
            future = pd.DataFrame({'ds': sector_test['Date']})
            predictions = model.predict(future)
            
            # Calculate returns
            actual_return = (sector_test['Close'].iloc[-1] / sector_test['Close'].iloc[0]) - 1
            prophet_return = np.exp(predictions['yhat'].iloc[-1] - predictions['yhat'].iloc[0]) - 1
            hybrid_return = np.exp(predictions['yhat_hybrid'].iloc[-1] - predictions['yhat_hybrid'].iloc[0]) - 1
            
            results['prophet_preds'][sector] = prophet_return
            results['hybrid_preds'][sector] = hybrid_return
            results['actuals'][sector] = actual_return
            
        except Exception as e:
            continue
    
    return results


def run_multiple_backtests(
    train_df: pd.DataFrame,
    test_years: List[int],
    sectors: List[str],
    alpha: float = 0.5
) -> List[Dict]:
    """
    Run backtests for multiple years
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Full training dataset
    test_years : List[int]
        Years to test
    sectors : List[str]
        List of sectors
    alpha : float
        Hybrid weight
    
    Returns:
    --------
    List[Dict] : list of results for each year
    """
    all_results = []
    
    for year in test_years:
        # Split data: train on years < year, test on year
        year_train = train_df[train_df['year'] < year].copy()
        year_test = train_df[train_df['year'] == year].copy()
        
        # Run backtest
        results = run_backtest(year_train, year_test, year, sectors, alpha)
        all_results.append(results)
    
    return all_results

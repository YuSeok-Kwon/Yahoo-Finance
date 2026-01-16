"""
Data Loading and Preprocessing Module
"""

import pandas as pd
import numpy as np
from typing import Tuple


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load and preprocess sector data
    
    Parameters:
    -----------
    csv_path : str
        Path to stock_features_clean.csv
    
    Returns:
    --------
    pd.DataFrame : Sector-level aggregated data
    """
    print("Loading data...")
    df_raw = pd.read_csv(csv_path, parse_dates=['Date'])
    
    print(f"Raw data loaded: {len(df_raw):,} rows")
    print(f"Date range: {df_raw['Date'].min().date()} to {df_raw['Date'].max().date()}")
    
    # Sector aggregation (daily average)
    sector_df = df_raw.groupby(['Date', 'Sector'], as_index=False).agg({
        'Close': 'mean',
        'Daily_Return_calc': 'mean'
    }).sort_values(by=['Sector', 'Date'])
    
    print(f"Sector aggregation: {len(sector_df):,} rows")
    print(f"Sectors: {sorted(sector_df['Sector'].unique())}")
    
    return sector_df


def split_train_test(
    df: pd.DataFrame,
    train_end_year: int = 2024,
    test_year: int = 2025
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets (no data leakage)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset
    train_end_year : int
        Last year for training (default: 2024)
    test_year : int
        Test year (default: 2025)
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame] : train_df, test_df
    """
    df['year'] = df['Date'].dt.year
    
    # Training: up to train_end_year
    train_df = df[df['year'] <= train_end_year].copy()
    
    # Testing: test_year only
    test_df = df[df['year'] == test_year].copy()
    
    print("\n" + "="*80)
    print("Train/Test Split (No Data Leakage)")
    print("="*80)
    print(f"Train: {train_df['year'].min()}-{train_df['year'].max()} ({len(train_df):,} rows)")
    print(f"Test:  {test_df['year'].unique()[0]} ({len(test_df):,} rows)")
    print("="*80 + "\n")
    
    return train_df, test_df

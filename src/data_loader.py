"""
데이터 로딩 및 전처리 모듈
"""

import pandas as pd
import numpy as np
from typing import Tuple


def load_data(csv_path: str) -> pd.DataFrame:
    """
    섹터 데이터 로드 및 전처리
    
    Parameters:
    -----------
    csv_path : str
        stock_features_clean.csv 파일 경로
    
    Returns:
    --------
    pd.DataFrame : 섹터 단위 집계 데이터
    """
    print("데이터 로딩 중...")
    df_raw = pd.read_csv(csv_path, parse_dates=['Date'])
    
    print(f"원본 데이터 로드 완료: {len(df_raw):,} 행")
    print(f"날짜 범위: {df_raw['Date'].min().date()} ~ {df_raw['Date'].max().date()}")
    
    # 섹터별 집계 (일별 평균)
    sector_df = df_raw.groupby(['Date', 'Sector'], as_index=False).agg({
        'Close': 'mean',
        'Daily_Return_calc': 'mean'
    }).sort_values(by=['Sector', 'Date'])
    
    print(f"섹터 집계 완료: {len(sector_df):,} 행")
    print(f"섹터: {sorted(sector_df['Sector'].unique())}")
    
    return sector_df


def split_train_test(
    df: pd.DataFrame,
    train_end_year: int = 2024,
    test_year: int = 2025
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    학습/테스트 데이터 분할 (데이터 누수 없음)
    
    Parameters:
    -----------
    df : pd.DataFrame
        전체 데이터셋
    train_end_year : int
        학습 데이터 마지막 연도 (기본값: 2024)
    test_year : int
        테스트 연도 (기본값: 2025)
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame] : train_df, test_df
    """
    df['year'] = df['Date'].dt.year
    
    # 학습: train_end_year까지
    train_df = df[df['year'] <= train_end_year].copy()
    
    # 테스트: test_year만
    test_df = df[df['year'] == test_year].copy()
    
    print("\n" + "="*80)
    print("학습/테스트 분할 (데이터 누수 없음)")
    print("="*80)
    print(f"학습: {train_df['year'].min()}-{train_df['year'].max()} ({len(train_df):,} 행)")
    print(f"테스트: {test_df['year'].unique()[0]} ({len(test_df):,} 행)")
    print("="*80 + "\n")
    
    return train_df, test_df

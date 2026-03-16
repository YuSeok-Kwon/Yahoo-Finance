"""
공통 테스트 fixture
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os


@pytest.fixture
def sample_sector_df():
    """100일치 섹터 데이터 생성"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='B')
    sectors = ['Technology', 'Energy', 'Financial Services', 'Healthcare', 'Consumer Cyclical']
    industries = {
        'Technology': ['Software', 'Semiconductors'],
        'Energy': ['Oil & Gas', 'Renewable Energy'],
        'Financial Services': ['Banks', 'Insurance'],
        'Healthcare': ['Biotech', 'Pharma'],
        'Consumer Cyclical': ['Retail', 'Auto'],
    }
    companies = {
        'Software': ['MSFT', 'ORCL'],
        'Semiconductors': ['NVDA', 'AMD'],
        'Oil & Gas': ['XOM', 'CVX'],
        'Renewable Energy': ['ENPH', 'SEDG'],
        'Banks': ['JPM', 'BAC'],
        'Insurance': ['BRK', 'PGR'],
        'Biotech': ['AMGN', 'GILD'],
        'Pharma': ['JNJ', 'PFE'],
        'Retail': ['AMZN', 'WMT'],
        'Auto': ['TSLA', 'F'],
    }

    rows = []
    for sector in sectors:
        base_price = np.random.uniform(50, 200)
        for date in dates:
            close = base_price * (1 + np.random.normal(0.0005, 0.02))
            base_price = close
            daily_return = np.random.normal(0.0005, 0.02)
            for industry in industries[sector]:
                for company in companies[industry]:
                    rows.append({
                        'Date': date,
                        'Sector': sector,
                        'Industry': industry,
                        'Company': company,
                        'Close': close * (1 + np.random.normal(0, 0.01)),
                        'Daily_Return_calc': daily_return + np.random.normal(0, 0.005),
                    })

    df = pd.DataFrame(rows)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


@pytest.fixture
def sample_prophet_output():
    """Prophet 형식 출력 데이터"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=60, freq='B')
    yhat = np.log(100) + np.cumsum(np.random.normal(0.001, 0.01, len(dates)))
    return pd.DataFrame({
        'ds': dates,
        'yhat': yhat,
    })


@pytest.fixture
def trained_xgb_corrector(sample_prophet_output):
    """학습 완료된 XGBoostCorrector 인스턴스"""
    from src.xgboost_model import XGBoostCorrector

    corrector = XGBoostCorrector()
    df = sample_prophet_output.copy()
    df['resid_rank'] = np.random.uniform(0, 1, len(df))
    features = corrector.create_features(df)
    corrector.train(features, target_col='resid_rank')
    return corrector


@pytest.fixture
def tmp_csv(sample_sector_df, tmp_path):
    """임시 CSV 파일 (data_loader 테스트용)"""
    csv_path = tmp_path / "test_data.csv"
    sample_sector_df.to_csv(csv_path, index=False)
    return str(csv_path)

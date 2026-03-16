"""
data_loader 테스트
"""

import pandas as pd
import numpy as np
import pytest
from src.data_loader import load_data, split_train_test


class TestLoadData:
    def test_load_data_columns(self, tmp_csv):
        """반환 컬럼 검증"""
        df = load_data(tmp_csv)
        assert 'Date' in df.columns
        assert 'Sector' in df.columns
        assert 'Close' in df.columns
        assert 'Daily_Return_calc' in df.columns


class TestSplitTrainTest:
    def _make_multi_year_df(self):
        """다중 연도 데이터 생성"""
        np.random.seed(42)
        rows = []
        for year in [2022, 2023, 2024, 2025]:
            dates = pd.date_range(f'{year}-01-01', f'{year}-06-30', freq='B')
            for date in dates:
                for sector in ['Technology', 'Energy']:
                    rows.append({
                        'Date': date,
                        'Sector': sector,
                        'Close': np.random.uniform(100, 200),
                        'Daily_Return_calc': np.random.normal(0, 0.02),
                    })
        return pd.DataFrame(rows)

    def test_split_train_test_no_leak(self):
        """학습/테스트 날짜 겹침 없음 검증"""
        df = self._make_multi_year_df()
        train_df, test_df = split_train_test(df, train_end_year=2024, test_year=2025)

        train_years = set(train_df['Date'].dt.year.unique())
        test_years = set(test_df['Date'].dt.year.unique())

        assert train_years.isdisjoint(test_years) or max(train_years) < min(test_years)
        assert 2025 not in train_years
        assert 2025 in test_years

    def test_split_year_filter(self):
        """연도 필터링 정확성"""
        df = self._make_multi_year_df()
        train_df, test_df = split_train_test(df, train_end_year=2023, test_year=2024)

        assert train_df['Date'].dt.year.max() <= 2023
        assert (test_df['Date'].dt.year == 2024).all()

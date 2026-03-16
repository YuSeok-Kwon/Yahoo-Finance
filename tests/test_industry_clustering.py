"""
IndustryClusterer 테스트
"""

import numpy as np
import pandas as pd
import pytest
from src.industry_clustering import IndustryClusterer


@pytest.fixture
def industry_features_df():
    """클러스터링용 산업 특성 데이터"""
    np.random.seed(42)
    n = 20
    return pd.DataFrame({
        'Industry': [f'Industry_{i}' for i in range(n)],
        'Sector': ['Technology'] * 10 + ['Energy'] * 10,
        'Return_Period': np.random.normal(0.05, 0.1, n),
        'Volatility_20d': np.abs(np.random.normal(0.3, 0.1, n)),
        'MDD': -np.abs(np.random.normal(0.15, 0.05, n)),
        'Sharpe_Ratio': np.random.normal(0.5, 0.5, n),
        'Num_Companies': np.random.randint(2, 20, n),
    })


class TestFitPredict:
    def test_fit_predict_cluster_count(self, industry_features_df):
        """클러스터 수 = n_clusters"""
        clusterer = IndustryClusterer(n_clusters=4)
        result = clusterer.fit_predict(industry_features_df)
        assert result['cluster'].nunique() <= 4

    def test_cluster_assignment_range(self, industry_features_df):
        """클러스터 ID 범위 검증"""
        clusterer = IndustryClusterer(n_clusters=4)
        result = clusterer.fit_predict(industry_features_df)
        assert result['cluster'].min() >= 0
        assert result['cluster'].max() < 4


class TestExtractFeatures:
    def test_extract_features_columns(self, sample_sector_df):
        """추출 컬럼 검증"""
        clusterer = IndustryClusterer(n_clusters=3)
        features = clusterer.extract_industry_features(
            df=sample_sector_df,
            selected_sectors=['Technology', 'Energy'],
            lookback_days=90,
        )
        expected_cols = ['Industry', 'Sector', 'Return_Period', 'Volatility_20d', 'MDD', 'Sharpe_Ratio', 'Num_Companies']
        for col in expected_cols:
            assert col in features.columns, f"Missing column: {col}"


class TestProfileClusters:
    def test_profile_clusters_shape(self, industry_features_df):
        """프로파일 shape 검증"""
        clusterer = IndustryClusterer(n_clusters=4)
        clustered = clusterer.fit_predict(industry_features_df)
        profile = clusterer.profile_clusters(clustered)
        n_actual_clusters = clustered['cluster'].nunique()
        assert len(profile) == n_actual_clusters
        # feature_cols + Sharpe_Ratio
        assert 'Return_Period' in profile.columns
        assert 'Volatility_20d' in profile.columns
        assert 'MDD' in profile.columns
        assert 'Sharpe_Ratio' in profile.columns

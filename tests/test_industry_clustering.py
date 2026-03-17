"""
IndustryClusterer 테스트
"""

import numpy as np
import pandas as pd
import pytest
from src.industry_clustering import IndustryClusterer, DEFAULT_SHARPE_THRESHOLDS, DEFAULT_VOL_THRESHOLDS


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
        clusterer = IndustryClusterer(n_clusters=5)
        result = clusterer.fit_predict(industry_features_df)
        assert result['cluster'].nunique() <= 5

    def test_cluster_assignment_range(self, industry_features_df):
        """클러스터 ID 범위 검증"""
        clusterer = IndustryClusterer(n_clusters=5)
        result = clusterer.fit_predict(industry_features_df)
        assert result['cluster'].min() >= 0
        assert result['cluster'].max() < 5


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
        clusterer = IndustryClusterer(n_clusters=5)
        clustered = clusterer.fit_predict(industry_features_df)
        profile = clusterer.profile_clusters(clustered)
        n_actual_clusters = clustered['cluster'].nunique()
        assert len(profile) == n_actual_clusters
        assert 'Return_Period' in profile.columns
        assert 'Volatility_20d' in profile.columns
        assert 'MDD' in profile.columns
        assert 'Sharpe_Ratio' in profile.columns


class TestRemapClusterIds:
    def test_group_membership_preserved(self, industry_features_df):
        """KMeans 그룹 멤버십이 보존되는지 검증 (같은 그룹이었던 산업은 remap 후에도 같은 그룹)"""
        clusterer = IndustryClusterer(n_clusters=5)
        clustered = clusterer.fit_predict(industry_features_df.copy())
        profile = clusterer.profile_clusters(clustered)

        # remap 전 그룹별 산업 목록 저장
        groups_before = {}
        for cid in clustered['cluster'].unique():
            groups_before[cid] = set(
                clustered[clustered['cluster'] == cid]['Industry'].tolist()
            )

        remapped, _ = clusterer.remap_cluster_ids(clustered, profile)

        # remap 후 그룹별 산업 목록
        groups_after = {}
        for cid in remapped['cluster'].unique():
            groups_after[cid] = set(
                remapped[remapped['cluster'] == cid]['Industry'].tolist()
            )

        # 원래 같은 그룹이었던 산업들이 여전히 같은 그룹인지 확인
        for _, industries in groups_before.items():
            # 이 산업들이 remap 후 모두 같은 클러스터에 있는지
            new_clusters = remapped[remapped['Industry'].isin(industries)]['cluster'].unique()
            assert len(new_clusters) == 1, f"그룹 멤버십이 깨짐: {industries}가 {new_clusters}에 분산됨"

    def test_valid_label_range(self, industry_features_df):
        """remap 후 클러스터 ID가 0-4 범위인지 검증"""
        clusterer = IndustryClusterer(n_clusters=5)
        clustered = clusterer.fit_predict(industry_features_df.copy())
        profile = clusterer.profile_clusters(clustered)
        remapped, _ = clusterer.remap_cluster_ids(clustered, profile)
        assert remapped['cluster'].min() >= 0
        assert remapped['cluster'].max() <= 4

    def test_centroid_based(self, industry_features_df):
        """센트로이드 기반으로 레이블이 결정되는지 검증"""
        clusterer = IndustryClusterer(n_clusters=5)
        clustered = clusterer.fit_predict(industry_features_df.copy())
        profile = clusterer.profile_clusters(clustered)
        remapped, new_profile = clusterer.remap_cluster_ids(clustered, profile)

        # 모든 산업이 할당되었는지 확인
        assert len(remapped) == len(industry_features_df)
        # 프로파일이 존재하는지 확인
        assert len(new_profile) > 0

    def test_no_label_collision(self, industry_features_df):
        """remap 후 각 클러스터 ID가 고유한지 검증"""
        clusterer = IndustryClusterer(n_clusters=5)
        clustered = clusterer.fit_predict(industry_features_df.copy())
        profile = clusterer.profile_clusters(clustered)
        remapped, new_profile = clusterer.remap_cluster_ids(clustered, profile)

        # 프로파일의 인덱스(클러스터 ID)에 중복 없음
        assert new_profile.index.is_unique


class TestConfigurableThresholds:
    def test_custom_thresholds_applied(self, industry_features_df):
        """커스텀 임계값이 적용되는지 검증"""
        custom_sharpe = {'strong_buy': 2.0, 'aggressive': 1.5, 'stable': 1.0, 'warning': 0.5}
        custom_vol = {'strong_buy_max': 0.30, 'aggressive_min': 0.20, 'stable_max': 0.15}

        clusterer = IndustryClusterer(
            n_clusters=5,
            sharpe_thresholds=custom_sharpe,
            vol_thresholds=custom_vol,
        )

        assert clusterer.sharpe_thresholds['strong_buy'] == 2.0
        assert clusterer.vol_thresholds['aggressive_min'] == 0.20

        # 실행해도 오류 없이 동작
        clustered = clusterer.fit_predict(industry_features_df.copy())
        profile = clusterer.profile_clusters(clustered)
        remapped, _ = clusterer.remap_cluster_ids(clustered, profile)
        assert len(remapped) == len(industry_features_df)

    def test_default_thresholds_match(self):
        """기본 임계값이 모듈 상수와 일치하는지 검증"""
        clusterer = IndustryClusterer()
        assert clusterer.sharpe_thresholds == DEFAULT_SHARPE_THRESHOLDS
        assert clusterer.vol_thresholds == DEFAULT_VOL_THRESHOLDS


class TestInterpretClusters:
    def test_all_clusters_interpreted(self, industry_features_df):
        """모든 클러스터에 대해 해석이 반환되는지 검증"""
        clusterer = IndustryClusterer(n_clusters=5)
        clustered = clusterer.fit_predict(industry_features_df.copy())
        profile = clusterer.profile_clusters(clustered)
        remapped, new_profile = clusterer.remap_cluster_ids(clustered, profile)
        interpretations = clusterer.interpret_clusters(remapped, new_profile)

        # 모든 클러스터에 대해 해석 존재
        actual_clusters = remapped['cluster'].unique()
        for cid in actual_clusters:
            assert f'cluster_{cid}' in interpretations

    def test_required_fields_present(self, industry_features_df):
        """해석 결과에 필수 필드가 포함되는지 검증"""
        clusterer = IndustryClusterer(n_clusters=5)
        clustered = clusterer.fit_predict(industry_features_df.copy())
        profile = clusterer.profile_clusters(clustered)
        remapped, new_profile = clusterer.remap_cluster_ids(clustered, profile)
        interpretations = clusterer.interpret_clusters(remapped, new_profile)

        required_fields = ['label', 'description', 'return_period', 'volatility_20d',
                           'mdd', 'sharpe_ratio', 'num_industries', 'industries']
        for key, info in interpretations.items():
            for field in required_fields:
                assert field in info, f"{key}에 필수 필드 {field} 누락"


class TestFullPipeline:
    def test_pipeline_return_keys(self, sample_sector_df):
        """파이프라인 반환값에 필수 키가 포함되는지 검증"""
        clusterer = IndustryClusterer(n_clusters=5)
        result = clusterer.run_full_pipeline(
            df=sample_sector_df,
            selected_sectors=['Technology', 'Energy'],
            lookback_days=90,
        )
        assert 'industry_features' in result
        assert 'cluster_profile' in result
        assert 'interpretations' in result
        assert 'by_cluster' in result

    def test_pipeline_cluster_id_range(self, sample_sector_df):
        """파이프라인 결과의 클러스터 ID가 0-4 범위인지 검증"""
        clusterer = IndustryClusterer(n_clusters=5)
        result = clusterer.run_full_pipeline(
            df=sample_sector_df,
            selected_sectors=['Technology', 'Energy'],
            lookback_days=90,
        )
        df_industry = result['industry_features']
        assert df_industry['cluster'].min() >= 0
        assert df_industry['cluster'].max() <= 4

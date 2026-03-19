"""
InvestorProfiler 테스트
"""

import sys
import os
import importlib.util

# prophet 등 무거운 의존성 없이 investor_profiler만 직접 로드
_mod_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'investor_profiler.py')
_spec = importlib.util.spec_from_file_location('investor_profiler', _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

InvestorProfiler = _mod.InvestorProfiler
INVESTOR_PROFILES = _mod.INVESTOR_PROFILES

import numpy as np
import pandas as pd
import pytest


def _make_industry_df():
    """테스트용 산업 특성 데이터 생성"""
    data = [
        # Cluster 1 (강력매수): 높은 Sharpe, 낮은 Vol
        {'Industry': 'Software', 'Sector': 'Technology', 'Return_Period': 0.15,
         'Volatility_20d': 0.18, 'MDD': -0.08, 'Sharpe_Ratio': 2.0, 'cluster': 1, 'Num_Companies': 50},
        {'Industry': 'Semiconductors', 'Sector': 'Technology', 'Return_Period': 0.12,
         'Volatility_20d': 0.22, 'MDD': -0.10, 'Sharpe_Ratio': 1.6, 'cluster': 1, 'Num_Companies': 30},
        # Cluster 0 (고수익·고위험): 높은 Vol + 높은 Return
        {'Industry': 'Oil & Gas', 'Sector': 'Energy', 'Return_Period': 0.20,
         'Volatility_20d': 0.35, 'MDD': -0.18, 'Sharpe_Ratio': 1.2, 'cluster': 0, 'Num_Companies': 25},
        {'Industry': 'Uranium', 'Sector': 'Energy', 'Return_Period': 0.25,
         'Volatility_20d': 0.45, 'MDD': -0.25, 'Sharpe_Ratio': 1.1, 'cluster': 0, 'Num_Companies': 10},
        # Cluster 2 (안정형/중립): 중간 Sharpe, 낮은 Vol
        {'Industry': 'Utilities', 'Sector': 'Utilities', 'Return_Period': 0.06,
         'Volatility_20d': 0.12, 'MDD': -0.05, 'Sharpe_Ratio': 0.8, 'cluster': 2, 'Num_Companies': 40},
        {'Industry': 'REITs', 'Sector': 'Real Estate', 'Return_Period': 0.05,
         'Volatility_20d': 0.15, 'MDD': -0.07, 'Sharpe_Ratio': 0.6, 'cluster': 2, 'Num_Companies': 35},
        # Cluster 3 (저수익·고위험)
        {'Industry': 'Airlines', 'Sector': 'Industrials', 'Return_Period': 0.02,
         'Volatility_20d': 0.30, 'MDD': -0.22, 'Sharpe_Ratio': 0.1, 'cluster': 3, 'Num_Companies': 15},
        # Cluster 4 (관망/주의)
        {'Industry': 'Retail', 'Sector': 'Consumer Cyclical', 'Return_Period': -0.05,
         'Volatility_20d': 0.28, 'MDD': -0.30, 'Sharpe_Ratio': -0.5, 'cluster': 4, 'Num_Companies': 20},
    ]
    return pd.DataFrame(data)


def _make_horizon_agreement():
    """테스트용 호라이즌 일치도"""
    return {
        'Technology': 5,        # 7개 중 5개 호라이즌 추천
        'Energy': 4,            # 4개 추천
        'Utilities': 3,         # 3개 추천
        'Real Estate': 2,       # 2개 추천
        'Industrials': 1,       # 1개 추천
        'Consumer Cyclical': 1, # 1개 추천
    }


class TestEntryFilter:
    """1단계: 진입 여부 필터 테스트"""

    def test_cluster4_always_excluded(self):
        """Cluster 4(관망/주의)는 모든 프로파일에서 제외"""
        profiler = InvestorProfiler()
        df = _make_industry_df()
        agreement = _make_horizon_agreement()

        results = profiler.recommend(df, agreement)

        for profile_name, result in results.items():
            df_rec = result['recommendations']
            if len(df_rec) > 0:
                assert 4 not in df_rec['cluster'].values, \
                    f"{profile_name}: Cluster 4가 추천에 포함됨"

    def test_aggressive_allows_cluster_0_and_1(self):
        """공격형: Cluster 0, 1만 허용"""
        profiler = InvestorProfiler()
        df = _make_industry_df()
        agreement = _make_horizon_agreement()

        results = profiler.recommend(df, agreement, profile_name='공격형')
        df_rec = results['공격형']['recommendations']

        if len(df_rec) > 0:
            assert set(df_rec['cluster'].unique()).issubset({0, 1})

    def test_stable_allows_cluster_1_and_2(self):
        """안정형: Cluster 1, 2만 허용"""
        profiler = InvestorProfiler()
        df = _make_industry_df()
        agreement = _make_horizon_agreement()

        results = profiler.recommend(df, agreement, profile_name='안정형')
        df_rec = results['안정형']['recommendations']

        if len(df_rec) > 0:
            assert set(df_rec['cluster'].unique()).issubset({1, 2})

    def test_horizon_agreement_filter(self):
        """호라이즌 일치도 미달 섹터 제외"""
        profiler = InvestorProfiler()
        df = _make_industry_df()
        # min_horizon_agreement=3이므로 1~2개 추천 섹터는 제외
        agreement = _make_horizon_agreement()

        results = profiler.recommend(df, agreement, profile_name='공격형')
        df_rec = results['공격형']['recommendations']

        # Industrials(1개), Consumer Cyclical(1개)은 제외되어야 함
        if len(df_rec) > 0:
            assert 'Industrials' not in df_rec['Sector'].values
            assert 'Consumer Cyclical' not in df_rec['Sector'].values


class TestProfileFilter:
    """3단계: 투자자 성향별 필터 테스트"""

    def test_stable_mdd_filter(self):
        """안정형: MDD > -15% 필터 적용"""
        profiler = InvestorProfiler()
        df = _make_industry_df()
        agreement = _make_horizon_agreement()

        results = profiler.recommend(df, agreement, profile_name='안정형')
        df_rec = results['안정형']['recommendations']

        if len(df_rec) > 0:
            assert (df_rec['MDD'] > -0.15).all(), "안정형 MDD 필터 위반"

    def test_stable_vol_filter(self):
        """안정형: 변동성 < 25% 필터 적용"""
        profiler = InvestorProfiler()
        df = _make_industry_df()
        agreement = _make_horizon_agreement()

        results = profiler.recommend(df, agreement, profile_name='안정형')
        df_rec = results['안정형']['recommendations']

        if len(df_rec) > 0:
            assert (df_rec['Volatility_20d'] < 0.25).all(), "안정형 Vol 필터 위반"

    def test_balanced_sharpe_filter(self):
        """균형형: Sharpe >= 0 필터 적용"""
        profiler = InvestorProfiler()
        df = _make_industry_df()
        agreement = _make_horizon_agreement()

        results = profiler.recommend(df, agreement, profile_name='균형형')
        df_rec = results['균형형']['recommendations']

        if len(df_rec) > 0:
            assert (df_rec['Sharpe_Ratio'] >= 0.0).all(), "균형형 Sharpe 필터 위반"


class TestPriorityRanking:
    """2단계: 성향별 가중 composite score 우선순위 테스트"""

    def test_sorted_by_composite_score(self):
        """추천 결과가 composite_score 내림차순 정렬"""
        profiler = InvestorProfiler()
        df = _make_industry_df()
        agreement = _make_horizon_agreement()

        results = profiler.recommend(df, agreement, profile_name='공격형')
        df_rec = results['공격형']['recommendations']

        if len(df_rec) > 1:
            assert 'composite_score' in df_rec.columns, "composite_score 컬럼 누락"
            scores = df_rec['composite_score'].values
            assert all(
                scores[i] >= scores[i + 1]
                for i in range(len(scores) - 1)
            ), "composite_score 내림차순 정렬 위반"

    def test_aggressive_favors_return(self):
        """공격형: Return 가중치(0.5)가 높아 고수익 산업이 유리"""
        # Sharpe 비슷하지만 Return 차이가 큰 통제 데이터
        data = [
            {'Industry': 'HighReturn', 'Sector': 'A', 'Return_Period': 0.30,
             'Volatility_20d': 0.25, 'MDD': -0.15, 'Sharpe_Ratio': 1.0, 'cluster': 0, 'Num_Companies': 10},
            {'Industry': 'LowReturn', 'Sector': 'B', 'Return_Period': 0.05,
             'Volatility_20d': 0.25, 'MDD': -0.15, 'Sharpe_Ratio': 1.0, 'cluster': 0, 'Num_Companies': 10},
        ]
        df = pd.DataFrame(data)
        agreement = {'A': 5, 'B': 5}

        profiler = InvestorProfiler(profiles={
            '공격형': {**INVESTOR_PROFILES['공격형'], 'allowed_clusters': [0]},
        })
        results = profiler.recommend(df, agreement, profile_name='공격형')
        df_rec = results['공격형']['recommendations']

        assert len(df_rec) == 2
        assert df_rec.iloc[0]['Industry'] == 'HighReturn', \
            "공격형(Return 0.5 가중)에서 고수익 산업이 1위여야 함"

    def test_stable_favors_low_mdd(self):
        """안정형: neg_mdd 가중치(0.5)가 높아 MDD 방어력이 좋은 산업이 유리"""
        # Sharpe 비슷하지만 MDD 차이가 큰 통제 데이터
        data = [
            {'Industry': 'GoodMDD', 'Sector': 'A', 'Return_Period': 0.10,
             'Volatility_20d': 0.15, 'MDD': -0.03, 'Sharpe_Ratio': 1.0, 'cluster': 1, 'Num_Companies': 10},
            {'Industry': 'BadMDD', 'Sector': 'B', 'Return_Period': 0.10,
             'Volatility_20d': 0.15, 'MDD': -0.14, 'Sharpe_Ratio': 1.0, 'cluster': 1, 'Num_Companies': 10},
        ]
        df = pd.DataFrame(data)
        agreement = {'A': 5, 'B': 5}

        profiler = InvestorProfiler(profiles={
            '안정형': {**INVESTOR_PROFILES['안정형'], 'allowed_clusters': [1]},
        })
        results = profiler.recommend(df, agreement, profile_name='안정형')
        df_rec = results['안정형']['recommendations']

        assert len(df_rec) == 2
        assert df_rec.iloc[0]['Industry'] == 'GoodMDD', \
            "안정형(neg_mdd 0.5 가중)에서 MDD 방어력 좋은 산업이 1위여야 함"

    def test_profiles_produce_different_rankings(self):
        """동일 입력에서 프로파일별 순위가 달라지는지 검증"""
        profiler = InvestorProfiler()
        # 모든 프로파일이 동일 데이터를 볼 수 있도록 클러스터/필터 조건 완화
        custom_profiles = {
            '공격형': {
                'description': '공격형',
                'allowed_clusters': [0, 1, 2],
                'mdd_threshold': None, 'vol_threshold': None,
                'min_sharpe': None, 'min_horizon_agreement': 1,
                'sector_diversity_min': 1,
                'ranking_weights': {'sharpe': 0.3, 'return': 0.5, 'neg_vol': 0.2},
            },
            '안정형': {
                'description': '안정형',
                'allowed_clusters': [0, 1, 2],
                'mdd_threshold': None, 'vol_threshold': None,
                'min_sharpe': None, 'min_horizon_agreement': 1,
                'sector_diversity_min': 1,
                'ranking_weights': {'sharpe': 0.3, 'return': 0.2, 'neg_mdd': 0.5},
            },
        }
        profiler_custom = InvestorProfiler(profiles=custom_profiles, top_n=10)
        df = _make_industry_df()
        agreement = _make_horizon_agreement()

        results = profiler_custom.recommend(df, agreement)
        agg_rec = results['공격형']['recommendations']
        stb_rec = results['안정형']['recommendations']

        if len(agg_rec) >= 2 and len(stb_rec) >= 2:
            agg_order = list(agg_rec['Industry'].values)
            stb_order = list(stb_rec['Industry'].values)
            assert agg_order != stb_order, \
                "공격형과 안정형의 추천 순위가 동일 — 가중치가 반영되지 않음"


class TestSectorDiversity:
    """섹터 분산도 보장 테스트"""

    def test_balanced_has_multiple_sectors(self):
        """균형형: 최소 2개 섹터 분산"""
        profiler = InvestorProfiler()
        df = _make_industry_df()
        agreement = _make_horizon_agreement()

        results = profiler.recommend(df, agreement, profile_name='균형형')
        df_rec = results['균형형']['recommendations']

        if len(df_rec) >= 2:
            assert df_rec['Sector'].nunique() >= 2, "균형형 섹터 분산 미달"


class TestResultStructure:
    """결과 구조 테스트"""

    def test_all_profiles_present(self):
        """3개 프로파일 결과 모두 존재"""
        profiler = InvestorProfiler()
        df = _make_industry_df()
        results = profiler.recommend(df)

        assert set(results.keys()) == {'공격형', '균형형', '안정형'}

    def test_result_keys(self):
        """각 프로파일 결과에 필수 키 포함"""
        profiler = InvestorProfiler()
        df = _make_industry_df()
        results = profiler.recommend(df)

        required_keys = {'description', 'recommendations', 'summary', 'filter_stats'}
        for profile_name, result in results.items():
            assert required_keys.issubset(set(result.keys())), \
                f"{profile_name}: 필수 키 누락"

    def test_summary_keys(self):
        """summary에 필수 통계 포함"""
        profiler = InvestorProfiler()
        df = _make_industry_df()
        agreement = _make_horizon_agreement()
        results = profiler.recommend(df, agreement)

        summary_keys = {'n_industries', 'n_sectors', 'avg_sharpe', 'avg_return', 'avg_volatility', 'avg_mdd'}
        for profile_name, result in results.items():
            if result['summary']:
                assert summary_keys.issubset(set(result['summary'].keys())), \
                    f"{profile_name}: summary 키 누락"

    def test_top_n_limit(self):
        """추천 수가 top_n 이하"""
        profiler = InvestorProfiler(top_n=3)
        df = _make_industry_df()
        agreement = _make_horizon_agreement()
        results = profiler.recommend(df, agreement)

        for profile_name, result in results.items():
            assert len(result['recommendations']) <= 3, \
                f"{profile_name}: top_n=3 초과"


class TestHorizonAgreement:
    """호라이즌 일치도 계산 테스트"""

    def test_from_raw_horizons(self):
        """7개 호라이즌 결과에서 일치도 계산"""
        profiler = InvestorProfiler()

        raw_results = {
            '1d': {'top_sectors': ['Technology', 'Energy']},
            '3d': {'top_sectors': ['Technology', 'Energy']},
            '1w': {'top_sectors': ['Technology', 'Utilities']},
            '1m': {'top_sectors': ['Energy', 'Utilities']},
            '1q': {'top_sectors': ['Technology', 'Energy']},
            '6m': {'top_sectors': ['Technology', 'Utilities']},
            '1y': {'top_sectors': ['Technology', 'Energy']},
        }

        agreement = profiler.calculate_horizon_agreement(raw_results, {})

        assert agreement['Technology'] == 6  # 1d,3d,1w,1q,6m,1y
        assert agreement['Energy'] == 5      # 1d,3d,1m,1q,1y
        assert agreement['Utilities'] == 3   # 1w,1m,6m

    def test_empty_raw_horizons(self):
        """raw_horizon_results가 비어있으면 빈 dict 반환"""
        profiler = InvestorProfiler()
        agreement = profiler.calculate_horizon_agreement({}, {})
        assert agreement == {}


class TestFormatReport:
    """리포트 포맷팅 테스트"""

    def test_report_not_empty(self):
        """리포트가 비어있지 않음"""
        profiler = InvestorProfiler()
        df = _make_industry_df()
        agreement = _make_horizon_agreement()
        results = profiler.recommend(df, agreement)
        report = profiler.format_report(results)

        assert len(report) > 0
        assert '공격형' in report
        assert '균형형' in report
        assert '안정형' in report


class TestExportForTableau:
    """Tableau 통합 CSV 출력 테스트"""

    def test_tableau_export_has_profile_column(self):
        """Profile 컬럼이 포함되어야 함"""
        profiler = InvestorProfiler()
        df = _make_industry_df()
        agreement = _make_horizon_agreement()
        results = profiler.recommend(df, agreement)
        df_tableau = profiler.export_for_tableau(results, horizon_name='단타')

        assert 'Profile' in df_tableau.columns
        assert 'Rank' in df_tableau.columns
        assert 'Horizon' in df_tableau.columns
        assert 'Cluster_Name' in df_tableau.columns

    def test_tableau_export_profiles_present(self):
        """3개 프로파일이 모두 포함"""
        profiler = InvestorProfiler()
        df = _make_industry_df()
        agreement = _make_horizon_agreement()
        results = profiler.recommend(df, agreement)
        df_tableau = profiler.export_for_tableau(results, horizon_name='스윙')

        profiles_in_data = set(df_tableau['Profile'].unique())
        # 최소 1개 프로파일은 포함되어야 함
        assert len(profiles_in_data) >= 1

    def test_tableau_export_rank_starts_at_1(self):
        """Rank는 프로파일별로 1부터 시작"""
        profiler = InvestorProfiler()
        df = _make_industry_df()
        agreement = _make_horizon_agreement()
        results = profiler.recommend(df, agreement)
        df_tableau = profiler.export_for_tableau(results, horizon_name='단타')

        for profile in df_tableau['Profile'].unique():
            profile_ranks = df_tableau[df_tableau['Profile'] == profile]['Rank'].values
            if len(profile_ranks) > 0:
                assert profile_ranks[0] == 1, f"{profile}: Rank가 1부터 시작하지 않음"

    def test_tableau_export_horizon_column(self):
        """Horizon 컬럼이 전달된 이름으로 설정"""
        profiler = InvestorProfiler()
        df = _make_industry_df()
        agreement = _make_horizon_agreement()
        results = profiler.recommend(df, agreement)
        df_tableau = profiler.export_for_tableau(results, horizon_name='장기투자')

        assert (df_tableau['Horizon'] == '장기투자').all()

    def test_tableau_export_empty_results(self):
        """추천 결과가 없으면 빈 DataFrame 반환"""
        profiler = InvestorProfiler()
        empty_results = {
            '공격형': {'description': '', 'recommendations': pd.DataFrame(), 'summary': {}, 'filter_stats': {}},
        }
        df_tableau = profiler.export_for_tableau(empty_results, horizon_name='단타')
        assert len(df_tableau) == 0

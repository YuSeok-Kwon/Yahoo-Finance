"""
MultiHorizonPredictor 테스트
"""

import numpy as np
import pytest
from src.multi_horizon_predictor import MultiHorizonPredictor


class TestRankingScores:
    def test_ranking_scores_sum_zero(self):
        """z-정규화 랭킹 점수 합 ~ 0 (gamma=0일 때 z_pred의 합은 0)"""
        predictor = MultiHorizonPredictor(alpha=0.6, gamma=0.0, top_k=3)
        predictions = {
            'Technology': 0.05,
            'Energy': 0.03,
            'Financial Services': 0.08,
            'Healthcare': -0.02,
            'Consumer Cyclical': 0.01,
        }
        confidences = {s: 0.5 for s in predictions}

        scores = predictor._calculate_ranking_scores(predictions, confidences)
        total = sum(scores.values())
        assert abs(total) < 1e-6

    def test_gamma_effect(self):
        """gamma=0이면 랭킹 = z(prediction)만"""
        predictor_0 = MultiHorizonPredictor(alpha=0.6, gamma=0.0, top_k=3)
        predictor_1 = MultiHorizonPredictor(alpha=0.6, gamma=1.0, top_k=3)

        predictions = {
            'Technology': 0.05,
            'Energy': 0.03,
            'Healthcare': -0.02,
        }
        # 신뢰도 순서를 예측 순서와 다르게 설정
        confidences = {
            'Technology': 0.3,
            'Energy': 0.9,
            'Healthcare': 0.6,
        }

        scores_0 = predictor_0._calculate_ranking_scores(predictions, confidences)
        scores_1 = predictor_1._calculate_ranking_scores(predictions, confidences)

        # gamma가 다르면 결과가 달라야 함
        assert scores_0 != scores_1

    def test_single_sector_returns_zero(self):
        """섹터 1개일 때 0.0 반환"""
        predictor = MultiHorizonPredictor()
        scores = predictor._calculate_ranking_scores(
            {'Technology': 0.05}, {'Technology': 0.8}
        )
        assert scores['Technology'] == 0.0


class TestAggregation:
    def _make_results(self):
        return {
            '1d': {'top_sectors': ['Technology', 'Energy', 'Healthcare']},
            '3d': {'top_sectors': ['Energy', 'Financial Services', 'Technology']},
            '1w': {'top_sectors': ['Healthcare', 'Energy', 'Consumer Cyclical']},
        }

    def test_aggregate_voting(self):
        """voting 통합: min_votes=2 이상 등장한 섹터만"""
        predictor = MultiHorizonPredictor(top_k=3)
        results = self._make_results()
        aggregated = predictor.aggregate_top_sectors(results, method='voting', min_votes=2)
        # Energy: 3회, Technology: 2회, Healthcare: 2회
        assert 'Energy' in aggregated
        assert 'Technology' in aggregated
        assert 'Healthcare' in aggregated
        assert 'Consumer Cyclical' not in aggregated

    def test_aggregate_union(self):
        """union 통합: 모든 상위 섹터 합집합"""
        predictor = MultiHorizonPredictor(top_k=3)
        results = self._make_results()
        aggregated = predictor.aggregate_top_sectors(results, method='union')
        expected = {'Technology', 'Energy', 'Healthcare', 'Financial Services', 'Consumer Cyclical'}
        assert set(aggregated) == expected


class TestStrategyGroupAggregation:
    """전략그룹 앙상블 (aggregate_by_strategy_groups) 테스트"""

    def _make_full_horizon_results(self):
        """7개 호라이즌 mock 결과 생성"""
        sectors = ['Technology', 'Energy', 'Healthcare', 'Financial Services', 'Consumer Cyclical']
        base = {
            s: 0.01 * (i + 1) for i, s in enumerate(sectors)
        }

        def make_result(top3, horizon_days):
            predictions = {s: base[s] for s in sectors}
            confidences = {s: 0.5 + 0.1 * i for i, s in enumerate(sectors)}
            ranking_scores = {s: float(len(sectors) - i) for i, s in enumerate(sectors)}
            return {
                'top_sectors': top3,
                'predictions': predictions,
                'confidences': confidences,
                'ranking_scores': ranking_scores,
                'horizon_days': horizon_days,
            }

        return {
            '1d': make_result(['Technology', 'Energy', 'Healthcare'], 1),
            '3d': make_result(['Energy', 'Technology', 'Financial Services'], 3),
            '1w': make_result(['Healthcare', 'Energy', 'Consumer Cyclical'], 7),
            '1m': make_result(['Energy', 'Healthcare', 'Technology'], 30),
            '1q': make_result(['Financial Services', 'Technology', 'Energy'], 90),
            '6m': make_result(['Technology', 'Energy', 'Healthcare'], 180),
            '1y': make_result(['Technology', 'Financial Services', 'Healthcare'], 365),
        }

    def test_strategy_group_keys(self):
        """반환값이 정확히 {'단타', '스윙', '장기투자'} 키를 가지는지"""
        predictor = MultiHorizonPredictor(top_k=3)
        results = self._make_full_horizon_results()
        strategy = predictor.aggregate_by_strategy_groups(results)
        assert set(strategy.keys()) == {'단타', '스윙', '장기투자'}

    def test_strategy_group_top_sectors_count(self):
        """각 그룹의 top_sectors가 top_k 이하인지"""
        predictor = MultiHorizonPredictor(top_k=3)
        results = self._make_full_horizon_results()
        strategy = predictor.aggregate_by_strategy_groups(results)
        for group_name, group_result in strategy.items():
            assert len(group_result['top_sectors']) <= 3, f"{group_name}: top_sectors가 3개 초과"

    def test_strategy_group_voting(self):
        """그룹 내 공통 추천 섹터가 우선 선정되는지"""
        predictor = MultiHorizonPredictor(top_k=3)
        results = self._make_full_horizon_results()
        strategy = predictor.aggregate_by_strategy_groups(results)

        # 단타 그룹: 1d=['Technology','Energy','Healthcare'], 3d=['Energy','Technology','Financial Services']
        # 공통: Technology, Energy → 반드시 포함
        danta = strategy['단타']['top_sectors']
        assert 'Technology' in danta
        assert 'Energy' in danta

        # 장기투자 그룹: 1q, 6m, 1y에서 2개 이상 공통인 섹터만
        # Technology: 1q,6m,1y = 3회, Energy: 1q,6m = 2회, Financial Services: 1q,1y = 2회
        # Healthcare: 6m,1y = 2회
        long_term = strategy['장기투자']['top_sectors']
        assert 'Technology' in long_term  # 3회 공통

    def test_strategy_group_result_structure(self):
        """각 그룹 결과에 필수 키가 포함되는지"""
        predictor = MultiHorizonPredictor(top_k=3)
        results = self._make_full_horizon_results()
        strategy = predictor.aggregate_by_strategy_groups(results)

        required_keys = {'top_sectors', 'predictions', 'confidences', 'ranking_scores', 'source_horizons', 'horizon_days'}
        for group_name, group_result in strategy.items():
            assert required_keys.issubset(set(group_result.keys())), \
                f"{group_name}: 필수 키 누락 - {required_keys - set(group_result.keys())}"

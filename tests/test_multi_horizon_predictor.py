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

"""
평가 지표 함수 테스트
"""

import numpy as np
import pytest
from src.evaluation import (
    calculate_top_k_hit_ratio,
    calculate_top_k_average_return,
    calculate_top_k_excess_return,
    ndcg_at_k,
    evaluate_predictions,
)


class TestTopKHitRatio:
    def test_top_k_hit_ratio_perfect(self):
        """완벽 예측 시 hit ratio = 1.0"""
        actuals = np.array([0.01, 0.05, 0.10, 0.03, 0.08])
        predictions = actuals.copy()  # 완벽히 동일한 예측
        assert calculate_top_k_hit_ratio(actuals, predictions, k=3) == 1.0

    def test_top_k_hit_ratio_worst(self):
        """완전 반대 예측 시 hit ratio 검증"""
        actuals = np.array([0.10, 0.08, 0.05, 0.03, 0.01])
        predictions = np.array([0.01, 0.03, 0.05, 0.08, 0.10])  # 역순
        # 실제 top-3: idx 0,1,2 / 예측 top-3: idx 2,3,4 → 교집합 {2} → 1/3
        ratio = calculate_top_k_hit_ratio(actuals, predictions, k=3)
        assert ratio == pytest.approx(1 / 3, abs=1e-6)

    def test_top_k_hit_ratio_insufficient_data(self):
        """데이터 부족 시 0.0 반환"""
        actuals = np.array([0.05, 0.10])
        predictions = np.array([0.10, 0.05])
        assert calculate_top_k_hit_ratio(actuals, predictions, k=5) == 0.0


class TestTopKAverageReturn:
    def test_top_k_average_return(self):
        """선택한 Top-K의 실제 평균 수익률 검증"""
        actuals = np.array([0.01, 0.05, 0.10, 0.03, 0.08])
        predictions = np.array([0.00, 0.04, 0.12, 0.02, 0.09])
        # 예측 top-2: idx 2 (0.12), idx 4 (0.09) → 실제: (0.10+0.08)/2 = 0.09
        result = calculate_top_k_average_return(actuals, predictions, k=2)
        assert result == pytest.approx(0.09, abs=1e-6)


class TestTopKExcessReturn:
    def test_top_k_excess_return(self):
        """초과수익 = Top-K평균 - 전체평균"""
        actuals = np.array([0.01, 0.05, 0.10, 0.03, 0.08])
        predictions = actuals.copy()
        excess = calculate_top_k_excess_return(actuals, predictions, k=2)
        # top-2: idx 2(0.10), idx 4(0.08) → 평균 0.09
        # 전체 평균: 0.054
        expected = 0.09 - actuals.mean()
        assert excess == pytest.approx(expected, abs=1e-6)


class TestNDCG:
    def test_ndcg_perfect_ranking(self):
        """완벽 순위 시 NDCG = 1.0"""
        actuals = np.array([0.01, 0.05, 0.10, 0.03, 0.08])
        predictions = actuals.copy()
        assert ndcg_at_k(actuals, predictions, k=3) == pytest.approx(1.0, abs=1e-6)

    def test_ndcg_random_ranking(self):
        """랜덤 순위 시 NDCG < 1.0 (일반적으로)"""
        np.random.seed(42)
        actuals = np.array([0.01, 0.05, 0.10, 0.03, 0.08])
        predictions = np.random.permutation(actuals)
        ndcg = ndcg_at_k(actuals, predictions, k=3)
        assert 0.0 <= ndcg <= 1.0

    def test_ndcg_insufficient_data(self):
        """데이터 부족 시 0.0"""
        actuals = np.array([0.05])
        predictions = np.array([0.05])
        assert ndcg_at_k(actuals, predictions, k=3) == 0.0


class TestEvaluatePredictions:
    def test_evaluate_predictions_keys(self):
        """반환 딕셔너리 키 검증"""
        actuals = np.array([0.01, 0.05, 0.10, 0.03, 0.08, 0.02])
        prophet_preds = actuals + np.random.normal(0, 0.01, len(actuals))
        hybrid_preds = actuals + np.random.normal(0, 0.005, len(actuals))

        result = evaluate_predictions(actuals, prophet_preds, hybrid_preds)

        expected_keys = {
            'prophet_spearman', 'hybrid_spearman',
            'prophet_top3_hit', 'hybrid_top3_hit',
            'prophet_top5_hit', 'hybrid_top5_hit',
            'prophet_top3_avg_return', 'hybrid_top3_avg_return',
            'prophet_top5_avg_return', 'hybrid_top5_avg_return',
            'prophet_top3_excess', 'hybrid_top3_excess',
            'prophet_top5_excess', 'hybrid_top5_excess',
            'prophet_ndcg3', 'hybrid_ndcg3',
            'prophet_ndcg5', 'hybrid_ndcg5',
        }
        assert set(result.keys()) == expected_keys

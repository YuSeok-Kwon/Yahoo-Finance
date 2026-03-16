"""
HybridModel 테스트
"""

import numpy as np
import pandas as pd
import pytest


class TestConfidence:
    def test_confidence_range(self):
        """confidence 값이 0~1 범위"""
        from src.hybrid_model import HybridModel

        model = HybridModel(alpha=0.6)
        prophet_yhat = np.array([4.5, 4.6, 4.7, 4.8])
        hybrid_yhat = np.array([4.51, 4.59, 4.72, 4.78])
        resid_rank = np.array([0.3, 0.7, 0.1, 0.9])

        conf = model.calculate_confidence(prophet_yhat, hybrid_yhat, resid_rank)
        assert np.all(conf >= 0.0)
        assert np.all(conf <= 1.0)

    def test_alpha_effect(self):
        """alpha=0 vs alpha=1 결과 차이 검증"""
        from src.hybrid_model import HybridModel

        # 최소한의 학습 데이터 생성
        np.random.seed(42)
        dates = pd.date_range('2022-01-01', periods=120, freq='B')
        close = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.01, len(dates))))
        train_data = pd.DataFrame({'Date': dates, 'Close': close})

        model_0 = HybridModel(alpha=0.0)
        model_1 = HybridModel(alpha=1.0)

        model_0.train(train_data)
        model_1.train(train_data)

        future = pd.DataFrame({'ds': pd.date_range('2022-07-01', periods=10, freq='B')})
        pred_0 = model_0.predict(future)
        pred_1 = model_1.predict(future)

        # alpha가 다르면 yhat_hybrid 값이 달라야 함
        assert not np.allclose(
            pred_0['yhat_hybrid'].values,
            pred_1['yhat_hybrid'].values,
            atol=1e-10,
        )


class TestTrainPredictPipeline:
    def test_train_predict_pipeline(self):
        """학습 -> 예측 파이프라인 정상 동작"""
        from src.hybrid_model import HybridModel

        np.random.seed(42)
        dates = pd.date_range('2022-01-01', periods=120, freq='B')
        close = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.01, len(dates))))
        train_data = pd.DataFrame({'Date': dates, 'Close': close})

        model = HybridModel(alpha=0.6)
        model.train(train_data)

        future = pd.DataFrame({'ds': pd.date_range('2022-07-01', periods=5, freq='B')})
        result = model.predict(future)

        assert len(result) == 5
        assert not result['yhat_hybrid'].isna().any()

    def test_predict_output_columns(self):
        """출력 컬럼 검증 (ds, yhat, yhat_hybrid, confidence)"""
        from src.hybrid_model import HybridModel

        np.random.seed(42)
        dates = pd.date_range('2022-01-01', periods=120, freq='B')
        close = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.01, len(dates))))
        train_data = pd.DataFrame({'Date': dates, 'Close': close})

        model = HybridModel(alpha=0.6)
        model.train(train_data)

        future = pd.DataFrame({'ds': pd.date_range('2022-07-01', periods=5, freq='B')})
        result = model.predict(future)

        for col in ['ds', 'yhat', 'yhat_hybrid', 'confidence']:
            assert col in result.columns, f"Missing column: {col}"

"""
XGBoostCorrector 테스트
"""

import numpy as np
import pandas as pd
import pytest
from src.xgboost_model import XGBoostCorrector


class TestCreateFeatures:
    def test_create_features_columns(self, sample_prophet_output):
        """생성되는 특징 컬럼 검증"""
        corrector = XGBoostCorrector()
        feat = corrector.create_features(sample_prophet_output)

        expected_cols = [
            'dayofweek', 'month', 'quarter',
            'volatility_5d', 'volatility_20d', 'volatility_ratio',
            'return_5d', 'return_10d', 'return_20d',
            'yhat_accel', 'trend_strength', 'range_position',
            'yhat_lag_1', 'yhat_diff',
        ]
        for col in expected_cols:
            assert col in feat.columns, f"Missing column: {col}"

    def test_create_features_no_nan(self, sample_prophet_output):
        """NaN 없음 검증 (fillna(0) 적용 후)"""
        corrector = XGBoostCorrector()
        feat = corrector.create_features(sample_prophet_output)
        assert feat.isna().sum().sum() == 0


class TestTrainPredict:
    def test_train_predict_shape(self, sample_prophet_output):
        """학습 후 예측 shape 검증"""
        corrector = XGBoostCorrector()
        df = sample_prophet_output.copy()
        df['resid_rank'] = np.random.uniform(0, 1, len(df))

        features = corrector.create_features(df)
        corrector.train(features, target_col='resid_rank')

        preds = corrector.predict(features)
        assert len(preds) == len(features)

    def test_predict_range(self, trained_xgb_corrector, sample_prophet_output):
        """resid_rank 예측값 범위 검증 (대략 0~1 근처)"""
        features = trained_xgb_corrector.create_features(sample_prophet_output)
        preds = trained_xgb_corrector.predict(features)
        # XGBoost regression은 정확히 [0,1]을 보장하지 않지만 근처에 있어야 함
        assert preds.min() > -1.0
        assert preds.max() < 2.0

    def test_untrained_model_raises(self, sample_prophet_output):
        """미학습 모델에서 predict 호출 시 ValueError"""
        corrector = XGBoostCorrector()
        features = corrector.create_features(sample_prophet_output)
        with pytest.raises(ValueError, match="학습되지 않았습니다"):
            corrector.predict(features)

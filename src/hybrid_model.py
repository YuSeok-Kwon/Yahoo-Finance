"""
하이브리드 모델 모듈 (Prophet + XGBoost)
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from .prophet_model import ProphetModel
from .xgboost_model import XGBoostCorrector


class HybridModel:
    """Prophet과 XGBoost를 결합한 하이브리드 모델"""
    
    def __init__(self, alpha: float = 0.6):
        """
        하이브리드 모델 초기화
        
        Parameters:
        -----------
        alpha : float
            XGBoost 보정 가중치 (0=Prophet만, 1=XGBoost만)
        """
        self.alpha = alpha
        self.prophet_model = ProphetModel()
        self.xgb_corrector = XGBoostCorrector()
        self.resid_mean = None
        self.resid_std = None
    
    def train(self, sector_train: pd.DataFrame) -> 'HybridModel':
        """
        하이브리드 모델 학습
        
        Parameters:
        -----------
        sector_train : pd.DataFrame
            Date와 Close 컬럼이 있는 학습 데이터
        
        Returns:
        --------
        self : 학습된 모델
        """
        # Prophet 학습
        self.prophet_model.train(sector_train)
        
        # 학습 데이터에 대한 Prophet 예측
        train_future = pd.DataFrame({'ds': sector_train['Date']})
        train_prophet_pred = self.prophet_model.predict(train_future)
        
        # 잔차 및 순위 계산
        sector_train_feat = sector_train.merge(
            train_prophet_pred, left_on='Date', right_on='ds', how='left'
        )
        sector_train_feat['y_log'] = np.log(sector_train_feat['Close'])
        sector_train_feat['resid'] = sector_train_feat['y_log'] - sector_train_feat['yhat']
        
        # 잔차의 평균/표준편차 저장 (예측 시 역변환용)
        self.resid_mean = sector_train_feat['resid'].mean()
        self.resid_std = sector_train_feat['resid'].std()
        
        # 잔차를 quantile rank로 변환 (0~1 범위의 상대적 순위)
        # 이렇게 하면 XGBoost가 "이 예측이 얼마나 좋을지"를 0~1 스케일로 학습
        sector_train_feat['resid_rank'] = sector_train_feat['resid'].rank(pct=True)
        
        # 특징 생성 및 XGBoost 학습
        train_features = self.xgb_corrector.create_features(sector_train_feat)
        self.xgb_corrector.train(train_features, target_col='resid_rank')
        
        return self
    
    def calculate_confidence(
        self, 
        prophet_yhat: np.ndarray, 
        hybrid_yhat: np.ndarray,
        resid_rank: np.ndarray
    ) -> np.ndarray:
        """
        Confidence weight calculation
        
        Higher confidence when:
        1. Prophet and Hybrid agree (similar predictions)
        2. XGBoost is confident (resid_rank far from 0.5)
        
        Parameters:
        -----------
        prophet_yhat : np.ndarray
            Prophet predictions
        hybrid_yhat : np.ndarray
            Hybrid predictions
        resid_rank : np.ndarray
            XGBoost predicted ranks (0~1)
        
        Returns:
        --------
        np.ndarray : Confidence weights (0~1)
        """
        agreement = 1.0 - np.abs(prophet_yhat - hybrid_yhat) / (
            np.abs(prophet_yhat) + np.abs(hybrid_yhat) + 1e-8
        )
        agreement = np.clip(agreement, 0, 1)
        
        xgb_confidence = np.abs(resid_rank - 0.5) * 2
        xgb_confidence = np.clip(xgb_confidence, 0, 1)
        
        confidence = np.sqrt(agreement * xgb_confidence)
        
        return confidence
    
    def predict(self, future_dates: pd.DataFrame) -> pd.DataFrame:
        """
        하이브리드 예측 수행
        
        Parameters:
        -----------
        future_dates : pd.DataFrame
            'ds' 컬럼이 있는 DataFrame (예측할 날짜)
        
        Returns:
        --------
        pd.DataFrame : 'ds', 'yhat', 'yhat_hybrid', 'confidence' 컬럼
        """
        prophet_pred = self.prophet_model.predict(future_dates)
        
        feat = self.xgb_corrector.create_features(prophet_pred)
        
        resid_rank_pred = self.xgb_corrector.predict(feat)
        
        resid_from_rank = norm.ppf(np.clip(resid_rank_pred, 0.01, 0.99))
        
        if self.resid_std is not None and self.resid_mean is not None:
            resid_correction = resid_from_rank * self.resid_std + self.resid_mean
        else:
            resid_correction = resid_from_rank * 0.1
        
        prophet_pred['yhat_hybrid'] = (
            (1 - self.alpha) * prophet_pred['yhat'] + 
            self.alpha * (prophet_pred['yhat'] + resid_correction)
        )
        
        confidence = self.calculate_confidence(
            prophet_pred['yhat'].values,
            prophet_pred['yhat_hybrid'].values,
            resid_rank_pred
        )
        prophet_pred['confidence'] = confidence
        
        return prophet_pred

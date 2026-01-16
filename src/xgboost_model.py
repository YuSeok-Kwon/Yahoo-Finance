"""
XGBoost 잔차 보정 모듈
"""

import pandas as pd
import numpy as np
import xgboost as xgb


class XGBoostCorrector:
    """Prophet 잔차 보정을 위한 XGBoost 모델"""
    
    def __init__(
        self,
        n_estimators: int = 75,
        max_depth: int = 3,
        learning_rate: float = 0.07,
        random_state: int = 42
    ):
        """
        XGBoost 보정기 초기화
        
        Parameters:
        -----------
        n_estimators : int
            부스팅 라운드 수
        max_depth : int
            트리 최대 깊이
        learning_rate : float
            학습률
        random_state : int
            랜덤 시드
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.feature_cols = None
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        XGBoost용 특징 생성
        
        Parameters:
        -----------
        df : pd.DataFrame
            'ds'와 'yhat' 컬럼 필수, 'resid' 선택
        
        Returns:
        --------
        pd.DataFrame : 특징이 추가된 DataFrame
        """
        feat = df.copy()
        
        # 시간 특징
        feat['dayofweek'] = pd.to_datetime(feat['ds']).dt.dayofweek
        feat['month'] = pd.to_datetime(feat['ds']).dt.month
        feat['quarter'] = pd.to_datetime(feat['ds']).dt.quarter
        
        # 변동성 특징
        feat['volatility_5d'] = feat['yhat'].rolling(5).std()
        feat['volatility_20d'] = feat['yhat'].rolling(20).std()
        feat['volatility_ratio'] = feat['volatility_5d'] / (feat['volatility_20d'] + 1e-8)
        
        # 모멘텀 특징
        feat['return_5d'] = feat['yhat'].pct_change(5)
        feat['return_10d'] = feat['yhat'].pct_change(10)
        feat['return_20d'] = feat['yhat'].pct_change(20)
        
        # 추세 특징
        feat['yhat_accel'] = feat['yhat'].diff(1).diff(1)
        feat['trend_strength'] = feat['yhat'].rolling(20).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0,
            raw=True
        )
        
        rolling_min = feat['yhat'].rolling(20).min()
        rolling_max = feat['yhat'].rolling(20).max()
        feat['range_position'] = (feat['yhat'] - rolling_min) / (rolling_max - rolling_min + 1e-8)
        
        # Prophet 예측 특징
        feat['yhat_lag_1'] = feat['yhat'].shift(1)
        feat['yhat_diff'] = feat['yhat'].diff(1)
        
        # NaN 채우기
        feat = feat.fillna(0)
        
        return feat
    
    def train(self, train_features: pd.DataFrame, target_col: str = 'resid') -> 'XGBoostCorrector':
        """
        XGBoost 모델 학습
        
        Parameters:
        -----------
        train_features : pd.DataFrame
            학습 특징 (타겟 컬럼 포함)
        target_col : str
            타겟 컬럼 이름
        
        Returns:
        --------
        self : 학습된 모델
        """
        # Top 7 가장 중요한 특징만 사용 (Feature Importance 분석 결과)
        feature_cols = [
            'yhat_lag_1',
            'range_position',
            'return_20d',
            'month',
            'trend_strength',
            'return_5d',
            'return_10d'
        ]
        
        # 사용 가능한 컬럼만 사용
        self.feature_cols = [c for c in feature_cols if c in train_features.columns]
        
        X = train_features[self.feature_cols].values
        y = train_features[target_col].values
        
        # XGBoost 학습
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state
        )
        
        self.model.fit(X, y)
        
        return self
    
    def predict(self, test_features: pd.DataFrame) -> np.ndarray:
        """
        잔차 보정 예측
        
        Parameters:
        -----------
        test_features : pd.DataFrame
            테스트 특징
        
        Returns:
        --------
        np.ndarray : 예측된 보정값
        """
        if self.model is None or self.feature_cols is None:
            raise ValueError("모델이 학습되지 않았습니다. train()을 먼저 호출하세요.")
        
        X = test_features[self.feature_cols].values
        return self.model.predict(X)

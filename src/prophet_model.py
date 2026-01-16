"""
Prophet 모델 모듈
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import warnings


class ProphetModel:
    """시계열 예측을 위한 Prophet 베이스라인 모델"""
    
    def __init__(
        self,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0
    ):
        """
        Prophet 모델 초기화
        
        Parameters:
        -----------
        changepoint_prior_scale : float
            추세 변화의 유연성
        seasonality_prior_scale : float
            계절성의 유연성
        """
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.model = None
    
    def train(self, sector_data: pd.DataFrame) -> 'ProphetModel':
        """
        섹터 데이터로 Prophet 모델 학습
        
        Parameters:
        -----------
        sector_data : pd.DataFrame
            'Date'와 'Close' 컬럼 필수
        
        Returns:
        --------
        self : 학습된 모델
        """
        # Prophet용 데이터 준비
        prophet_df = sector_data[['Date', 'Close']].copy()
        prophet_df.columns = ['ds', 'y']
        
        # 로그 변환
        prophet_df['y'] = np.log(prophet_df['y'])
        
        # 모델 생성
        self.model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        
        # 학습
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(prophet_df)
        
        return self
    
    def predict(self, future_dates: pd.DataFrame) -> pd.DataFrame:
        """
        예측 수행
        
        Parameters:
        -----------
        future_dates : pd.DataFrame
            'ds' 컬럼이 있는 DataFrame (예측할 날짜)
        
        Returns:
        --------
        pd.DataFrame : 'ds'와 'yhat' 컬럼이 있는 예측 결과
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다. train()을 먼저 호출하세요.")
        
        forecast = self.model.predict(future_dates)
        return forecast[['ds', 'yhat']].copy()

"""
Hybrid Model Module (Prophet + XGBoost)
"""

import pandas as pd
import numpy as np
from .prophet_model import ProphetModel
from .xgboost_model import XGBoostCorrector


class HybridModel:
    """Hybrid model combining Prophet and XGBoost"""
    
    def __init__(self, alpha: float = 0.5):
        """
        Initialize hybrid model
        
        Parameters:
        -----------
        alpha : float
            Weight for XGBoost correction (0=Prophet only, 1=XGBoost only)
        """
        self.alpha = alpha
        self.prophet_model = ProphetModel()
        self.xgb_corrector = XGBoostCorrector()
    
    def train(self, sector_train: pd.DataFrame) -> 'HybridModel':
        """
        Train hybrid model
        
        Parameters:
        -----------
        sector_train : pd.DataFrame
            Training data with Date and Close columns
        
        Returns:
        --------
        self : trained model
        """
        # Train Prophet
        self.prophet_model.train(sector_train)
        
        # Get Prophet predictions on training data
        train_future = pd.DataFrame({'ds': sector_train['Date']})
        train_prophet_pred = self.prophet_model.predict(train_future)
        
        # Calculate residuals
        sector_train_feat = sector_train.merge(
            train_prophet_pred, left_on='Date', right_on='ds', how='left'
        )
        sector_train_feat['y_log'] = np.log(sector_train_feat['Close'])
        sector_train_feat['resid'] = sector_train_feat['y_log'] - sector_train_feat['yhat']
        
        # Create features and train XGBoost
        train_features = self.xgb_corrector.create_features(sector_train_feat)
        self.xgb_corrector.train(train_features, target_col='resid')
        
        return self
    
    def predict(self, future_dates: pd.DataFrame) -> pd.DataFrame:
        """
        Make hybrid predictions
        
        Parameters:
        -----------
        future_dates : pd.DataFrame
            DataFrame with 'ds' column (dates to predict)
        
        Returns:
        --------
        pd.DataFrame : predictions with 'ds', 'yhat', 'yhat_hybrid' columns
        """
        # Prophet prediction
        prophet_pred = self.prophet_model.predict(future_dates)
        
        # Create features
        feat = self.xgb_corrector.create_features(prophet_pred)
        
        # XGBoost correction
        xgb_correction = self.xgb_corrector.predict(feat)
        
        # Hybrid prediction
        prophet_pred['yhat_hybrid'] = (
            (1 - self.alpha) * prophet_pred['yhat'] + 
            self.alpha * (prophet_pred['yhat'] + xgb_correction)
        )
        
        return prophet_pred

"""
XGBoost Residual Correction Module
"""

import pandas as pd
import numpy as np
import xgboost as xgb


class XGBoostCorrector:
    """XGBoost model for correcting Prophet residuals"""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize XGBoost corrector
        
        Parameters:
        -----------
        n_estimators : int
            Number of boosting rounds
        max_depth : int
            Maximum tree depth
        learning_rate : float
            Learning rate
        random_state : int
            Random seed
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.feature_cols = None
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for XGBoost
        
        Parameters:
        -----------
        df : pd.DataFrame
            Must have 'ds' and 'yhat' columns, optionally 'resid'
        
        Returns:
        --------
        pd.DataFrame : DataFrame with features
        """
        feat = df.copy()
        
        # Time features
        feat['dayofweek'] = pd.to_datetime(feat['ds']).dt.dayofweek
        feat['month'] = pd.to_datetime(feat['ds']).dt.month
        feat['quarter'] = pd.to_datetime(feat['ds']).dt.quarter
        
        # Residual-based features
        if 'resid' in feat.columns:
            feat['resid_lag_1'] = feat['resid'].shift(1)
            feat['resid_lag_5'] = feat['resid'].shift(5)
            feat['resid_roll_mean'] = feat['resid'].rolling(5).mean()
            feat['resid_roll_std'] = feat['resid'].rolling(5).std()
        
        # Prophet prediction features
        feat['yhat_lag_1'] = feat['yhat'].shift(1)
        feat['yhat_diff'] = feat['yhat'].diff(1)
        
        # Fill NaN
        feat = feat.fillna(0)
        
        return feat
    
    def train(self, train_features: pd.DataFrame, target_col: str = 'resid') -> 'XGBoostCorrector':
        """
        Train XGBoost model
        
        Parameters:
        -----------
        train_features : pd.DataFrame
            Training features (with target column)
        target_col : str
            Target column name
        
        Returns:
        --------
        self : trained model
        """
        feature_cols = [
            'dayofweek', 'month', 'quarter',
            'resid_lag_1', 'resid_lag_5', 'resid_roll_mean', 'resid_roll_std',
            'yhat_lag_1', 'yhat_diff'
        ]
        
        # Use only available columns
        self.feature_cols = [c for c in feature_cols if c in train_features.columns]
        
        X = train_features[self.feature_cols].values
        y = train_features[target_col].values
        
        # Train XGBoost
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
        Predict residual correction
        
        Parameters:
        -----------
        test_features : pd.DataFrame
            Test features
        
        Returns:
        --------
        np.ndarray : predicted corrections
        """
        if self.model is None or self.feature_cols is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X = test_features[self.feature_cols].values
        return self.model.predict(X)

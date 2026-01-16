"""
Prophet Model Module
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import warnings


class ProphetModel:
    """Prophet baseline model for time series forecasting"""
    
    def __init__(
        self,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0
    ):
        """
        Initialize Prophet model
        
        Parameters:
        -----------
        changepoint_prior_scale : float
            Flexibility of trend changes
        seasonality_prior_scale : float
            Flexibility of seasonality
        """
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.model = None
    
    def train(self, sector_data: pd.DataFrame) -> 'ProphetModel':
        """
        Train Prophet model on sector data
        
        Parameters:
        -----------
        sector_data : pd.DataFrame
            Must have 'Date' and 'Close' columns
        
        Returns:
        --------
        self : trained model
        """
        # Prepare data for Prophet
        prophet_df = sector_data[['Date', 'Close']].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Log transformation
        prophet_df['y'] = np.log(prophet_df['y'])
        
        # Create model
        self.model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        
        # Train
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(prophet_df)
        
        return self
    
    def predict(self, future_dates: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions
        
        Parameters:
        -----------
        future_dates : pd.DataFrame
            DataFrame with 'ds' column (dates to predict)
        
        Returns:
        --------
        pd.DataFrame : predictions with 'ds' and 'yhat' columns
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        forecast = self.model.predict(future_dates)
        return forecast[['ds', 'yhat']].copy()

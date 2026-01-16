"""
Multi-Horizon Prediction Wrapper
Prophet + XGBoost 기반 멀티 호라이즌 예측

- 예외 처리 추가 (부분 실패 허용)
- 수익률 계산 로직 수정

Horizons:
- 3d: 3 trading days
- 1w: 5 trading days
- 2w: 10 trading days
- 1m: 21 trading days  
- 1q: 63 trading days
- 1y: 252 trading days
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta


class MultiHorizonPredictor:
    """멀티 호라이즌 예측 wrapper (FIXED 버전)"""
    
    HORIZONS = {
        '3d': 3,
        '1w': 5,
        '2w': 10,
        '1m': 21,
        '1q': 63,
        '1y': 252
    }
    
    def __init__(
        self,
        predict_func,
        sector_final: pd.DataFrame,
        test_years: List[int],
        sectors: List[str],
        holidays_df: pd.DataFrame,
        alpha: float = 0.5
    ):
        self.predict_func = predict_func
        self.sector_final = sector_final
        self.test_years = test_years
        self.sectors = sectors
        self.holidays_df = holidays_df
        self.alpha = alpha
        
        self.results = {}
        self.failures = []
    
    def predict_all_horizons(self) -> Dict[str, pd.DataFrame]:
        """
        모든 horizon에 대해 예측 수행
        
        개선사항:
        - 섹터별 데이터 캐싱
        - 예외 처리 추가
        """
        print("=" * 80)
        print("Multi-Horizon Prediction (FIXED)")
        print("=" * 80)
        print(f"Horizons: {list(self.HORIZONS.keys())}")
        print(f"Test years: {self.test_years}")
        print(f"Sectors: {len(self.sectors)}")
        print("")
        
        sector_data_cache = {}
        for sector in self.sectors:
            sector_data_cache[sector] = self.sector_final[
                self.sector_final['Sector'] == sector
            ].copy()
        print(f"✓ 섹터 데이터 캐싱 완료 ({len(sector_data_cache)} sectors)")
        
        self.failures = []
        
        for horizon_name, horizon_days in self.HORIZONS.items():
            print(f"\n[{horizon_name}] horizon={horizon_days} days")
            print("-" * 80)
            
            horizon_predictions = []
            
            for year in self.test_years:
                for sector in self.sectors:
                    try:
                        sector_data = sector_data_cache[sector]
                        
                        pred = self.predict_func(
                            sector_data=sector_data,
                            year=year,
                            holidays_df=self.holidays_df,
                            alpha=self.alpha,
                            horizon=horizon_days
                        )
                        
                        if pred is None or len(pred) == 0:
                            continue
                        
                        pred = pred.copy()
                        pred['Sector'] = sector
                        pred['test_year'] = year
                        pred['horizon'] = horizon_name
                        pred['horizon_days'] = horizon_days
                        
                        horizon_predictions.append(pred)
                    
                    except Exception as e:
                        error_msg = f"{horizon_name}|{sector}|{year}: {str(e)}"
                        self.failures.append(error_msg)
                        if len(self.failures) <= 3:
                            print(f"  ⚠ {error_msg}")
                        continue
            
            if horizon_predictions:
                df_horizon = pd.concat(horizon_predictions, ignore_index=True)
                self.results[horizon_name] = df_horizon
                print(f"  ✓ {len(df_horizon)} predictions")
            else:
                print(f"  ✗ No predictions")
        
        print("\n" + "=" * 80)
        print(f"✓ Multi-horizon prediction complete")
        print(f"  Total horizons: {len(self.results)}")
        if self.failures:
            print(f"  ⚠ Failures: {len(self.failures)}")
            if len(self.failures) > 3:
                print(f"    (showing first 3)")
                for f in self.failures[:3]:
                    print(f"    - {f}")
        print("=" * 80)
        
        return self.results
    
    def build_integrated_table(self, strategy='first_point') -> pd.DataFrame:
        """
        모든 horizon 결과를 통합
        
        Args:
            strategy: 
                'first_point': 연초 t0 기준 (t0 → t0+h) 수익률 (권장)
                'full_period': 연도 전체 (연초 → 연말) 수익률 (기존 방식)
        """
        if not self.results:
            raise ValueError("Run predict_all_horizons() first")
        
        print(f"\nBuilding integrated table (strategy={strategy})...")
        
        all_returns = []
        
        for year in self.test_years:
            for sector in self.sectors:
                row = {
                    'test_year': year,
                    'Sector': sector
                }
                
                for horizon_name in self.HORIZONS.keys():
                    if horizon_name not in self.results:
                        continue
                    
                    df_h = self.results[horizon_name]
                    df_sector_year = df_h[
                        (df_h['test_year'] == year) & 
                        (df_h['Sector'] == sector)
                    ]
                    
                    if len(df_sector_year) == 0:
                        continue
                    
                    pred_col = self._find_prediction_column(df_sector_year, sector, year, horizon_name)
                    if pred_col is None:
                        continue
                    
                    if strategy == 'first_point':
                        actual_log_t0 = df_sector_year['y_actual'].iloc[0]
                        pred_log_t0 = df_sector_year[pred_col].iloc[0]
                        
                        actual_return = np.exp(actual_log_t0) - 1 if not np.isnan(actual_log_t0) else np.nan
                        pred_return = np.exp(pred_log_t0) - 1 if not np.isnan(pred_log_t0) else np.nan
                    
                    elif strategy == 'full_period':
                        actual_log_start = df_sector_year['y_actual'].iloc[0]
                        actual_log_end = df_sector_year['y_actual'].iloc[-1]
                        pred_log_start = df_sector_year[pred_col].iloc[0]
                        pred_log_end = df_sector_year[pred_col].iloc[-1]
                        
                        actual_return = np.exp(actual_log_end - actual_log_start) - 1
                        pred_return = np.exp(pred_log_end - pred_log_start) - 1
                    
                    else:
                        raise ValueError(f"Unknown strategy: {strategy}")
                    
                    row[f'actual_return_{horizon_name}'] = actual_return
                    row[f'pred_return_{horizon_name}'] = pred_return
                
                if len(row) > 2:
                    all_returns.append(row)
        
        df_integrated = pd.DataFrame(all_returns)
        
        print(f"  ✓ Integrated table shape: {df_integrated.shape}")
        
        return df_integrated
    
    def _find_prediction_column(self, df, sector, year, horizon_name):
        """예측 컬럼 자동 탐지"""
        candidates = ['yhat_final', 'yhat_hybrid_h', 'yhat_hybrid', 'yhat_prophet', 'yhat']
        
        for col in candidates:
            if col in df.columns:
                return col
        
        print(f"  Warning: No prediction column for {sector} {year} {horizon_name}")
        return None
    
    def build_curve_features(self, df_integrated: pd.DataFrame) -> pd.DataFrame:
        """커브 형태 파생 변수 생성"""
        df_features = df_integrated.copy()
        
        print("\nBuilding curve features...")
        
        if 'pred_return_3d' in df_features.columns and 'pred_return_1y' in df_features.columns:
            df_features['term_slope'] = (
                df_features['pred_return_1y'] - df_features['pred_return_3d']
            )
            print("  ✓ term_slope (장단기 스프레드)")
        
        if 'pred_return_1m' in df_features.columns and 'pred_return_1q' in df_features.columns:
            df_features['curve_shape'] = (
                df_features['pred_return_1q'] - df_features['pred_return_1m']
            )
            print("  ✓ curve_shape (중기 기울기)")
        
        if 'pred_return_3d' in df_features.columns and 'pred_return_1w' in df_features.columns:
            df_features['short_momentum'] = (
                df_features['pred_return_1w'] - df_features['pred_return_3d']
            )
            print("  ✓ short_momentum (초단기 모멘텀)")
        
        if 'pred_return_1q' in df_features.columns and 'pred_return_1y' in df_features.columns:
            df_features['long_momentum'] = (
                df_features['pred_return_1y'] - df_features['pred_return_1q']
            )
            print("  ✓ long_momentum (장기 모멘텀)")
        
        pred_cols = [c for c in df_features.columns if c.startswith('pred_return_')]
        if len(pred_cols) >= 2:
            df_features['return_volatility'] = df_features[pred_cols].std(axis=1)
            print("  ✓ return_volatility (예측 변동성)")
        
        print(f"\n  Total features: {len(df_features.columns)}")
        
        return df_features


if __name__ == '__main__':
    print(__doc__)

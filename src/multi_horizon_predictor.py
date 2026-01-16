"""
멀티 호라이즌 예측 모듈
여러 시간대(1d, 3d, 1w, 1m, 1q, 6m, 1y)에 걸쳐 상위 섹터를 예측합니다
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import timedelta
from .hybrid_model import HybridModel


class MultiHorizonPredictor:
    """
    Prophet + XGBoost 하이브리드 모델을 사용한 멀티 호라이즌 섹터 예측
    """
    
    HORIZONS = {
        '1d': 1,
        '3d': 3,
        '1w': 7,
        '1m': 30,
        '1q': 90,
        '6m': 180,
        '1y': 365
    }
    
    def __init__(
        self, 
        alpha: float = 0.6,
        gamma: float = 0.5,
        top_k: int = 3
    ):
        """
        멀티 호라이즌 예측기 초기화
        
        Parameters:
        -----------
        alpha : float
            하이브리드 모델 가중치 (0=Prophet만, 1=XGBoost만)
        gamma : float
            랭킹에서 신뢰도 가중치 (z_hybrid + gamma * z_confidence)
        top_k : int
            호라이즌별 선택할 상위 섹터 수
        """
        self.alpha = alpha
        self.gamma = gamma
        self.top_k = top_k
        self.models = {}
    
    def predict_all_horizons(
        self,
        df: pd.DataFrame,
        prediction_date: pd.Timestamp,
        sectors: List[str],
        train_years: int = 4
    ) -> Dict:
        """
        모든 호라이즌에 대해 상위 섹터 예측
        
        Parameters:
        -----------
        df : pd.DataFrame
            Date, Sector, Close 컬럼이 있는 전체 데이터셋
        prediction_date : pd.Timestamp
            예측 기준일
        sectors : List[str]
            예측할 섹터 리스트
        train_years : int
            학습에 사용할 연도 수
        
        Returns:
        --------
        Dict : 모든 호라이즌의 결과
            {
                '1d': {
                    'top_sectors': ['Energy', 'Financial Services', 'Technology'],
                    'predictions': {'Energy': 0.05, ...},
                    'confidences': {'Energy': 0.85, ...},
                    'ranking_scores': {'Energy': 2.3, ...}
                },
                ...
            }
        """
        results = {}
        
        train_start = prediction_date - pd.Timedelta(days=train_years * 365)
        train_df = df[(df['Date'] >= train_start) & (df['Date'] < prediction_date)].copy()
        
        print(f"\n{prediction_date.date()} 기준 멀티 호라이즌 예측")
        print(f"학습 데이터: {train_df['Date'].min().date()} ~ {train_df['Date'].max().date()}")
        print(f"호라이즌: {list(self.HORIZONS.keys())}")
        print("="*80)
        
        for horizon_name, horizon_days in self.HORIZONS.items():
            print(f"\n[{horizon_name}] {horizon_days}일 호라이즌 예측 중...")
            
            horizon_result = self._predict_single_horizon(
                train_df=train_df,
                prediction_date=prediction_date,
                horizon_days=horizon_days,
                sectors=sectors
            )
            
            results[horizon_name] = horizon_result
            
            print(f"  상위 {self.top_k}개 섹터: {horizon_result['top_sectors']}")
        
        print("\n" + "="*80)
        print("멀티 호라이즌 예측 완료")
        
        return results
    
    def _predict_single_horizon(
        self,
        train_df: pd.DataFrame,
        prediction_date: pd.Timestamp,
        horizon_days: int,
        sectors: List[str]
    ) -> Dict:
        """
        단일 호라이즌에 대한 예측
        
        Parameters:
        -----------
        train_df : pd.DataFrame
            학습 데이터
        prediction_date : pd.Timestamp
            예측 시작일
        horizon_days : int
            예측할 일수
        sectors : List[str]
            섹터 리스트
        
        Returns:
        --------
        Dict : 단일 호라이즌 결과
        """
        predictions = {}
        confidences = {}
        
        target_date = prediction_date + pd.Timedelta(days=horizon_days)
        
        for sector in sectors:
            sector_train = train_df[train_df['Sector'] == sector].copy()
            
            if len(sector_train) < 30:
                continue
            
            try:
                model = HybridModel(alpha=self.alpha)
                model.train(sector_train)
                
                future_dates = pd.date_range(
                    start=prediction_date,
                    end=target_date,
                    freq='D'
                )
                future = pd.DataFrame({'ds': future_dates})
                
                pred = model.predict(future)
                
                predicted_return = np.exp(pred['yhat_hybrid'].iloc[-1] - pred['yhat_hybrid'].iloc[0]) - 1
                avg_confidence = pred['confidence'].mean() if 'confidence' in pred.columns else 0.5
                
                predictions[sector] = predicted_return
                confidences[sector] = avg_confidence
                
            except Exception as e:
                print(f"    {sector}: 실패 - {type(e).__name__}")
                continue
        
        ranking_scores = self._calculate_ranking_scores(predictions, confidences)
        
        sorted_sectors = sorted(
            ranking_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_sectors = [s[0] for s in sorted_sectors[:self.top_k]]
        
        return {
            'top_sectors': top_sectors,
            'predictions': predictions,
            'confidences': confidences,
            'ranking_scores': ranking_scores,
            'horizon_days': horizon_days
        }
    
    def _calculate_ranking_scores(
        self,
        predictions: Dict[str, float],
        confidences: Dict[str, float]
    ) -> Dict[str, float]:
        """
        랭킹 점수 계산: z(prediction) + gamma * z(confidence)
        
        Parameters:
        -----------
        predictions : Dict[str, float]
            섹터별 예측값
        confidences : Dict[str, float]
            섹터별 신뢰도
        
        Returns:
        --------
        Dict[str, float] : 랭킹 점수
        """
        if len(predictions) < 2:
            return {s: 0.0 for s in predictions.keys()}
        
        pred_arr = np.array(list(predictions.values()))
        conf_arr = np.array(list(confidences.values()))
        
        z_pred = (pred_arr - pred_arr.mean()) / (pred_arr.std() + 1e-8)
        z_conf = (conf_arr - conf_arr.mean()) / (conf_arr.std() + 1e-8)
        
        ranking = z_pred + self.gamma * z_conf
        
        ranking_scores = {
            sector: ranking[i]
            for i, sector in enumerate(predictions.keys())
        }
        
        return ranking_scores
    
    def aggregate_top_sectors(
        self,
        multi_horizon_results: Dict,
        method: str = 'voting',
        min_votes: int = 2
    ) -> List[str]:
        """
        호라이즌 전체에 걸쳐 상위 섹터 통합
        
        Parameters:
        -----------
        multi_horizon_results : Dict
            predict_all_horizons의 결과
        method : str
            'union': 모든 호라이즌의 상위 섹터 합집합
            'voting': 최소 min_votes 이상의 호라이즌에 등장한 섹터
            'weighted': 호라이즌 중요도 가중치 적용
        min_votes : int
            'voting' 방식의 최소 투표 수
        
        Returns:
        --------
        List[str] : 통합된 상위 섹터
        """
        if method == 'union':
            all_sectors = set()
            for result in multi_horizon_results.values():
                all_sectors.update(result['top_sectors'])
            return sorted(list(all_sectors))
        
        elif method == 'voting':
            from collections import Counter
            sector_votes = Counter()
            for result in multi_horizon_results.values():
                sector_votes.update(result['top_sectors'])
            
            return sorted([
                sector for sector, votes in sector_votes.items()
                if votes >= min_votes
            ])
        
        elif method == 'weighted':
            weights = {
                '1d': 0.30,
                '3d': 0.25,
                '1w': 0.20,
                '1m': 0.15,
                '1q': 0.05,
                '6m': 0.03,
                '1y': 0.02
            }
            
            sector_weighted_score = {}
            for horizon_name, result in multi_horizon_results.items():
                weight = weights.get(horizon_name, 0.1)
                for i, sector in enumerate(result['top_sectors']):
                    score = (self.top_k - i) * weight
                    sector_weighted_score[sector] = sector_weighted_score.get(sector, 0) + score
            
            sorted_sectors = sorted(
                sector_weighted_score.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            return [s[0] for s in sorted_sectors]
        
        else:
            raise ValueError(f"알 수 없는 방법: {method}")

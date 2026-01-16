"""
백테스팅 모듈
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from .hybrid_model import HybridModel


def run_backtest(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_year: int,
    sectors: List[str],
    alpha: float = 0.6,
    gamma: float = 0.5,
    adaptive_gamma: bool = False
) -> Dict:
    """
    단일 연도 백테스트 실행
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        학습 데이터 (test_year 이전)
    test_df : pd.DataFrame
        테스트 데이터 (test_year만)
    test_year : int
        테스트 연도
    sectors : List[str]
        섹터 리스트
    alpha : float
        하이브리드 가중치
    
    Returns:
    --------
    Dict : prophet_preds, hybrid_preds, actuals가 포함된 결과
    """
    results = {
        'year': test_year,
        'prophet_preds': {},
        'hybrid_preds': {},
        'hybrid_ranking_preds': {},
        'confidences': {},
        'actuals': {}
    }
    
    print(f"\n{test_year} 백테스팅 중...")
    
    success_count = 0
    fail_count = 0
    
    for sector in sectors:
        sector_train = train_df[train_df['Sector'] == sector].copy()
        sector_test = test_df[test_df['Sector'] == sector].copy()
        
        if len(sector_train) < 30:
            print(f"  {sector}: 건너뜀 - 학습 데이터 부족 ({len(sector_train)} 행)")
            fail_count += 1
            continue
        
        if len(sector_test) == 0:
            print(f"  {sector}: 건너뜀 - 테스트 데이터 없음")
            fail_count += 1
            continue
        
        try:
            # 하이브리드 모델 학습
            model = HybridModel(alpha=alpha)
            model.train(sector_train)
            
            future = pd.DataFrame({'ds': sector_test['Date']})
            predictions = model.predict(future)
            
            actual_return = (sector_test['Close'].iloc[-1] / sector_test['Close'].iloc[0]) - 1
            prophet_return = np.exp(predictions['yhat'].iloc[-1] - predictions['yhat'].iloc[0]) - 1
            hybrid_return = np.exp(predictions['yhat_hybrid'].iloc[-1] - predictions['yhat_hybrid'].iloc[0]) - 1
            
            avg_confidence = predictions['confidence'].mean() if 'confidence' in predictions.columns else 0.5
            
            results['prophet_preds'][sector] = prophet_return
            results['hybrid_preds'][sector] = hybrid_return
            results['hybrid_ranking_preds'][sector] = hybrid_return
            results['confidences'][sector] = avg_confidence
            results['actuals'][sector] = actual_return
            
            success_count += 1
            print(f"  {sector}: 성공 (실제={actual_return:.2%}, prophet={prophet_return:.2%}, hybrid={hybrid_return:.2%})")
            
        except Exception as e:
            fail_count += 1
            print(f"  {sector}: 오류 - {type(e).__name__}: {str(e)}")
            import traceback
            print(f"    추적: {traceback.format_exc()[:200]}")
            continue
    
    print(f"\n요약: {success_count} 성공, {fail_count} 실패")
    
    if len(results['hybrid_preds']) > 0 and len(results['confidences']) > 0:
        hybrid_arr = np.array(list(results['hybrid_preds'].values()))
        conf_arr = np.array(list(results['confidences'].values()))
        
        final_gamma = gamma
        
        if adaptive_gamma and len(hybrid_arr) >= 4:
            sorted_hybrid = np.sort(hybrid_arr)[::-1]
            gap_3_4 = sorted_hybrid[2] - sorted_hybrid[3]
            gap_threshold = 0.05
            
            if abs(gap_3_4) < gap_threshold:
                final_gamma = 1.5
                print(f"  [Adaptive Gamma] Top-3/4 갭 작음 ({gap_3_4:.4f}) → gamma={final_gamma}")
            else:
                print(f"  [Adaptive Gamma] Top-3/4 갭 충분 ({gap_3_4:.4f}) → gamma={gamma} 유지")
        
        if final_gamma > 0:
            z_hybrid = (hybrid_arr - hybrid_arr.mean()) / (hybrid_arr.std() + 1e-8)
            z_conf = (conf_arr - conf_arr.mean()) / (conf_arr.std() + 1e-8)
            
            ranking_score = z_hybrid + final_gamma * z_conf
            
            for i, sector in enumerate(results['hybrid_preds'].keys()):
                results['hybrid_ranking_preds'][sector] = ranking_score[i]
        else:
            for sector in results['hybrid_preds'].keys():
                results['hybrid_ranking_preds'][sector] = results['hybrid_preds'][sector]
    
    return results


def run_multiple_backtests(
    train_df: pd.DataFrame,
    test_years: List[int],
    sectors: List[str],
    alpha: float = 0.6,
    gamma: float = 0.5,
    adaptive_gamma: bool = False
) -> List[Dict]:
    """
    다중 연도 백테스트 실행
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        전체 학습 데이터셋
    test_years : List[int]
        테스트할 연도들
    sectors : List[str]
        섹터 리스트
    alpha : float
        하이브리드 가중치
    
    Returns:
    --------
    List[Dict] : 각 연도별 결과 리스트
    """
    all_results = []
    
    for year in test_years:
        # 데이터 분할: year 이전은 학습, year는 테스트
        year_train = train_df[train_df['year'] < year].copy()
        year_test = train_df[train_df['year'] == year].copy()
        
        results = run_backtest(year_train, year_test, year, sectors, alpha, gamma, adaptive_gamma)
        all_results.append(results)
    
    return all_results

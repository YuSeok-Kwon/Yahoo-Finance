"""
평가 지표 모듈
"""

import numpy as np
from scipy.stats import spearmanr
from typing import Dict
from evaluation import ndcg_at_k


def calculate_top_k_hit_ratio(actuals: np.ndarray, predictions: np.ndarray, k: int) -> float:
    """
    Top-K Hit Ratio 계산
    
    상위 K개 섹터를 얼마나 정확히 예측했는지 측정
    
    Parameters:
    -----------
    actuals : np.ndarray
        실제 수익률
    predictions : np.ndarray
        예측 수익률
    k : int
        상위 K개
    
    Returns:
    --------
    float : Hit Ratio (0~1, 1이 완벽)
    """
    if len(actuals) < k:
        return 0.0
    
    actual_top_k = set(np.argsort(actuals)[-k:])
    pred_top_k = set(np.argsort(predictions)[-k:])
    
    hit_count = len(actual_top_k.intersection(pred_top_k))
    return hit_count / k


def calculate_top_k_average_return(actuals: np.ndarray, predictions: np.ndarray, k: int) -> float:
    """
    예측한 Top-K 섹터들의 실제 평균 수익률
    
    "내가 고른 상위 K개가 실제로 얼마나 벌었는가?"
    
    Parameters:
    -----------
    actuals : np.ndarray
        실제 수익률
    predictions : np.ndarray
        예측 수익률
    k : int
        상위 K개
    
    Returns:
    --------
    float : Top-K의 평균 실제 수익률
    """
    if len(actuals) < k:
        return 0.0
    
    # 예측 기준 상위 K개 인덱스
    pred_top_k_indices = np.argsort(predictions)[-k:]
    
    # 그 섹터들의 실제 수익률 평균
    return actuals[pred_top_k_indices].mean()


def calculate_top_k_excess_return(actuals: np.ndarray, predictions: np.ndarray, k: int) -> float:
    """
    Top-K 초과수익 = Top-K 평균 수익률 - 전체 평균 수익률
    
    "내가 고른 상위 K개가 시장 평균 대비 얼마나 더 벌었는가?"
    
    Parameters:
    -----------
    actuals : np.ndarray
        실제 수익률
    predictions : np.ndarray
        예측 수익률
    k : int
        상위 K개
    
    Returns:
    --------
    float : 초과수익 (양수면 시장보다 좋음)
    """
    top_k_return = calculate_top_k_average_return(actuals, predictions, k)
    market_average = actuals.mean()
    return top_k_return - market_average


def calculate_ndcg_at_k(actuals: np.ndarray, predictions: np.ndarray, k: int) -> float:
    """
    NDCG@K (Normalized Discounted Cumulative Gain)
    
    순위 품질을 측정. 상위에 높은 수익률이 올수록 점수가 높음.
    1위가 가장 중요, 2위가 그 다음, 3위... 순으로 가중치 감소
    
    Parameters:
    -----------
    actuals : np.ndarray
        실제 수익률 (relevance score)
    predictions : np.ndarray
        예측 수익률
    k : int
        상위 K개
    
    Returns:
    --------
    float : NDCG@K (0~1, 1이 완벽)
    """
    if len(actuals) < k:
        return 0.0
    
    # DCG (Discounted Cumulative Gain)
    # 예측 기준 상위 K개의 실제 수익률을 1/log2(rank+1) 가중치로 합산
    pred_top_k_indices = np.argsort(predictions)[-k:][::-1]  # 내림차순
    dcg = 0.0
    for i, idx in enumerate(pred_top_k_indices):
        relevance = actuals[idx]
        dcg += relevance / np.log2(i + 2)  # i+2 because rank starts at 1, log2(1)=0
    
    # IDCG (Ideal DCG) - 실제 수익률 기준 최적 순서
    actual_top_k_indices = np.argsort(actuals)[-k:][::-1]  # 내림차순
    idcg = 0.0
    for i, idx in enumerate(actual_top_k_indices):
        relevance = actuals[idx]
        idcg += relevance / np.log2(i + 2)
    
    # NDCG = DCG / IDCG
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_predictions(
    actuals: np.ndarray,
    prophet_preds: np.ndarray,
    hybrid_preds: np.ndarray
) -> Dict[str, float]:
    """
    여러 지표를 사용한 예측 평가
    
    Parameters:
    -----------
    actuals : np.ndarray
        실제 수익률
    prophet_preds : np.ndarray
        Prophet 예측값
    hybrid_preds : np.ndarray
        Hybrid 예측값
    
    Returns:
    --------
    Dict[str, float] : 평가 지표
        - prophet_spearman, hybrid_spearman: 순위 상관계수
        - prophet_top3_hit, hybrid_top3_hit: 상위 3개 적중률
        - prophet_top5_hit, hybrid_top5_hit: 상위 5개 적중률
        - prophet_top3_avg_return, hybrid_top3_avg_return: Top-3 평균 실제 수익률
        - prophet_top5_avg_return, hybrid_top5_avg_return: Top-5 평균 실제 수익률
        - prophet_top3_excess, hybrid_top3_excess: Top-3 초과수익
        - prophet_top5_excess, hybrid_top5_excess: Top-5 초과수익
        - prophet_ndcg3, hybrid_ndcg3: NDCG@3
        - prophet_ndcg5, hybrid_ndcg5: NDCG@5
    """
    prophet_spearman = spearmanr(actuals, prophet_preds)[0]
    hybrid_spearman = spearmanr(actuals, hybrid_preds)[0]
    
    prophet_top3_hit = calculate_top_k_hit_ratio(actuals, prophet_preds, k=3)
    hybrid_top3_hit = calculate_top_k_hit_ratio(actuals, hybrid_preds, k=3)
    
    prophet_top5_hit = calculate_top_k_hit_ratio(actuals, prophet_preds, k=5)
    hybrid_top5_hit = calculate_top_k_hit_ratio(actuals, hybrid_preds, k=5)
    
    prophet_top3_avg_return = calculate_top_k_average_return(actuals, prophet_preds, k=3)
    hybrid_top3_avg_return = calculate_top_k_average_return(actuals, hybrid_preds, k=3)
    
    prophet_top5_avg_return = calculate_top_k_average_return(actuals, prophet_preds, k=5)
    hybrid_top5_avg_return = calculate_top_k_average_return(actuals, hybrid_preds, k=5)
    
    prophet_top3_excess = calculate_top_k_excess_return(actuals, prophet_preds, k=3)
    hybrid_top3_excess = calculate_top_k_excess_return(actuals, hybrid_preds, k=3)
    
    prophet_top5_excess = calculate_top_k_excess_return(actuals, prophet_preds, k=5)
    hybrid_top5_excess = calculate_top_k_excess_return(actuals, hybrid_preds, k=5)
    
    prophet_ndcg3 = ndcg_at_k(actuals, prophet_preds, k=3)
    hybrid_ndcg3  = ndcg_at_k(actuals, hybrid_preds,  k=3)

    prophet_ndcg5 = ndcg_at_k(actuals, prophet_preds, k=5)
    hybrid_ndcg5  = ndcg_at_k(actuals, hybrid_preds,  k=5)
    
    return {
        'prophet_spearman': prophet_spearman,
        'hybrid_spearman': hybrid_spearman,
        'prophet_top3_hit': prophet_top3_hit,
        'hybrid_top3_hit': hybrid_top3_hit,
        'prophet_top5_hit': prophet_top5_hit,
        'hybrid_top5_hit': hybrid_top5_hit,
        'prophet_top3_avg_return': prophet_top3_avg_return,
        'hybrid_top3_avg_return': hybrid_top3_avg_return,
        'prophet_top5_avg_return': prophet_top5_avg_return,
        'hybrid_top5_avg_return': hybrid_top5_avg_return,
        'prophet_top3_excess': prophet_top3_excess,
        'hybrid_top3_excess': hybrid_top3_excess,
        'prophet_top5_excess': prophet_top5_excess,
        'hybrid_top5_excess': hybrid_top5_excess,
        'prophet_ndcg3': prophet_ndcg3,
        'hybrid_ndcg3': hybrid_ndcg3,
        'prophet_ndcg5': prophet_ndcg5,
        'hybrid_ndcg5': hybrid_ndcg5
    }


def print_metrics(metrics: Dict[str, float], year: int):
    """
    형식화된 테이블로 지표 출력
    
    Parameters:
    -----------
    metrics : Dict[str, float]
        evaluate_predictions()의 지표
    year : int
        평가 연도
    """
    print("\n" + "="*100)
    print(f"{year}년 성능 평가")
    print("="*100)
    print(f"{'지표':<30s} | {'Prophet':>18s} | {'Hybrid':>18s} | {'개선도':>18s}")
    print("-"*100)
    
    print(f"{'Spearman 상관계수':<30s} | {metrics['prophet_spearman']:>18.4f} | {metrics['hybrid_spearman']:>18.4f} | {metrics['hybrid_spearman']-metrics['prophet_spearman']:>+18.4f}")
    
    print(f"{'Top-3 적중률':<30s} | {metrics['prophet_top3_hit']:>18.2%} | {metrics['hybrid_top3_hit']:>18.2%} | {metrics['hybrid_top3_hit']-metrics['prophet_top3_hit']:>+18.2%}")
    print(f"{'Top-3 평균 수익률':<30s} | {metrics['prophet_top3_avg_return']:>18.2%} | {metrics['hybrid_top3_avg_return']:>18.2%} | {metrics['hybrid_top3_avg_return']-metrics['prophet_top3_avg_return']:>+18.2%}")
    print(f"{'Top-3 초과수익':<30s} | {metrics['prophet_top3_excess']:>18.2%} | {metrics['hybrid_top3_excess']:>18.2%} | {metrics['hybrid_top3_excess']-metrics['prophet_top3_excess']:>+18.2%}")
    print(f"{'NDCG@3':<30s} | {metrics['prophet_ndcg3']:>18.4f} | {metrics['hybrid_ndcg3']:>18.4f} | {metrics['hybrid_ndcg3']-metrics['prophet_ndcg3']:>+18.4f}")
    
    print(f"{'Top-5 적중률':<30s} | {metrics['prophet_top5_hit']:>18.2%} | {metrics['hybrid_top5_hit']:>18.2%} | {metrics['hybrid_top5_hit']-metrics['prophet_top5_hit']:>+18.2%}")
    print(f"{'Top-5 평균 수익률':<30s} | {metrics['prophet_top5_avg_return']:>18.2%} | {metrics['hybrid_top5_avg_return']:>18.2%} | {metrics['hybrid_top5_avg_return']-metrics['prophet_top5_avg_return']:>+18.2%}")
    print(f"{'Top-5 초과수익':<30s} | {metrics['prophet_top5_excess']:>18.2%} | {metrics['hybrid_top5_excess']:>18.2%} | {metrics['hybrid_top5_excess']-metrics['prophet_top5_excess']:>+18.2%}")
    print(f"{'NDCG@5':<30s} | {metrics['prophet_ndcg5']:>18.4f} | {metrics['hybrid_ndcg5']:>18.4f} | {metrics['hybrid_ndcg5']-metrics['prophet_ndcg5']:>+18.4f}")
    
    print("="*100)


def print_sector_rankings(actuals: dict, prophet_preds: dict, hybrid_preds: dict, year: int, top_k: int = 5):
    """
    섹터별 순위 비교 출력
    
    Parameters:
    -----------
    actuals : dict
        섹터: 실제 수익률
    prophet_preds : dict
        섹터: Prophet 예측 수익률
    hybrid_preds : dict
        섹터: Hybrid 예측 수익률
    year : int
        연도
    top_k : int
        표시할 상위 섹터 수
    """
    import numpy as np
    
    # 순위 계산
    actual_sorted = sorted(actuals.items(), key=lambda x: x[1], reverse=True)
    prophet_sorted = sorted(prophet_preds.items(), key=lambda x: x[1], reverse=True)
    hybrid_sorted = sorted(hybrid_preds.items(), key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*100)
    print(f"{year}년 섹터 순위 비교 (Top {top_k})")
    print("="*100)
    print(f"{'순위':<6s} | {'실제 (수익률)':<35s} | {'Prophet 예측':<28s} | {'Hybrid 예측':<28s}")
    print("-"*100)
    
    for i in range(min(top_k, len(actual_sorted))):
        actual_sector, actual_return = actual_sorted[i]
        prophet_sector, prophet_return = prophet_sorted[i] if i < len(prophet_sorted) else ('-', 0)
        hybrid_sector, hybrid_return = hybrid_sorted[i] if i < len(hybrid_sorted) else ('-', 0)
        
        # 실제 상위 섹터를 맞췄는지 표시
        prophet_hit = "✓" if prophet_sector == actual_sector else " "
        hybrid_hit = "✓" if hybrid_sector == actual_sector else " "
        
        print(f"{i+1:<6d} | {actual_sector:<25s} ({actual_return:>7.2%}) | {prophet_hit} {prophet_sector:<25s} | {hybrid_hit} {hybrid_sector:<25s}")
    
    print("="*100)
    print("✓ = 실제 상위 섹터를 정확히 예측")

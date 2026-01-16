"""
군집 + LTR 통합 분석 스크립트 (FIXED)

군집별 LTR 성능 분석 및 다각화 포트폴리오 구성
rank_actual, rank_final 컬럼 자동 생성
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from typing import Dict, List, Tuple


def analyze_cluster_ltr_performance(
    df_sector_year: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    test_years: List[int]
) -> pd.DataFrame:
    """군집별 LTR 성능 분석 (rank 컬럼 자동 생성)"""
    
    df_with_cluster = df_sector_year.merge(
        cluster_assignments,
        on=['test_year', 'Sector'],
        how='left'
    )
    
    if 'rank_actual' not in df_with_cluster.columns:
        print(" 'rank_actual' 컬럼 없음 - 자동 생성 중...")
        df_with_cluster['rank_actual'] = df_with_cluster.groupby('test_year')['actual_return'].rank(
            ascending=False, method='first'
        ).astype(int)
        print("   rank_actual 생성 완료 (actual_return 기준)")
    
    if 'rank_final' not in df_with_cluster.columns:
        print(" 'rank_final' 컬럼 없음 - 자동 생성 중...")
        score_col = 'ltr_score_raw' if 'ltr_score_raw' in df_with_cluster.columns else 'pred_return_final'
        df_with_cluster['rank_final'] = df_with_cluster.groupby('test_year')[score_col].rank(
            ascending=False, method='first'
        ).astype(int)
        print(f"   rank_final 생성 완료 ({score_col} 기준)")
    
    print("\n" + "=" * 80)
    print("군집별 평균 Spearman 상관계수 (LTR)")
    print("=" * 80)
    
    cluster_performance = []
    
    for cluster_id in sorted(cluster_assignments['cluster'].unique()):
        cluster_data = df_with_cluster[
            df_with_cluster['cluster'] == cluster_id
        ]
        
        correlations = []
        for year in test_years:
            year_data = cluster_data[cluster_data['test_year'] == year]
            if len(year_data) >= 2:
                actual_rank = year_data['rank_actual']
                pred_rank = year_data['rank_final']
                corr, _ = spearmanr(actual_rank, pred_rank)
                correlations.append(corr)
        
        if correlations:
            avg_corr = np.mean(correlations)
            std_corr = np.std(correlations)
            min_corr = np.min(correlations)
            max_corr = np.max(correlations)
            
            cluster_performance.append({
                'cluster': cluster_id,
                'avg_spearman': avg_corr,
                'std_spearman': std_corr,
                'min_spearman': min_corr,
                'max_spearman': max_corr,
                'n_sectors': len(cluster_data['Sector'].unique()),
                'n_observations': len(cluster_data)
            })
            
            print(f"  Cluster {cluster_id}: {avg_corr:6.3f} ± {std_corr:.3f} "
                  f"(range: [{min_corr:.3f}, {max_corr:.3f}]) - {len(cluster_data['Sector'].unique())} sectors")
    
    print("=" * 80)
    
    return pd.DataFrame(cluster_performance)


def build_cluster_diversified_portfolio(
    df_sector_year: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    year: int,
    n_per_cluster: int = 2,
    score_col: str = 'ltr_score_raw'
) -> pd.DataFrame:
    """군집 다각화 포트폴리오 구성"""
    
    df = df_sector_year[df_sector_year['test_year'] == year].copy()
    df = df.merge(cluster_assignments[cluster_assignments['test_year'] == year],
                  on=['test_year', 'Sector'], how='left')
    
    if 'cluster' not in df.columns or df['cluster'].isna().all():
        print(f"Warning: No cluster info for year {year}")
        fallback_col = score_col if score_col in df.columns else 'pred_return_final'
        return df.nlargest(n_per_cluster * 3, fallback_col)[['Sector', fallback_col]]
    
    portfolio = []
    
    for cluster_id in sorted(df['cluster'].dropna().unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        
        score_to_use = score_col if score_col in cluster_data.columns else 'pred_return_final'
        top_sectors = cluster_data.nlargest(n_per_cluster, score_to_use)
        
        for _, row in top_sectors.iterrows():
            portfolio.append({
                'Sector': row['Sector'],
                'Cluster': int(cluster_id),
                'LTR_Score': row.get(score_col, row.get('pred_return_final', np.nan)),
                'Actual_Return': row.get('actual_return', np.nan),
                'Pred_Return': row.get('pred_return_final', np.nan)
            })
    
    portfolio_df = pd.DataFrame(portfolio)
    
    print(f"\n{year} 포트폴리오 (군집 다각화):")
    print("=" * 80)
    print(portfolio_df.sort_values('LTR_Score', ascending=False).to_string(index=False))
    print("=" * 80)
    
    return portfolio_df


def compare_portfolio_strategies(
    df_sector_year: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    test_years: List[int],
    n_stocks: int = 5
) -> pd.DataFrame:
    """포트폴리오 전략 비교: 단순 Top-N vs 군집 다각화"""
    
    results = []
    
    score_col = 'ltr_score_raw' if 'ltr_score_raw' in df_sector_year.columns else 'pred_return_final'
    
    for year in test_years:
        year_data = df_sector_year[df_sector_year['test_year'] == year].copy()
        
        top_n = year_data.nlargest(n_stocks, score_col)
        top_n_return = top_n['actual_return'].mean()
        top_n_std = top_n['actual_return'].std()
        
        cluster_portfolio = build_cluster_diversified_portfolio(
            df_sector_year, cluster_assignments, year,
            n_per_cluster=max(1, n_stocks // 3),
            score_col=score_col
        )
        
        if len(cluster_portfolio) > 0:
            cluster_return = cluster_portfolio['Actual_Return'].mean()
            cluster_std = cluster_portfolio['Actual_Return'].std()
            cluster_n_clusters = cluster_portfolio['Cluster'].nunique()
        else:
            cluster_return = np.nan
            cluster_std = np.nan
            cluster_n_clusters = 0
        
        results.append({
            'Year': year,
            'TopN_Return': top_n_return,
            'TopN_Std': top_n_std,
            'TopN_Sharpe': top_n_return / top_n_std if top_n_std > 0 else np.nan,
            'Cluster_Return': cluster_return,
            'Cluster_Std': cluster_std,
            'Cluster_Sharpe': cluster_return / cluster_std if cluster_std > 0 else np.nan,
            'N_Clusters': cluster_n_clusters
        })
    
    comparison_df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("포트폴리오 전략 비교")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("\n평균:")
    print(comparison_df[['TopN_Return', 'TopN_Sharpe', 'Cluster_Return', 'Cluster_Sharpe']].mean())
    print("=" * 80)
    
    return comparison_df


def main_cluster_ltr_analysis(
    df_sector_year: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    test_years: List[int]
):
    """전체 군집+LTR 통합 분석 실행"""
    
    print("\n" + "=" * 80)
    print("군집 + LTR 통합 분석")
    print("=" * 80)
    
    cluster_perf = analyze_cluster_ltr_performance(
        df_sector_year, cluster_assignments, test_years
    )
    
    portfolio_comp = compare_portfolio_strategies(
        df_sector_year, cluster_assignments, test_years, n_stocks=5
    )
    
    print("\n 군집 + LTR 통합 분석 완료")
    
    return {
        'cluster_performance': cluster_perf,
        'portfolio_comparison': portfolio_comp
    }


if __name__ == '__main__':
    print(__doc__)
    print("\n사용법:")
    print("""
# 노트북에서:
from cluster_ltr_integration import main_cluster_ltr_analysis

results = main_cluster_ltr_analysis(
    df_sector_year=df_sector_year,
    cluster_assignments=cluster_assignments,
    test_years=TEST_YEARS
)
""")

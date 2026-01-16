"""
클러스터링 수익률 정렬 테스트
"""
import pandas as pd
import numpy as np
import warnings
import sys
from pathlib import Path
warnings.filterwarnings('ignore')

# 직접 파일 import
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from industry_clustering import IndustryClusterer

# 테스트 데이터 생성
print("=" * 80)
print("클러스터링 수익률 정렬 테스트")
print("=" * 80)

# 샘플 산업 데이터 생성 (32개 산업)
np.random.seed(42)
industries = [
    'Software', 'Semiconductors', 'Hardware', 'IT Services',
    'Oil & Gas', 'Energy Equipment', 'Coal & Fuel', 'Renewable Energy',
    'Banks', 'Insurance', 'Investment', 'Real Estate',
    'Pharma', 'Biotech', 'Healthcare Equip', 'Healthcare Services',
    'Retail', 'Consumer Staples', 'Restaurants', 'Hotels',
    'Aerospace', 'Defense', 'Industrial Machinery', 'Construction',
    'Telecom', 'Media', 'Entertainment', 'Advertising',
    'Auto Manufacturers', 'Auto Parts', 'Transportation', 'Logistics'
]

df_industry = pd.DataFrame({
    'Industry': industries,
    'Sector': ['Technology'] * 4 + ['Energy'] * 4 + ['Financials'] * 4 +
              ['Healthcare'] * 4 + ['Consumer'] * 4 + ['Industrials'] * 4 +
              ['Communication'] * 4 + ['Consumer Cyclical'] * 4,
    'Return_3M': np.random.uniform(-0.20, 0.30, 32),  # -20% ~ +30%
    'Volatility_20d': np.random.uniform(0.10, 0.40, 32),  # 10% ~ 40%
    'MDD': np.random.uniform(-0.30, -0.05, 32),  # -30% ~ -5%
    'Sharpe_Ratio': np.random.uniform(-0.5, 1.5, 32),
    'Num_Companies': np.random.randint(5, 50, 32)
})

print(f"\n생성된 산업 수: {len(df_industry)}")
print(f"수익률 범위: {df_industry['Return_3M'].min():.2%} ~ {df_industry['Return_3M'].max():.2%}")

# 클러스터링 실행
clusterer = IndustryClusterer(n_clusters=4, random_state=42)

print("\n" + "-" * 80)
print("1. KMeans 클러스터링 수행")
print("-" * 80)
df_industry = clusterer.fit_predict(df_industry)

print("\n" + "-" * 80)
print("2. 클러스터 프로파일 생성")
print("-" * 80)
cluster_profile = clusterer.profile_clusters(df_industry)

print("\n" + "-" * 80)
print("3. 수익률 기준 클러스터 재매핑")
print("-" * 80)
df_industry, cluster_profile = clusterer.remap_cluster_ids(df_industry, cluster_profile)

print("\n" + "=" * 80)
print("재매핑 후 클러스터 프로파일")
print("=" * 80)
print(cluster_profile[['Return_3M', 'Volatility_20d', 'MDD', 'Sharpe_Ratio']])
print("=" * 80)

# 각 클러스터별 산업 목록
print("\n" + "=" * 80)
print("클러스터별 산업 목록 (수익률 순)")
print("=" * 80)
for cluster_id in sorted(df_industry['cluster'].unique()):
    cluster_data = df_industry[df_industry['cluster'] == cluster_id].sort_values('Return_3M', ascending=False)
    avg_return = cluster_data['Return_3M'].mean()
    print(f"\nCluster {cluster_id} (평균 수익률: {avg_return:.2%}, {len(cluster_data)}개 산업):")
    for idx, row in cluster_data.head(5).iterrows():
        print(f"  - {row['Industry']}: {row['Return_3M']:.2%} (변동성: {row['Volatility_20d']:.2%})")
    if len(cluster_data) > 5:
        print(f"  ... 외 {len(cluster_data) - 5}개")

print("\n" + "=" * 80)
print("검증: 클러스터 순서가 수익률 순인지 확인")
print("=" * 80)
cluster_returns = cluster_profile['Return_3M'].to_dict()
print("클러스터별 평균 수익률:")
for cluster_id in sorted(cluster_returns.keys()):
    print(f"  Cluster {cluster_id}: {cluster_returns[cluster_id]:.2%}")

is_sorted = all(
    cluster_returns[i] >= cluster_returns[i+1]
    for i in range(len(cluster_returns)-1)
)
if is_sorted:
    print("\n✓ 성공: 클러스터가 수익률 높은 순으로 정렬되었습니다!")
else:
    print("\n✗ 실패: 클러스터 정렬에 문제가 있습니다.")

print("=" * 80)

# Industry Analysis Module Usage Guide

## Overview

`industry_analysis.py` 모듈은 섹터 LTR 결과를 기반으로 산업(Industry) 레벨 분석을 수행합니다.

**Pipeline**:
```
Sector LTR → TOP-N Sectors → Industry Data → Risk-Return Clustering → Portfolio
```

---

## Installation Check

```python
# Required packages (in py_study conda environment)
# - pandas
# - numpy
# - scikit-learn (sklearn)

# Verify imports
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
print("✓ All required packages available")
```

---

## Quick Start (Single Function)

### Example 1: Complete Pipeline for Single Year

```python
from industry_analysis import main_industry_analysis

# Run complete pipeline for 2025
results = main_industry_analysis(
    df_sector_year=df_sector_year,          # From Section 7.5 (LTR results)
    stock_data_path='Data_set/stock_features_clean.csv',
    year=2025,
    top_n_sectors=5,                        # Select TOP-5 sectors
    n_clusters=4,                           # 4 risk profiles
    profile_weights={2: 0.5, 0: 0.3, 3: 0.2},  # Portfolio allocation
    top_n_per_profile=3                     # 3 industries per profile
)

# Extract results
top_sectors = results['top_sectors']        # List[str]
industry_data = results['industry_data']    # DataFrame (Date x Industry)
clusterer = results['clusterer']            # IndustryClusterer object
portfolio = results['portfolio']            # DataFrame (Portfolio composition)

print(portfolio)
```

**Expected Output**:
```
================================================================================
2025년 TOP-5 섹터 선택 (ltr_score_raw 기준)
================================================================================
1. Technology                       | ltr_score_raw:   0.1234 | Actual:  15.23%
2. Healthcare                       | ltr_score_raw:   0.0987 | Actual:  12.45%
...
================================================================================

산업 데이터 추출 중...
  - 대상 섹터: 5개
  - 필터링 후: 85,432 행, 127 기업
  - 날짜 범위: 2025-01-01 ~ 2025-12-31
  - 산업 수: 23
✓ 추출 완료

================================================================================
산업 리스크-수익률 클러스터링
================================================================================
  - 분석 기간: 전체 (2025-01-01 ~ 2025-12-31)
✓ 클러스터링 완료: 23 산업 → 4 클러스터

================================================================================
다각화 포트폴리오 구성
================================================================================
프로파일별 가중치:
  [2] Low Risk - High Return          :  50.0%
  [0] High Risk - High Return         :  30.0%
  [3] Low Risk - Low Return           :  20.0%
...

    Industry                             Sector  Risk_Profile Weight  Expected_Return
0   Software - Infrastructure         Technology            2  0.167           0.1523
1   Medical Devices                   Healthcare            2  0.167           0.1445
...
```

---

## Step-by-Step Usage

### Step 1: Select TOP-N Sectors

```python
from industry_analysis import select_top_sectors

# Select TOP-5 sectors for 2025 based on LTR scores
top_sectors_2025 = select_top_sectors(
    df_sector_year=df_sector_year,
    year=2025,
    top_n=5,
    score_col='ltr_score_raw'  # or 'pred_return_final'
)

print(f"Selected sectors: {top_sectors_2025}")
# Output: ['Technology', 'Healthcare', 'Finance', 'Consumer Discretionary', 'Industrials']
```

### Step 2: Extract Industry Data

```python
from industry_analysis import extract_industry_data

# Extract industry-level data from TOP sectors
industry_data = extract_industry_data(
    stock_data_path='Data_set/stock_features_clean.csv',
    top_sectors=top_sectors_2025,
    start_date='2025-01-01',
    end_date='2025-12-31'
)

print(industry_data.head())
```

**Output Columns**:
- `Date`: 날짜
- `Industry`: 산업명
- `Sector`: 섹터명
- `Daily_Return`: 일일 수익률 (평균)
- `Volatility_20d`: 20일 변동성 (평균)
- `Return_1M`, `Return_3M`, `Return_6M`: 1/3/6개월 수익률
- `Company_Count`: 해당 산업 기업 수

### Step 3: Risk-Return Clustering

```python
from industry_analysis import IndustryClusterer

# Initialize clusterer
clusterer = IndustryClusterer(n_clusters=4, random_state=42)

# Fit clustering model
clusterer.fit(
    industry_agg=industry_data,
    risk_col='Volatility_20d',
    return_col='Return_3M',
    window='recent'  # 'recent': 최근 3개월, 'all': 전체 기간
)

# View cluster summary
cluster_summary = clusterer.get_cluster_summary()
print(cluster_summary)
```

**Cluster Labels**:
- **0**: High Risk - High Return (공격적)
- **1**: High Risk - Low Return (회피)
- **2**: Low Risk - High Return (이상적) ⭐
- **3**: Low Risk - Low Return (안정적)

### Step 4: Get Industries by Risk Profile

```python
# Get industries in "Low Risk - High Return" profile
low_risk_high_return = clusterer.get_industries_by_profile(
    profile=2,  # Low Risk - High Return
    top_n=5     # Top 5 industries
)

print(low_risk_high_return[['Industry', 'Sector', 'Return_3M', 'Volatility_20d']])
```

### Step 5: Build Diversified Portfolio

```python
from industry_analysis import build_industry_portfolio

# Build portfolio with custom weights
portfolio = build_industry_portfolio(
    industry_clusterer=clusterer,
    profile_weights={
        2: 0.50,  # 50% in Low Risk - High Return
        0: 0.30,  # 30% in High Risk - High Return
        3: 0.20   # 20% in Low Risk - Low Return
    },
    top_n_per_profile=3  # 3 industries per profile
)

print(portfolio[['Industry', 'Sector', 'Weight', 'Expected_Return', 'Profile_Desc']])

# Portfolio statistics
total_expected_return = (portfolio['Expected_Return'] * portfolio['Weight']).sum()
portfolio_volatility = (portfolio['Volatility'] * portfolio['Weight']).sum()

print(f"\nPortfolio Expected Return: {total_expected_return:.2%}")
print(f"Portfolio Volatility: {portfolio_volatility:.2%}")
```

---

## Advanced Usage

### Example 2: Multi-Year Analysis

```python
from industry_analysis import main_industry_analysis

# Analyze multiple years
results_by_year = {}

for year in [2023, 2024, 2025]:
    print(f"\n{'#'*80}")
    print(f"# {year}년 분석")
    print(f"{'#'*80}\n")
    
    results = main_industry_analysis(
        df_sector_year=df_sector_year,
        stock_data_path='Data_set/stock_features_clean.csv',
        year=year,
        top_n_sectors=5,
        n_clusters=4
    )
    
    results_by_year[year] = results

# Compare portfolios across years
for year, results in results_by_year.items():
    portfolio = results['portfolio']
    exp_return = (portfolio['Expected_Return'] * portfolio['Weight']).sum()
    print(f"{year}: Expected Return = {exp_return:.2%}, Industries = {len(portfolio)}")
```

### Example 3: Custom Risk Profiles

```python
# Conservative portfolio (heavy on Low Risk - Low Return)
conservative_portfolio = build_industry_portfolio(
    clusterer,
    profile_weights={
        3: 0.60,  # 60% Low Risk - Low Return
        2: 0.30,  # 30% Low Risk - High Return
        0: 0.10   # 10% High Risk - High Return
    },
    top_n_per_profile=2
)

# Aggressive portfolio (heavy on High Risk - High Return)
aggressive_portfolio = build_industry_portfolio(
    clusterer,
    profile_weights={
        0: 0.50,  # 50% High Risk - High Return
        2: 0.30,  # 30% Low Risk - High Return
        3: 0.20   # 20% Low Risk - Low Return
    },
    top_n_per_profile=4
)

# Compare
print("Conservative:", (conservative_portfolio['Expected_Return'] * conservative_portfolio['Weight']).sum())
print("Aggressive:", (aggressive_portfolio['Expected_Return'] * aggressive_portfolio['Weight']).sum())
```

### Example 4: Industry Trend Analysis

```python
# Analyze industry performance over time
industry_data = results['industry_data']

# Get specific industry trend
software_trend = industry_data[
    industry_data['Industry'] == 'Software - Infrastructure'
].sort_values('Date')

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(software_trend['Date'], software_trend['Return_3M'])
plt.title('Software - Infrastructure: 3-Month Return Trend')
plt.xlabel('Date')
plt.ylabel('Return (%)')
plt.grid(True)
plt.show()
```

---

## Integration with Main Notebook

### Recommended Notebook Structure

```
Section 10: Industry-Level Analysis (NEW)

Cell 10.1: Import Module
────────────────────────────────────────────────
from industry_analysis import (
    select_top_sectors,
    extract_industry_data,
    IndustryClusterer,
    build_industry_portfolio,
    main_industry_analysis
)

Cell 10.2: Run Industry Analysis (2025)
────────────────────────────────────────────────
results_2025 = main_industry_analysis(
    df_sector_year=df_sector_year,
    stock_data_path='Data_set/stock_features_clean.csv',
    year=2025,
    top_n_sectors=5,
    n_clusters=4,
    profile_weights={2: 0.5, 0: 0.3, 3: 0.2},
    top_n_per_profile=3
)

portfolio_2025 = results_2025['portfolio']
print(portfolio_2025)

Cell 10.3: Visualize Risk-Return Clusters
────────────────────────────────────────────────
import matplotlib.pyplot as plt

clusterer = results_2025['clusterer']
industry_profile = clusterer.industry_profile

plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    industry_profile['Volatility_20d'],
    industry_profile['Return_3M'],
    c=industry_profile['risk_profile'],
    cmap='viridis',
    s=100,
    alpha=0.6
)
plt.xlabel('Volatility (Risk)')
plt.ylabel('3-Month Return (%)')
plt.title('Industry Risk-Return Profile (2025)')
plt.colorbar(scatter, label='Risk Profile')
plt.grid(True, alpha=0.3)

# Annotate industries
for _, row in industry_profile.iterrows():
    plt.annotate(
        row['Industry'][:20],  # Truncate long names
        (row['Volatility_20d'], row['Return_3M']),
        fontsize=8,
        alpha=0.7
    )

plt.show()

Cell 10.4: Portfolio Performance Summary
────────────────────────────────────────────────
# Calculate portfolio metrics
def calculate_portfolio_metrics(portfolio):
    exp_return = (portfolio['Expected_Return'] * portfolio['Weight']).sum()
    volatility = (portfolio['Volatility'] * portfolio['Weight']).sum()
    sharpe = exp_return / volatility if volatility > 0 else 0
    
    return {
        'Expected Return': exp_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe,
        'N_Industries': len(portfolio),
        'N_Sectors': portfolio['Sector'].nunique()
    }

metrics = calculate_portfolio_metrics(portfolio_2025)
print(pd.Series(metrics))
```

---

## Data Requirements

### Input Data

#### 1. `df_sector_year` (from Section 7.5)

Required columns:
- `test_year` (int): 연도
- `Sector` (str): 섹터명
- `ltr_score_raw` (float): LTR 스코어
- `actual_return` (float): 실제 수익률 (optional, for display)

#### 2. `stock_features_clean.csv`

Required columns:
- `Date` (datetime): 날짜
- `Company` (str): 기업 ticker
- `Sector` (str): 섹터명
- `Industry` (str): 산업명
- `Daily_Return_calc` (float): 일일 수익률
- `Volatility_20d` (float): 20일 변동성
- `Return_1M`, `Return_3M`, `Return_6M` (float): 1/3/6개월 수익률

---

## Troubleshooting

### Issue 1: sklearn Not Found

```bash
# Install scikit-learn in py_study environment
conda activate py_study
conda install scikit-learn

# Or with pip
pip install scikit-learn
```

### Issue 2: Empty TOP Sectors

```python
# Check if year exists in df_sector_year
print(df_sector_year['test_year'].unique())

# Check LTR scores
print(df_sector_year[df_sector_year['test_year'] == 2025]['ltr_score_raw'].describe())

# If all NaN, run Section 7.5 LTR first
```

### Issue 3: No Industries in Cluster

```python
# Check cluster distribution
print(clusterer.industry_profile['risk_profile'].value_counts())

# If imbalanced, try different n_clusters
clusterer = IndustryClusterer(n_clusters=3)  # Try 3 instead of 4
```

### Issue 4: KeyError on 'Return_3M'

```python
# Check available columns in industry_data
print(industry_data.columns.tolist())

# Update clustering to use available column
clusterer.fit(
    industry_data,
    risk_col='Volatility_20d',
    return_col='Return_1M',  # Use 1M instead of 3M
    window='recent'
)
```

---

## Performance Tips

### Tip 1: Cache Industry Data

```python
# Save extracted industry data to avoid re-processing
industry_data.to_csv('Data_set/industry_data_2025.csv', index=False)

# Load cached data
industry_data = pd.read_csv('Data_set/industry_data_2025.csv', parse_dates=['Date'])
```

### Tip 2: Parallel Multi-Year Analysis

```python
from multiprocessing import Pool

def analyze_year(year):
    return main_industry_analysis(
        df_sector_year, 'Data_set/stock_features_clean.csv',
        year, top_n_sectors=5
    )

# Run in parallel
with Pool(3) as pool:
    results = pool.map(analyze_year, [2023, 2024, 2025])
```

### Tip 3: Profile Only Recent Data

```python
# For faster clustering, use recent data only
clusterer.fit(
    industry_data,
    window='recent'  # Last 3 months only
)
```

---

## Expected Runtime

| Operation | Time (Typical) |
|-----------|----------------|
| `select_top_sectors` | < 1 sec |
| `extract_industry_data` | 2-5 sec (depends on data size) |
| `IndustryClusterer.fit` | 1-3 sec |
| `build_industry_portfolio` | < 1 sec |
| **Full Pipeline** | **5-10 sec** |

---

## Next Steps

After running industry analysis:

1. **Backtesting**: Test portfolio performance on historical data
2. **Daily Updates**: Automate data refresh and re-run clustering
3. **Alert System**: Monitor when industries change risk profiles
4. **Integration**: Connect to trading API for automated execution

---

## Questions?

Check:
- Main notebook: `07_Integrated_Prophet_Analysis.ipynb`
- Module source: `industry_analysis.py`
- Project README: `README.md`

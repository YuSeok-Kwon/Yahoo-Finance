"""
Industry-Level Analysis Module
Industry-level risk-return clustering within TOP-N sectors selected by LTR

Main Functions:
1. select_top_sectors() - TOP-N 섹터 선택
2. extract_industry_data() - 산업별 데이터 추출
3. IndustryClusterer - 산업별 리스크-수익률 클러스터링
4. build_industry_portfolio() - 산업 포트폴리오 구성

Flow:
Sector LTR → TOP-N Sectors → Industry Data → Risk-Return Clustering → Portfolio
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta


def select_top_sectors(
    df_sector_year: pd.DataFrame,
    year: int,
    top_n: int = 5,
    score_col: str = 'ltr_score_raw'
) -> List[str]:
    """
    LTR 스코어 기준 TOP-N 섹터 선택
    
    Parameters:
    -----------
    df_sector_year : DataFrame
        LTR 결과 데이터 (test_year, Sector, ltr_score_raw 포함)
    year : int
        선택할 연도
    top_n : int
        상위 N개 섹터 (default: 5)
    score_col : str
        정렬 기준 컬럼 (default: 'ltr_score_raw')
    
    Returns:
    --------
    List[str] : 선택된 섹터 리스트
    
    Example:
    --------
    >>> top_sectors_2025 = select_top_sectors(df_sector_year, 2025, top_n=5)
    >>> print(top_sectors_2025)
    ['Technology', 'Healthcare', 'Finance', 'Consumer Discretionary', 'Industrials']
    """
    year_data = df_sector_year[df_sector_year['test_year'] == year].copy()
    
    if year_data.empty:
        print(f"⚠️  경고: {year}년 데이터 없음")
        return []
    
    # 스코어 기준 정렬
    year_data = year_data.sort_values(by=score_col, ascending=False)
    top_sectors_df = year_data.head(top_n)
    top_sectors = top_sectors_df['Sector'].tolist()
    
    print(f"\n{'='*80}")
    print(f"{year}년 TOP-{top_n} 섹터 선택 ({score_col} 기준)")
    print(f"{'='*80}")
    for rank, (idx, row) in enumerate(top_sectors_df.iterrows(), 1):
        print(f"{rank}. {row['Sector']:30s} | {score_col}: {row[score_col]:8.4f} | Actual: {row.get('actual_return', np.nan):7.2%}")
    print(f"{'='*80}\n")
    
    return top_sectors


def extract_industry_data(
    stock_data_path: str,
    top_sectors: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    TOP-N 섹터 내 산업별 데이터 추출
    
    Parameters:
    -----------
    stock_data_path : str
        stock_features_clean.csv 경로
    top_sectors : List[str]
        선택된 섹터 리스트
    start_date : str, optional
        시작일 (YYYY-MM-DD)
    end_date : str, optional
        종료일 (YYYY-MM-DD)
    
    Returns:
    --------
    DataFrame : 산업별 집계 데이터
        컬럼: Date, Industry, Sector, Daily_Return, Volatility_20d, Return_1M, Return_3M, Company_Count
    
    Example:
    --------
    >>> industry_data = extract_industry_data(
    ...     'Data_set/stock_features_clean.csv',
    ...     ['Technology', 'Healthcare'],
    ...     start_date='2024-01-01'
    ... )
    """
    print(f"\n산업 데이터 추출 중...")
    print(f"  - 대상 섹터: {len(top_sectors)}개")
    
    # 데이터 로드
    df = pd.read_csv(stock_data_path, parse_dates=['Date'])
    
    # 섹터 필터링
    df = df[df['Sector'].isin(top_sectors)].copy()
    print(f"  - 필터링 후: {len(df):,} 행, {df['Company'].nunique()} 기업")
    
    # 날짜 필터링
    if start_date:
        df = df[df['Date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['Date'] <= pd.to_datetime(end_date)]
    print(f"  - 날짜 범위: {df['Date'].min().date()} ~ {df['Date'].max().date()}")
    
    # 산업별 집계
    industry_agg = df.groupby(['Date', 'Industry', 'Sector']).agg({
        'Daily_Return_calc': 'mean',
        'Volatility_20d': 'mean',
        'Return_1M': 'mean',
        'Return_3M': 'mean',
        'Return_6M': 'mean',
        'Company': 'count'  # 기업 수
    }).reset_index()
    
    # 컬럼명 정리
    industry_agg.rename(columns={
        'Daily_Return_calc': 'Daily_Return',
        'Company': 'Company_Count'
    }, inplace=True)
    
    print(f"  - 산업 수: {industry_agg['Industry'].nunique()}")
    print(f"  - 날짜 수: {industry_agg['Date'].nunique()}")
    print(f"\n✓ 추출 완료\n")
    
    return industry_agg


class IndustryClusterer:
    """
    산업별 리스크-수익률 클러스터링
    
    4개 프로파일로 분류:
    - 0: High Risk - High Return
    - 1: High Risk - Low Return
    - 2: Low Risk - High Return
    - 3: Low Risk - Low Return
    """
    
    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = None
        self.cluster_profiles = None
    
    def fit(
        self,
        industry_agg: pd.DataFrame,
        risk_col: str = 'Volatility_20d',
        return_col: str = 'Return_3M',
        window: str = 'recent'  # 'recent', 'all', or specific date
    ) -> 'IndustryClusterer':
        """
        리스크-수익률 기준 클러스터링 수행
        
        Parameters:
        -----------
        industry_agg : DataFrame
            extract_industry_data() 결과
        risk_col : str
            리스크 컬럼 (default: Volatility_20d)
        return_col : str
            수익률 컬럼 (default: Return_3M)
        window : str
            분석 기간 ('recent': 최근 3개월, 'all': 전체)
        
        Returns:
        --------
        self
        """
        print(f"\n{'='*80}")
        print("산업 리스크-수익률 클러스터링")
        print(f"{'='*80}")
        
        # 산업별 평균 계산
        if window == 'recent':
            # 최근 3개월 데이터
            recent_date = industry_agg['Date'].max()
            cutoff_date = recent_date - pd.Timedelta(days=90)
            df_window = industry_agg[industry_agg['Date'] >= cutoff_date]
            print(f"  - 분석 기간: 최근 3개월 ({cutoff_date.date()} ~ {recent_date.date()})")
        else:
            df_window = industry_agg
            print(f"  - 분석 기간: 전체 ({industry_agg['Date'].min().date()} ~ {industry_agg['Date'].max().date()})")
        
        industry_profile = df_window.groupby(['Industry', 'Sector']).agg({
            risk_col: 'mean',
            return_col: 'mean',
            'Company_Count': 'mean'
        }).reset_index()
        
        # 결측치 제거
        industry_profile = industry_profile.dropna(subset=[risk_col, return_col])
        
        # 특징 추출 및 정규화
        X = industry_profile[[risk_col, return_col]].values
        X_scaled = self.scaler.fit_transform(X)
        
        # KMeans 클러스터링
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        industry_profile['cluster'] = self.kmeans.fit_predict(X_scaled)
        
        # 클러스터 프로파일 계산
        self._compute_cluster_profiles(industry_profile, risk_col, return_col)
        
        # 클러스터 레이블링 (Risk-Return 기준)
        industry_profile = self._label_clusters(industry_profile, risk_col, return_col)
        
        self.industry_profile = industry_profile
        
        print(f"\n✓ 클러스터링 완료: {len(industry_profile)} 산업 → {self.n_clusters} 클러스터\n")
        
        return self
    
    def _compute_cluster_profiles(
        self,
        industry_profile: pd.DataFrame,
        risk_col: str,
        return_col: str
    ):
        """클러스터별 프로파일 계산"""
        self.cluster_profiles = industry_profile.groupby('cluster').agg({
            risk_col: ['mean', 'std', 'min', 'max'],
            return_col: ['mean', 'std', 'min', 'max'],
            'Industry': 'count'
        }).round(4)
        
        self.cluster_profiles.columns = ['_'.join(col).strip() for col in self.cluster_profiles.columns.values]
        self.cluster_profiles.rename(columns={'Industry_count': 'n_industries'}, inplace=True)
    
    def _label_clusters(
        self,
        industry_profile: pd.DataFrame,
        risk_col: str,
        return_col: str
    ) -> pd.DataFrame:
        """
        클러스터 레이블링: Risk-Return 중앙값 기준
        
        0: High Risk - High Return
        1: High Risk - Low Return
        2: Low Risk - High Return (BEST)
        3: Low Risk - Low Return
        """
        risk_median = industry_profile[risk_col].median()
        return_median = industry_profile[return_col].median()
        
        cluster_means = industry_profile.groupby('cluster').agg({
            risk_col: 'mean',
            return_col: 'mean'
        })
        
        cluster_mapping = {}
        for cluster_id in cluster_means.index:
            risk = cluster_means.loc[cluster_id, risk_col]
            ret = cluster_means.loc[cluster_id, return_col]
            
            if risk >= risk_median and ret >= return_median:
                label = 0  # High-High
                desc = 'High Risk - High Return'
            elif risk >= risk_median and ret < return_median:
                label = 1  # High-Low
                desc = 'High Risk - Low Return'
            elif risk < risk_median and ret >= return_median:
                label = 2  # Low-High (BEST)
                desc = 'Low Risk - High Return'
            else:
                label = 3  # Low-Low
                desc = 'Low Risk - Low Return'
            
            cluster_mapping[cluster_id] = (label, desc)
        
        industry_profile['risk_profile'] = industry_profile['cluster'].map(
            lambda x: cluster_mapping[x][0]
        )
        industry_profile['profile_desc'] = industry_profile['cluster'].map(
            lambda x: cluster_mapping[x][1]
        )
        
        return industry_profile
    
    def get_cluster_summary(self) -> pd.DataFrame:
        """클러스터별 요약 통계"""
        if self.cluster_profiles is None:
            raise ValueError("fit()을 먼저 실행하세요")
        
        print(f"\n{'='*80}")
        print("클러스터별 프로파일")
        print(f"{'='*80}")
        print(self.cluster_profiles.to_string())
        print(f"{'='*80}\n")
        
        return self.cluster_profiles
    
    def get_industries_by_profile(
        self,
        profile: int,
        top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        특정 리스크 프로파일 산업 조회
        
        Parameters:
        -----------
        profile : int
            0: High-High, 1: High-Low, 2: Low-High, 3: Low-Low
        top_n : int, optional
            상위 N개 (Return_3M 기준)
        
        Returns:
        --------
        DataFrame : 해당 프로파일 산업 목록
        """
        if self.industry_profile is None:
            raise ValueError("fit()을 먼저 실행하세요")
        
        result = self.industry_profile[
            self.industry_profile['risk_profile'] == profile
        ].copy()
        
        if top_n:
            result = result.nlargest(top_n, 'Return_3M')
        
        return result.sort_values(by='Return_3M', ascending=False)


def build_industry_portfolio(
    industry_clusterer: IndustryClusterer,
    profile_weights: Dict[int, float] = None,
    top_n_per_profile: int = 3
) -> pd.DataFrame:
    """
    다각화 산업 포트폴리오 구성
    
    Parameters:
    -----------
    industry_clusterer : IndustryClusterer
        fit() 완료된 클러스터러
    profile_weights : Dict[int, float]
        프로파일별 가중치 (default: {2: 0.5, 0: 0.3, 3: 0.2})
        2: Low-High (50%), 0: High-High (30%), 3: Low-Low (20%)
    top_n_per_profile : int
        프로파일별 상위 N개 선택
    
    Returns:
    --------
    DataFrame : 포트폴리오 구성 (Industry, Sector, Weight, Risk_Profile)
    
    Example:
    --------
    >>> portfolio = build_industry_portfolio(
    ...     clusterer,
    ...     profile_weights={2: 0.5, 0: 0.3, 3: 0.2},
    ...     top_n_per_profile=3
    ... )
    """
    if profile_weights is None:
        profile_weights = {
            2: 0.50,  # Low Risk - High Return (BEST)
            0: 0.30,  # High Risk - High Return
            3: 0.20   # Low Risk - Low Return (STABLE)
        }
    
    print(f"\n{'='*80}")
    print("다각화 포트폴리오 구성")
    print(f"{'='*80}")
    print("프로파일별 가중치:")
    profile_names = {
        0: 'High Risk - High Return',
        1: 'High Risk - Low Return',
        2: 'Low Risk - High Return',
        3: 'Low Risk - Low Return'
    }
    for profile, weight in profile_weights.items():
        print(f"  [{profile}] {profile_names[profile]:30s} : {weight:5.1%}")
    print(f"{'='*80}\n")
    
    portfolio_components = []
    
    for profile, weight in profile_weights.items():
        # 프로파일별 상위 N개 산업 선택
        industries = industry_clusterer.get_industries_by_profile(profile, top_n=top_n_per_profile)
        
        if industries.empty:
            print(f"⚠️  프로파일 {profile} 산업 없음")
            continue
        
        # 프로파일 내 균등 배분
        n_selected = len(industries)
        weight_per_industry = weight / n_selected
        
        for _, row in industries.iterrows():
            portfolio_components.append({
                'Industry': row['Industry'],
                'Sector': row['Sector'],
                'Risk_Profile': profile,
                'Profile_Desc': row['profile_desc'],
                'Weight': weight_per_industry,
                'Expected_Return': row['Return_3M'],
                'Volatility': row['Volatility_20d']
            })
        
        print(f"프로파일 {profile} ({profile_names[profile]}):")
        print(f"  선택된 산업: {n_selected}개")
        for idx, row in industries.iterrows():
            print(f"    - {row['Industry']:40s} ({row['Sector']:20s}) | Return: {row['Return_3M']:7.2%} | Vol: {row['Volatility_20d']:6.2%}")
        print()
    
    portfolio_df = pd.DataFrame(portfolio_components)
    
    # 포트폴리오 통계
    print(f"{'='*80}")
    print("포트폴리오 요약")
    print(f"{'='*80}")
    print(f"  총 산업 수: {len(portfolio_df)}")
    print(f"  총 섹터 수: {portfolio_df['Sector'].nunique()}")
    print(f"  예상 수익률: {(portfolio_df['Expected_Return'] * portfolio_df['Weight']).sum():7.2%}")
    print(f"  포트폴리오 변동성: {(portfolio_df['Volatility'] * portfolio_df['Weight']).sum():6.2%}")
    print(f"{'='*80}\n")
    
    return portfolio_df


def main_industry_analysis(
    df_sector_year: pd.DataFrame,
    stock_data_path: str,
    year: int,
    top_n_sectors: int = 5,
    n_clusters: int = 4,
    profile_weights: Dict[int, float] = None,
    top_n_per_profile: int = 3
) -> Dict[str, pd.DataFrame]:
    """
    산업 분석 전체 파이프라인
    
    Sector LTR → TOP-N Sectors → Industry Extraction → Clustering → Portfolio
    
    Parameters:
    -----------
    df_sector_year : DataFrame
        LTR 결과 (test_year, Sector, ltr_score_raw, actual_return)
    stock_data_path : str
        stock_features_clean.csv 경로
    year : int
        분석 연도
    top_n_sectors : int
        TOP-N 섹터 (default: 5)
    n_clusters : int
        클러스터 수 (default: 4)
    profile_weights : Dict[int, float]
        포트폴리오 가중치
    top_n_per_profile : int
        프로파일별 산업 수
    
    Returns:
    --------
    Dict:
        'top_sectors': List[str]
        'industry_data': DataFrame
        'clusterer': IndustryClusterer
        'portfolio': DataFrame
    
    Example:
    --------
    >>> results = main_industry_analysis(
    ...     df_sector_year,
    ...     'Data_set/stock_features_clean.csv',
    ...     year=2025,
    ...     top_n_sectors=5
    ... )
    >>> portfolio = results['portfolio']
    """
    print(f"\n{'#'*80}")
    print(f"# 산업 분석 파이프라인 - {year}년")
    print(f"{'#'*80}\n")
    
    # Step 1: TOP-N 섹터 선택
    top_sectors = select_top_sectors(df_sector_year, year, top_n=top_n_sectors)
    
    if not top_sectors:
        print("❌ TOP 섹터 선택 실패")
        return {}
    
    # Step 2: 산업 데이터 추출
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    industry_data = extract_industry_data(
        stock_data_path,
        top_sectors,
        start_date=start_date,
        end_date=end_date
    )
    
    # Step 3: 리스크-수익률 클러스터링
    clusterer = IndustryClusterer(n_clusters=n_clusters)
    clusterer.fit(industry_data, window='all')
    clusterer.get_cluster_summary()
    
    # Step 4: 포트폴리오 구성
    portfolio = build_industry_portfolio(
        clusterer,
        profile_weights=profile_weights,
        top_n_per_profile=top_n_per_profile
    )
    
    print(f"\n{'#'*80}")
    print(f"# 분석 완료")
    print(f"{'#'*80}\n")
    
    return {
        'top_sectors': top_sectors,
        'industry_data': industry_data,
        'clusterer': clusterer,
        'portfolio': portfolio
    }


if __name__ == '__main__':
    print("Industry Analysis Module")
    print("=" * 80)
    print(__doc__)

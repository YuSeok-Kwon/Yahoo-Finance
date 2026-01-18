"""
인더스트리 클러스터링 모듈
KMeans를 사용하여 상위 섹터 내 산업을 리스크-수익률 기반으로 클러스터링합니다
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class IndustryClusterer:
    """
    리스크-수익률 프로파일 기반 산업 KMeans 클러스터링
    
    클러스터:
    - Cluster 2: 고수익, 고위험 (공격형)
    - Cluster 1: 최고 리스크-보상 (에이스 클러스터)
    - Cluster 0: 중립/안정
    - Cluster 3: 저수익, 고낙폭 (가치 함정)
    """
    
    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        """
        인더스트리 클러스터러 초기화

        Parameters:
        -----------
        n_clusters : int
            클러스터 개수 (기본값: 4)
        random_state : int
            재현성을 위한 랜덤 시드
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = None
        self.feature_cols = ['Return_Period', 'Volatility_20d', 'MDD']
        self._cluster_id_mapped = False
    
    def extract_industry_features(
        self,
        df: pd.DataFrame,
        selected_sectors: List[str],
        lookback_days: int = 90,
        end_date: pd.Timestamp = None
    ) -> pd.DataFrame:
        """
        주식 데이터에서 산업별 특성 추출
        
        Parameters:
        -----------
        df : pd.DataFrame
            Date, Company, Sector, Industry, Close, Daily_Return 컬럼이 있는 전체 데이터셋
        selected_sectors : List[str]
            분석할 상위 섹터
        lookback_days : int
            특성 계산 기간 (일수)
        end_date : pd.Timestamp
            특성 계산 종료일 (기본값: df의 최신 날짜)
        
        Returns:
        --------
        pd.DataFrame : 산업 특성 데이터프레임
            Industry, Sector, Return_3M, Volatility_20d, MDD, Sharpe_Ratio, Num_Companies 컬럼 포함
        """
        if end_date is None:
            end_date = df['Date'].max()
        
        start_date = end_date - pd.Timedelta(days=lookback_days)
        
        df_period = df[
            (df['Date'] >= start_date) & 
            (df['Date'] <= end_date) &
            (df['Sector'].isin(selected_sectors))
        ].copy()
        
        print(f"\n산업 특성 추출 중...")
        print(f"기간: {start_date.date()} ~ {end_date.date()}")
        print(f"섹터: {selected_sectors}")
        
        industries = df_period['Industry'].unique()
        print(f"산업 수: {len(industries)}")
        
        industry_features = []
        
        for industry in industries:
            ind_data = df_period[df_period['Industry'] == industry].copy()
            
            if len(ind_data) < 20:
                continue
            
            ind_data = ind_data.sort_values('Date')
            
            sector = ind_data['Sector'].iloc[0]
            num_companies = ind_data['Company'].nunique()
            
            close_by_date = ind_data.groupby('Date')['Close'].mean().sort_index()
            
            if len(close_by_date) < 20:
                continue
            
            daily_returns = close_by_date.pct_change().dropna()
            
            if len(daily_returns) < 10:
                continue
            
            return_3m = (close_by_date.iloc[-1] / close_by_date.iloc[0]) - 1
            
            volatility_20d = daily_returns.tail(20).std() * np.sqrt(252) if len(daily_returns) >= 20 else daily_returns.std() * np.sqrt(252)
            
            cum_max = close_by_date.cummax()
            drawdown = (close_by_date / cum_max) - 1
            mdd = drawdown.min()
            
            mean_return_annual = daily_returns.mean() * 252
            sharpe_ratio = (mean_return_annual - 0.02) / (volatility_20d + 1e-8) if volatility_20d > 0 else 0
            
            industry_features.append({
                'Industry': industry,
                'Sector': sector,
                'Return_Period': return_3m,
                'Volatility_20d': volatility_20d,
                'MDD': mdd,
                'Sharpe_Ratio': sharpe_ratio,
                'Num_Companies': num_companies
            })
        
        df_industry = pd.DataFrame(industry_features)
        
        print(f"{len(df_industry)}개 산업의 특성 추출 완료")
        
        return df_industry
    
    def fit_predict(self, df_industry: pd.DataFrame) -> pd.DataFrame:
        """
        KMeans 학습 및 클러스터 예측
        
        Parameters:
        -----------
        df_industry : pd.DataFrame
            extract_industry_features에서 추출한 산업 특성
        
        Returns:
        --------
        pd.DataFrame : 'cluster' 컬럼이 추가된 df_industry
        """
        if len(df_industry) < self.n_clusters:
            print(f"경고: {len(df_industry)}개 산업만 있음, {self.n_clusters}개 클러스터보다 적음")
            df_industry['cluster'] = 0
            return df_industry
        
        X = df_industry[self.feature_cols].copy()
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=20
        )
        
        df_industry['cluster'] = self.kmeans.fit_predict(X_scaled)
        
        print(f"\nKMeans 클러스터링 완료")
        print(f"클러스터 수: {df_industry['cluster'].nunique()}")
        print(f"클러스터 분포: {df_industry['cluster'].value_counts().sort_index().to_dict()}")
        
        return df_industry
    
    def _label_from_metrics(
        self,
        return_val: float,
        vol_val: float,
        mdd_val: float,
        sharpe_val: float
    ) -> Dict[str, str]:
        if return_val > 0.05 and vol_val > 0.05:
            return {
                'label': "공격형(고수익·고위험)",
                'description': "공격적 성장, 높은 변동성, 모멘텀 트레이딩에 적합"
            }
        if sharpe_val > 0.4 and mdd_val > -0.20:
            return {
                'label': "최적형(최적 리스크-보상)",
                'description': "최적의 위험 조정 수익률, 핵심 포트폴리오 보유 자산"
            }
        if abs(return_val) < 0.05 and vol_val < 0.05:
            return {
                'label': "안정형(중립/저변동)",
                'description': "평균 수준 성과, 포트폴리오 완충재, 리밸런싱 후보"
            }
        return {
            'label': "주의형(가치 함정)",
            'description': "낮은 수익률과 높은 낙폭 위험, 회피 또는 언더웨이트"
        }

    def _label_scores(self, cluster_profile: pd.DataFrame) -> Dict[str, pd.Series]:

        scores = {}
        n_clusters = len(cluster_profile)
        if n_clusters == 0:
            return scores

        def high_score(series: pd.Series) -> pd.Series:
            ranks = series.rank(ascending=False, method='first')
            return (n_clusters + 1) - ranks

        def low_score(series: pd.Series) -> pd.Series:
            ranks = series.rank(ascending=True, method='first')
            return (n_clusters + 1) - ranks

        scores["공격형(고수익·고위험)"] = high_score(cluster_profile['Return_Period']) + high_score(
            cluster_profile['Volatility_20d']
        )
        scores["최적 리스크-보상"] = high_score(cluster_profile['Sharpe_Ratio']) + high_score(
            cluster_profile['MDD']
        )
        scores["중립/안정"] = low_score(cluster_profile['Return_Period'].abs()) + low_score(
            cluster_profile['Volatility_20d']
        )
        scores["가치 함정"] = low_score(cluster_profile['Return_Period']) + low_score(
            cluster_profile['MDD']
        )
        return scores

    def remap_cluster_ids(
        self,
        df_industry: pd.DataFrame,
        cluster_profile: pd.DataFrame
    ) -> pd.DataFrame:
        """
        샤프 비율 구간 기반 클러스터 레이블링 (5개 클러스터)
        각 산업을 개별 지표 기준으로 직접 할당
        0=고수익·고위험, 1=강력매수, 2=안정형/중립, 3=저수익·고위험, 4=관망/주의
        """
        df_industry = df_industry.copy()

        # 각 산업을 개별적으로 샤프 비율 구간에 따라 클러스터 할당
        new_clusters = []
        for idx, row in df_industry.iterrows():
            sharpe = row['Sharpe_Ratio']
            vol = row['Volatility_20d']
            ret = row['Return_Period']

            # 샤프 비율 구간별 레이블 결정
            if sharpe >= 1.5 and vol < 0.40:
                # 강력 매수: 높은 샤프 + 관리 가능한 변동성 (40% 미만)
                cluster = 1  # 강력 매수
            elif (sharpe >= 1.0 and vol >= 0.30 and ret >= 0.12) or (sharpe >= 1.5 and vol >= 0.40):
                # 고수익·고위험: (샤프 1.0+ & 변동성 30%+ & 수익률 12%+) 또는 (샤프는 높지만 변동성이 40% 이상)
                cluster = 0  # 고수익·고위험
            elif 0.5 <= sharpe < 1.5 and vol < 0.25:
                # 안정형: 샤프 0.5~1.5 + 낮은 변동성
                cluster = 2  # 안정형/중립
            elif sharpe >= 0:
                # 나머지 양수 샤프: 변동성 대비 수익률이 낮거나 수익률 자체가 낮음
                cluster = 3  # 저수익·고위험
            else:  # sharpe < 0
                cluster = 4  # 관망/주의

            new_clusters.append(cluster)

        df_industry['cluster'] = new_clusters

        # 새로운 cluster_profile 재계산
        cluster_profile = df_industry.groupby('cluster')[self.feature_cols].mean()
        cluster_profile.columns = ['Return_Period', 'Volatility_20d', 'MDD']

        # Sharpe ratio 재계산
        sharpe_ratios = []
        for cluster_id in cluster_profile.index:
            ret = cluster_profile.at[cluster_id, 'Return_Period']
            vol = cluster_profile.at[cluster_id, 'Volatility_20d']
            sharpe = ret / vol if vol != 0 else 0
            sharpe_ratios.append(sharpe)
        cluster_profile['Sharpe_Ratio'] = sharpe_ratios

        print("\n클러스터 재매핑 (샤프 비율 구간 기반):")
        for cluster_id in sorted(df_industry['cluster'].unique()):
            if cluster_id in cluster_profile.index:
                sharpe_val = cluster_profile.at[cluster_id, 'Sharpe_Ratio']
                return_val = cluster_profile.at[cluster_id, 'Return_Period']
                vol_val = cluster_profile.at[cluster_id, 'Volatility_20d']
                mdd_val = cluster_profile.at[cluster_id, 'MDD']

                # 레이블 결정 기준 표시
                if cluster_id == 1:
                    criteria = f"Sharpe >= 1.5 && Vol < 40%"
                elif cluster_id == 0:
                    criteria = f"고수익·고위험 (Sharpe >= 1.0 || 고변동성)"
                elif cluster_id == 2:
                    criteria = f"0.5 <= Sharpe < 1.5 && Vol < 25%"
                elif cluster_id == 3:
                    criteria = f"Sharpe >= 0 (저수익)"
                else:  # 4
                    criteria = f"Sharpe < 0"

                print(f"  Cluster {cluster_id}: Sharpe={sharpe_val:6.2f} | Return={return_val:7.2%} | Vol={vol_val:6.2%} | MDD={mdd_val:7.2%} ({criteria})")

        return df_industry, cluster_profile

    def profile_clusters(self, df_industry: pd.DataFrame) -> pd.DataFrame:
        """
        클러스터 프로파일 생성 (클러스터별 특성 평균)

        Parameters:
        -----------
        df_industry : pd.DataFrame
            'cluster' 컬럼이 있는 산업 특성 데이터

        Returns:
        --------
        pd.DataFrame : 클러스터 프로파일
        """
        cluster_profile = (
            df_industry
            .groupby('cluster')[self.feature_cols + ['Sharpe_Ratio']]
            .mean()
            .round(4)
        )

        # cluster_id 순서대로 정렬 (0, 1, 2, 3)
        cluster_profile = cluster_profile.sort_index()

        print("\n" + "="*80)
        print("클러스터 프로파일")
        print("="*80)
        print(cluster_profile)
        print("="*80)

        return cluster_profile

    def _label_from_metrics(
        self,
        return_val: float,
        vol_val: float,
        mdd_val: float,
        sharpe_val: float
    ) -> Dict[str, str]:
        if return_val > 0.05 and vol_val > 0.05:
            return {
                'label': '공격형(고수익·고위험)',
                'description': '공격적 성장, 높은 변동성, 모멘텀 트레이딩에 적합'
            }
        if sharpe_val > 0.4 and mdd_val > -0.20:
            return {
                'label': '최적형(최적 리스크-보상)',
                'description': '최적의 위험 조정 수익률, 핵심 포트폴리오 보유 자산'
            }
        if abs(return_val) < 0.05 and vol_val < 0.05:
            return {
                'label': '안정형(중립/저변동)',
                'description': '평균 수준 성과, 포트폴리오 완충재, 리밸런싱 후보'
            }
        return {
            'label': '주의형(가치 함정)',
            'description': '낮은 수익률과 높은 낙폭 위험, 회피 또는 언더웨이트'
        }

    def _remap_clusters_by_label(
        self,
        df_industry: pd.DataFrame,
        cluster_profile: pd.DataFrame
    ) -> pd.DataFrame:
        label_priority = [
            '공격형(고수익·고위험)',
            '최적형(최적 리스크-보상)',
            '안정형(중립/저변동)',
            '주의형(가치 함정)'
        ]
        label_order = {label: index for index, label in enumerate(label_priority)}

        labeled_clusters = []
        for cluster_id in cluster_profile.index:
            profile = cluster_profile.loc[cluster_id]
            label_info = self._label_from_metrics(
                float(profile['Return_Period']),
                float(profile['Volatility_20d']),
                float(profile['MDD']),
                float(profile['Sharpe_Ratio'])
            )
            labeled_clusters.append({
                'cluster_id': cluster_id,
                'label': label_info['label'],
                'return_period': float(profile['Return_Period']),
                'volatility_20d': float(profile['Volatility_20d'])
            })

        labeled_clusters.sort(
            key=lambda row: (
                label_order.get(row['label'], len(label_order)),
                -row['return_period'],
                -row['volatility_20d']
            )
        )

        target_ids = {
            '공격형(고수익·고위험)': 2,
            '최적형(최적 리스크-보상)': 1,
            '안정형(중립/저변동)': 0,
            '주의형(가치 함정)': 3
        }

        remap = {
            row['cluster_id']: target_ids.get(row['label'], row['cluster_id'])
            for row in labeled_clusters
        }
        df_industry['cluster'] = df_industry['cluster'].map(remap)
        cluster_profile = cluster_profile.rename(index=remap).sort_index()

        return df_industry, cluster_profile
    
    def interpret_clusters(
        self,
        df_industry: pd.DataFrame,
        cluster_profile: pd.DataFrame
    ) -> Dict[str, Dict]:
        """
        클러스터 해석 및 레이블 할당
        수익률 순서에 따라 고정된 레이블 할당 (0=공격형, 1=최적형, 2=안정형, 3=주의형)

        Parameters:
        -----------
        df_industry : pd.DataFrame
            'cluster' 컬럼이 있는 산업 특성 데이터
        cluster_profile : pd.DataFrame
            profile_clusters에서 생성한 클러스터 프로파일

        Returns:
        --------
        Dict : 클러스터 해석 정보
            {
                'cluster_0': {
                    'label': '공격형(고수익·고위험)',
                    'description': '...',
                    'industries': [...]
                },
                ...
            }
        """
        # 5개 클러스터 레이블 매핑
        label_mapping = {
            0: {
                'label': '고수익·고위험',
                'description': '높은 변동성 + 양의 샤프 비율, 공격적 투자 성향에 적합'
            },
            1: {
                'label': '강력 매수',
                'description': '최고 샤프 비율, 최적의 위험 조정 수익률, 핵심 포트폴리오 자산'
            },
            2: {
                'label': '안정형/중립',
                'description': '낮은 변동성과 안정적 성과, 포트폴리오 완충재'
            },
            3: {
                'label': '저수익·고위험',
                'description': '높은 변동성 + 낮은/음의 샤프 비율, 투자 주의 필요'
            },
            4: {
                'label': '관망/주의',
                'description': '최저 샤프 비율, 높은 리스크 대비 낮은 수익, 투자 회피 권장'
            }
        }

        interpretations = {}

        for cluster_id in sorted(df_industry['cluster'].unique()):
            profile = cluster_profile.loc[cluster_id]
            industries = df_industry[df_industry['cluster'] == cluster_id]['Industry'].tolist()

            return_val = profile['Return_Period']
            vol_val = profile['Volatility_20d']
            mdd_val = profile['MDD']
            sharpe_val = profile['Sharpe_Ratio']

            # 수익률 순서에 따른 고정 레이블 사용
            label_info = label_mapping.get(cluster_id, label_mapping[3])

            interpretations[f'cluster_{cluster_id}'] = {
                'label': label_info['label'],
                'description': label_info['description'],
                'return_period': float(return_val),
                'volatility_20d': float(vol_val),
                'mdd': float(mdd_val),
                'sharpe_ratio': float(sharpe_val),
                'num_industries': len(industries),
                'industries': industries
            }

        print("\n" + "="*80)
        print("클러스터 해석")
        print("="*80)
        for cluster_id in sorted(interpretations.keys()):
            info = interpretations[cluster_id]
            print(f"\n{cluster_id.upper()}: {info['label']}")
            print(f"  샤프: {info['sharpe_ratio']:.2f} | 수익률: {info['return_period']:.2%} | 변동성: {info['volatility_20d']:.2%} | MDD: {info['mdd']:.2%}")
            print(f"  산업 ({info['num_industries']}개): {', '.join(info['industries'][:5])}...")
            print(f"  설명: {info['description']}")
        print("="*80)

        return interpretations


    
    def run_full_pipeline(
        self,
        df: pd.DataFrame,
        selected_sectors: List[str],
        lookback_days: int = 90,
        end_date: pd.Timestamp = None
    ) -> Dict:
        """
        전체 산업 클러스터링 파이프라인 실행
        
        Parameters:
        -----------
        df : pd.DataFrame
            전체 주식 데이터셋
        selected_sectors : List[str]
            멀티 호라이즌 예측에서 선정된 상위 섹터
        lookback_days : int
            특성 계산 윈도우
        end_date : pd.Timestamp
            분석 종료일
        
        Returns:
        --------
        Dict : 완전한 클러스터링 결과
            {
                'industry_features': df_industry,
                'cluster_profile': cluster_profile,
                'interpretations': interpretations,
                'by_cluster': {
                    'cluster_0': df_cluster_0,
                    'cluster_1': df_cluster_1,
                    ...
                }
            }
        """
        df_industry = self.extract_industry_features(
            df=df,
            selected_sectors=selected_sectors,
            lookback_days=lookback_days,
            end_date=end_date
        )
        
        df_industry = self.fit_predict(df_industry)

        cluster_profile = self.profile_clusters(df_industry)

        # 클러스터 ID를 수익률 기준으로 재매핑
        df_industry, cluster_profile = self.remap_cluster_ids(df_industry, cluster_profile)

        interpretations = self.interpret_clusters(df_industry, cluster_profile)
        
        by_cluster = {}
        for cluster_id in sorted(df_industry['cluster'].unique()):
            by_cluster[f'cluster_{cluster_id}'] = df_industry[
                df_industry['cluster'] == cluster_id
            ].sort_values('Return_Period', ascending=False)
        
        return {
            'industry_features': df_industry,
            'cluster_profile': cluster_profile,
            'interpretations': interpretations,
            'by_cluster': by_cluster
        }

"""
인더스트리 클러스터링 모듈 (하이브리드: KMeans + 센트로이드 규칙 레이블링)
KMeans가 그룹 경계를 결정하고, 센트로이드 지표 기반 규칙이 레이블만 부여합니다.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# 기본 임계값 상수
DEFAULT_SHARPE_THRESHOLDS = {
    'strong_buy': 1.5,
    'aggressive': 1.0,
    'stable': 0.5,
    'warning': 0.0,
}

DEFAULT_VOL_THRESHOLDS = {
    'strong_buy_max': 0.40,
    'aggressive_min': 0.30,
    'stable_max': 0.25,
}


class IndustryClusterer:
    """
    리스크-수익률 프로파일 기반 산업 하이브리드 클러스터링

    KMeans(k=5)로 그룹 경계를 결정한 뒤, 각 센트로이드의
    Sharpe/Vol/Return 지표를 기반으로 의미 있는 레이블을 할당합니다.

    클러스터 (5개):
    - Cluster 0: 고수익·고위험 — Sharpe >= 1.0, Vol >= 30%, Return >= 12%
    - Cluster 1: 강력 매수 — Sharpe >= 1.5, Vol < 40%
    - Cluster 2: 안정형/중립 — 0.5 <= Sharpe < 1.5, Vol < 25%
    - Cluster 3: 저수익·고위험 — Sharpe >= 0 (나머지)
    - Cluster 4: 관망/주의 — Sharpe < 0
    """

    def __init__(
        self,
        n_clusters: int = 5,
        random_state: int = 42,
        sharpe_thresholds: Optional[Dict[str, float]] = None,
        vol_thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        인더스트리 클러스터러 초기화

        Parameters:
        -----------
        n_clusters : int
            클러스터 개수 (기본값: 5)
        random_state : int
            재현성을 위한 랜덤 시드
        sharpe_thresholds : dict, optional
            Sharpe 임계값 (strong_buy, aggressive, stable, warning)
        vol_thresholds : dict, optional
            변동성 임계값 (strong_buy_max, aggressive_min, stable_max)
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = None
        self.feature_cols = ['Return_Period', 'Volatility_20d', 'MDD']
        self._cluster_id_mapped = False

        self.sharpe_thresholds = {**DEFAULT_SHARPE_THRESHOLDS, **(sharpe_thresholds or {})}
        self.vol_thresholds = {**DEFAULT_VOL_THRESHOLDS, **(vol_thresholds or {})}

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

    def _classify_centroid(self, sharpe: float, vol: float, ret: float) -> int:
        """
        센트로이드 지표로 레이블(0-4) 결정

        Parameters:
        -----------
        sharpe : float
            클러스터 센트로이드의 평균 Sharpe Ratio
        vol : float
            클러스터 센트로이드의 평균 Volatility
        ret : float
            클러스터 센트로이드의 평균 Return

        Returns:
        --------
        int : 클러스터 레이블 (0-4)
        """
        st = self.sharpe_thresholds
        vt = self.vol_thresholds

        if sharpe >= st['strong_buy'] and vol < vt['strong_buy_max']:
            return 1  # 강력 매수
        if (sharpe >= st['aggressive'] and vol >= vt['aggressive_min'] and ret >= 0.12) or \
           (sharpe >= st['strong_buy'] and vol >= vt['strong_buy_max']):
            return 0  # 고수익·고위험
        if st['stable'] <= sharpe < st['strong_buy'] and vol < vt['stable_max']:
            return 2  # 안정형/중립
        if sharpe >= st['warning']:
            return 3  # 저수익·고위험
        return 4  # 관망/주의

    def _resolve_label_collisions(
        self,
        centroid_labels: Dict[int, int],
        cluster_profile: pd.DataFrame,
    ) -> Dict[int, int]:
        """
        2개 이상의 센트로이드가 같은 레이블을 받았을 때 Sharpe 순위로 분화

        Parameters:
        -----------
        centroid_labels : dict
            {kmeans_cluster_id: proposed_label}
        cluster_profile : pd.DataFrame
            클러스터 프로파일 (Sharpe_Ratio 컬럼 필요)

        Returns:
        --------
        dict : 충돌 없는 {kmeans_cluster_id: final_label}
        """
        # 레이블 → 해당 레이블을 받은 클러스터 ID 목록
        from collections import defaultdict
        label_to_clusters: Dict[int, List[int]] = defaultdict(list)
        for cid, label in centroid_labels.items():
            label_to_clusters[label].append(cid)

        used_labels = set()
        final = {}

        # 충돌 없는 레이블 먼저 확정
        for label, cids in sorted(label_to_clusters.items()):
            if len(cids) == 1:
                final[cids[0]] = label
                used_labels.add(label)

        # 충돌 있는 레이블 처리: Sharpe 가장 높은 클러스터에 원래 레이블 부여,
        # 나머지는 비어있는 레이블 중 Sharpe 순위에 맞게 할당
        all_labels = set(range(5))
        for label, cids in sorted(label_to_clusters.items()):
            if len(cids) <= 1:
                continue
            # Sharpe 내림차순 정렬
            sorted_cids = sorted(
                cids,
                key=lambda c: cluster_profile.at[c, 'Sharpe_Ratio'],
                reverse=True,
            )
            # 가장 Sharpe 높은 클러스터에 원래 레이블 부여
            final[sorted_cids[0]] = label
            used_labels.add(label)
            # 나머지 클러스터에 빈 레이블 할당
            for cid in sorted_cids[1:]:
                available = sorted(all_labels - used_labels)
                if available:
                    chosen = available[0]
                    final[cid] = chosen
                    used_labels.add(chosen)
                else:
                    final[cid] = label  # fallback

        return final

    def remap_cluster_ids(
        self,
        df_industry: pd.DataFrame,
        cluster_profile: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        센트로이드 기반 클러스터 레이블링 (하이브리드 방식)

        KMeans가 결정한 그룹 멤버십은 유지하면서, 각 클러스터 센트로이드의
        Sharpe/Vol/Return 지표를 보고 의미 있는 레이블(0-4)을 할당합니다.

        0=고수익·고위험, 1=강력매수, 2=안정형/중립, 3=저수익·고위험, 4=관망/주의
        """
        df_industry = df_industry.copy()

        # 각 KMeans 센트로이드에 대해 레이블 결정
        centroid_labels = {}
        for cluster_id in cluster_profile.index:
            sharpe = float(cluster_profile.at[cluster_id, 'Sharpe_Ratio'])
            vol = float(cluster_profile.at[cluster_id, 'Volatility_20d'])
            ret = float(cluster_profile.at[cluster_id, 'Return_Period'])
            centroid_labels[cluster_id] = self._classify_centroid(sharpe, vol, ret)

        # 레이블 충돌 해결
        final_labels = self._resolve_label_collisions(centroid_labels, cluster_profile)

        # KMeans 클러스터 ID → 새 레이블로 매핑 (그룹 멤버십 유지)
        df_industry['cluster'] = df_industry['cluster'].map(final_labels)

        # 새로운 cluster_profile 재계산
        cluster_profile = (
            df_industry
            .groupby('cluster')[self.feature_cols + ['Sharpe_Ratio']]
            .mean()
            .round(4)
        )

        print("\n클러스터 재매핑 (센트로이드 기반 하이브리드):")
        for cluster_id in sorted(df_industry['cluster'].unique()):
            if cluster_id in cluster_profile.index:
                sharpe_val = cluster_profile.at[cluster_id, 'Sharpe_Ratio']
                return_val = cluster_profile.at[cluster_id, 'Return_Period']
                vol_val = cluster_profile.at[cluster_id, 'Volatility_20d']
                mdd_val = cluster_profile.at[cluster_id, 'MDD']

                label_names = {
                    0: "고수익·고위험",
                    1: "강력 매수",
                    2: "안정형/중립",
                    3: "저수익·고위험",
                    4: "관망/주의",
                }
                label = label_names.get(cluster_id, "미분류")
                print(f"  Cluster {cluster_id} ({label}): Sharpe={sharpe_val:6.4f} | Return={return_val:7.2%} | Vol={vol_val:6.2%} | MDD={mdd_val:7.2%}")

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

        # cluster_id 순서대로 정렬 (0, 1, 2, 3, 4)
        cluster_profile = cluster_profile.sort_index()

        print("\n" + "="*80)
        print("클러스터 프로파일")
        print("="*80)
        print(cluster_profile)
        print("="*80)

        return cluster_profile

    def interpret_clusters(
        self,
        df_industry: pd.DataFrame,
        cluster_profile: pd.DataFrame
    ) -> Dict[str, Dict]:
        """
        클러스터 해석 및 레이블 할당
        센트로이드 기반으로 할당된 클러스터 ID에 따라 고정 레이블 사용
        (0=고수익·고위험, 1=강력매수, 2=안정형/중립, 3=저수익·고위험, 4=관망/주의)

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
                    'label': '고수익·고위험',
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

        # 센트로이드 기반으로 클러스터 ID 재매핑
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

"""
투자자 성향별 추천 모듈

3단계 추천 프레임워크:
  1단계 (진입 여부): 클러스터 레이블 + 호라이즌 일치도 기반 필터
  2단계 (우선순위): 성향별 가중 composite score 기반 랭킹
  3단계 (성향 필터): 공격형/균형형/안정형별 MDD·Vol·Cluster 조건 적용
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from collections import Counter


# 투자자 프로파일 정의
INVESTOR_PROFILES = {
    '공격형': {
        'description': '높은 수익을 위해 높은 리스크를 감수하는 투자자',
        'allowed_clusters': [0, 1],         # 고수익·고위험 + 강력매수
        'mdd_threshold': None,              # MDD 제한 없음
        'vol_threshold': None,              # 변동성 제한 없음
        'min_sharpe': None,                 # Sharpe 하한 없음
        'min_horizon_agreement': 3,         # 7개 중 3개+ 호라이즌 추천
        'sector_diversity_min': 1,          # 최소 섹터 수
        'ranking_weights': {'sharpe': 0.3, 'return': 0.5, 'neg_vol': 0.2},
    },
    '균형형': {
        'description': '적절한 리스크로 안정적 수익을 추구하는 투자자',
        'allowed_clusters': [0, 1, 2],      # 고수익·고위험 + 강력매수 + 안정형
        'mdd_threshold': -0.20,             # MDD > -20%
        'vol_threshold': None,              # 변동성 제한 없음
        'min_sharpe': 0.0,                  # Sharpe > 0
        'min_horizon_agreement': 3,         # 7개 중 3개+ 호라이즌 추천
        'sector_diversity_min': 2,          # 최소 2개 섹터 분산
        'ranking_weights': {'sharpe': 0.5, 'return': 0.3, 'neg_mdd': 0.2},
    },
    '안정형': {
        'description': '원금 보존을 최우선으로 하는 보수적 투자자',
        'allowed_clusters': [1, 2],         # 강력매수 + 안정형
        'mdd_threshold': -0.15,             # MDD > -15%
        'vol_threshold': 0.25,              # 변동성 < 25%
        'min_sharpe': 0.5,                  # Sharpe > 0.5
        'min_horizon_agreement': 3,         # 7개 중 3개+ 호라이즌 추천
        'sector_diversity_min': 2,          # 최소 2개 섹터 분산
        'ranking_weights': {'sharpe': 0.3, 'return': 0.2, 'neg_mdd': 0.5},
    },
}


class InvestorProfiler:
    """
    투자자 성향별 산업 추천 시스템

    클러스터링 결과 + 호라이즌 일치도 + 재무 지표를 조합하여
    공격형/균형형/안정형 투자자에게 맞춤 추천을 제공합니다.
    """

    def __init__(
        self,
        profiles: Optional[Dict] = None,
        top_n: int = 5,
    ):
        """
        Parameters:
        -----------
        profiles : dict, optional
            투자자 프로파일 정의 (기본: INVESTOR_PROFILES)
        top_n : int
            프로파일당 최종 추천 산업 수 (기본: 5)
        """
        self.profiles = profiles or INVESTOR_PROFILES
        self.top_n = top_n

    def calculate_horizon_agreement(
        self,
        raw_horizon_results: Dict,
        strategy_group_results: Dict,
    ) -> Dict[str, int]:
        """
        산업이 몇 개 호라이즌에서 추천되었는지 계산 (호라이즌 일치도)

        Parameters:
        -----------
        raw_horizon_results : dict
            predict_all_horizons()의 7개 호라이즌 결과
            (없으면 strategy_group_results의 source_horizons로 추정)
        strategy_group_results : dict
            aggregate_by_strategy_groups()의 3개 전략그룹 결과

        Returns:
        --------
        dict : {sector: 추천 횟수} — 7개 호라이즌 기준
        """
        sector_votes = Counter()

        if raw_horizon_results:
            # 7개 호라이즌 결과가 있으면 직접 계산
            for result in raw_horizon_results.values():
                sector_votes.update(result['top_sectors'])
        else:
            # 전략그룹 결과에서 추정 (정확도 낮음)
            for group_result in strategy_group_results.values():
                n_horizons = len(group_result.get('source_horizons', []))
                for sector in group_result['top_sectors']:
                    sector_votes[sector] += n_horizons

        return dict(sector_votes)

    def _apply_entry_filter(
        self,
        df_industry: pd.DataFrame,
        horizon_agreement: Dict[str, int],
        min_agreement: int,
        allowed_clusters: List[int],
    ) -> pd.DataFrame:
        """
        1단계: 진입 여부 필터

        - Cluster 4(관망/주의)는 모든 프로파일에서 제외
        - allowed_clusters에 해당하는 클러스터만 통과
        - 최소 호라이즌 일치도 충족

        Parameters:
        -----------
        df_industry : pd.DataFrame
            클러스터링이 완료된 산업 특성 데이터
        horizon_agreement : dict
            섹터별 호라이즌 추천 횟수
        min_agreement : int
            최소 호라이즌 일치도
        allowed_clusters : list
            허용되는 클러스터 ID 목록

        Returns:
        --------
        pd.DataFrame : 필터링된 산업
        """
        df = df_industry.copy()

        # 클러스터 필터: Cluster 4는 항상 제외, allowed_clusters만 통과
        df = df[df['cluster'].isin(allowed_clusters)]

        # 호라이즌 일치도 필터: Sector 기준
        if horizon_agreement:
            qualifying_sectors = {
                sector for sector, votes in horizon_agreement.items()
                if votes >= min_agreement
            }
            df = df[df['Sector'].isin(qualifying_sectors)]

        return df

    def _apply_priority_ranking(
        self,
        df_filtered: pd.DataFrame,
        profile: Dict,
    ) -> pd.DataFrame:
        """
        2단계: 성향별 가중 composite score 기반 우선순위 랭킹

        z-score 정규화 후 프로파일별 가중합산으로 composite_score를 계산합니다.
        공격형은 수익률, 안정형은 MDD 방어력에 높은 가중치를 부여합니다.

        Parameters:
        -----------
        df_filtered : pd.DataFrame
            1단계 필터 통과 산업
        profile : dict
            투자자 프로파일 설정 (ranking_weights 포함)

        Returns:
        --------
        pd.DataFrame : composite_score 내림차순 정렬
        """
        weights = profile.get('ranking_weights')
        if not weights or len(df_filtered) == 0:
            return df_filtered.sort_values('Sharpe_Ratio', ascending=False)

        df = df_filtered.copy()

        # 지표 추출 (높을수록 좋은 방향으로 통일)
        # Vol: 낮을수록 좋으므로 부호 반전
        # MDD: 이미 음수이며, 0에 가까울수록(높을수록) 좋으므로 그대로 사용
        metric_map = {
            'sharpe': df['Sharpe_Ratio'],
            'return': df['Return_Period'],
            'neg_vol': -df['Volatility_20d'],
            'neg_mdd': df['MDD'],
        }

        # z-score 정규화 (2개 미만이면 raw값 사용)
        def _zscore(s: pd.Series) -> pd.Series:
            if len(s) < 2 or s.std() == 0:
                return s
            return (s - s.mean()) / s.std()

        composite = pd.Series(0.0, index=df.index)
        for key, weight in weights.items():
            if key in metric_map:
                composite += weight * _zscore(metric_map[key])

        df['composite_score'] = composite
        return df.sort_values('composite_score', ascending=False)

    def _apply_profile_filter(
        self,
        df_ranked: pd.DataFrame,
        profile: Dict,
    ) -> pd.DataFrame:
        """
        3단계: 투자자 성향별 필터

        MDD, Volatility, Sharpe 조건을 적용합니다.

        Parameters:
        -----------
        df_ranked : pd.DataFrame
            2단계 랭킹 결과
        profile : dict
            투자자 프로파일 설정

        Returns:
        --------
        pd.DataFrame : 성향 필터 적용 결과
        """
        df = df_ranked.copy()

        # MDD 필터: threshold보다 높아야 함 (MDD는 음수이므로 > 비교)
        if profile.get('mdd_threshold') is not None:
            df = df[df['MDD'] > profile['mdd_threshold']]

        # 변동성 필터
        if profile.get('vol_threshold') is not None:
            df = df[df['Volatility_20d'] < profile['vol_threshold']]

        # Sharpe 하한 필터
        if profile.get('min_sharpe') is not None:
            df = df[df['Sharpe_Ratio'] >= profile['min_sharpe']]

        return df

    def _ensure_sector_diversity(
        self,
        df_filtered: pd.DataFrame,
        min_sectors: int,
        top_n: int,
    ) -> pd.DataFrame:
        """
        섹터 분산도 보장: 추천 산업이 한 섹터에 몰리지 않도록 조정

        Parameters:
        -----------
        df_filtered : pd.DataFrame
            필터 통과 산업 (Sharpe 내림차순)
        min_sectors : int
            최소 섹터 수
        top_n : int
            최종 추천 수

        Returns:
        --------
        pd.DataFrame : 섹터 분산이 보장된 추천
        """
        if len(df_filtered) <= top_n:
            return df_filtered

        selected = []
        sectors_covered = set()
        remaining = df_filtered.copy()

        # 1차: 각 섹터에서 Sharpe 최고 산업 1개씩 선택
        for sector in remaining['Sector'].unique():
            if len(selected) >= top_n:
                break
            sector_best = remaining[remaining['Sector'] == sector].iloc[0]
            selected.append(sector_best)
            sectors_covered.add(sector)
            remaining = remaining.drop(sector_best.name)

        # 2차: 나머지 슬롯을 Sharpe 내림차순으로 채움
        for _, row in remaining.iterrows():
            if len(selected) >= top_n:
                break
            selected.append(row)

        if not selected:
            return df_filtered.head(top_n)

        result = pd.DataFrame(selected)
        sort_col = 'composite_score' if 'composite_score' in result.columns else 'Sharpe_Ratio'
        result = result.sort_values(sort_col, ascending=False)
        return result.head(top_n)

    def recommend(
        self,
        df_industry: pd.DataFrame,
        horizon_agreement: Dict[str, int] = None,
        profile_name: str = None,
    ) -> Dict:
        """
        투자자 성향별 산업 추천 실행

        Parameters:
        -----------
        df_industry : pd.DataFrame
            클러스터링이 완료된 산업 특성 데이터 (cluster 컬럼 필수)
        horizon_agreement : dict, optional
            섹터별 호라이즌 추천 횟수
        profile_name : str, optional
            특정 프로파일만 실행 (기본: 전체 프로파일)

        Returns:
        --------
        dict : 프로파일별 추천 결과
            {
                '공격형': {
                    'description': '...',
                    'recommendations': pd.DataFrame,
                    'summary': {
                        'n_industries': 5,
                        'n_sectors': 3,
                        'avg_sharpe': 1.23,
                        'avg_return': 0.15,
                        'avg_mdd': -0.12,
                    }
                },
                '균형형': {...},
                '안정형': {...},
            }
        """
        if horizon_agreement is None:
            horizon_agreement = {}

        profiles_to_run = (
            {profile_name: self.profiles[profile_name]}
            if profile_name and profile_name in self.profiles
            else self.profiles
        )

        results = {}

        for name, profile in profiles_to_run.items():
            # 1단계: 진입 여부
            df_entry = self._apply_entry_filter(
                df_industry=df_industry,
                horizon_agreement=horizon_agreement,
                min_agreement=profile['min_horizon_agreement'],
                allowed_clusters=profile['allowed_clusters'],
            )

            # 2단계: 우선순위 랭킹
            df_ranked = self._apply_priority_ranking(df_entry, profile)

            # 3단계: 성향 필터
            df_profiled = self._apply_profile_filter(df_ranked, profile)

            # 섹터 분산도 보장
            df_final = self._ensure_sector_diversity(
                df_profiled,
                min_sectors=profile['sector_diversity_min'],
                top_n=self.top_n,
            )

            # 요약 통계
            summary = {}
            if len(df_final) > 0:
                summary = {
                    'n_industries': len(df_final),
                    'n_sectors': df_final['Sector'].nunique(),
                    'avg_sharpe': round(float(df_final['Sharpe_Ratio'].mean()), 4),
                    'avg_return': round(float(df_final['Return_Period'].mean()), 4),
                    'avg_volatility': round(float(df_final['Volatility_20d'].mean()), 4),
                    'avg_mdd': round(float(df_final['MDD'].mean()), 4),
                }

            results[name] = {
                'description': profile['description'],
                'recommendations': df_final,
                'summary': summary,
                'filter_stats': {
                    'total_industries': len(df_industry),
                    'after_entry_filter': len(df_entry),
                    'after_profile_filter': len(df_profiled),
                    'final_recommendations': len(df_final),
                },
            }

        return results

    def format_report(self, results: Dict) -> str:
        """
        추천 결과를 텍스트 리포트로 포맷팅

        Parameters:
        -----------
        results : dict
            recommend()의 반환값

        Returns:
        --------
        str : 포맷팅된 리포트
        """
        lines = []
        lines.append("=" * 80)
        lines.append("투자자 성향별 산업 추천 리포트")
        lines.append("=" * 80)

        cluster_labels = {
            0: '고수익·고위험', 1: '강력매수', 2: '안정형/중립',
            3: '저수익·고위험', 4: '관망/주의',
        }

        for profile_name, result in results.items():
            lines.append(f"\n{'─' * 40}")
            lines.append(f"[{profile_name}] {result['description']}")
            lines.append(f"{'─' * 40}")

            stats = result['filter_stats']
            lines.append(
                f"  필터링: {stats['total_industries']}개 → "
                f"{stats['after_entry_filter']}개 (진입) → "
                f"{stats['after_profile_filter']}개 (성향) → "
                f"{stats['final_recommendations']}개 (최종)"
            )

            df_rec = result['recommendations']
            if len(df_rec) == 0:
                lines.append("  추천 산업 없음")
                continue

            summary = result['summary']
            lines.append(
                f"  평균 Sharpe: {summary['avg_sharpe']:.2f} | "
                f"수익률: {summary['avg_return']:.2%} | "
                f"변동성: {summary['avg_volatility']:.2%} | "
                f"MDD: {summary['avg_mdd']:.2%}"
            )
            lines.append(f"  섹터 분산: {summary['n_sectors']}개 섹터")

            lines.append(f"\n  {'순위':<4} {'산업':<30} {'섹터':<20} {'클러스터':<12} {'Sharpe':>8} {'수익률':>8} {'MDD':>8}")
            lines.append(f"  {'─' * 92}")

            for rank, (_, row) in enumerate(df_rec.iterrows(), 1):
                cluster_name = cluster_labels.get(int(row['cluster']), '미분류')
                lines.append(
                    f"  {rank:<4} {row['Industry']:<30} {row['Sector']:<20} "
                    f"{cluster_name:<12} {row['Sharpe_Ratio']:>8.2f} "
                    f"{row['Return_Period']:>7.2%} {row['MDD']:>7.2%}"
                )

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)

    def export_for_tableau(
        self,
        results: Dict,
        horizon_name: str = '',
    ) -> pd.DataFrame:
        """
        Tableau 연동용 단일 DataFrame 생성 (추천 산업만 포함, 레거시 호환)

        3개 프로파일 추천 결과를 Profile 컬럼으로 구분하여 단일 테이블로 합침.
        Tableau에서 매개변수로 Profile을 선택하면 해당 행만 필터링됨.

        Parameters:
        -----------
        results : dict
            recommend()의 반환값
        horizon_name : str
            전략그룹 이름 (단타/스윙/장기투자)

        Returns:
        --------
        pd.DataFrame : Tableau용 통합 테이블
            기존 컬럼 + Profile, Rank, Horizon, Cluster_Name, Profile_Description
        """
        cluster_labels = {
            0: '고수익·고위험', 1: '강력매수', 2: '안정형/중립',
            3: '저수익·고위험', 4: '관망/주의',
        }

        frames = []

        for profile_name, result in results.items():
            df_rec = result['recommendations'].copy()

            if len(df_rec) == 0:
                continue

            # 순위 부여 (Sharpe 내림차순 기준)
            df_rec = df_rec.sort_values('Sharpe_Ratio', ascending=False).reset_index(drop=True)
            df_rec['Rank'] = range(1, len(df_rec) + 1)

            # 메타 컬럼 추가
            df_rec['Profile'] = profile_name
            df_rec['Profile_Description'] = result['description']
            df_rec['Horizon'] = horizon_name
            df_rec['Cluster_Name'] = df_rec['cluster'].map(cluster_labels).fillna('미분류')

            frames.append(df_rec)

        if not frames:
            return pd.DataFrame()

        df_combined = pd.concat(frames, ignore_index=True)

        # 컬럼 순서 정리
        col_order = [
            'Horizon', 'Profile', 'Rank',
            'Industry', 'Sector', 'cluster', 'Cluster_Name',
            'Sharpe_Ratio', 'Return_Period', 'Volatility_20d', 'MDD',
            'Num_Companies', 'Profile_Description',
        ]
        existing_cols = [c for c in col_order if c in df_combined.columns]
        remaining_cols = [c for c in df_combined.columns if c not in col_order]
        df_combined = df_combined[existing_cols + remaining_cols]

        return df_combined

    def export_unified_for_tableau(
        self,
        df_industry: pd.DataFrame,
        results: Dict,
        horizon_name: str = '',
    ) -> pd.DataFrame:
        """
        기존 industry_features + 성향별 추천 정보를 통합한 단일 DataFrame 생성

        모든 산업이 포함되며, 추천 여부·순위·점수가 성향별 컬럼으로 추가됨.
        Tableau에서 단일 데이터 원본으로 사용하여 기존 차트(산점도, 랭킹)에
        매개변수 기반 추천 하이라이트를 자연스럽게 통합 가능.

        Parameters:
        -----------
        df_industry : pd.DataFrame
            클러스터링 완료된 산업 특성 데이터 (전체 산업 포함)
        results : dict
            recommend()의 반환값
        horizon_name : str
            전략그룹 이름 (단타/스윙/장기투자)

        Returns:
        --------
        pd.DataFrame : 통합 테이블
            기존 컬럼 + Cluster_Name + 성향별 (Is_추천, Rank, Score) 컬럼
        """
        CLUSTER_LABELS = {
            0: '고수익·고위험', 1: '강력매수', 2: '안정형/중립',
            3: '저수익·고위험', 4: '관망/주의',
        }

        df = df_industry.copy()

        # Cluster_Name 추가
        if 'Cluster_Name' not in df.columns and 'cluster' in df.columns:
            df['Cluster_Name'] = df['cluster'].map(CLUSTER_LABELS).fillna('미분류')

        # Horizon 보장
        if 'Horizon' not in df.columns:
            df.insert(0, 'Horizon', horizon_name)

        # 성향별 추천 정보 추가
        for profile_name, result in results.items():
            df_rec = result['recommendations']

            if len(df_rec) == 0:
                df[f'Is_추천_{profile_name}'] = False
                df[f'Rank_{profile_name}'] = 0
                df[f'Score_{profile_name}'] = 0.0
                continue

            # 순위 부여 (composite_score > Sharpe 내림차순)
            sort_col = 'composite_score' if 'composite_score' in df_rec.columns else 'Sharpe_Ratio'
            df_rec_sorted = df_rec.sort_values(sort_col, ascending=False).reset_index(drop=True)

            # 추천 여부
            rec_industries = set(df_rec_sorted['Industry'].values)
            df[f'Is_추천_{profile_name}'] = df['Industry'].isin(rec_industries)

            # 순위 매핑 (추천 산업만 1~N, 비추천은 0)
            rank_map = {
                row['Industry']: rank
                for rank, (_, row) in enumerate(df_rec_sorted.iterrows(), 1)
            }
            df[f'Rank_{profile_name}'] = df['Industry'].map(rank_map).fillna(0).astype(int)

            # composite_score 매핑 (비추천은 0.0)
            if 'composite_score' in df_rec.columns:
                score_map = dict(zip(df_rec['Industry'], df_rec['composite_score']))
                df[f'Score_{profile_name}'] = df['Industry'].map(score_map).fillna(0.0)
            else:
                df[f'Score_{profile_name}'] = 0.0

        return df

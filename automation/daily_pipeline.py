"""
일일 자동화 파이프라인: 데이터 수집 -> 멀티 호라이즌 예측 -> 클러스터링
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

# 상위 디렉토리를 Python 경로에 추가 (src 모듈 import를 위해)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daily_data_fetcher import DailyDataFetcher
from src import MultiHorizonPredictor, IndustryClusterer, InvestorProfiler


class DailyPipeline:
    """일일 자동화 파이프라인"""

    # 전략그룹 한글 이름 매핑
    STRATEGY_GROUP_NAMES = {
        '단타': '단타',
        '스윙': '스윙',
        '장기투자': '장기투자',
    }

    def __init__(self, config: dict = None, config_file: str = None):
        """
        Parameters:
        -----------
        config : dict
            파이프라인 설정
        config_file : str
            설정 파일 경로 (JSON)
        """
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = json.load(f)

            # config.json의 상대 경로를 절대 경로로 변환
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if not os.path.isabs(self.config.get('data_path', '')):
                self.config['data_path'] = os.path.join(base_dir, self.config['data_path'])
            if not os.path.isabs(self.config.get('output_dir', '')):
                self.config['output_dir'] = os.path.join(base_dir, self.config['output_dir'])
            if not os.path.isabs(self.config.get('log_dir', '')):
                self.config['log_dir'] = os.path.join(base_dir, self.config['log_dir'])
        elif config:
            self.config = config
        else:
            self.config = self.default_config()

        self.setup_logging()

    def default_config(self) -> dict:
        """기본 설정"""
        # 프로젝트 루트 디렉토리 (automation의 상위 디렉토리)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        return {
            # 데이터 경로
            'data_path': os.path.join(base_dir, 'Data_set/stock_features_clean.csv'),
            'output_dir': os.path.join(base_dir, 'Data_set/Cluster_Results'),

            # 데이터 수집
            'fetch_days_back': 7,  # 며칠 전부터 가져올지

            # 예측 파라미터
            'train_years': 4,
            'alpha': 0.6,
            'gamma': 0.5,
            'top_k': 3,

            # 클러스터링 파라미터
            'n_clusters': 5,
            'horizon_lookback_map': {
                '1d': 60,
                '3d': 75,
                '1w': 90,
                '1m': 105,
                '1q': 120,
                '6m': 150,
                '1y': 180
            },
            'strategy_group_lookback_map': {
                '단타': 75,
                '스윙': 105,
                '장기투자': 180,
            },

            # 로깅
            'log_dir': os.path.join(base_dir, 'logs'),
            'log_level': 'INFO'
        }

    def setup_logging(self):
        """로깅 설정"""
        os.makedirs(self.config['log_dir'], exist_ok=True)

        log_file = os.path.join(
            self.config['log_dir'],
            f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"
        )

        logging.basicConfig(
            level=getattr(logging, self.config['log_level']),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)

    def run(self):
        """전체 파이프라인 실행"""
        self.logger.info("="*80)
        self.logger.info("일일 자동화 파이프라인 시작")
        self.logger.info("="*80)

        try:
            # 1. 데이터 수집 및 업데이트
            self.logger.info("\n[1단계] 데이터 수집 및 업데이트")
            df_updated = self.fetch_and_update_data()

            # 2. 멀티 호라이즌 예측
            self.logger.info("\n[2단계] 멀티 호라이즌 섹터 예측")
            multi_horizon_results, raw_horizon_results = self.run_multi_horizon_prediction(df_updated)

            # 3. 클러스터링
            self.logger.info("\n[3단계] 호라이즌별 인더스트리 클러스터링")
            clustering_results = self.run_clustering(df_updated, multi_horizon_results)

            # 4. 투자자 성향별 추천
            self.logger.info("\n[4단계] 투자자 성향별 산업 추천")
            investor_results = self.run_investor_profiling(
                clustering_results, raw_horizon_results
            )

            # 5. 결과 저장
            self.logger.info("\n[5단계] 결과 저장")
            self.save_results(multi_horizon_results, clustering_results, investor_results)

            self.logger.info("\n" + "="*80)
            self.logger.info("✓ 파이프라인 완료")
            self.logger.info("="*80)

            return {
                'status': 'success',
                'multi_horizon_results': multi_horizon_results,
                'clustering_results': clustering_results,
                'investor_results': investor_results,
            }

        except Exception as e:
            self.logger.error(f"파이프라인 실행 중 오류 발생: {str(e)}", exc_info=True)
            return {
                'status': 'failed',
                'error': str(e)
            }

    def fetch_and_update_data(self) -> pd.DataFrame:
        """데이터 수집 및 업데이트"""
        fetcher = DailyDataFetcher(self.config['data_path'])
        fetcher.load_existing_data()

        # 최신 데이터 가져오기
        df_new = fetcher.fetch_recent_data(days_back=self.config['fetch_days_back'])

        if df_new.empty:
            self.logger.info("새로운 데이터가 없습니다. 기존 데이터로 진행합니다.")
            return fetcher.df_existing

        # 특성 계산
        df_new_features = fetcher.calculate_features(df_new)

        # 데이터셋 업데이트
        df_updated = fetcher.update_dataset(df_new_features)

        return df_updated

    def run_multi_horizon_prediction(self, df_raw: pd.DataFrame) -> dict:
        """멀티 호라이즌 예측"""
        # 섹터 단위로 집계
        sector_df = df_raw.groupby(['Date', 'Sector'], as_index=False).agg({
            'Close': 'mean'
        }).sort_values(['Sector', 'Date'])

        self.logger.info(f"섹터 집계: {len(sector_df):,} 행")

        # 섹터 목록
        sectors = sorted(sector_df['Sector'].unique())
        self.logger.info(f"섹터: {sectors}")

        # 예측기 초기화
        predictor = MultiHorizonPredictor(
            alpha=self.config['alpha'],
            gamma=self.config['gamma'],
            top_k=self.config['top_k']
        )

        # 7개 호라이즌 예측 실행
        prediction_date = pd.Timestamp(datetime.now().date())
        raw_horizon_results = predictor.predict_all_horizons(
            df=sector_df,
            prediction_date=prediction_date,
            sectors=sectors,
            train_years=self.config['train_years']
        )

        # 3개 전략그룹으로 앙상블 병합
        strategy_results = predictor.aggregate_by_strategy_groups(raw_horizon_results)

        # 결과 출력
        self.logger.info("\n전략그룹별 상위 섹터:")
        for group_name, result in strategy_results.items():
            top_sectors = result['top_sectors']
            source = ', '.join(result['source_horizons'])
            self.logger.info(f"  {group_name} ({source}): {', '.join(top_sectors)}")

        return strategy_results, raw_horizon_results

    def run_clustering(self, df_raw: pd.DataFrame, multi_horizon_results: dict) -> dict:
        """호라이즌별 클러스터링"""
        clusterer = IndustryClusterer(
            n_clusters=self.config['n_clusters'],
            random_state=42,
            sharpe_thresholds=self.config.get('sharpe_thresholds'),
            vol_thresholds=self.config.get('vol_thresholds'),
        )

        clustering_results = {}
        prediction_date = pd.Timestamp(datetime.now().date())

        for horizon_name, horizon_result in multi_horizon_results.items():
            self.logger.info(f"\n[{horizon_name}] 클러스터링 시작")

            top_sectors = horizon_result['top_sectors']
            self.logger.info(f"  선정 섹터: {', '.join(top_sectors)}")

            # 전략그룹용 lookback 우선, fallback으로 horizon_lookback_map
            strategy_map = self.config.get('strategy_group_lookback_map', {})
            horizon_map = self.config.get('horizon_lookback_map', {})
            lookback_days = strategy_map.get(horizon_name, horizon_map.get(horizon_name, 90))
            self.logger.info(f"  특성 계산 기간: {lookback_days}일")

            # 클러스터링 수행
            clustering_result = clusterer.run_full_pipeline(
                df=df_raw,
                selected_sectors=top_sectors,
                lookback_days=lookback_days,
                end_date=prediction_date
            )

            clustering_results[horizon_name] = clustering_result

            # 클러스터별 산업 수 출력
            self.logger.info("  클러스터 분포:")
            for cluster_name, df_cluster in clustering_result['by_cluster'].items():
                interp = clustering_result['interpretations'][cluster_name]
                self.logger.info(f"    {cluster_name}: {len(df_cluster)}개 산업 - {interp['label']}")

        return clustering_results

    def run_investor_profiling(
        self, clustering_results: dict, raw_horizon_results: dict
    ) -> dict:
        """전략그룹별 투자자 성향 추천"""
        profiler = InvestorProfiler(
            profiles=self.config.get('investor_profiles'),
            top_n=self.config.get('investor_top_n', 5),
        )

        investor_results = {}

        for group_name, clustering_result in clustering_results.items():
            self.logger.info(f"\n[{group_name}] 투자자 성향별 추천")

            df_industry = clustering_result['industry_features']

            # 호라이즌 일치도 계산
            horizon_agreement = profiler.calculate_horizon_agreement(
                raw_horizon_results=raw_horizon_results,
                strategy_group_results={},
            )

            # 추천 실행
            profile_results = profiler.recommend(
                df_industry=df_industry,
                horizon_agreement=horizon_agreement,
            )

            investor_results[group_name] = profile_results

            # 결과 출력
            report = profiler.format_report(profile_results)
            for line in report.split('\n'):
                self.logger.info(line)

        return investor_results

    def save_results(self, multi_horizon_results: dict, clustering_results: dict, investor_results: dict = None):
        """결과 저장"""
        os.makedirs(self.config['output_dir'], exist_ok=True)

        date_str = datetime.now().strftime('%Y%m%d')

        profiler = InvestorProfiler(
            profiles=self.config.get('investor_profiles'),
            top_n=self.config.get('investor_top_n', 5),
        ) if investor_results else None

        unified_frames = []  # 통합 CSV (industry_features + 추천 정보)

        for horizon_name, clustering_result in clustering_results.items():
            # 전략그룹 한글 이름 가져오기
            horizon_name_kr = self.STRATEGY_GROUP_NAMES.get(horizon_name, horizon_name)
            df_industry = clustering_result['industry_features'].copy()

            # 1. 통합 CSV 생성 (industry_features + 추천 컬럼)
            if investor_results and horizon_name in investor_results and profiler:
                df_unified = profiler.export_unified_for_tableau(
                    df_industry=df_industry,
                    results=investor_results[horizon_name],
                    horizon_name=horizon_name_kr,
                )
            else:
                df_unified = df_industry.copy()
                if 'Horizon' not in df_unified.columns:
                    df_unified.insert(0, 'Horizon', horizon_name_kr)

            unified_frames.append(df_unified)

            # 2. 개별 industry_features CSV (기존 호환용)
            df_industry_save = df_industry.copy()
            df_industry_save.insert(0, 'Horizon', horizon_name_kr)
            industry_filename = os.path.join(
                self.config['output_dir'],
                f"{date_str}_{horizon_name}_industry_features.csv"
            )
            df_industry_save.to_csv(industry_filename, index=False, encoding='utf-8-sig')
            self.logger.info(f"  저장: {industry_filename} ({len(df_industry_save)}개 산업)")

            # 3. 클러스터별 파일
            for cluster_name, df_cluster in clustering_result['by_cluster'].items():
                df_cluster_copy = df_cluster.copy()
                df_cluster_copy.insert(0, 'Horizon', horizon_name_kr)
                cluster_filename = os.path.join(
                    self.config['output_dir'],
                    f"{date_str}_{horizon_name}_{cluster_name}.csv"
                )
                df_cluster_copy.to_csv(cluster_filename, index=False, encoding='utf-8-sig')

        # 4. Tableau 통합 CSV (전 호라이즌 × 전 산업 × 추천 컬럼)
        if unified_frames:
            df_all_unified = pd.concat(unified_frames, ignore_index=True)
            unified_filename = os.path.join(
                self.config['output_dir'],
                f"{date_str}_unified_industry_features.csv"
            )
            df_all_unified.to_csv(unified_filename, index=False, encoding='utf-8-sig')
            self.logger.info(
                f"  Tableau 통합 저장: {unified_filename} "
                f"({len(df_all_unified)}행 × {len(df_all_unified.columns)}컬럼)"
            )

        # 5. 투자자 성향별 개별 CSV (레거시 호환)
        if investor_results and profiler:
            tableau_frames = []

            for group_name, profile_results in investor_results.items():
                for profile_name, result in profile_results.items():
                    df_rec = result['recommendations']
                    if len(df_rec) > 0:
                        rec_filename = os.path.join(
                            self.config['output_dir'],
                            f"{date_str}_{group_name}_{profile_name}_recommendations.csv"
                        )
                        df_rec.to_csv(rec_filename, index=False, encoding='utf-8-sig')
                        self.logger.info(f"  저장: {rec_filename} ({len(df_rec)}개 산업)")

                horizon_name_kr = self.STRATEGY_GROUP_NAMES.get(group_name, group_name)
                df_tableau = profiler.export_for_tableau(profile_results, horizon_name=horizon_name_kr)
                if len(df_tableau) > 0:
                    tableau_frames.append(df_tableau)

            if tableau_frames:
                df_all_tableau = pd.concat(tableau_frames, ignore_index=True)
                tableau_filename = os.path.join(
                    self.config['output_dir'],
                    f"{date_str}_investor_recommendations_all.csv"
                )
                df_all_tableau.to_csv(tableau_filename, index=False, encoding='utf-8-sig')
                self.logger.info(f"  레거시 추천 저장: {tableau_filename} ({len(df_all_tableau)}행)")

        self.logger.info(f"\n✓ 모든 결과 저장 완료: {self.config['output_dir']}")


def main():
    """메인 실행"""
    # config.json 파일이 있으면 사용, 없으면 기본 설정
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    config_file = config_path if os.path.exists(config_path) else None

    pipeline = DailyPipeline(config_file=config_file)
    result = pipeline.run()

    if result['status'] == 'success':
        print("\n✓ 파이프라인 성공적으로 완료")
    else:
        print(f"\n✗ 파이프라인 실패: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()

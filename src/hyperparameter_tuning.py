"""
GridSearch 하이퍼파라미터 튜닝 모듈
alpha(하이브리드 가중치)와 gamma(신뢰도 가중치)의 최적 조합을 탐색합니다.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from itertools import product

from .backtest import run_backtest
from .evaluation import evaluate_predictions


class GridSearchTuner:
    """
    alpha, gamma 하이퍼파라미터 GridSearch 튜너

    기존 run_backtest()를 활용하여 각 (alpha, gamma) 조합의 성능을 측정하고
    최적 파라미터를 반환합니다.
    """

    DEFAULT_ALPHA_RANGE = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    DEFAULT_GAMMA_RANGE = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]

    def __init__(
        self,
        alpha_range: Optional[List[float]] = None,
        gamma_range: Optional[List[float]] = None,
        metric: str = 'hybrid_ndcg5',
    ):
        """
        Parameters
        ----------
        alpha_range : list of float
            탐색할 alpha 값 리스트
        gamma_range : list of float
            탐색할 gamma 값 리스트
        metric : str
            최적화 지표. evaluate_predictions 반환 키 중 하나.
            기본값 'hybrid_ndcg5' (NDCG@5)
        """
        self.alpha_range = alpha_range or self.DEFAULT_ALPHA_RANGE
        self.gamma_range = gamma_range or self.DEFAULT_GAMMA_RANGE
        self.metric = metric
        self.results: List[Dict] = []
        self.best_params: Optional[Dict] = None

    def search(
        self,
        df: pd.DataFrame,
        sectors: List[str],
        test_years: List[int],
    ) -> Dict:
        """
        모든 (alpha, gamma) 조합을 탐색하고 최적 파라미터를 반환합니다.

        Parameters
        ----------
        df : pd.DataFrame
            'Date', 'Sector', 'Close', 'year' 컬럼이 있는 전체 데이터셋
        sectors : list of str
            섹터 리스트
        test_years : list of int
            테스트 연도 리스트

        Returns
        -------
        dict : {'alpha': best_alpha, 'gamma': best_gamma, 'score': best_score}
        """
        self.results = []
        total = len(self.alpha_range) * len(self.gamma_range)

        print(f"\nGridSearch 시작: {total}개 조합 탐색")
        print(f"Alpha 범위: {self.alpha_range}")
        print(f"Gamma 범위: {self.gamma_range}")
        print(f"최적화 지표: {self.metric}")
        print(f"테스트 연도: {test_years}")
        print("=" * 80)

        for i, (alpha, gamma) in enumerate(product(self.alpha_range, self.gamma_range), 1):
            score = self._evaluate_single(df, sectors, test_years, alpha, gamma)
            self.results.append({
                'alpha': alpha,
                'gamma': gamma,
                'score': score,
            })
            print(f"  [{i}/{total}] alpha={alpha:.2f}, gamma={gamma:.2f} → {self.metric}={score:.4f}")

        best = max(self.results, key=lambda x: x['score'])
        self.best_params = best

        print("\n" + "=" * 80)
        print(f"최적 파라미터: alpha={best['alpha']:.2f}, gamma={best['gamma']:.2f}")
        print(f"최적 {self.metric}: {best['score']:.4f}")
        print("=" * 80)

        return best

    def _evaluate_single(
        self,
        df: pd.DataFrame,
        sectors: List[str],
        test_years: List[int],
        alpha: float,
        gamma: float,
    ) -> float:
        """
        단일 (alpha, gamma) 조합 평가

        여러 test_year에 대해 백테스트를 수행하고 지표의 평균을 반환합니다.
        """
        scores = []

        for year in test_years:
            train_df = df[df['year'] < year].copy()
            test_df = df[df['year'] == year].copy()

            if len(test_df) == 0:
                continue

            result = run_backtest(
                train_df=train_df,
                test_df=test_df,
                test_year=year,
                sectors=sectors,
                alpha=alpha,
                gamma=gamma,
            )

            if len(result['actuals']) < 3:
                continue

            actuals = np.array(list(result['actuals'].values()))
            prophet_preds = np.array(list(result['prophet_preds'].values()))
            hybrid_preds = np.array(list(result['hybrid_ranking_preds'].values()))

            metrics = evaluate_predictions(actuals, prophet_preds, hybrid_preds)
            scores.append(metrics.get(self.metric, 0.0))

        return float(np.mean(scores)) if scores else 0.0

    def get_results_df(self) -> pd.DataFrame:
        """결과를 DataFrame으로 반환"""
        if not self.results:
            return pd.DataFrame(columns=['alpha', 'gamma', 'score'])
        return pd.DataFrame(self.results).sort_values('score', ascending=False)

    def print_results(self) -> None:
        """결과 요약 출력"""
        df = self.get_results_df()
        if df.empty:
            print("결과 없음. search()를 먼저 실행하세요.")
            return

        print("\n" + "=" * 60)
        print("GridSearch 결과 (상위 10개)")
        print("=" * 60)
        print(f"{'순위':<6s} {'alpha':<8s} {'gamma':<8s} {self.metric:<15s}")
        print("-" * 60)

        for i, row in df.head(10).iterrows():
            rank = df.index.get_loc(i) + 1
            print(f"{rank:<6d} {row['alpha']:<8.2f} {row['gamma']:<8.2f} {row['score']:<15.4f}")

        print("=" * 60)

        if self.best_params:
            print(f"\n최적: alpha={self.best_params['alpha']:.2f}, "
                  f"gamma={self.best_params['gamma']:.2f}, "
                  f"score={self.best_params['score']:.4f}")

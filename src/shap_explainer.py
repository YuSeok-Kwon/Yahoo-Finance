"""
SHAP 모델 해석 모듈
XGBoost 모델의 SHAP 값을 계산하고 시각화합니다.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class ShapExplainer:
    """
    XGBoost 모델의 SHAP 기반 해석기

    SHAP(SHapley Additive exPlanations) 값을 계산하여
    각 특징이 예측에 기여하는 정도를 분석합니다.
    """

    def __init__(self):
        self.shap_values = None
        self.explainer = None
        self.feature_names = None
        self.X = None

    def explain(
        self,
        model,
        X: np.ndarray,
        feature_names: list,
    ) -> np.ndarray:
        """
        SHAP 값 계산

        Parameters
        ----------
        model : xgboost.XGBRegressor
            학습된 XGBoost 모델
        X : np.ndarray
            특징 행렬
        feature_names : list of str
            특징 이름 리스트

        Returns
        -------
        np.ndarray : SHAP 값 행렬 (n_samples, n_features)
        """
        self.feature_names = feature_names
        self.X = X

        self.explainer = shap.TreeExplainer(model)
        self.shap_values = self.explainer.shap_values(X)

        return self.shap_values

    def summary_plot(self, save_path: str = 'shap_summary.png') -> str:
        """
        SHAP summary bar plot 저장

        Parameters
        ----------
        save_path : str
            저장 경로

        Returns
        -------
        str : 저장 경로
        """
        if self.shap_values is None:
            raise ValueError("explain()을 먼저 호출하세요.")

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            self.shap_values,
            self.X,
            feature_names=self.feature_names,
            plot_type='bar',
            show=False,
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"SHAP summary plot 저장: {save_path}")
        return save_path

    def feature_importance_df(self) -> pd.DataFrame:
        """
        특징 중요도 DataFrame 반환

        Returns
        -------
        pd.DataFrame : 'feature', 'mean_abs_shap' 컬럼
        """
        if self.shap_values is None:
            raise ValueError("explain()을 먼저 호출하세요.")

        importance = np.abs(self.shap_values).mean(axis=0)
        df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': importance,
        }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

        return df

    def waterfall_plot(self, idx: int = 0, save_path: str = 'shap_waterfall.png') -> str:
        """
        개별 샘플 waterfall plot

        Parameters
        ----------
        idx : int
            분석할 샘플 인덱스
        save_path : str
            저장 경로

        Returns
        -------
        str : 저장 경로
        """
        if self.shap_values is None:
            raise ValueError("explain()을 먼저 호출하세요.")

        explanation = shap.Explanation(
            values=self.shap_values[idx],
            base_values=self.explainer.expected_value,
            data=self.X[idx],
            feature_names=self.feature_names,
        )

        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"SHAP waterfall plot 저장: {save_path}")
        return save_path

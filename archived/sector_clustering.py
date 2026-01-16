"""
Sector Clustering Pipeline
멀티 호라이즌 예측 결과 기반 섹터 군집화

Features:
- Term structure (2w → 1y curve)
- Slope/shape metrics
- KMeans clustering with optimal k selection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import Dict, List, Tuple, Optional


class SectorClusterer:
    """
    섹터 군집화 파이프라인
    """
    
    def __init__(
        self,
        df_features: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        scaler_type: str = 'standard'
    ):
        """
        Args:
            df_features: 멀티 호라이즌 피처 테이블
            feature_cols: 군집화에 사용할 컬럼 (None이면 자동 선택)
            scaler_type: 'standard' or 'robust'
        """
        self.df_features = df_features.copy()
        self.feature_cols = feature_cols
        self.scaler_type = scaler_type
        
        if self.feature_cols is None:
            self.feature_cols = [
                c for c in df_features.columns 
                if c.startswith('pred_return_') or 
                   c in ['term_slope', 'curve_shape', 'short_momentum', 
                         'long_momentum', 'return_volatility']
            ]
        
        self.scaler = None
        self.X_scaled = None
        self.best_k = None
        self.best_model = None
        self.cluster_labels = None
    
    def prepare_features(self) -> np.ndarray:
        """
        피처 스케일링 및 전처리
        
        Returns:
            Scaled feature matrix
        """
        print("=" * 80)
        print("Feature Preparation")
        print("=" * 80)
        
        missing_cols = [c for c in self.feature_cols if c not in self.df_features.columns]
        if missing_cols:
            print(f"⚠ Missing columns: {missing_cols}")
            self.feature_cols = [c for c in self.feature_cols if c in self.df_features.columns]
        
        print(f"Feature columns: {len(self.feature_cols)}")
        for i, col in enumerate(self.feature_cols, 1):
            print(f"  {i}. {col}")
        
        X = self.df_features[self.feature_cols].values
        
        print(f"\nOriginal shape: {X.shape}")
        print(f"Missing values: {np.isnan(X).sum()}")
        
        if np.isnan(X).any():
            print("  Filling NaN with column mean...")
            for i in range(X.shape[1]):
                col_mean = np.nanmean(X[:, i])
                X[np.isnan(X[:, i]), i] = col_mean
        
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler: {self.scaler_type}")
        
        self.X_scaled = self.scaler.fit_transform(X)
        
        print(f"\n✓ Scaled shape: {self.X_scaled.shape}")
        print(f"  Scaler: {self.scaler_type}")
        print("=" * 80)
        
        return self.X_scaled
    
    def find_optimal_k(
        self,
        k_range: Tuple[int, int] = (2, 8),
        random_state: int = 42
    ) -> Dict[int, Dict[str, float]]:
        """
        최적 군집 수 탐색
        
        Args:
            k_range: (min_k, max_k)
            random_state: 재현성
            
        Returns:
            Dict[k, metrics]
        """
        if self.X_scaled is None:
            self.prepare_features()
        
        print("\n" + "=" * 80)
        print("Finding Optimal K")
        print("=" * 80)
        
        results = {}
        
        for k in range(k_range[0], k_range[1] + 1):
            kmeans = KMeans(
                n_clusters=k,
                random_state=random_state,
                n_init=10,
                max_iter=300
            )
            
            labels = kmeans.fit_predict(self.X_scaled)
            
            inertia = kmeans.inertia_
            silhouette = silhouette_score(self.X_scaled, labels)
            davies_bouldin = davies_bouldin_score(self.X_scaled, labels)
            
            results[k] = {
                'inertia': inertia,
                'silhouette': silhouette,
                'davies_bouldin': davies_bouldin,
                'model': kmeans
            }
            
            print(f"k={k}: silhouette={silhouette:.3f}, "
                  f"davies_bouldin={davies_bouldin:.3f}, inertia={inertia:.1f}")
        
        best_k_by_silhouette = max(results.keys(), key=lambda k: results[k]['silhouette'])
        best_k_by_db = min(results.keys(), key=lambda k: results[k]['davies_bouldin'])
        
        print("\n" + "-" * 80)
        print(f"Best k by Silhouette: {best_k_by_silhouette} "
              f"(score={results[best_k_by_silhouette]['silhouette']:.3f})")
        print(f"Best k by Davies-Bouldin: {best_k_by_db} "
              f"(score={results[best_k_by_db]['davies_bouldin']:.3f})")
        
        if best_k_by_silhouette == best_k_by_db:
            self.best_k = best_k_by_silhouette
            print(f"\n✓ Optimal k: {self.best_k} (both metrics agree)")
        else:
            self.best_k = best_k_by_silhouette
            print(f"\n✓ Optimal k: {self.best_k} (using Silhouette)")
        
        print("=" * 80)
        
        return results
    
    def fit_clusters(self, k: Optional[int] = None, random_state: int = 42):
        """
        최종 군집 모델 학습
        
        Args:
            k: 군집 수 (None이면 find_optimal_k 결과 사용)
            random_state: 재현성
        """
        if self.X_scaled is None:
            self.prepare_features()
        
        if k is None:
            if self.best_k is None:
                self.find_optimal_k()
            k = self.best_k
        
        print(f"\nFitting KMeans with k={k}...")
        
        self.best_model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=20,
            max_iter=500
        )
        
        self.cluster_labels = self.best_model.fit_predict(self.X_scaled)
        
        self.df_features['cluster'] = self.cluster_labels
        
        print(f"✓ Clustering complete")
        print(f"  Cluster distribution:")
        for i in range(k):
            count = (self.cluster_labels == i).sum()
            pct = count / len(self.cluster_labels) * 100
            print(f"    Cluster {i}: {count} samples ({pct:.1f}%)")
    
    def get_cluster_profiles(self) -> pd.DataFrame:
        """
        군집별 프로파일 생성
        
        Returns:
            군집별 평균 피처값
        """
        if 'cluster' not in self.df_features.columns:
            raise ValueError("Run fit_clusters() first")
        
        cluster_profiles = self.df_features.groupby('cluster')[self.feature_cols].mean()
        
        return cluster_profiles
    
    def plot_cluster_curves(
        self,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ):
        """
        군집별 수익률 커브 시각화
        
        Args:
            figsize: Figure size
            save_path: 저장 경로 (None이면 표시만)
        """
        if 'cluster' not in self.df_features.columns:
            raise ValueError("Run fit_clusters() first")
        
        horizon_cols = [c for c in self.feature_cols if c.startswith('pred_return_')]
        
        if len(horizon_cols) < 2:
            print("⚠ Not enough horizon columns for curve plot")
            return
        
        horizon_order = ['pred_return_3d', 'pred_return_1w', 'pred_return_2w', 
                         'pred_return_1m', 'pred_return_1q', 'pred_return_1y']
        horizon_cols = [h for h in horizon_order if h in horizon_cols]
        horizon_labels = [h.replace('pred_return_', '').upper() for h in horizon_cols]
        
        cluster_profiles = self.get_cluster_profiles()
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        for cluster_id in sorted(cluster_profiles.index):
            curve = cluster_profiles.loc[cluster_id, horizon_cols].values
            axes[0].plot(
                range(len(curve)),
                curve,
                marker='o',
                label=f'Cluster {cluster_id}',
                linewidth=2
            )
        
        axes[0].set_xticks(range(len(horizon_labels)))
        axes[0].set_xticklabels(horizon_labels)
        axes[0].set_xlabel('Horizon')
        axes[0].set_ylabel('Expected Return')
        axes[0].set_title('Cluster Return Curves')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        cluster_sizes = self.df_features['cluster'].value_counts().sort_index()
        axes[1].bar(cluster_sizes.index, cluster_sizes.values)
        axes[1].set_xlabel('Cluster')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Cluster Sizes')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")
        else:
            plt.show()
    
    def get_cluster_assignments(self) -> pd.DataFrame:
        """
        군집 할당 결과 반환
        
        Returns:
            DataFrame with Sector, test_year, cluster
        """
        if 'cluster' not in self.df_features.columns:
            raise ValueError("Run fit_clusters() first")
        
        id_cols = ['test_year', 'Sector']
        existing_id_cols = [c for c in id_cols if c in self.df_features.columns]
        
        return self.df_features[existing_id_cols + ['cluster']].copy()


def example_usage():
    """
    사용 예시
    """
    
    clusterer = SectorClusterer(
        df_features=df_multi_horizon_features,
        scaler_type='robust'
    )
    
    clusterer.prepare_features()
    
    metrics = clusterer.find_optimal_k(k_range=(3, 6))
    
    clusterer.fit_clusters()
    
    cluster_profiles = clusterer.get_cluster_profiles()
    print("\nCluster Profiles:")
    print(cluster_profiles)
    
    clusterer.plot_cluster_curves(save_path='cluster_curves.png')
    
    cluster_assignments = clusterer.get_cluster_assignments()
    
    return cluster_assignments


if __name__ == '__main__':
    print(__doc__)

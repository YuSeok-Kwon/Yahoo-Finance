"""
Model Validation with Proper Train/Test Split
데이터 누수 없는 모델 검증 시스템

데이터 분리:
- 학습: 2020-2024년 (Prophet 학습 + XGBoost 보정)
- 테스트: 2025년 (Hold-out test, 학습에 전혀 사용 안 함)
- 백테스트: 2022, 2023, 2024 각각 독립적으로

성능 비교:
1. Prophet 단독
2. Prophet + XGBoost 하이브리드

평가 지표:
- MAPE (Mean Absolute Percentage Error)
- RMSE (Root Mean Squared Error)
- Spearman Correlation (섹터 순위 예측)
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import xgboost as xgb
from scipy.stats import spearmanr
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


# ==================== 1. 데이터 로딩 ====================

def load_sector_data(csv_path: str) -> pd.DataFrame:
    """섹터 데이터 로딩 및 전처리"""
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    
    # 필수 컬럼 확인
    required = ['Date', 'Sector', 'Close', 'Daily_Return_calc']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # 정렬
    df = df.sort_values(['Sector', 'Date']).reset_index(drop=True)
    
    print(f"✓ 데이터 로딩 완료: {len(df):,} 행")
    print(f"  섹터: {df['Sector'].nunique()}개")
    print(f"  날짜 범위: {df['Date'].min().date()} ~ {df['Date'].max().date()}")
    
    return df


def split_train_test(
    df: pd.DataFrame,
    train_end_year: int = 2024,
    test_year: int = 2025
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    학습/테스트 데이터 분리 (데이터 누수 방지)
    
    Parameters:
    -----------
    df : DataFrame
        전체 데이터
    train_end_year : int
        학습 데이터 마지막 연도 (default: 2024)
    test_year : int
        테스트 연도 (default: 2025)
    
    Returns:
    --------
    train_df, test_df
    """
    df['year'] = df['Date'].dt.year
    
    # 2020-2024: 학습
    train_df = df[df['year'] <= train_end_year].copy()
    
    # 2025: 테스트
    test_df = df[df['year'] == test_year].copy()
    
    print(f"\n{'='*80}")
    print("데이터 분리 (데이터 누수 방지)")
    print(f"{'='*80}")
    print(f"학습 데이터: {train_df['year'].min()}-{train_df['year'].max()} ({len(train_df):,} 행)")
    print(f"테스트 데이터: {test_df['year'].unique()[0]} ({len(test_df):,} 행)")
    print(f"{'='*80}\n")
    
    return train_df, test_df


# ==================== 2. Prophet 모델 ====================

def train_prophet(
    sector_data: pd.DataFrame,
    sector_name: str
) -> Prophet:
    """
    섹터별 Prophet 모델 학습
    
    Parameters:
    -----------
    sector_data : DataFrame
        특정 섹터 데이터 (Date, Close 필수)
    sector_name : str
        섹터명
    
    Returns:
    --------
    Prophet model (fitted)
    """
    # Prophet 데이터 형식 변환
    prophet_df = sector_data[['Date', 'Close']].copy()
    prophet_df.columns = ['ds', 'y']
    
    # 로그 변환
    prophet_df['y'] = np.log(prophet_df['y'])
    
    # Prophet 모델 생성
    model = Prophet(
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True
    )
    
    # 학습
    model.fit(prophet_df)
    
    return model


def predict_prophet(
    model: Prophet,
    future_dates: pd.DataFrame
) -> pd.DataFrame:
    """
    Prophet 모델 예측
    
    Parameters:
    -----------
    model : Prophet
        학습된 Prophet 모델
    future_dates : DataFrame
        예측할 날짜 (ds 컬럼)
    
    Returns:
    --------
    DataFrame with predictions
    """
    forecast = model.predict(future_dates)
    return forecast[['ds', 'yhat']]


# ==================== 3. XGBoost 보정 모델 ====================

def make_xgb_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    XGBoost 특징 생성
    
    Parameters:
    -----------
    df : DataFrame
        Prophet 예측이 포함된 데이터 (ds, y, yhat, resid)
    
    Returns:
    --------
    DataFrame with features
    """
    feat = df.copy()
    
    # 1. 시계열 특징
    feat['dayofweek'] = pd.to_datetime(feat['ds']).dt.dayofweek
    feat['month'] = pd.to_datetime(feat['ds']).dt.month
    feat['quarter'] = pd.to_datetime(feat['ds']).dt.quarter
    
    # 2. 잔차 기반 특징
    if 'resid' in feat.columns:
        feat['resid_lag_1'] = feat['resid'].shift(1)
        feat['resid_lag_5'] = feat['resid'].shift(5)
        feat['resid_roll_mean_5'] = feat['resid'].rolling(5).mean()
        feat['resid_roll_std_5'] = feat['resid'].rolling(5).std()
    
    # 3. yhat 기반 특징
    feat['yhat_lag_1'] = feat['yhat'].shift(1)
    feat['yhat_diff_1'] = feat['yhat'].diff(1)
    
    # 결측치 제거
    feat = feat.fillna(0)
    
    return feat


def train_xgb_corrector(
    train_features: pd.DataFrame,
    target_col: str = 'resid'
) -> xgb.XGBRegressor:
    """
    XGBoost 잔차 보정 모델 학습
    
    Parameters:
    -----------
    train_features : DataFrame
        학습 특징
    target_col : str
        타겟 컬럼 (default: 'resid')
    
    Returns:
    --------
    XGBRegressor (fitted)
    """
    feature_cols = [
        'dayofweek', 'month', 'quarter',
        'resid_lag_1', 'resid_lag_5', 'resid_roll_mean_5', 'resid_roll_std_5',
        'yhat_lag_1', 'yhat_diff_1'
    ]
    
    # 사용 가능한 컬럼만 선택
    available_cols = [c for c in feature_cols if c in train_features.columns]
    
    X = train_features[available_cols].values
    y = train_features[target_col].values
    
    # XGBoost 모델
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X, y)
    
    return model


def predict_hybrid(
    prophet_model: Prophet,
    xgb_model: xgb.XGBRegressor,
    future_dates: pd.DataFrame,
    alpha: float = 0.5
) -> pd.DataFrame:
    """
    Prophet + XGBoost 하이브리드 예측
    
    Parameters:
    -----------
    prophet_model : Prophet
        학습된 Prophet 모델
    xgb_model : XGBRegressor
        학습된 XGBoost 모델
    future_dates : DataFrame
        예측할 날짜
    alpha : float
        하이브리드 가중치 (0: Prophet only, 1: XGBoost only)
    
    Returns:
    --------
    DataFrame with hybrid predictions
    """
    # Prophet 예측
    prophet_pred = predict_prophet(prophet_model, future_dates)
    
    # XGBoost 특징 생성
    feat = make_xgb_features(prophet_pred)
    
    feature_cols = [
        'dayofweek', 'month', 'quarter',
        'resid_lag_1', 'resid_lag_5', 'resid_roll_mean_5', 'resid_roll_std_5',
        'yhat_lag_1', 'yhat_diff_1'
    ]
    available_cols = [c for c in feature_cols if c in feat.columns]
    
    # XGBoost 보정
    xgb_correction = xgb_model.predict(feat[available_cols].values)
    
    # 하이브리드
    prophet_pred['yhat_hybrid'] = (1 - alpha) * prophet_pred['yhat'] + alpha * (prophet_pred['yhat'] + xgb_correction)
    
    return prophet_pred


# ==================== 4. 백테스트 시스템 ====================

def backtest_year(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_year: int,
    sectors: List[str],
    alpha: float = 0.5
) -> Dict:
    """
    특정 연도에 대한 백테스트
    
    Parameters:
    -----------
    train_df : DataFrame
        학습 데이터 (test_year 이전)
    test_df : DataFrame
        테스트 데이터 (test_year)
    test_year : int
        테스트 연도
    sectors : List[str]
        섹터 리스트
    alpha : float
        하이브리드 가중치
    
    Returns:
    --------
    Dict with results
    """
    results = {
        'year': test_year,
        'prophet_predictions': {},
        'hybrid_predictions': {},
        'actuals': {}
    }
    
    print(f"\n{'='*80}")
    print(f"{test_year}년 백테스트")
    print(f"{'='*80}")
    
    for sector in sectors:
        # 섹터 데이터
        sector_train = train_df[train_df['Sector'] == sector].copy()
        sector_test = test_df[test_df['Sector'] == sector].copy()
        
        if len(sector_train) < 30 or len(sector_test) == 0:
            continue
        
        try:
            # Prophet 학습
            prophet_model = train_prophet(sector_train, sector)
            
            # Prophet 예측
            future = pd.DataFrame({'ds': sector_test['Date']})
            prophet_pred = predict_prophet(prophet_model, future)
            
            # XGBoost 학습을 위한 잔차 계산
            sector_train_prophet = sector_train.copy()
            sector_train_prophet['y_log'] = np.log(sector_train_prophet['Close'])
            
            # 학습 데이터에 대한 Prophet 예측
            train_future = pd.DataFrame({'ds': sector_train['Date']})
            train_prophet_pred = predict_prophet(prophet_model, train_future)
            
            sector_train_prophet = sector_train_prophet.merge(
                train_prophet_pred, left_on='Date', right_on='ds', how='left'
            )
            sector_train_prophet['resid'] = sector_train_prophet['y_log'] - sector_train_prophet['yhat']
            
            # XGBoost 특징 생성 및 학습
            train_features = make_xgb_features(sector_train_prophet)
            xgb_model = train_xgb_corrector(train_features)
            
            # 하이브리드 예측
            hybrid_pred = predict_hybrid(prophet_model, xgb_model, future, alpha)
            
            # 실제값 (연말 기준 수익률)
            actual_return = (sector_test['Close'].iloc[-1] / sector_test['Close'].iloc[0]) - 1
            
            # 예측값 (연말 기준 수익률)
            prophet_return = np.exp(prophet_pred['yhat'].iloc[-1] - prophet_pred['yhat'].iloc[0]) - 1
            hybrid_return = np.exp(hybrid_pred['yhat_hybrid'].iloc[-1] - hybrid_pred['yhat_hybrid'].iloc[0]) - 1
            
            results['prophet_predictions'][sector] = prophet_return
            results['hybrid_predictions'][sector] = hybrid_return
            results['actuals'][sector] = actual_return
            
            print(f"  {sector:30s} | Actual: {actual_return:7.2%} | Prophet: {prophet_return:7.2%} | Hybrid: {hybrid_return:7.2%}")
        
        except Exception as e:
            print(f"  {sector:30s} | ERROR: {str(e)[:50]}")
            continue
    
    return results


def evaluate_performance(results: Dict) -> Dict:
    """
    백테스트 결과 평가
    
    Parameters:
    -----------
    results : Dict
        백테스트 결과
    
    Returns:
    --------
    Dict with metrics
    """
    actuals = np.array(list(results['actuals'].values()))
    prophet_preds = np.array(list(results['prophet_predictions'].values()))
    hybrid_preds = np.array(list(results['hybrid_predictions'].values()))
    
    # MAPE
    prophet_mape = np.mean(np.abs((actuals - prophet_preds) / actuals)) * 100
    hybrid_mape = np.mean(np.abs((actuals - hybrid_preds) / actuals)) * 100
    
    # RMSE
    prophet_rmse = np.sqrt(np.mean((actuals - prophet_preds) ** 2))
    hybrid_rmse = np.sqrt(np.mean((actuals - hybrid_preds) ** 2))
    
    # Spearman Correlation
    prophet_spearman = spearmanr(actuals, prophet_preds)[0]
    hybrid_spearman = spearmanr(actuals, hybrid_preds)[0]
    
    metrics = {
        'prophet_mape': prophet_mape,
        'hybrid_mape': hybrid_mape,
        'prophet_rmse': prophet_rmse,
        'hybrid_rmse': hybrid_rmse,
        'prophet_spearman': prophet_spearman,
        'hybrid_spearman': hybrid_spearman
    }
    
    return metrics


# ==================== 5. 메인 실행 ====================

def main():
    """메인 실행 함수"""
    
    # 데이터 로딩
    data_path = 'Data_set/stock_features_clean.csv'
    df = load_sector_data(data_path)
    
    # 섹터별 집계 (일별 평균)
    sector_df = df.groupby(['Date', 'Sector'], as_index=False).agg({
        'Close': 'mean',
        'Daily_Return_calc': 'mean'
    })
    
    # 학습/테스트 분리
    train_df, test_df = split_train_test(sector_df, train_end_year=2024, test_year=2025)
    
    # 섹터 리스트
    sectors = sorted(train_df['Sector'].unique())
    
    # 백테스트 연도
    backtest_years = [2022, 2023, 2024]
    
    # 각 연도별 백테스트
    all_results = []
    all_metrics = []
    
    for year in backtest_years:
        # 해당 연도 이전 데이터로만 학습
        year_train = train_df[train_df['year'] < year].copy()
        year_test = train_df[train_df['year'] == year].copy()
        
        # 백테스트
        results = backtest_year(year_train, year_test, year, sectors, alpha=0.5)
        
        # 평가
        metrics = evaluate_performance(results)
        
        all_results.append(results)
        all_metrics.append(metrics)
        
        print(f"\n{'='*80}")
        print(f"{year}년 성능 평가")
        print(f"{'='*80}")
        print(f"{'Metric':<20s} | {'Prophet':>10s} | {'Hybrid':>10s} | {'Improvement':>12s}")
        print(f"{'-'*80}")
        print(f"{'MAPE (%)':<20s} | {metrics['prophet_mape']:>10.2f} | {metrics['hybrid_mape']:>10.2f} | {metrics['prophet_mape'] - metrics['hybrid_mape']:>11.2f}%")
        print(f"{'RMSE':<20s} | {metrics['prophet_rmse']:>10.4f} | {metrics['hybrid_rmse']:>10.4f} | {metrics['prophet_rmse'] - metrics['hybrid_rmse']:>11.4f}")
        print(f"{'Spearman Corr':<20s} | {metrics['prophet_spearman']:>10.4f} | {metrics['hybrid_spearman']:>10.4f} | {metrics['hybrid_spearman'] - metrics['prophet_spearman']:>11.4f}")
        print(f"{'='*80}\n")
    
    # 전체 평균 성능
    print(f"\n{'='*80}")
    print("전체 평균 성능 (2022-2024)")
    print(f"{'='*80}")
    
    avg_prophet_mape = np.mean([m['prophet_mape'] for m in all_metrics])
    avg_hybrid_mape = np.mean([m['hybrid_mape'] for m in all_metrics])
    avg_prophet_rmse = np.mean([m['prophet_rmse'] for m in all_metrics])
    avg_hybrid_rmse = np.mean([m['hybrid_rmse'] for m in all_metrics])
    avg_prophet_spearman = np.mean([m['prophet_spearman'] for m in all_metrics])
    avg_hybrid_spearman = np.mean([m['hybrid_spearman'] for m in all_metrics])
    
    print(f"{'Metric':<20s} | {'Prophet':>10s} | {'Hybrid':>10s} | {'Improvement':>12s}")
    print(f"{'-'*80}")
    print(f"{'MAPE (%)':<20s} | {avg_prophet_mape:>10.2f} | {avg_hybrid_mape:>10.2f} | {avg_prophet_mape - avg_hybrid_mape:>11.2f}%")
    print(f"{'RMSE':<20s} | {avg_prophet_rmse:>10.4f} | {avg_hybrid_rmse:>10.4f} | {avg_prophet_rmse - avg_hybrid_rmse:>11.4f}")
    print(f"{'Spearman Corr':<20s} | {avg_prophet_spearman:>10.4f} | {avg_hybrid_spearman:>10.4f} | {avg_hybrid_spearman - avg_prophet_spearman:>11.4f}")
    print(f"{'='*80}\n")
    
    # 2025년 최종 테스트
    print(f"\n{'#'*80}")
    print("2025년 최종 테스트 (Hold-out Test)")
    print(f"{'#'*80}\n")
    
    final_results = backtest_year(train_df, test_df, 2025, sectors, alpha=0.5)
    final_metrics = evaluate_performance(final_results)
    
    print(f"\n{'='*80}")
    print("2025년 성능 평가 (Hold-out Test)")
    print(f"{'='*80}")
    print(f"{'Metric':<20s} | {'Prophet':>10s} | {'Hybrid':>10s} | {'Improvement':>12s}")
    print(f"{'-'*80}")
    print(f"{'MAPE (%)':<20s} | {final_metrics['prophet_mape']:>10.2f} | {final_metrics['hybrid_mape']:>10.2f} | {final_metrics['prophet_mape'] - final_metrics['hybrid_mape']:>11.2f}%")
    print(f"{'RMSE':<20s} | {final_metrics['prophet_rmse']:>10.4f} | {final_metrics['hybrid_rmse']:>10.4f} | {final_metrics['prophet_rmse'] - final_metrics['hybrid_rmse']:>11.4f}")
    print(f"{'Spearman Corr':<20s} | {final_metrics['prophet_spearman']:>10.4f} | {final_metrics['hybrid_spearman']:>10.4f} | {final_metrics['hybrid_spearman'] - final_metrics['prophet_spearman']:>11.4f}")
    print(f"{'='*80}\n")
    
    # 섹터 순위 비교
    print(f"\n{'='*80}")
    print("2025년 섹터 순위 비교 (실제 vs 예측)")
    print(f"{'='*80}\n")
    
    # 정렬
    sorted_actual = sorted(final_results['actuals'].items(), key=lambda x: x[1], reverse=True)
    sorted_prophet = sorted(final_results['prophet_predictions'].items(), key=lambda x: x[1], reverse=True)
    sorted_hybrid = sorted(final_results['hybrid_predictions'].items(), key=lambda x: x[1], reverse=True)
    
    print(f"{'Rank':<6s} | {'Actual':<30s} | {'Prophet':<30s} | {'Hybrid':<30s}")
    print(f"{'-'*100}")
    
    for i in range(min(11, len(sorted_actual))):
        actual_sector = sorted_actual[i][0] if i < len(sorted_actual) else '-'
        prophet_sector = sorted_prophet[i][0] if i < len(sorted_prophet) else '-'
        hybrid_sector = sorted_hybrid[i][0] if i < len(sorted_hybrid) else '-'
        
        print(f"{i+1:<6d} | {actual_sector:<30s} | {prophet_sector:<30s} | {hybrid_sector:<30s}")
    
    print(f"{'='*80}\n")
    
    print("✓ 검증 완료!")
    print("\n다음 단계:")
    print("1. XGBoost 파라미터 튜닝 (alpha, n_estimators, max_depth 등)")
    print("2. 추가 특징 엔지니어링 (변동성, 모멘텀 등)")
    print("3. Learning-to-Rank (LTR) 적용")
    print("4. Industry-level 분석")


if __name__ == '__main__':
    main()

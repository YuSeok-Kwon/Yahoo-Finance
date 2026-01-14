"""
섹터 로테이션 분석을 위한 통합 피처 생성 모듈

이 모듈은 Prophet + XGBoost 하이브리드 모델을 위한 모든 피처 생성 함수를 포함합니다.

주요 기능:
1. 거시경제 데이터 수집 (FRED API)
2. v2 피처: 거시경제, 레짐, 모멘텀
3. v3 피처: 섹터 상대 모멘텀, 테마 감지
4. v3 통합: Prophet/XGBoost 모델 생성

사용법:
    from sector_rotation_features import upgrade_to_v3_features, create_v3_prophet_model

    # v3 피처 추가
    sector_panel = upgrade_to_v3_features(df, sector_panel)

    # v3 모델 생성
    prophet_model = create_v3_prophet_model(US_HOLIDAYS, sector_data)

작성자: nbCamp Project
버전: v3.0
날짜: 2026-01-14
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Optional imports (필요한 경우에만 사용)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


# =============================================================================
# 섹션 1: 거시경제 데이터 수집
# =============================================================================

def fetch_fred_data(series_id, start_date='2018-01-01', end_date=None):
    """
    FRED API를 통한 거시경제 데이터 수집

    Parameters:
    -----------
    series_id : str
        FRED 시리즈 ID (예: 'DGS10', 'DCOILWTICO')
    start_date : str
        시작 날짜 (YYYY-MM-DD)
    end_date : str, optional
        종료 날짜 (기본값: 현재)

    Returns:
    --------
    DataFrame : Date, series_id 컬럼
    """
    try:
        from pandas_datareader import data as web

        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        print(f"  Fetching {series_id} from FRED...")
        df = web.DataReader(series_id, 'fred', start_date, end_date)
        df = df.reset_index()
        df.columns = ['Date', series_id]
        df[series_id] = df[series_id].ffill()

        print(f"   {series_id}: {len(df)} records ({df['Date'].min()} ~ {df['Date'].max()})")
        return df

    except ImportError:
        print("    pandas-datareader not installed. Install with: pip install pandas-datareader")
        return None
    except Exception as e:
        print(f"   Error fetching {series_id}: {str(e)}")
        return None


def create_fallback_macro_data(start_date='2018-01-01', end_date=None):
    """
    FRED API 실패 시 더미 거시경제 데이터 생성

    Parameters:
    -----------
    start_date : str
        시작 날짜
    end_date : str, optional
        종료 날짜

    Returns:
    --------
    DataFrame : Date, DGS10, DCOILWTICO
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    print("    더미 데이터 생성 중...")

    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # DGS10: 3% → 4.5% 추세
    np.random.seed(42)
    base_rate = 3.0
    trend = np.linspace(0, 1.5, len(dates))
    noise = np.random.randn(len(dates)) * 0.1
    dgs10 = base_rate + trend + noise
    dgs10 = np.clip(dgs10, 0.5, 5.0)

    # WTI: 60$ 중심으로 변동
    base_oil = 60.0
    oil_trend = np.sin(np.linspace(0, 4*np.pi, len(dates))) * 20
    oil_noise = np.random.randn(len(dates)) * 5
    wti = base_oil + oil_trend + oil_noise
    wti = np.clip(wti, 30, 120)

    df = pd.DataFrame({
        'Date': dates,
        'DGS10': dgs10,
        'DCOILWTICO': wti
    })

    print(f"  Fallback data: {len(df)} records")
    return df


def fetch_and_save_macro_data(data_dir='Data_set', start_date='2018-01-01', end_date=None):
    """
    거시경제 데이터를 FRED에서 가져와 CSV로 저장

    Parameters:
    -----------
    data_dir : str or Path
        저장 디렉토리
    start_date : str
        시작 날짜
    end_date : str, optional
        종료 날짜

    Returns:
    --------
    DataFrame : 저장된 거시경제 데이터
    """
    print("=" * 70)
    print("FRED 거시경제 데이터 수집")
    print("=" * 70)

    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    # DGS10, WTI 수집
    print("\n[1/2] DGS10 (10-Year Treasury Rate)")
    dgs10_df = fetch_fred_data('DGS10', start_date, end_date)

    print("\n[2/2] DCOILWTICO (WTI Oil Price)")
    wti_df = fetch_fred_data('DCOILWTICO', start_date, end_date)

    # 병합
    if dgs10_df is not None and wti_df is not None:
        macro_df = pd.merge(dgs10_df, wti_df, on='Date', how='outer').sort_values('Date')
        macro_df['DGS10'] = macro_df['DGS10'].ffill().bfill()
        macro_df['DCOILWTICO'] = macro_df['DCOILWTICO'].ffill().bfill()
    else:
        print("\n  FRED API 실패 - 더미 데이터 생성")
        macro_df = create_fallback_macro_data(start_date, end_date)

    # 저장
    output_path = data_path / 'macro_data.csv'
    macro_df.to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print(f"저장 완료: {output_path}")
    print(f"레코드: {len(macro_df)}, 기간: {macro_df['Date'].min()} ~ {macro_df['Date'].max()}")
    print("=" * 70)

    return macro_df


# =============================================================================
# 섹션 2: v2 피처 생성
# =============================================================================

def load_and_prepare_macro(data_dir='Data_set', verbose=True):
    """
    거시경제 데이터 로드 및 파생 피처 생성

    생성 피처:
    - DGS10_level, DGS10_change_20d, DGS10_zscore_252
    - WTI_log, WTI_mom_3M, WTI_change_20d

    Parameters:
    -----------
    data_dir : str or Path
        데이터 디렉토리
    verbose : bool
        출력 여부

    Returns:
    --------
    DataFrame : 거시경제 파생 피처
    """
    data_path = Path(data_dir) / 'macro_data.csv'

    if not data_path.exists():
        raise FileNotFoundError(f"Macro data not found: {data_path}")

    macro = pd.read_csv(data_path)
    macro['Date'] = pd.to_datetime(macro['Date'])
    macro = macro.sort_values('Date').reset_index(drop=True)

    if verbose:
        print("=" * 70)
        print("거시경제 데이터 로드")
        print("=" * 70)
        print(f"  파일: {data_path}")
        print(f"  기간: {macro['Date'].min()} ~ {macro['Date'].max()}")
        print(f"  레코드: {len(macro):,}")

    # DGS10 파생 피처
    macro['DGS10_level'] = macro['DGS10']
    macro['DGS10_change_20d'] = macro['DGS10'].diff(20)

    rolling_mean_252 = macro['DGS10'].rolling(window=252, min_periods=60).mean()
    rolling_std_252 = macro['DGS10'].rolling(window=252, min_periods=60).std()
    macro['DGS10_zscore_252'] = (macro['DGS10'] - rolling_mean_252) / rolling_std_252

    # WTI 파생 피처
    macro['WTI_log'] = np.log(macro['DCOILWTICO'].replace(0, np.nan))
    macro['WTI_mom_3M'] = macro['DCOILWTICO'].pct_change(63)
    macro['WTI_change_20d'] = macro['DCOILWTICO'].diff(20)

    # 결측치 처리 및 극단값 클리핑
    macro = macro.ffill().bfill()
    macro['DGS10_zscore_252'] = macro['DGS10_zscore_252'].clip(-3, 3)
    macro['WTI_mom_3M'] = macro['WTI_mom_3M'].clip(-0.5, 0.5)

    if verbose:
        print("\n생성된 파생 피처:")
        print("  [DGS10] DGS10_level, DGS10_change_20d, DGS10_zscore_252")
        print("  [WTI] WTI_log, WTI_mom_3M, WTI_change_20d")

    return macro


def merge_macro_to_sector_panel(sector_panel, macro_df, verbose=True):
    """
    섹터 패널에 거시경제 데이터 병합 (거래일 캘린더 기준)

    Parameters:
    -----------
    sector_panel : DataFrame
        섹터 패널 (ds 또는 Date 필수)
    macro_df : DataFrame
        거시경제 데이터
    verbose : bool
        출력 여부

    Returns:
    --------
    DataFrame : 병합된 섹터 패널
    """
    panel = sector_panel.copy()
    macro = macro_df.copy()

    # Date 컬럼 통일
    if 'ds' in panel.columns and 'Date' not in panel.columns:
        panel['Date'] = pd.to_datetime(panel['ds'])
    elif 'Date' not in panel.columns:
        raise ValueError("sector_panel must have 'ds' or 'Date' column")

    panel['Date'] = pd.to_datetime(panel['Date'])
    macro['Date'] = pd.to_datetime(macro['Date'])

    # 거래일 기준 병합
    trading_dates = panel['Date'].unique()
    macro_trading = macro.set_index('Date').reindex(trading_dates, method='ffill').reset_index()
    macro_trading.columns = ['Date'] + [c for c in macro_trading.columns if c != 'Date']

    macro_cols = [c for c in macro_trading.columns if c not in ['Date', 'DGS10', 'DCOILWTICO']]
    panel = panel.merge(macro_trading[['Date'] + macro_cols], on='Date', how='left')

    # 결측치 처리
    if 'Sector' in panel.columns:
        panel = panel.groupby('Sector', group_keys=False).apply(lambda x: x.ffill().bfill())
    else:
        panel = panel.ffill().bfill()

    if verbose:
        print("\n 거시경제 데이터 병합 완료")
        print(f"  추가된 컬럼: {len(macro_cols)}개")
        print(f"  최종 Shape: {panel.shape}")

    return panel


def add_regime_features(sector_panel, window=252, verbose=True):
    """
    레짐 피처 추가

    생성 피처:
    - Market_Vol: 시장 전체 변동성
    - CrossSection_Dispersion: 섹터 간 수익률 분산

    Parameters:
    -----------
    sector_panel : DataFrame
        섹터 패널
    window : int
        계산 윈도우
    verbose : bool
        출력 여부

    Returns:
    --------
    DataFrame : 레짐 피처 추가된 패널
    """
    panel = sector_panel.copy()

    if 'Date' not in panel.columns and 'ds' in panel.columns:
        panel['Date'] = panel['ds']

    panel['Date'] = pd.to_datetime(panel['Date'])

    # Market_Vol
    if 'Avg_Volatility_20d' in panel.columns:
        market_vol = panel.groupby('Date')['Avg_Volatility_20d'].mean().to_frame('Market_Vol')
    elif 'Volatility_20d' in panel.columns:
        market_vol = panel.groupby('Date')['Volatility_20d'].mean().to_frame('Market_Vol')
    else:
        daily_ret = panel.groupby('Date')['Daily_Return'].mean()
        market_vol = daily_ret.rolling(window=20).std().to_frame('Market_Vol') * np.sqrt(252)
        market_vol = market_vol.reset_index()

    # CrossSection_Dispersion
    if 'Daily_Return' in panel.columns:
        cross_dispersion = panel.groupby('Date')['Daily_Return'].std().to_frame('CrossSection_Dispersion')
    elif 'Avg_Daily_Return_raw' in panel.columns:
        cross_dispersion = panel.groupby('Date')['Avg_Daily_Return_raw'].std().to_frame('CrossSection_Dispersion')
    else:
        cross_dispersion = pd.DataFrame({
            'Date': panel['Date'].unique(),
            'CrossSection_Dispersion': 0.01
        })

    # 병합
    if not isinstance(market_vol, pd.DataFrame):
        market_vol = market_vol.reset_index()
    if not isinstance(cross_dispersion, pd.DataFrame):
        cross_dispersion = cross_dispersion.reset_index()

    panel = panel.merge(market_vol, on='Date', how='left')
    panel = panel.merge(cross_dispersion, on='Date', how='left')

    panel['Market_Vol'] = panel['Market_Vol'].ffill().fillna(0.2)
    panel['CrossSection_Dispersion'] = panel['CrossSection_Dispersion'].ffill().fillna(0.01)

    if verbose:
        print("\n 레짐 피처 추가 완료")
        print("  - Market_Vol, CrossSection_Dispersion")

    return panel


def add_momentum_features(sector_panel, verbose=True):
    """
    단기 모멘텀 피처 추가

    생성 피처:
    - Mom_1M, Mom_6M, Mom_Accel

    Parameters:
    -----------
    sector_panel : DataFrame
        섹터 패널
    verbose : bool
        출력 여부

    Returns:
    --------
    DataFrame : 모멘텀 피처 추가된 패널
    """
    panel = sector_panel.copy()

    def calc_momentum(group):
        group['Mom_1M'] = group['Index'].pct_change(21)
        group['Mom_6M'] = group['Index'].pct_change(126)
        group['Mom_Accel'] = group['Mom_1M'] - group['Mom_6M']
        return group

    if 'Sector' in panel.columns:
        panel = panel.groupby('Sector', group_keys=False).apply(calc_momentum)
    else:
        panel = calc_momentum(panel)

    # 결측치 및 극단값 처리
    for col in ['Mom_1M', 'Mom_6M', 'Mom_Accel']:
        panel[col] = panel[col].fillna(0).clip(-0.5, 0.5)

    if verbose:
        print("\n 모멘텀 피처 추가 완료")
        print("  - Mom_1M, Mom_6M, Mom_Accel")

    return panel


def calculate_dynamic_cps(sector_data, base_cps=0.05, window=252):
    """
    섹터 변동성 기반 동적 changepoint_prior_scale 계산

    Parameters:
    -----------
    sector_data : DataFrame
        섹터 데이터
    base_cps : float
        기본 cps
    window : int
        계산 윈도우

    Returns:
    --------
    float : 계산된 cps 값
    """
    if 'Avg_Volatility_20d' in sector_data.columns:
        recent_vol = sector_data.tail(window)['Avg_Volatility_20d'].mean()
    elif 'Volatility_20d' in sector_data.columns:
        recent_vol = sector_data.tail(window)['Volatility_20d'].mean()
    else:
        return base_cps

    if recent_vol > 0.40:
        return 0.10
    elif recent_vol > 0.30:
        return 0.05
    else:
        return 0.03


# =============================================================================
# 섹션 3: v3 피처 생성
# =============================================================================

def add_relative_momentum_features(sector_panel, windows=[21, 63, 126], verbose=True):
    """
    🥇 1순위: 섹터 상대 모멘텀 피처 추가

    핵심 개념:
    - 절대 수익률이 아닌 시장 대비 상대 수익률 사용
    - 2023년 Tech의 "시장 대비 압도적 강세" 포착

    생성 피처:
    - RelMom_1M, RelMom_3M, RelMom_6M, RelMom_Accel
    - Relative_Mom_{window}d_diff, Relative_Mom_{window}d_ratio

    Parameters:
    -----------
    sector_panel : DataFrame
        섹터 패널 (Date, Sector, Index 필요)
    windows : list
        모멘텀 윈도우 (일 단위) [21=1M, 63=3M, 126=6M]
    verbose : bool
        출력 여부

    Returns:
    --------
    DataFrame : 상대 모멘텀 피처 추가된 패널
    """
    panel = sector_panel.copy()

    if 'Date' not in panel.columns and 'ds' in panel.columns:
        panel['Date'] = panel['ds']

    panel['Date'] = pd.to_datetime(panel['Date'])
    panel = panel.sort_values(['Date', 'Sector'])

    # 1. 시장 전체 수익률 (모든 섹터 평균)
    market_index = panel.groupby('Date')['Index'].mean().to_frame('Market_Index')

    # 2. 시장 수익률 계산
    for window in windows:
        market_index[f'Market_Return_{window}d'] = market_index['Market_Index'].pct_change(window)

    # 3. 섹터 패널에 시장 수익률 병합
    panel = panel.merge(market_index, on='Date', how='left')

    # 4. 섹터별 절대 수익률 계산
    def calc_sector_returns(group):
        for window in windows:
            group[f'Sector_Return_{window}d'] = group['Index'].pct_change(window)
        return group

    panel = panel.groupby('Sector', group_keys=False).apply(calc_sector_returns)

    # 5. 상대 모멘텀 계산 (Sector - Market)
    for window in windows:
        sector_col = f'Sector_Return_{window}d'
        market_col = f'Market_Return_{window}d'

        # 차이 (Difference)
        panel[f'Relative_Mom_{window}d_diff'] = panel[sector_col] - panel[market_col]

        # 비율 (Ratio) - 로그 변환
        panel[f'Relative_Mom_{window}d_ratio'] = (
            np.log1p(panel[sector_col]) - np.log1p(panel[market_col])
        )

    # 6. 결측치 및 극단값 처리
    relative_cols = [c for c in panel.columns if 'Relative_Mom' in c]
    for col in relative_cols:
        panel[col] = panel[col].fillna(0).clip(-0.5, 0.5)

    # 7. 단축 컬럼명
    panel['RelMom_1M'] = panel['Relative_Mom_21d_diff']
    panel['RelMom_3M'] = panel['Relative_Mom_63d_diff']
    panel['RelMom_6M'] = panel['Relative_Mom_126d_diff']

    # 8. 상대 모멘텀 가속도
    panel['RelMom_Accel'] = panel['RelMom_1M'] - panel['RelMom_6M']

    if verbose:
        print("\n 섹터 상대 모멘텀 피처 추가 완료")
        print("  - Relative_Mom_21d/63d/126d_diff/ratio")
        print("  - RelMom_1M/3M/6M, RelMom_Accel")
        print("   2023년 Tech 같은 '시장 대비 강세' 포착!")

    return panel


def add_theme_detection_features(df, sector_panel, verbose=True):
    """
     3순위: 테마 감지 피처 추가

    핵심 개념:
    - 2023 AI 붐처럼 특정 테마로 쏠림 감지
    - 상위 소수 종목에 거래 집중도 측정

    생성 피처:
    - BigTech_Concentration: 섹터 내 상위 3종목 거래대금 비중
    - Hot_Stock_Ratio: Vol_Z_Score 상위 10% 종목 비중
    - Relative_Theme_Strength: 섹터 vs 시장 테마 강도

    Parameters:
    -----------
    df : DataFrame
        종목별 데이터 (Company, Sector, Volume, Close 필요)
    sector_panel : DataFrame
        섹터 패널
    verbose : bool
        출력 여부

    Returns:
    --------
    DataFrame : 테마 감지 피처 추가된 패널
    """
    panel = sector_panel.copy()

    if 'Date' not in panel.columns and 'ds' in panel.columns:
        panel['Date'] = panel['ds']

    panel['Date'] = pd.to_datetime(panel['Date'])

    # 1. 거래대금 계산
    df_theme = df.copy()
    df_theme['Trade_Value'] = df_theme['Volume'] * df_theme['Close']

    # 2. BigTech Concentration
    def calc_bigtech_concentration(group):
        if len(group) < 3:
            return 0.0
        total_value = group['Trade_Value'].sum()
        if total_value == 0:
            return 0.0
        top3_value = group.nlargest(3, 'Trade_Value')['Trade_Value'].sum()
        return top3_value / total_value

    bigtech_conc = df_theme.groupby(['Date', 'Sector']).apply(
        calc_bigtech_concentration
    ).reset_index(name='BigTech_Concentration')

    # 3. Hot Stock Ratio
    if 'Vol_Z_Score' in df_theme.columns:
        def calc_hot_stock_ratio(group):
            if len(group) == 0:
                return 0.0
            threshold = group['Vol_Z_Score'].quantile(0.90)
            hot_count = (group['Vol_Z_Score'] >= threshold).sum()
            return hot_count / len(group)

        hot_stock = df_theme.groupby(['Date', 'Sector']).apply(
            calc_hot_stock_ratio
        ).reset_index(name='Hot_Stock_Ratio')
    else:
        # Fallback: 거래량 기반
        def calc_hot_stock_ratio_vol(group):
            if len(group) == 0:
                return 0.0
            threshold = group['Volume'].quantile(0.90)
            hot_count = (group['Volume'] >= threshold).sum()
            return hot_count / len(group)

        hot_stock = df_theme.groupby(['Date', 'Sector']).apply(
            calc_hot_stock_ratio_vol
        ).reset_index(name='Hot_Stock_Ratio')

    # 4. 섹터 패널에 병합
    panel = panel.merge(bigtech_conc, on=['Date', 'Sector'], how='left')
    panel = panel.merge(hot_stock, on=['Date', 'Sector'], how='left')

    # 5. 결측치 처리
    panel['BigTech_Concentration'] = panel['BigTech_Concentration'].fillna(0.33)
    panel['Hot_Stock_Ratio'] = panel['Hot_Stock_Ratio'].fillna(0.10)

    # 6. 시장 전체 테마 강도
    market_theme = panel.groupby('Date')[['BigTech_Concentration', 'Hot_Stock_Ratio']].mean()
    market_theme.columns = ['Market_BigTech_Conc', 'Market_Hot_Ratio']

    panel = panel.merge(market_theme, on='Date', how='left')

    # 7. 상대 테마 강도
    panel['Relative_Theme_Strength'] = (
        panel['BigTech_Concentration'] - panel['Market_BigTech_Conc']
    )

    if verbose:
        print("\n 테마 감지 피처 추가 완료")
        print("  - BigTech_Concentration, Hot_Stock_Ratio")
        print("  - Relative_Theme_Strength")
        print("   2023 AI 붐 같은 테마장 탐지!")

    return panel


def prepare_rank_signal_target(train_data, year):
    """
     2순위: XGBoost Rank Signal 목표 준비

    핵심 개념:
    - 절대 수익률이 아닌 상대 강도를 목표로 사용
    - "누가 더 나은가?" 순위 예측에 최적화

    Parameters:
    -----------
    train_data : DataFrame
        학습 데이터 (ds, Sector, y 필요)
    year : int
        예측 연도

    Returns:
    --------
    DataFrame : rank signal이 추가된 데이터
    """
    data = train_data.copy()
    data['Date'] = pd.to_datetime(data['ds'])
    data['Year'] = data['Date'].dt.year

    # 연도별 섹터 누적 수익률
    def calc_annual_rank(group):
        if len(group) < 2:
            return group

        start_idx = group['y'].iloc[0]
        end_idx = group['y'].iloc[-1]
        cumulative_return = np.exp(end_idx) - np.exp(start_idx)

        group['Annual_Return'] = cumulative_return
        return group

    data = data.groupby(['Year', 'Sector'], group_keys=False).apply(calc_annual_rank)

    # 연도별 시장 평균
    market_return = data.groupby('Year')['Annual_Return'].mean().to_frame('Market_Return_Year')
    data = data.merge(market_return, on='Year', how='left')

    # 상대 강도 = 섹터 - 시장
    data['Relative_Strength'] = data['Annual_Return'] - data['Market_Return_Year']

    # 순위
    data['Rank_Signal'] = data.groupby('Year')['Relative_Strength'].rank(ascending=False, method='min')

    return data


# =============================================================================
# 섹션 4: v3 통합 함수
# =============================================================================

def upgrade_to_v3_features(df, sector_panel, verbose=True):
    """
    기존 v2 sector_panel을 v3으로 업그레이드

    한 줄로 v3 피처 추가!

    Parameters:
    -----------
    df : DataFrame
        종목별 데이터
    sector_panel : DataFrame
        v2 섹터 패널 (거시경제/레짐/모멘텀 포함)
    verbose : bool
        출력 여부

    Returns:
    --------
    DataFrame : v3 피처 추가된 섹터 패널
    """
    if verbose:
        print("=" * 70)
        print(" v3 업그레이드 시작")
        print("=" * 70)
        print(f"입력 Shape: {sector_panel.shape}")

    # 1. 섹터 상대 모멘텀
    panel = add_relative_momentum_features(sector_panel, verbose=verbose)

    # 2. 테마 감지
    panel = add_theme_detection_features(df, panel, verbose=verbose)

    if verbose:
        print("\n" + "=" * 70)
        print(" v3 업그레이드 완료")
        print("=" * 70)
        print(f"최종 Shape: {panel.shape}")

        v3_cols = [c for c in panel.columns if any(x in c for x in
                   ['RelMom', 'Relative_Mom', 'BigTech', 'Hot_Stock', 'Theme'])]
        print(f"\n추가된 v3 피처 ({len(v3_cols)}개):")
        for col in v3_cols[:15]:
            print(f"  - {col}")
        if len(v3_cols) > 15:
            print(f"  ... 외 {len(v3_cols) - 15}개")

    return panel


def apply_v3_enhancements(df, sector_panel, verbose=True):
    """
    v3 모든 개선사항 적용 (upgrade_to_v3_features의 alias)

    기존 노트북 호환성을 위한 함수입니다.
    내부적으로 upgrade_to_v3_features()를 호출합니다.

    Parameters:
    -----------
    df : DataFrame
        종목별 데이터
    sector_panel : DataFrame
        섹터 패널
    verbose : bool
        출력 여부

    Returns:
    --------
    DataFrame : v3 개선 피처가 모두 추가된 섹터 패널
    """
    if verbose:
        print("=" * 70)
        print(" v3 핵심 개선사항 적용")
        print("=" * 70)

    # upgrade_to_v3_features와 동일한 기능
    panel = add_relative_momentum_features(sector_panel, verbose=verbose)
    panel = add_theme_detection_features(df, panel, verbose=verbose)

    if verbose:
        print("\n" + "=" * 70)
        print(" v3 개선 완료")
        print("=" * 70)
        print(f"최종 Shape: {panel.shape}")
        print(f"\n추가된 v3 피처:")
        v3_cols = [c for c in panel.columns if any(x in c for x in ['Relative_Mom', 'RelMom', 'BigTech', 'Hot_Stock', 'Theme'])]
        for col in v3_cols[:10]:
            print(f"  - {col}")
        if len(v3_cols) > 10:
            print(f"  ... 외 {len(v3_cols) - 10}개")

    return panel


def create_v3_prophet_model(holidays_df, sector_data=None, base_cps=0.05):
    """
    v3 Prophet 모델 생성 (v2 + v3 regressor)

    Parameters:
    -----------
    holidays_df : DataFrame
        휴일 데이터
    sector_data : DataFrame, optional
        섹터 데이터 (dynamic cps 계산용)
    base_cps : float
        기본 changepoint_prior_scale

    Returns:
    --------
    Prophet : v3 regressor가 추가된 Prophet 모델
    """
    if not PROPHET_AVAILABLE:
        raise ImportError("Prophet not installed. Install with: pip install prophet")

    # Dynamic cps
    if sector_data is not None:
        cps = calculate_dynamic_cps(sector_data, base_cps)
    else:
        cps = base_cps

    # Prophet 모델
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=cps,
        seasonality_prior_scale=10,
        holidays=holidays_df,
        holidays_prior_scale=10
    )

    # v2 기본 regressor
    base_regressors = [
        'Avg_Volatility_20d', 'Avg_RSI_14', 'Avg_ATR_14',
        'DGS10_level', 'DGS10_change_20d', 'WTI_log', 'WTI_mom_3M',
        'Market_Vol', 'CrossSection_Dispersion',
        'Mom_1M', 'Mom_6M', 'Mom_Accel'
    ]

    # v3 regressor
    v3_regressors = [
        'RelMom_1M', 'RelMom_3M', 'RelMom_6M', 'RelMom_Accel',
        'BigTech_Concentration', 'Hot_Stock_Ratio', 'Relative_Theme_Strength'
    ]

    all_regressors = base_regressors + v3_regressors

    for reg in all_regressors:
        model.add_regressor(reg)

    print(f" v3 Prophet 모델 생성 완료 (cps={cps:.3f})")
    print(f"   - 기본 regressor: {len(base_regressors)}개")
    print(f"   - v3 regressor: {len(v3_regressors)}개")
    print(f"   - 총 regressor: {len(all_regressors)}개")

    return model


def train_v3_xgb_model(train_data, prophet_forecast,
                       use_rank_signal=True,
                       use_early_stopping=True,
                       val_ratio=0.2,
                       verbose=True):
    """
    v3 XGBoost 모델 학습 (Rank Signal 목표)

     핵심: target을 residual이 아닌 relative_strength로 변경

    Parameters:
    -----------
    train_data : DataFrame
        학습 데이터
    prophet_forecast : DataFrame
        Prophet 예측
    use_rank_signal : bool
        Rank Signal 사용 여부
    use_early_stopping : bool
        조기 종료 사용 여부
    val_ratio : float
        검증 세트 비율
    verbose : bool
        출력 여부

    Returns:
    --------
    tuple : (xgb_model, feature_cols)
    """
    if not XGB_AVAILABLE:
        raise ImportError("XGBoost not installed. Install with: pip install xgboost")

    merged = train_data.merge(
        prophet_forecast[['ds', 'yhat']],
        on='ds',
        how='inner'
    )

    if verbose:
        print("=" * 70)
        print(" v3 XGBoost 학습 시작")
        print("=" * 70)
        print(f"학습 데이터: {len(merged)} rows")

    # Rank Signal 목표
    if use_rank_signal:
        market_y = merged.groupby('ds')['y'].mean().to_frame('market_y')
        merged = merged.merge(market_y, on='ds', how='left')
        merged['relative_strength'] = merged['y'] - merged['market_y']

        market_yhat = merged.groupby('ds')['yhat'].mean().to_frame('market_yhat')
        merged = merged.merge(market_yhat, on='ds', how='left')
        merged['relative_yhat'] = merged['yhat'] - merged['market_yhat']

        merged['target'] = merged['relative_strength'] - merged['relative_yhat']

        if verbose:
            print(" Rank Signal 목표 사용")
    else:
        merged['target'] = merged['y'] - merged['yhat']
        if verbose:
            print("  기존 Residual 목표 사용")

    # 피처 컬럼
    feature_cols = [
        'yhat',
        'Avg_Volatility_20d', 'Avg_RSI_14', 'Avg_ATR_14',
        'DGS10_level', 'DGS10_change_20d', 'WTI_log', 'WTI_mom_3M',
        'Market_Vol', 'CrossSection_Dispersion',
        'Mom_1M', 'Mom_6M', 'Mom_Accel',
        'RelMom_1M', 'RelMom_3M', 'RelMom_6M', 'RelMom_Accel',
        'BigTech_Concentration', 'Hot_Stock_Ratio', 'Relative_Theme_Strength'
    ]

    available_features = [f for f in feature_cols if f in merged.columns]

    if verbose:
        print(f"사용 가능한 피처: {len(available_features)}개")

    X = merged[available_features].values
    y = merged['target'].values
    X = np.nan_to_num(X, nan=0.0)

    # Train/Val 분할
    if use_early_stopping:
        n_train = int(len(X) * (1 - val_ratio))
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
    else:
        X_train, y_train = X, y

    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    if use_early_stopping:
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        if verbose:
            print(f" Early Stopping: {xgb_model.best_iteration} iterations")
    else:
        xgb_model.fit(X_train, y_train)

    if verbose:
        print("=" * 70)
        print(" v3 XGBoost 학습 완료")
        print("=" * 70)

    return xgb_model, available_features


# =============================================================================
# 메인 함수
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Sector Rotation Features Module (v3)")
    print("=" * 70)
    print("\n 사용 가능한 함수:")
    print("\n[데이터 수집]")
    print("  - fetch_and_save_macro_data()")
    print("\n[v2 피처]")
    print("  - load_and_prepare_macro()")
    print("  - merge_macro_to_sector_panel()")
    print("  - add_regime_features()")
    print("  - add_momentum_features()")
    print("\n[v3 피처]")
    print("  - add_relative_momentum_features()  ")
    print("  - add_theme_detection_features()    ")
    print("\n[v3 통합]")
    print("  - upgrade_to_v3_features()")
    print("  - apply_v3_enhancements()           (alias)")
    print("  - create_v3_prophet_model()")
    print("  - train_v3_xgb_model()              ")
    print("\n=" * 70)

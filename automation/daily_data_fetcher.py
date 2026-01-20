"""
Yahoo Finance API로 최신 주식 데이터 수집 및 업데이트
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')


class DailyDataFetcher:
    """Yahoo Finance API로 일일 데이터를 가져와서 기존 데이터셋을 업데이트"""

    def __init__(self, existing_data_path: str = None):
        """
        Parameters:
        -----------
        existing_data_path : str
            기존 stock_features_clean.csv 파일 경로
        """
        if existing_data_path is None:
            # automation 폴더에서 상위 디렉토리의 Data_set 참조
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            existing_data_path = os.path.join(base_dir, 'Data_set/stock_features_clean.csv')

        self.existing_data_path = existing_data_path
        self.df_existing = None
        self.ticker_list = []

    def load_existing_data(self):
        """기존 데이터 로드"""
        print("기존 데이터 로딩 중...")
        self.df_existing = pd.read_csv(self.existing_data_path, parse_dates=['Date'])

        # 티커 목록 추출 (Company 컬럼에서)
        self.ticker_list = self.df_existing['Company'].unique().tolist()

        last_date = self.df_existing['Date'].max()
        print(f"기존 데이터: {len(self.df_existing):,} 행")
        print(f"마지막 날짜: {last_date.date()}")
        print(f"티커 수: {len(self.ticker_list)}")

        return self.df_existing

    def fetch_recent_data(self, days_back: int = 7) -> pd.DataFrame:
        """
        Yahoo Finance API로 최근 데이터 가져오기

        Parameters:
        -----------
        days_back : int
            며칠 전부터 가져올지 (기본값: 7일, 주말/공휴일 고려)

        Returns:
        --------
        pd.DataFrame : 새로운 데이터
        """
        if self.df_existing is None:
            self.load_existing_data()

        last_date = self.df_existing['Date'].max()
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')

        print(f"\nYahoo Finance API로 데이터 다운로드 중...")
        print(f"기간: {start_date} ~ {end_date}")
        print(f"티커 수: {len(self.ticker_list)}")

        new_data_list = []
        failed_tickers = []

        for i, ticker in enumerate(self.ticker_list, 1):
            try:
                # Yahoo Finance에서 데이터 다운로드
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)

                if hist.empty:
                    continue

                # 기존 데이터에서 Sector, Industry 정보 가져오기
                ticker_info = self.df_existing[self.df_existing['Company'] == ticker].iloc[0]
                sector = ticker_info['Sector']
                industry = ticker_info['Industry']

                # 데이터프레임 생성
                for date, row in hist.iterrows():
                    new_data_list.append({
                        'Date': date.date(),
                        'Company': ticker,
                        'Sector': sector,
                        'Industry': industry,
                        'Open': row['Open'],
                        'High': row['High'],
                        'Low': row['Low'],
                        'Close': row['Close'],
                        'Volume': row['Volume'],
                        'Dividends': row['Dividends'],
                        'Stock Splits': row['Stock Splits']
                    })

                if (i % 50) == 0:
                    print(f"  진행: {i}/{len(self.ticker_list)} 티커 완료...")

            except Exception as e:
                failed_tickers.append(ticker)
                continue

        if failed_tickers:
            print(f"\n⚠️  다운로드 실패: {len(failed_tickers)}개 티커")

        if not new_data_list:
            print("새로운 데이터가 없습니다.")
            return pd.DataFrame()

        df_new = pd.DataFrame(new_data_list)
        df_new['Date'] = pd.to_datetime(df_new['Date'])

        # 기존 데이터에 이미 있는 날짜 제외
        df_new = df_new[df_new['Date'] > last_date]

        print(f"\n✓ 새로운 데이터: {len(df_new)} 행")
        if len(df_new) > 0:
            print(f"  날짜 범위: {df_new['Date'].min().date()} ~ {df_new['Date'].max().date()}")

        return df_new

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        새로운 데이터에 대해 기술적 지표 계산

        Parameters:
        -----------
        df : pd.DataFrame
            원본 데이터 (Open, High, Low, Close, Volume 포함)

        Returns:
        --------
        pd.DataFrame : 특성이 추가된 데이터
        """
        print("\n기술적 지표 계산 중...")

        # 전체 데이터 결합 (특성 계산을 위해 기존 데이터 필요)
        df_combined = pd.concat([self.df_existing, df], ignore_index=True)
        df_combined = df_combined.sort_values(['Company', 'Date'])

        # 티커별로 특성 계산
        result_list = []

        for ticker in df['Company'].unique():
            ticker_data = df_combined[df_combined['Company'] == ticker].copy()
            ticker_data = ticker_data.sort_values('Date')

            # Daily Return
            ticker_data['Daily_Return_raw'] = ticker_data['Close'].pct_change()
            ticker_data['Daily_Return_calc'] = (ticker_data['Close'] - ticker_data['Close'].shift(1)) / ticker_data['Close'].shift(1)
            ticker_data['Daily_Return'] = ticker_data['Daily_Return_calc'].fillna(0)

            # Cumulative Return
            ticker_data['Cum_Return'] = (1 + ticker_data['Daily_Return']).cumprod() - 1

            # Moving Averages
            ticker_data['MA_5'] = ticker_data['Close'].rolling(window=5, min_periods=1).mean()
            ticker_data['MA_20'] = ticker_data['Close'].rolling(window=20, min_periods=1).mean()
            ticker_data['MA_60'] = ticker_data['Close'].rolling(window=60, min_periods=1).mean()

            # Volatility (20-day)
            ticker_data['Volatility_20d'] = ticker_data['Daily_Return'].rolling(window=20, min_periods=1).std() * np.sqrt(252)

            # Drawdown
            ticker_data['Cum_Max'] = ticker_data['Close'].expanding().max()
            ticker_data['Drawdown'] = (ticker_data['Close'] - ticker_data['Cum_Max']) / ticker_data['Cum_Max']
            ticker_data['MDD'] = ticker_data['Drawdown'].expanding().min()
            ticker_data['DD_Short'] = ticker_data['Drawdown'].rolling(window=60, min_periods=1).min()

            # Volume indicators
            ticker_data['Vol_MA_20'] = ticker_data['Volume'].rolling(window=20, min_periods=1).mean()
            ticker_data['Vol_Ratio'] = ticker_data['Volume'] / ticker_data['Vol_MA_20']
            ticker_data['Vol_Std_20'] = ticker_data['Volume'].rolling(window=20, min_periods=1).std()
            ticker_data['Vol_Z_Score'] = (ticker_data['Volume'] - ticker_data['Vol_MA_20']) / ticker_data['Vol_Std_20']
            ticker_data['Log_Volume'] = np.log1p(ticker_data['Volume'])
            ticker_data['Log_Volume_W'] = ticker_data['Log_Volume'].rolling(window=5, min_periods=1).mean()

            # RSI
            delta = ticker_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss
            ticker_data['RSI_14'] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            ticker_data['BB_Middle'] = ticker_data['Close'].rolling(window=20, min_periods=1).mean()
            bb_std = ticker_data['Close'].rolling(window=20, min_periods=1).std()
            ticker_data['BB_Upper'] = ticker_data['BB_Middle'] + (bb_std * 2)
            ticker_data['BB_Lower'] = ticker_data['BB_Middle'] - (bb_std * 2)
            ticker_data['BB_Width'] = (ticker_data['BB_Upper'] - ticker_data['BB_Lower']) / ticker_data['BB_Middle']

            # Price Gap
            ticker_data['Prev_Close'] = ticker_data['Close'].shift(1)
            ticker_data['Gap'] = ticker_data['Open'] - ticker_data['Prev_Close']
            ticker_data['Gap_Pct'] = ticker_data['Gap'] / ticker_data['Prev_Close']

            # Period Returns
            ticker_data['Return_1M'] = ticker_data['Close'].pct_change(periods=21)
            ticker_data['Return_3M'] = ticker_data['Close'].pct_change(periods=63)
            ticker_data['Return_6M'] = ticker_data['Close'].pct_change(periods=126)

            # Extreme changes
            ticker_data['Is_Extreme_Change'] = (ticker_data['Daily_Return'].abs() > 0.05).astype(int)

            # 새로운 데이터만 추출
            new_dates = df[df['Company'] == ticker]['Date'].values
            ticker_new = ticker_data[ticker_data['Date'].isin(new_dates)]

            result_list.append(ticker_new)

        df_result = pd.concat(result_list, ignore_index=True)

        print(f"✓ 특성 계산 완료: {len(df_result)} 행")

        return df_result

    def update_dataset(self, df_new: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
        """
        기존 데이터셋에 새로운 데이터 추가

        Parameters:
        -----------
        df_new : pd.DataFrame
            새로운 데이터
        output_path : str
            저장할 파일 경로 (기본값: 기존 파일 덮어쓰기)

        Returns:
        --------
        pd.DataFrame : 업데이트된 전체 데이터셋
        """
        if df_new.empty:
            print("업데이트할 새로운 데이터가 없습니다.")
            return self.df_existing

        print("\n데이터셋 업데이트 중...")

        # 기존 데이터와 병합
        df_updated = pd.concat([self.df_existing, df_new], ignore_index=True)
        df_updated = df_updated.sort_values(['Company', 'Date'])
        df_updated = df_updated.drop_duplicates(subset=['Date', 'Company'], keep='last')

        # 저장
        if output_path is None:
            output_path = self.existing_data_path

        df_updated.to_csv(output_path, index=False)

        print(f"✓ 데이터셋 업데이트 완료")
        print(f"  전체: {len(df_updated):,} 행")
        print(f"  날짜 범위: {df_updated['Date'].min().date()} ~ {df_updated['Date'].max().date()}")
        print(f"  저장 위치: {output_path}")

        return df_updated


def main():
    """테스트 실행"""
    data_path = 'Data_set/stock_features_clean.csv'

    fetcher = DailyDataFetcher(data_path)
    fetcher.load_existing_data()

    # 최신 데이터 가져오기
    df_new = fetcher.fetch_recent_data(days_back=7)

    if not df_new.empty:
        # 특성 계산
        df_new_features = fetcher.calculate_features(df_new)

        # 데이터셋 업데이트
        fetcher.update_dataset(df_new_features)


if __name__ == "__main__":
    main()

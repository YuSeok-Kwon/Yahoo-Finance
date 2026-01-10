# Data Dictionary

---

## 1. 개요

프로젝트에서 생성된 주요 데이터셋의 스키마와 컬럼 의미를 정의

---

## 2. 파일 목록

| 파일명                      | 설명                                                                 | 주요 용도                      |
| --------------------------- | -------------------------------------------------------------------- | ------------------------------ |
| `stock_daily_master.csv`    | **메인 데이터셋**. 500개 기업의 일별 시세 + 기술적 지표 + 섹터 정보. | 개별 종목 분석, 상세 차트      |
| `stock_sector_summary.csv`  | 섹터별 일간 평균 통계 (수익률, 변동성 등).                           | Sector Heatmap, 시장 흐름 파악 |
| `stock_monthly_summary.csv` | 월간 단위로 집계된 OHLCV 데이터.                                     | 장기 추세 분석, 월봉 차트      |

---

## 3. 상세 스키마

### A. `stock_daily_master.csv`

**Granularity**: 1행 = 1기업 / 1일

| 컬럼명 (Column)    | 타입  | 설명 (Description)     | 비고 (Formula)                              |
| ------------------ | ----- | ---------------------- | ------------------------------------------- |
| **Date**           | Date  | 거래일자 (YYYY-MM-DD)  | 시간 정보 제거됨                            |
| **Company**        | Str   | 기업 티커              | 예: AAPL, MSFT                              |
| **Sector**         | Str   | 대분류 섹터            | 예: Technology, Healthcare                  |
| **Industry**       | Str   | 중분류 산업군          | 예: Consumer Electronics                    |
| **Open**           | Float | 시가                   |                                             |
| **High**           | Float | 고가                   |                                             |
| **Low**            | Float | 저가                   |                                             |
| **Close**          | Float | 종가                   | 수정 주가 반영 여부 확인 필요               |
| **Volume**         | Int   | 거래량                 |                                             |
| **Daily_Return**   | Float | 일간 수익률            | `(Today_Close / Prev_Close) - 1`            |
| **Cum_Return**     | Float | 누적 수익률            | `(Today_Close / Start_Close) - 1`           |
| **MA_5**           | Float | 5일 이동평균선         | 단기 추세                                   |
| **MA_20**          | Float | 20일 이동평균선        | 중기 추세 (생명선)                          |
| **MA_60**          | Float | 60일 이동평균선        | 장기 추세 (수급선)                          |
| **Volatility_20d** | Float | 20일 변동성 (표준편차) | 리스크 지표                                 |
| **MDD**            | Float | 최대 낙폭              | `(Close - Historical_Max) / Historical_Max` |
| **Vol_Ratio**      | Float | 거래량 급증 비율       | `Volume / Vol_MA_20`                        |
| **Vol_Z_Score**    | Float | 거래량 이상치 점수     | `(Volume - Mean) / Std` (3 이상시 급증)     |
| **Gap_Pct**        | Float | 시가 갭 비율           | `(Open / Prev_Close) - 1`                   |

### B. `stock_sector_summary.csv`

**Granularity**: 1행 = 1섹터 / 1일

| 컬럼명                | 타입  | 설명                              |
| --------------------- | ----- | --------------------------------- |
| **Date**              | Date  | 거래일자                          |
| **Sector**            | Str   | 섹터명                            |
| **Sector_Return**     | Float | 해당 섹터 내 기업들의 평균 수익률 |
| **Sector_Price_Avg**  | Float | 해당 섹터 내 기업들의 평균 주가   |
| **Sector_Vol_Ratio**  | Float | 해당 섹터의 평균 거래 활기도      |
| **Sector_Volatility** | Float | 해당 섹터의 평균 변동성           |
| **Sector_MDD**        | Float | 해당 섹터의 평균 낙폭             |
| **Stock_Count**       | Int   | 집계에 포함된 기업 수             |

### C. `stock_monthly_summary.csv`

**Granularity**: 1행 = 1기업 / 1월

| 컬럼명             | 타입  | 설명                         |
| ------------------ | ----- | ---------------------------- |
| **Date**           | Date  | 해당 월의 마지막 거래일      |
| **Company**        | Str   | 기업 티커                    |
| **Monthly_Return** | Float | 월간 수익률 (%)              |
| **Monthly_Range**  | Float | 월간 등락폭 (고가-저가) 비율 |
| **Volume**         | Int   | 월간 총 거래량               |

---

# 데이터 처리 작업 로그

---

## 1. Phase 1: 초기 데이터 로드 및 QA

**목표**: 원본 데이터(`stock_details_5_years.csv`)를 로드하고, 결측치 처리 및 1차 파생변수를 생성함.

### 수행 내용

- **스크립트**: `01_Data_Loader_QA.ipynb` (실행: `run_qa_loader.py`)
- **전처리**:
  - 시간 정보 제거 (`YYYY-MM-DD` 포맷 통일)
  - 결측치 및 중복값 검사 (Result: **0건, Clean**)
- **1차 파생변수 생성**:
  - `Daily_Return` (일간 수익률)
  - `MA_5`, `MA_20`, `MA_60` (이동평균)
  - `Volatility_20d` (20일 변동성)
  - `MDD` (최대 낙폭)

### 결과 데이터

- **Row Count**: 602,962행
- **저장 파일**: `stock_daily_master.csv`, `stock_daily_master.pkl`

### 발견된 문제점

- **섹터 정보 누락**: 분석 대상 기업의 **90% 이상**이 `Sector = 'Others'`로 분류됨.
- **원인**: 초기 매핑 테이블이 소수 대형주(20개)에 대해서만 정의되어 있었음.

---

## 2. Phase 2: 섹터 정보 보완

**목표**: 외부 API를 활용하여 미분류 기업(Others)의 메타데이터(Sector, Industry)를 확보함.

### 수행 내용

- **스크립트**: `02_Sector_Update.ipynb` (실행: `run_sector_update.py`)
- **방법**: `yfinance` 라이브러리를 사용하여 Ticker별 메타데이터 수집 (Batch Processing).
- **대상**: 'Others'로 분류된 471개 기업.

### 업데이트 결과

| 순위 | 섹터 (Sector)          | 데이터 수 (Rows) | 비고                |
| :--: | :--------------------- | ---------------: | :------------------ |
|  1   | **Financial Services** |           97,900 | 금융주 비중 최다    |
|  2   | **Technology**         |           93,282 |                     |
|  3   | **Healthcare**         |           69,462 |                     |
|  4   | **Industrials**        |           69,226 |                     |
|  5   | **Consumer Cyclical**  |           53,616 | 경기민감 소비재     |
|  6   | **Energy**             |           47,804 |                     |
|  -   | ...                    |              ... |                     |
|  -   | **Unknown**            |           11,322 | 식별 불가 (약 1.8%) |

### 최종 산출물

- **파일명**: `stock_daily_master_v2.csv` (및 `.pkl`)

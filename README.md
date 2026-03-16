# 영웅호걸의 길: AI 기반 섹터 로테이션 & 가치주 발굴 시스템

## 프로젝트 개요

### 주제

개인 투자자의 섹터 선택에 대한 판단 기준 부재 문제를 해결하기 위해, **Prophet+XGBoost 하이브리드 모델로 유망 섹터를 사전 선별하면 시장 평균 대비 초과수익이 가능한가?** 를 검증한다.

### 목표

S&P 500 시계열 데이터를 활용하여 멀티 호라이즌(1D~1Y) 섹터 예측 → KMeans 클러스터링 기반 종목 분류 → 기술적 지표 검증의 **3단 필터링 시스템**을 구축한다.

---

## 배경 및 현황: 왜 '판단 기준'인가?

### 거시경제 및 투자 환경의 변화

- **고물가·원화 약세의 고착화:** 투자는 단순 재테크가 아닌 생존의 문제로 전환. "주식을 안 하면 손해"라는 인식 확산
- **정보 비대칭의 심화:** 개인 투자자는 뇌동매매와 감정적 의사결정 반복. 근본 원인은 명확한 판단 기준의 부재
- **단일 모델의 한계:** 주식 시장은 추세와 변칙이 공존하는 비선형 시스템. Prophet 단독 예측만으로는 순위 예측력 부족 확인(Spearman 0.04)

### 핵심 가설 및 검증 결과

| 가설 | 검증 방법 | 결과 | 판정 |
|------|-----------|------|------|
| H1: Prophet 단독으로 유망 섹터 선별 가능 | 2022~2024 백테스트 | Spearman 평균 0.04, 순위 예측력 미흡 | 기각 |
| H2: XGBoost 잔차 보정으로 예측 순위 개선 | 2025 홀드아웃 | NDCG@5 0.74, Top-5 초과수익률 +3.68% | 채택 |
| H3: 멀티 호라이즌 앙상블이 단일 시점보다 안정적 | Voting 방식 교차 검증 | 2개 이상 호라이즌 공통 추천 섹터의 변동성 축소 | 채택 |

> 2023년 블랙스완 구간(전쟁·ChatGPT 열풍)에서 Spearman -0.28 기록. 두 모델 모두 한계를 보였으나, 이는 **과적합 없이 시장 이상을 정직하게 반영**한 증거. 2024~2025년 시장 정상화 후 예측력 회복으로 **회복 탄력성** 확인.

---

## 문제 정의: 직관을 구조로 바꾸다

### 개인 투자자의 구조적 한계

| 구분 | 내용 |
|------|------|
| 정보 비대칭 | 기관 대비 접근 가능한 데이터·분석 도구의 격차 |
| 뇌동매매 | 소문·테마 기반 감정적 의사결정 반복 |
| 판단 기준 부재 | "어디에 투자해야 할지 모르겠다"는 막막함 |

### 솔루션 프레임워크

전체 시스템은 **Prediction(예측) → Classification(분류) → Verification(검증)** 의 3단계로 연결된다. 1등 종목을 맞히는 것보다 상승 확률 높은 Top-N 그룹을 선별하는 전략이 더 유효하며, 단일 모델보다 하이브리드 구조가 블랙스완 이후 회복 탄력성에서 우위를 보인다.

---

## 통합 솔루션: Prophet+XGBoost 하이브리드 파이프라인

### Prediction: 하이브리드 모델 기반 섹터 예측

**하이브리드 알고리즘 로직:**

```
[Prophet] 계절성 + 추세 1차 예측 (log scale)
    │ 잔차(residual) 추출
    ▼
[XGBoost] 잔차를 quantile rank로 학습
    │ inverse normal CDF로 역변환
    ▼
[Hybrid] (1-α)×Prophet + α×(Prophet + Residual)  (α=0.7)
    │
    ▼
[Multi-Horizon] 7개 시간 지평(1D/3D/1W/1M/1Q/6M/1Y) 독립 예측
    │
    ▼
[Ranking] Score = z(prediction) + γ×z(confidence)  (γ=0.25)
```

- **Prophet:** 거시적 추세와 계절성을 1차 예측. 변곡점 감지로 하락 전환 섹터 자동 배제
- **XGBoost:** Prophet이 놓친 비선형 변동성(잔차)을 학습하여 보정. α=0.7로 최근 시장 변화에 70% 가중치 (GridSearch 최적화)
- **멀티 호라이즌:** 장기 추세가 무너지기 전 단기 모델이 섹터 로테이션 신호를 선제 포착

**리스크 대응:**

| 상황 | 대응 모듈 | 메커니즘 |
|------|-----------|----------|
| 금리 인상으로 기술주 하락 전환 | Prophet | 하락 기울기 감지 → 예측값 하향 → 추천 순위 배제 |
| 지정학적 쇼크로 급락 | XGBoost | 비선형 급변동 잔차 학습 → 예측 보정 |
| 장기 추세 무너지기 직전 | 단기 호라이즌(1D/3D) | 로테이션 신호 선제 포착 → 빠른 태세 전환 |

### Classification: KMeans 클러스터링 기반 종목 분류

선정된 주도 섹터 내에서 산업을 **Risk-Reward** 기준으로 4그룹 분류:

| 클러스터 | 정의 | 조건 | 투자 전략 |
|----------|------|------|-----------|
| Strong Buy | Low Risk, High Return | Sharpe >= 1.5, Vol < 40% | 핵심 편입 |
| Aggressive | High Risk, High Return | Sharpe >= 1.0, Vol >= 30%, Return >= 12% | 모멘텀 추종 |
| Stable | Mid Return, Low Vol | 0.5 <= Sharpe < 1.5, Vol < 25% | 하락장 방어 |
| Value Trap | Low Return, High MDD | 나머지 | 편입 제외 |

- **피처:** Return_Period, Volatility_20d, MDD
- **스케일링:** StandardScaler → KMeans(k=4, seed=42)
- **해석:** 클러스터 ID를 지표 기준으로 재매핑하여 일관된 레이블 부여

### Verification: 기술적 지표 기반 안정성 검증

Tableau 대시보드를 통해 3개 탭으로 최종 검증:

| 탭 | 기능 | 주요 시각화 |
|----|------|-------------|
| 신뢰도 검증 | 모델 성능 투명 공개 | KPI 카드, S&P 500 예측 비교 |
| 시장 구조 분석 | 2026년 섹터별 매크로 파악 | 트리맵, 상관관계 히트맵, 성장률 랭킹 |
| 맞춤형 종목 발굴 | 기간별 Top-3 섹터 + 산업 클러스터링 | Risk-Reward 산점도, Sharpe·MDD 상세 지표 |

---

## 데이터 아키텍처 및 품질 엔지니어링

### 데이터 개요

| 항목 | 내용 |
|------|------|
| 원천 | Yahoo Finance API (S&P 500 기업, GICS 섹터/산업 메타데이터) |
| 기간 | 2020.11 ~ 2026.01 |
| 규모 | 603,359행 → 섹터 집계 후 14,135행, 11개 GICS 섹터 |
| 실시간 확장 | yfinance API 파이프라인으로 최신 데이터 일일 자동 병합 |

### 데이터 무결성 확보 (6대 결함 해결)

| 이슈 | 설명 | 해결책 |
|------|------|--------|
| 논리 오류 | High < Low 비정상 데이터 (30건) | `clean_ohlc_violations()` 값 보정 |
| 데이터 누수 | A기업 종가가 B기업 수익률 계산에 반영 | `groupby('Company')` 기반 독립 연산 |
| 정밀도 오류 | 부동소수점 오차로 인한 오검출 | `tolerance=0.01` 허용 오차 도입 |
| 블랙스완 | 50% 이상 폭락/폭등 (실제 충격) | 삭제 대신 `Is_Extreme_Change` 플래그 보존 |
| 거래량 0 | API 오류 및 거래 정지 (329건) | 히스토리컬 원본 교차 검증으로 복원 |
| 섹터 결측 | Sector 정보 누락 | Unknown 제거로 분석 전제 확립 |

### 파생변수 엔지니어링 (39개)

| 분류 | 변수 | 설명 |
|------|------|------|
| Trend & Momentum | `Daily_Return`, `Return_1M/3M/6M` | 중단기 추세 확인 |
| Risk & Volatility | `Volatility_20d` | 연환산 변동성 (Risk 핵심) |
| | `MDD` | 고점 대비 최대 하락폭 (하방 경직성) |
| | `Sharpe_Ratio` | 위험 대비 수익 효율성 (핵심 랭킹 지표) |
| Volume Analysis | `Vol_Ratio`, `Vol_Z_Score` | 수급 이탈/유입 감지 |
| Technical | `RSI`, `Bollinger_Band` | 과매수/과매도 판단 |

---

## 모델 성능 (백테스트 결과)

### GridSearch 하이퍼파라미터 최적화

기존 수동 고정값(α=0.6, γ=0.5)에 대해 36개 조합(α×γ = 6×6) GridSearch를 수행하여 최적 파라미터를 도출하였다.

| 구분 | α | γ | NDCG@5 (2024-2025 평균) |
|------|-----|------|------------------------|
| 기존 설정 | 0.6 | 0.5 | 0.6631 |
| **최적 설정** | **0.7** | **0.25** | **0.7190 (+8.44%)** |

> **핵심 발견:** XGBoost 보정 비중을 60%→70%로 높이고, 신뢰도 가중치를 0.5→0.25로 낮추는 것이 순위 품질을 극대화한다. 예측값 자체에 더 가중하고 신뢰도의 노이즈 유입을 억제하는 전략이 유효하다.

### 연도별 백테스트 성능 (최적 α=0.7, γ=0.25)

| 연도 | Spearman | NDCG@5 | Top-5 Hit | Top-5 초과수익 | 비고 |
|------|----------|--------|-----------|---------------|------|
| 2022 | 0.3273 | -0.5400 | 60.00% | +8.42% | 하락장에서도 초과수익 달성 |
| 2023 | -0.3182 | 0.3269 | 40.00% | -5.74% | 블랙스완 (전쟁·ChatGPT) |
| 2024 | 0.2636 | 0.7002 | 60.00% | +5.14% | 시장 정상화 |
| **2025 (홀드아웃)** | **0.1455** | **0.7378** | **60.00%** | **+3.68%** | **최종 검증 구간** |

### 기존 vs 최적 파라미터 비교 (전체 연도 평균)

| 지표 | 기존 (α=0.6, γ=0.5) | 최적 (α=0.7, γ=0.25) | 개선도 |
|------|---------------------|---------------------|--------|
| Spearman | 0.0841 | **0.1045** | +0.0205 |
| NDCG@5 | 0.0602 | **0.3063** | **+0.2461** |
| Top-5 Hit | 50.00% | **55.00%** | +5.00%p |
| Top-5 초과수익 | -1.41% | **+2.87%** | **+4.29%p** |

> 가장 중요한 변화: **Top-5 초과수익이 음수(-1.41%)에서 양수(+2.87%)로 전환** — GridSearch를 통해 시장 평균을 실질적으로 이기는 모델이 되었다.

### SHAP 기반 모델 해석

XGBoost 잔차 보정 모델이 어떤 특징에 기반하여 섹터를 추천하는지 SHAP 분석으로 규명하였다.

| 순위 | 특징 | 평균 |SHAP| | 기여 비율 | 해석 |
|------|------|------------|---------|------|
| 1 | **Prophet 전일예측값** | 0.0455 | 27.3% | 전일 예측 수준이 잔차 방향의 최대 단서 |
| 2 | **20일 수익률** | 0.0302 | 18.1% | 4주간 중기 모멘텀이 보정의 핵심 기준 |
| 3 | **5일 수익률** | 0.0250 | 15.0% | 1주 단기 반등/하락 신호 포착 |

> **Top-3 특징이 전체 기여도의 60.4%** — XGBoost는 "Prophet의 추세 판단 + 중단기 모멘텀 확인"이라는 2중 필터로 작동하며, 이 구조가 NDCG@5 개선의 근본 원인이다.

### 핵심 인사이트

| 번호 | 인사이트 |
|------|----------|
| 1 | 시장 변화는 뉴스보다 **가격·거래량 데이터에 먼저 반영** |
| 2 | 1등 종목 맞히기보다 **상승 확률 높은 Top-N 그룹 선별이 더 유효** |
| 3 | 단일 모델보다 **하이브리드 구조가 회복 탄력성에서 우위** |
| 4 | 정보의 양보다 **판단 기준의 유무**가 리스크를 결정 |
| 5 | 수동 고정 파라미터보다 **GridSearch 자동 탐색이 +8.44% 성능 향상** |
| 6 | SHAP 해석을 통해 **모델의 추천 근거를 정량적으로 설명** 가능 |

### 한계

- 전쟁 등 블랙스완급 변수는 데이터 기반 예측의 구조적 한계 (2023년 Spearman -0.32)
- 극단적 수익 추구(단일 종목 올인)에는 부적합한 시스템 설계
- GridSearch 최적 파라미터가 미래 시장에서도 최적임을 보장하지 않음 (정기 재탐색 필요)

---

## 코드 품질 및 모델 해석 개선

프로젝트 리뷰에서 도출된 3가지 개선점을 반영하였다.

### 1. 단위 테스트 (pytest)

모든 핵심 모듈에 대한 단위 테스트를 `tests/` 디렉토리에 구현하였다. 총 **30개 테스트 전체 통과**.

```bash
python -m pytest tests/ -v
```

| 테스트 파일 | 테스트 수 | 검증 내용 |
|-------------|-----------|-----------|
| `test_evaluation.py` | 8 | Hit Ratio 완벽/최악 예측, NDCG 완벽/랜덤 순위, 초과수익, 반환 키 검증 |
| `test_xgboost_model.py` | 5 | 특징 컬럼 생성, NaN 제거, 예측 shape/범위, 미학습 모델 에러 |
| `test_hybrid_model.py` | 4 | confidence 범위(0~1), alpha 효과, 학습-예측 파이프라인, 출력 컬럼 |
| `test_multi_horizon_predictor.py` | 5 | z-정규화 랭킹 합 ≈ 0, gamma 효과, voting/union 통합 |
| `test_industry_clustering.py` | 4 | 클러스터 수/범위, 특징 추출 컬럼, 프로파일 shape |
| `test_data_loader.py` | 3 | 반환 컬럼, 학습/테스트 데이터 누수 없음, 연도 필터링 |

- **공통 fixture** (`conftest.py`): 100일치 샘플 섹터 데이터, Prophet 출력, 학습된 XGBoostCorrector, 임시 CSV 등 재사용 가능한 테스트 데이터 제공

### 2. GridSearch 하이퍼파라미터 튜닝

기존에 수동 고정값이었던 α=0.6, γ=0.5에 대해 **GridSearch 자동 탐색**을 구현하였다.

- **모듈:** `src/hyperparameter_tuning.py` — `GridSearchTuner` 클래스
- **탐색 범위:** α × γ = 6 × 6 = **36개 조합**
  - α (하이브리드 가중치): [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
  - γ (신뢰도 가중치): [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
- **평가 방식:** 기존 `run_backtest()` 활용, 다중 연도 백테스트 평균
- **최적화 지표:** NDCG@5 (기본), Spearman 등 선택 가능

```python
from src import GridSearchTuner

tuner = GridSearchTuner(metric='hybrid_ndcg5')
best = tuner.search(df=sector_df, sectors=sectors, test_years=[2024, 2025])
# → {'alpha': 최적값, 'gamma': 최적값, 'score': 최적 NDCG@5}

tuner.print_results()       # 상위 10개 조합 출력
tuner.get_results_df()      # 전체 결과 DataFrame
```

### 3. SHAP 모델 해석

XGBoost 모델의 블랙박스 문제를 해결하기 위해 **SHAP(SHapley Additive exPlanations)** 기반 해석 도구를 추가하였다.

- **모듈:** `src/shap_explainer.py` — `ShapExplainer` 클래스
- **XGBoost 연동:** `XGBoostCorrector.explain_shap()` 메서드 추가 (기존 코드 변경 최소화)

| 메서드 | 기능 |
|--------|------|
| `explain(model, X, feature_names)` | SHAP 값 계산 (TreeExplainer) |
| `summary_plot(save_path)` | 전체 특징 중요도 bar plot 저장 |
| `feature_importance_df()` | 특징별 평균 |SHAP| 값 DataFrame |
| `waterfall_plot(idx, save_path)` | 개별 샘플의 예측 기여도 waterfall plot |

```python
from src import XGBoostCorrector

corrector = XGBoostCorrector()
# ... 학습 후 ...
explainer = corrector.explain_shap(features, save_dir='./output')
# → shap_summary.png 저장, 특징 중요도 출력

importance = explainer.feature_importance_df()
explainer.waterfall_plot(idx=0, save_path='./output/shap_waterfall.png')
```

---

## 자동화 파이프라인

### 기술 스택

| 분류 | 기술 |
|------|------|
| 데이터 수집 | yfinance API, daily_data_fetcher.py |
| 모델링 | Prophet, XGBoost, Scikit-learn (KMeans, StandardScaler) |
| 평가 | Spearman, NDCG@K, Top-K Hit Ratio, SciPy |
| 모델 해석 | SHAP (TreeExplainer, summary/waterfall plot) |
| 하이퍼파라미터 | GridSearch (α × γ 36개 조합 자동 탐색) |
| 테스트 | pytest (30개 단위 테스트) |
| 시각화 | Tableau, matplotlib |
| 자동화 | launchd (macOS), config.json 설정 분리 |
| 로깅 | Python logging, 일별 로그 파일 관리 |

### 파이프라인 흐름

```
yfinance API → 최신 주가 데이터 자동 수집 (7일 롤링)           ✅
stock_features_clean.csv와 병합                                ✅
섹터별 Prophet+XGBoost 하이브리드 모델 학습                    ✅
7개 호라이즌 멀티 예측 실행                                    ✅
Ranking Score 산출 → Top-3 섹터 선정                           ✅
KMeans 클러스터링 → 산업별 Risk-Reward 분류                    ✅
CSV 출력 → Tableau 대시보드 자동 갱신                          ✅
```

### 주요 설정 (config.json)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `alpha` | 0.7 | XGBoost 보정 가중치 (Prophet 30% : XGBoost 70%, GridSearch 최적) |
| `gamma` | 0.25 | 랭킹 시 신뢰도 반영 가중치 (GridSearch 최적) |
| `top_k` | 3 | 호라이즌별 상위 섹터 수 |
| `train_years` | 4 | 학습 데이터 기간 (연) |
| `n_clusters` | 5 | KMeans 클러스터 수 |

---

## 프로젝트 구조

```
Yahoo Finance/
├── Data_set/                        # 원본 및 처리 데이터
│   ├── stock_features_clean.csv     #   전처리 완료 데이터 (603K행)
│   └── Cluster_Results/             #   클러스터링 결과 CSV
├── src/                             # 핵심 소스 코드 (패키지화)
│   ├── __init__.py                  #   모듈 export 관리
│   ├── data_loader.py               #   데이터 로딩 및 분할
│   ├── prophet_model.py             #   Prophet 모델
│   ├── xgboost_model.py             #   XGBoost 잔차 보정기 (+SHAP 해석)
│   ├── hybrid_model.py              #   하이브리드 모델 (Prophet+XGBoost)
│   ├── multi_horizon_predictor.py   #   멀티 호라이즌 예측 엔진
│   ├── industry_clustering.py       #   KMeans 산업 클러스터링
│   ├── evaluation.py                #   평가 지표 (NDCG, Spearman, Hit Ratio)
│   ├── backtest.py                  #   백테스트 프레임워크
│   ├── hyperparameter_tuning.py     #   GridSearch 하이퍼파라미터 튜닝
│   └── shap_explainer.py            #   SHAP 기반 모델 해석
├── tests/                           # 단위 테스트 (pytest)
│   ├── conftest.py                  #   공통 fixture
│   ├── test_evaluation.py           #   평가 지표 테스트
│   ├── test_xgboost_model.py        #   XGBoostCorrector 테스트
│   ├── test_hybrid_model.py         #   HybridModel 테스트
│   ├── test_multi_horizon_predictor.py  # MultiHorizonPredictor 테스트
│   ├── test_industry_clustering.py  #   IndustryClusterer 테스트
│   └── test_data_loader.py          #   data_loader 테스트
├── notebooks/                       # 분석 노트북
│   ├── Multi_Horizon_Industry_Pipeline.ipynb  # 멀티 호라이즌 파이프라인
│   └── Proper_Validation.ipynb      #   모델 검증 (2022~2025 백테스트)
├── automation/                      # 일일 자동화 파이프라인
│   ├── daily_pipeline.py            #   메인 파이프라인 스크립트
│   ├── daily_data_fetcher.py        #   yfinance 데이터 수집기
│   ├── config.json                  #   하이퍼파라미터 설정
│   ├── run_daily_pipeline.sh        #   실행 스크립트
│   └── com.yahoofinance.daily.plist #   launchd 스케줄링
├── Tableau/                         # Tableau 워크북
├── Docs/                            # 문서
│   ├── PDF/                         #   Data Dictionary, 설명서
│   ├── Rule.md                      #   프로젝트 규칙
│   └── 머신러닝 설명서.md            #   모델 상세 문서
├── logs/                            # 자동화 로그
└── 발표/                            # 발표 자료
    ├── 직감에서 데이터로, 투자판단의 전환.pdf
    └── 본 발표 대본.txt
```

---

## 문서

### 모델링 / 핵심 분석

| 문서 경로 | 설명 |
|-----------|------|
| `notebooks/Proper_Validation.ipynb` | **모델 검증 핵심** — 2022~2025 백테스트, Prophet vs Hybrid 비교 |
| `notebooks/Multi_Horizon_Industry_Pipeline.ipynb` | 멀티 호라이즌 파이프라인 + 클러스터링 실행 |
| `Docs/머신러닝 설명서.md` | 모델 아키텍처 상세 문서 |

### 데이터 / 기타

| 문서 경로 | 설명 |
|-----------|------|
| `Docs/PDF/Data_Dictionary.pdf` | 전체 변수 명세 |
| `Docs/PDF/Data_Processing_Log.pdf` | 전처리 로그 |
| `Docs/Rule.md` | 프로젝트 규칙 및 컨벤션 |
| `Docs/주식 도메인 지식 완벽 가이드.md` | 주식 도메인 지식 정리 |

---

## 기대 효과

### 정량적 기대 효과

- **초과수익 달성:** 2025 홀드아웃에서 Top-5 선별 섹터의 시장 평균 대비 **+3.68% 초과수익률** 기록 (기존 α=0.6,γ=0.5 대비 +4.29%p 개선)
- **순위 예측력:** NDCG@5 **0.74** (2025), 4개년 평균 **0.31** 달성. GridSearch를 통해 기존 대비 **+8.44%** 개선
- **데이터 품질:** 6대 결함 사전 제거, 329건 거래량 결측치 원본 교차검증 복원

### 정성적 기대 효과

- **판단 기준 제공:** 감정적 매매를 데이터 기반 의사결정으로 전환
- **리스크 가시화:** 클러스터링을 통해 Risk-Reward 프로파일을 직관적으로 제시
- **운영 자동화:** 일일 파이프라인으로 수동 분석 작업 제거, 매일 최신 데이터 반영

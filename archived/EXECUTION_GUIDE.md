# 07_Integrated_Prophet_Analysis.ipynb 실행 가이드

**마지막 업데이트**: 2026-01-16  
**변경사항**: Section 9와 Section 7.5 순서 재배치 완료

---

## ✅ 이제 순차 실행 가능합니다!

셀 순서가 수정되어 **"Run All Cells" 가능**해졌습니다.

---

## 🎯 실행 방법

### Option 1: Run All (권장)

```bash
# 1. Jupyter 실행
cd "/Users/yu_seok/Documents/workspace/nbCamp/Project/Yahoo Finance"
conda activate py_study
jupyter notebook 07_Integrated_Prophet_Analysis.ipynb

# 2. Jupyter 메뉴에서
Cell → Run All
```

**예상 소요시간**: 약 5-7분

---

### Option 2: 섹션별 실행

여전히 섹션별로 나눠서 실행하고 싶다면:

```
1. Cell 0~29 실행
   (데이터 로딩, 변수 설정)
   ↓
2. Cell 30~40 실행  ⭐ Section 9 (새 위치!)
   (Multi-Horizon Prediction)
   → df_multi_horizon_integrated 생성
   ↓
3. Cell 41~57 실행
   (Section 7.5: LTR 학습 및 평가)
   → df_sector_year 생성 (df_multi_horizon_integrated 사용)
   ↓
4. Cell 58~80 실행
   (나머지: 시각화, 2026 예측 등)
```

---

## 📊 주요 변경사항

### 변경 전 (문제 상황)

```
Cell 30-46: Section 7.5 (LTR)
  ↓ Cell 32가 df_multi_horizon_integrated 필요
  ❌ 하지만 아직 생성 안 됨!
  
Cell 63-73: Section 9 (Multi-Horizon)
  ↓ 여기서 df_multi_horizon_integrated 생성
  ⚠️ 너무 늦음!
```

**문제**: Cell 32 실행 시 `NameError` 발생

### 변경 후 (해결!)

```
Cell 30-40: Section 9 (Multi-Horizon) ← 이동됨!
  ↓ df_multi_horizon_integrated 생성
  ✓ 먼저 생성!
  
Cell 41-57: Section 7.5 (LTR)
  ↓ Cell 43가 df_multi_horizon_integrated 사용
  ✓ 이미 생성되어 있음!
```

**해결**: 의존성 순서 맞춤

---

## 🔍 검증 방법

각 섹션 실행 후 확인:

### Section 9 실행 후 (Cell 30-40)

```python
# 새 셀에서 확인
print('df_multi_horizon_integrated' in dir())  
# 출력: True

print(df_multi_horizon_integrated.shape)       
# 출력: (44, 15) 또는 비슷한 크기

print(df_multi_horizon_integrated.columns.tolist())
# 출력: ['test_year', 'Sector', 'pred_return_3d', 'pred_return_1w', ...]
```

### Section 7.5 실행 후 (Cell 41-57)

```python
# 새 셀에서 확인
print('df_sector_year' in dir())
# 출력: True

print('ltr_score_raw' in df_sector_year.columns)
# 출력: True

print(df_sector_year['ltr_score_raw'].describe())
# 출력: std가 0.01 이상이어야 함
```

---

## ⚠️ 주의사항

### 1. 환경 확인

```bash
# py_study 환경 필수!
conda activate py_study

# 패키지 확인
python -c "import prophet, xgboost, sklearn; print('OK')"
```

### 2. 데이터 파일 확인

```bash
# stock_features_clean.csv 존재 확인
ls -lh "Data_set/stock_features_clean.csv"
# 출력: ~280MB 파일이 있어야 함
```

### 3. 실행 시간

| 섹션 | 예상 시간 | 주요 작업 |
|------|-----------|-----------|
| Cell 0-29 | 30초 | 데이터 로딩 |
| Cell 30-40 (Section 9) | 2-3분 | 6개 horizon 예측 |
| Cell 41-57 (Section 7.5) | 10초 | LTR 학습 |
| Cell 58-80 | 1-2분 | 시각화, 예측 |
| **전체** | **5-7분** | |

---

## 🆘 문제 해결

### 에러 1: `NameError: name 'df_multi_horizon_integrated' is not defined`

**원인**: Section 9 (Cell 30-40)을 실행하지 않음

**해결**:
```python
# Cell 30-40을 실행하고 확인
print('df_multi_horizon_integrated' in dir())
```

### 에러 2: `ModuleNotFoundError: No module named 'multi_horizon_predictor'`

**원인**: Python 경로 문제

**해결**:
```python
# Notebook 첫 셀에 추가
import sys
sys.path.append('/Users/yu_seok/Documents/workspace/nbCamp/Project/Yahoo Finance')
```

### 에러 3: LTR 점수가 모두 비슷함 (std < 0.01)

**원인**: Multi-horizon 예측이 제대로 안 됨

**해결**:
```python
# Section 9 결과 확인
print(df_multi_horizon_integrated.groupby('Sector')['pred_return_1m'].describe())

# 각 섹터별로 pred_return이 다르게 나와야 함
# 만약 모두 비슷하면 Section 9 재실행
```

---

## 🎓 백업 파일

재배치 전 원본은 백업되었습니다:

```bash
ls -lh 07_Integrated_Prophet_Analysis_backup_*.ipynb
# 출력: 백업 파일들이 보임
```

문제 발생 시 백업으로 복원:

```bash
# 최신 백업 찾기
ls -lt 07_Integrated_Prophet_Analysis_backup_*.ipynb | head -1

# 복원 (백업 파일명을 실제 파일명으로 교체)
cp 07_Integrated_Prophet_Analysis_backup_20260116_143812.ipynb 07_Integrated_Prophet_Analysis.ipynb
```

---

## 📝 다음 단계: Industry Analysis 추가

Notebook이 정상 실행된 후, 맨 아래에 Section 10 추가:

```python
# ============================================================================
# Section 10: Industry-Level Analysis (NEW)
# ============================================================================

from industry_analysis import main_industry_analysis

# 2025년 산업 분석
results_2025 = main_industry_analysis(
    df_sector_year=df_sector_year,
    stock_data_path='Data_set/stock_features_clean.csv',
    year=2025,
    top_n_sectors=5,
    n_clusters=4
)

# 포트폴리오 출력
portfolio = results_2025['portfolio']
print("\n" + "="*80)
print("2025 Industry Portfolio")
print("="*80)
print(portfolio.to_string())

# 통계
exp_return = (portfolio['Expected_Return'] * portfolio['Weight']).sum()
volatility = (portfolio['Volatility'] * portfolio['Weight']).sum()
print(f"\nExpected Return: {exp_return:.2%}")
print(f"Volatility: {volatility:.2%}")
```

---

## 📚 추가 참고자료

- **산업 분석 사용법**: `industry_analysis_usage.md`
- **프로젝트 상태**: `PROJECT_STATUS.md`
- **버그 수정 이력**: `BUGFIX_SUMMARY_KR.md` (있는 경우)

---

**최종 확인**: 2026-01-16  
**작성자**: Sisyphus (OhMyOpenCode)

# 일일 자동화 파이프라인 설정 가이드

Yahoo Finance API로 매일 자동으로 데이터를 수집하고 멀티 호라이즌 클러스터링을 수행하는 자동화 시스템입니다.

## 📁 파일 구성

모든 자동화 파일은 `automation/` 폴더에 있습니다:

- `automation/daily_data_fetcher.py`: Yahoo Finance API 데이터 수집기
- `automation/daily_pipeline.py`: 전체 파이프라인 (데이터 수집 → 예측 → 클러스터링)
- `automation/run_daily_pipeline.sh`: 실행 스크립트
- `automation/com.yahoofinance.daily.plist`: macOS 자동 실행 설정 (launchd)
- `automation/config.json`: 설정 파일

## 🚀 설정 방법

### 1. 필요한 패키지 설치

```bash
pip install yfinance pandas numpy
```

### 2. 수동 실행 테스트

먼저 파이프라인이 정상 작동하는지 확인:

```bash
cd "/Users/yu_seok/Documents/workspace/nbCamp/Project/Yahoo Finance"
python automation/daily_pipeline.py
```

### 3. 자동 실행 설정 (macOS - launchd 사용)

#### 3-1. plist 파일을 LaunchAgents에 복사

```bash
cp automation/com.yahoofinance.daily.plist ~/Library/LaunchAgents/
```

#### 3-2. launchd에 등록

```bash
launchctl load ~/Library/LaunchAgents/com.yahoofinance.daily.plist
```

#### 3-3. 등록 확인

```bash
launchctl list | grep yahoofinance
```

#### 3-4. 즉시 실행 테스트 (예약 시간 기다리지 않고)

```bash
launchctl start com.yahoofinance.daily
```

#### 3-5. 로그 확인

```bash
# 실행 로그
tail -f logs/pipeline_$(date +%Y%m%d).log

# 표준 출력
tail -f logs/stdout.log

# 에러 로그
tail -f logs/stderr.log
```

### 4. 실행 시간 변경

기본 설정: **매일 오전 9시**

다른 시간으로 변경하려면 `automation/com.yahoofinance.daily.plist` 파일 수정:

```xml
<key>StartCalendarInterval</key>
<dict>
    <key>Hour</key>
    <integer>14</integer>  <!-- 오후 2시로 변경 -->
    <key>Minute</key>
    <integer>30</integer>  <!-- 30분으로 변경 -->
</dict>
```

변경 후 다시 로드:

```bash
launchctl unload ~/Library/LaunchAgents/com.yahoofinance.daily.plist
launchctl load ~/Library/LaunchAgents/com.yahoofinance.daily.plist
```

### 5. 자동 실행 중지

```bash
launchctl unload ~/Library/LaunchAgents/com.yahoofinance.daily.plist
```

### 6. 완전 제거

```bash
launchctl unload ~/Library/LaunchAgents/com.yahoofinance.daily.plist
rm ~/Library/LaunchAgents/com.yahoofinance.daily.plist
```

## 🛠️ 대안: cron 사용 (선택사항)

launchd 대신 cron을 사용하려면:

### cron 설정

```bash
crontab -e
```

다음 라인 추가 (매일 오전 9시 실행):

```
0 9 * * * cd "/Users/yu_seok/Documents/workspace/nbCamp/Project/Yahoo Finance" && /bin/bash run_daily_pipeline.sh
```

### cron 확인

```bash
crontab -l
```

### cron 로그 확인

```bash
tail -f logs/execution.log
```

## ⚙️ 설정 커스터마이징

`automation/config.json` 파일에서 쉽게 설정 변경 가능:

```python
def default_config(self) -> dict:
    return {
        # 데이터 수집
        'fetch_days_back': 7,  # 며칠 전부터 가져올지

        # 예측 파라미터
        'train_years': 4,      # 학습 데이터 연도
        'alpha': 0.6,          # 하이브리드 모델 가중치
        'gamma': 0.5,          # 랭킹 신뢰도 가중치
        'top_k': 3,            # 상위 섹터 수

        # 클러스터링
        'n_clusters': 5,       # 클러스터 개수

        # 호라이즌별 lookback 기간
        'horizon_lookback_map': {
            '1d': 60,
            '3d': 75,
            '1w': 90,
            '1m': 105,
            '1q': 120,
            '6m': 150,
            '1y': 180
        }
    }
```

## 📊 결과 확인

실행 결과는 다음 위치에 저장됩니다:

### CSV 파일
```
Data_set/Cluster_Results/
├── YYYYMMDD_1d_industry_features.csv
├── YYYYMMDD_1d_cluster_0.csv
├── YYYYMMDD_1d_cluster_1.csv
├── ...
```

### 로그 파일
```
logs/
├── pipeline_YYYYMMDD.log    # 파이프라인 실행 로그
├── execution.log              # cron 실행 로그
├── stdout.log                 # 표준 출력
└── stderr.log                 # 에러 로그
```

## 🔍 문제 해결

### 1. 파이프라인이 실행되지 않음

```bash
# 권한 확인
ls -la run_daily_pipeline.sh

# 권한 부여
chmod +x run_daily_pipeline.sh

# 수동 실행 테스트
./run_daily_pipeline.sh
```

### 2. Yahoo Finance API 오류

- 네트워크 연결 확인
- 티커 심볼이 유효한지 확인
- API 호출 제한 확인 (너무 많은 요청 시 대기)

### 3. 새로운 데이터가 없음

- 주말/공휴일에는 새로운 데이터가 없을 수 있음
- `fetch_days_back` 값을 늘려서 더 긴 기간 조회

### 4. launchd 작동 확인

```bash
# 상태 확인
launchctl list | grep yahoofinance

# 로그 확인
cat logs/stderr.log
cat logs/stdout.log
```

## 📅 실행 스케줄 예시

- **매일 오전 9시**: 장 시작 전 업데이트
- **매일 오후 4시**: 장 마감 후 업데이트
- **평일만 실행**: plist에 `Weekday` 추가

```xml
<key>StartCalendarInterval</key>
<dict>
    <key>Hour</key>
    <integer>9</integer>
    <key>Minute</key>
    <integer>0</integer>
    <key>Weekday</key>
    <integer>1</integer>  <!-- 월요일=1, 금요일=5 -->
</dict>
```

## 📧 알림 설정 (선택사항)

파이프라인 완료 시 이메일 알림을 받으려면 `daily_pipeline.py`에 이메일 전송 코드 추가 가능.

## 🔐 보안 주의사항

- API 키가 필요한 경우 환경 변수로 관리
- 로그 파일에 민감한 정보 기록하지 않기
- 정기적으로 로그 파일 정리

---

## 문의

문제가 발생하면 로그 파일을 확인하거나 이슈를 등록하세요.

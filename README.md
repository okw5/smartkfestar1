# 축제 방문객 예측 웹 서비스

이 프로젝트는 축제의 위치, 날짜, 유형, 예산 등의 정보를 입력하면 예측 모델을 통해 해당 축제의 예상 방문객 수를 제공하는 Flask 기반 웹 애플리케이션입니다.

## 주요 기능

- 축제 정보(지역, 날짜, 유형, 예산 등) 입력 폼 제공
- 입력 정보 기반 방문객 수 예측 결과 반환
- 예측 모델 및 데이터 전처리 자동 로드
- 웹 UI 및 RESTful API 제공 (기본적으로 웹폼 제공)

## 사용 기술

- Python, Flask
- scikit-learn, xgboost, pandas, numpy
- joblib (모델 및 전처리기 로드)
- HTML 템플릿 (Jinja2)
- Docker 지원

## 폴더 구조

```
├── app.py               # Flask 메인 애플리케이션
├── requirements.txt     # 필요 패키지 목록
├── Dockerfile           # Docker 이미지 빌드 설정
├── app.yaml             # 배포 환경(yaml)
├── model/               # 사전 학습된 모델, 전처리기, PCA 등
├── static/              # 정적 파일
└── templates/           # 웹 템플릿 (예: index.html)
```

## 설치 및 실행 방법

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 모델 파일 준비

`model/` 폴더에 아래 파일들이 준비되어 있어야 합니다:
- `best_model.joblib`
- `preprocessor.joblib`
- `pca_transformer.joblib`
- `address_freq_maps.joblib`
- `축제_데이터셋_업로드용.xlsx`

### 3. 개발 서버 실행

```bash
python app.py
```
- 기본적으로 `localhost:5000`에서 실행됩니다.

### 4. Docker로 실행 (옵션)

```bash
docker build -t smartkfestar .
docker run -p 5000:5000 smartkfestar
```

## 사용법

1. 브라우저에서 `localhost:5000` 접속
2. 폼에 축제 정보를 입력 후 예측 버튼 클릭
3. 예상 방문객 수가 화면에 표시됨

## 주요 입력 항목

- 광역자치단체, 시/군/구, 읍/면/동 (지역 정보)
- 축제 시작일
- 축제 종류
- 예산(백만원 단위)

## 참고/주의사항

- 예측 품질은 학습 데이터와 제공된 모델에 따라 달라질 수 있습니다.
- 모델 파일이 없으면 서비스가 정상적으로 작동하지 않습니다.
- 입력값이 누락되거나 잘못된 경우 에러 메시지가 표시됩니다.

---

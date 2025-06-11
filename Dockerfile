# 1. 베이스 이미지 설정: 가벼운 버전의 공식 파이썬 이미지를 사용합니다.
FROM python:3.12-slim

# 2. 환경 변수 설정: 파이썬 로그가 버퍼링 없이 즉시 출력되도록 합니다.
ENV PYTHONUNBUFFERED True

# 3. 작업 디렉토리 설정: 컨테이너 내에서 작업할 공간을 만듭니다.
WORKDIR /app

# 4. 의존성 파일 복사 및 설치:
# 먼저 requirements.txt만 복사하여 설치합니다.
# 이렇게 하면 앱 코드만 변경되었을 때, 매번 라이브러리를 새로 설치하지 않아 빌드 속도가 빨라집니다.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 애플리케이션 소스 코드 복사:
# 현재 디렉토리의 모든 파일(app.py, model/, static/, templates/ 등)을 컨테이너의 /app 디렉토리로 복사합니다.
COPY . .

# 6. 컨테이너가 리스닝할 포트 지정
EXPOSE 8080

# 7. 애플리케이션 실행: 컨테이너가 시작될 때 실행할 명령입니다.
# Gunicorn을 사용하여 0.0.0.0:8080 주소로 앱을 실행합니다.
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
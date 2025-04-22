# Python 3.11 슬림 버전을 기반 이미지로 사용 (버전은 필요에 따라 조정)
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1 \
    # Flask가 실행될 포트 (Fly.io는 내부적으로 8080을 선호)
    PORT=8080 \
    # Gunicorn 설정 (선택 사항)
    # WEB_CONCURRENCY=4 \
    # Sentence Transformers 캐시 경로 (index.py의 설정과 일치)
    SENTENCE_TRANSFORMERS_HOME=/data/sentence_transformers_cache

# 시스템 패키지 설치 (필요한 경우, 예: 특정 라이브러리가 C 컴파일러 등을 요구할 때)
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# 파이썬 의존성 설치
# requirements.txt 파일을 먼저 복사하여 레이어 캐싱 활용
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사 (api 폴더와 혹시 필요한 다른 파일)
# 여기서는 api 폴더 안의 내용만 app 폴더로 복사한다고 가정
COPY ./api /app

# Sentence Transformers 캐시 및 ChromaDB 런타임 데이터 디렉토리 생성 (볼륨 마운트 지점)
# 이 디렉토리들은 볼륨이 마운트될 위치이므로 미리 만들어두는 것이 좋음
RUN mkdir -p /data/chroma_db_runtime && \
    mkdir -p /data/sentence_transformers_cache && \
    # 해당 디렉토리에 쓰기 권한 부여 (non-root 유저로 실행 시 필요할 수 있음)
    chown -R nobody:nogroup /data

# Flask 앱 실행 (Gunicorn 사용)
# Gunicorn이 0.0.0.0:$PORT 에서 실행되도록 설정
# index: Flask 앱 인스턴스(app = Flask(__name__))가 있는 파일 이름 (index.py -> index)
# app: Flask 앱 인스턴스 변수 이름
# 워커 수(-w)는 앱의 I/O 특성과 사용 가능한 메모리에 따라 조정
CMD ["gunicorn", "-b", "0.0.0.0:8080", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "index:app"]
# 또는 gunicorn 기본 워커 사용 시:
# CMD ["gunicorn", "-b", "0.0.0.0:8080", "-w", "2", "index:app"]
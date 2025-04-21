# -*- coding: utf-8 -*-
import google.generativeai as genai
import os
import sys
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import shutil # 파일/디렉토리 복사 위해 추가
import threading # 백그라운드 초기화 위해 추가 (선택적)

# --- 로깅 및 설정 로드 (이전과 동일) ---
# ... (이전 logging, load_dotenv, 설정 변수 로드 부분) ...
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)
# ... (GEMINI_MODEL_NAME, N_RESULTS 등 설정 로드) ...

# --- !!! 중요: 프로젝트 루트의 chroma_db 경로 ---
# 이 경로는 빌드 시 포함된 DB 원본 위치를 가리킵니다.
PROJECT_ROOT_CHROMA_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'chroma_db'))

# --- !!! 중요: Vercel의 임시 저장 공간 경로 ---
# 서버리스 함수가 실행될 때 사용할 쓰기 가능한 경로입니다.
RUNTIME_CHROMA_DB_PATH = "/tmp/chroma_db_runtime" # /tmp 아래 경로 사용

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "doctor_duck_collection_lc")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "jhgan/ko-sbert-nli")
# --- !!! 중요: 임베딩 모델 캐시 경로 지정 ---
# Sentence Transformers가 모델을 다운로드할 경로를 /tmp로 지정합니다.
SENTENCE_TRANSFORMERS_HOME = '/tmp/sentence_transformers_cache'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = SENTENCE_TRANSFORMERS_HOME


logging.info("--- 설정 로드 완료 ---")
# ... (로그 설정 정보 출력) ...
logging.info(f"원본 ChromaDB 경로 (빌드 시점): {PROJECT_ROOT_CHROMA_DB_PATH}")
logging.info(f"런타임 ChromaDB 경로 (사용 시점): {RUNTIME_CHROMA_DB_PATH}")
logging.info(f"임베딩 모델 캐시 경로: {SENTENCE_TRANSFORMERS_HOME}")
logging.info("----------------------")


# --- Flask 앱 초기화 (이전과 동일) ---
app = Flask(__name__)
CORS(app)
logging.info("Flask 앱 초기화 및 CORS 설정 완료.")

# --- 전역 변수 및 초기화 상태 플래그 ---
model = None
chroma_collection = None
services_initialized = False
initialization_lock = threading.Lock() # 동시 초기화 방지용 락
initialization_error = None # 초기화 에러 저장

# --- 서비스 초기화 함수 (런타임 로딩 로직 추가) ---
def initialize_services():
    global model, chroma_collection, services_initialized, initialization_error

    # 이미 초기화되었거나 다른 스레드가 초기화 중이면 반환
    if services_initialized:
        logging.info("서비스는 이미 초기화되어 있습니다.")
        return
    if not initialization_lock.acquire(blocking=False): # 락 획득 실패 시 (다른 스레드 진행 중)
        logging.info("다른 스레드에서 서비스 초기화 진행 중... 대기.")
        initialization_lock.acquire() # 락 풀릴 때까지 대기
        initialization_lock.release()
        if services_initialized: return # 기다리는 동안 완료되었을 수 있음
        if initialization_error: raise initialization_error # 기다리는 동안 에러 발생 시 전파
        # 만약 아직도 초기화 안됐으면 뭔가 문제 -> 에러 발생시키기 (이론상 드문 경우)
        raise RuntimeError("초기화 대기 후에도 서비스가 준비되지 않았습니다.")


    logging.info("서비스 초기화 시작...")
    try:
        # 1. Google AI 설정 (이전과 동일)
        google_api_key = os.getenv("GOOGLE_API_KEY")
        # ... (Google AI 모델 로드 로직) ...
        if not google_api_key: raise ValueError("GOOGLE_API_KEY 환경 변수를 찾을 수 없습니다.")
        genai.configure(api_key=google_api_key)
        safety_settings = [ /* ... */ ] # 생략 - 이전 설정과 동일
        model = genai.GenerativeModel(GEMINI_MODEL_NAME, safety_settings=safety_settings)
        logging.info(f"Google AI 모델 '{GEMINI_MODEL_NAME}' 로드 완료.")

        # 2. ChromaDB 준비 (원본 복사)
        logging.info(f"런타임 ChromaDB 준비 시작: {RUNTIME_CHROMA_DB_PATH}")
        if os.path.exists(RUNTIME_CHROMA_DB_PATH):
            logging.info(f"기존 런타임 ChromaDB({RUNTIME_CHROMA_DB_PATH}) 삭제 시도.")
            try:
                shutil.rmtree(RUNTIME_CHROMA_DB_PATH)
            except Exception as e:
                logging.warning(f"기존 런타임 DB 삭제 실패: {e}. 계속 진행.")

        if not os.path.exists(PROJECT_ROOT_CHROMA_DB_PATH):
            logging.error(f"원본 ChromaDB 경로({PROJECT_ROOT_CHROMA_DB_PATH})를 찾을 수 없습니다! 빌드 시 포함되었는지 확인하세요.")
            raise FileNotFoundError("원본 ChromaDB 데이터 없음")

        try:
            # 원본 DB 디렉토리를 /tmp 아래로 복사합니다.
            logging.info(f"원본 DB ({PROJECT_ROOT_CHROMA_DB_PATH})를 런타임 경로 ({RUNTIME_CHROMA_DB_PATH})로 복사 중...")
            shutil.copytree(PROJECT_ROOT_CHROMA_DB_PATH, RUNTIME_CHROMA_DB_PATH)
            logging.info("DB 복사 완료.")
        except Exception as e:
            logging.error(f"DB 복사 중 오류 발생: {e}", exc_info=True)
            raise RuntimeError(f"ChromaDB 데이터 준비 실패: {e}")

        # 3. 임베딩 함수 로드 (런타임에 모델 다운로드 유도)
        logging.info(f"임베딩 함수 로드 시작 ({EMBEDDING_MODEL_NAME}). 모델 다운로드가 필요할 수 있음...")
        # SENTENCE_TRANSFORMERS_HOME 환경변수가 설정되어 있으므로 /tmp 아래 캐시 사용
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
        logging.info("임베딩 함수 로드 완료 (모델 다운로드 완료되었거나 캐시 사용).")

        # 4. ChromaDB 컬렉션 로드 (복사된 런타임 경로 사용)
        logging.info(f"런타임 ChromaDB 클라이언트 생성 및 컬렉션 로드 시도 (경로: {RUNTIME_CHROMA_DB_PATH})...")
        chroma_client = chromadb.PersistentClient(path=RUNTIME_CHROMA_DB_PATH)
        chroma_collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_function)
        collection_count = chroma_collection.count()
        logging.info(f"런타임 ChromaDB 컬렉션 '{COLLECTION_NAME}' 로드 완료. (데이터 수: {collection_count})")
        if collection_count == 0: logging.warning(f"주의: 컬렉션 '{COLLECTION_NAME}'에 데이터가 없습니다.")

        services_initialized = True
        initialization_error = None # 성공 시 에러 상태 초기화
        logging.info("모든 서비스 초기화 성공.")

    except Exception as e:
        logging.critical(f"서비스 초기화 중 심각한 오류 발생: {e}", exc_info=True)
        initialization_error = e # 에러 상태 저장
        # 여기서 바로 에러를 raise 해야 상위 호출자(요청 처리 등)가 알 수 있음
        raise RuntimeError(f"서비스 초기화 실패: {e}")
    finally:
        # 초기화 시도가 끝났으므로 락 해제 (성공/실패 무관)
        initialization_lock.release()


# --- 검색 및 답변 생성 함수 (이전과 동일) ---
# search_similar_chunks, generate_answer 함수는 수정 없이 그대로 사용
# 내부에서 사용하는 chroma_collection 변수는 initialize_services()를 통해 설정됨
def search_similar_chunks(query):
    # ... (함수 내용 동일) ...
    if not services_initialized or chroma_collection is None: # 초기화 확인 추가
        logging.error("검색 시도 실패: ChromaDB 서비스가 초기화되지 않았습니다.")
        raise RuntimeError("서비스가 준비되지 않았습니다.") # 에러 발생시켜 처리 중단
    # ... (나머지 검색 로직 동일) ...

def generate_answer(query, retrieved_chunks):
    # ... (함수 내용 동일) ...
    if not services_initialized or model is None: # 초기화 확인 추가
        logging.error("답변 생성 시도 실패: Google AI 모델 서비스가 초기화되지 않았습니다.")
        raise RuntimeError("서비스가 준비되지 않았습니다.") # 에러 발생시켜 처리 중단
    # ... (나머지 답변 생성 로직 동일) ...

# --- API 엔드포인트 정의 (초기화 호출 방식 변경) ---
@app.route('/api/ask', methods=['POST'])
def ask_question():
    logging.info(f"'/api/ask' 요청 수신: {request.remote_addr}")

    # --- 서비스 초기화 확인 및 시도 (매 요청 시 확인) ---
    if not services_initialized:
        logging.info("서비스가 초기화되지 않아 초기화 시도...")
        try:
            # initialize_services() 내부에서 락을 사용하므로 동시 호출 문제 없음
            initialize_services()
        except Exception as init_e:
            logging.critical(f"요청 처리 중 서비스 초기화 실패: {init_e}", exc_info=True)
            return jsonify({"error": f"서버 내부 오류: 서비스 초기화 실패 ({type(init_e).__name__}). 잠시 후 다시 시도해주세요."}), 503

    # --- 요청 처리 (이전과 동일) ---
    if not request.is_json: # ... (JSON 형식 검사) ...
    data = request.get_json() # ... (데이터 가져오기) ...
    user_query = data.get('query') # ... (쿼리 추출) ...
    if not user_query or not isinstance(user_query, str) or not user_query.strip(): # ... (쿼리 유효성 검사) ...

    sanitized_query = user_query.strip()
    logging.info(f"처리할 질문: '{sanitized_query}'")

    try:
        # 1. 검색
        retrieved_chunks = search_similar_chunks(sanitized_query)
        # 2. 답변 생성
        answer = generate_answer(sanitized_query, retrieved_chunks)

        logging.info(f"생성된 답변 전송 (일부): {answer[:100]}...")
        return jsonify({"answer": answer})

    except RuntimeError as service_err: # 서비스 미준비 에러 처리
         logging.error(f"서비스 미준비 오류: {service_err}")
         return jsonify({"error": "서버가 아직 준비 중입니다. 잠시 후 다시 시도해주세요."}), 503
    except Exception as e: # 기타 예외 처리
        logging.error(f"요청 처리 중 예외 발생: {e}", exc_info=True)
        return jsonify({"error": "내부 서버 오류: 요청을 처리하는 중 문제가 발생했습니다."}), 500


# --- 상태 확인 엔드포인트 (이전과 동일) ---
@app.route('/api/health', methods=['GET'])
def health_check():
    # ... (이전 health_check 코드) ...

# --- 앱 시작 시 초기화 시도 제거 ---
# Vercel에서는 요청이 들어올 때 함수 인스턴스가 생성/재사용되므로,
# 첫 요청 시 initialize_services()가 호출되도록 하는 것이 일반적입니다.
# 로컬 테스트 시에는 첫 요청이 들어올 때 초기화됩니다.

# --- 로컬 테스트용 실행 구문 (이전과 동일) ---
if __name__ == '__main__':
    logging.info("Flask 개발 서버 시작 (http://127.0.0.1:5000)")
    # 로컬 테스트 시에도 첫 요청 시 초기화되도록 debug=False 유지 권장
    app.run(host='0.0.0.0', port=5000, debug=False)
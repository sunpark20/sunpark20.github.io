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

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- .env 파일 로드 (로컬 개발 시 유용, Vercel에서는 환경변수 직접 설정) ---
# Vercel 배포 시에는 .env 파일이 없을 수 있으므로, 오류 없이 진행되도록 처리
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env') # api 폴더 기준 상위 폴더의 .env
load_dotenv(dotenv_path=dotenv_path)
logging.info(f".env 파일 로드 시도: {dotenv_path} (파일 존재 여부: {os.path.exists(dotenv_path)})")

# --- 설정 로드 (환경 변수 우선) ---
GEMINI_MODEL_NAME = os.getenv('GEMINI_MODEL_NAME', 'gemini-1.5-flash-latest')
try: N_RESULTS = int(os.getenv("CHROMA_N_RESULTS", "15")); N_RESULTS = max(1, N_RESULTS)
except ValueError: N_RESULTS = 15
try: TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.6")); TEMPERATURE = max(0.0, min(1.0, TEMPERATURE))
except ValueError: TEMPERATURE = 0.6
try: MAX_CONTEXT_LENGTH = int(os.getenv("GEMINI_MAX_CONTEXT_LENGTH", "30000")); MAX_CONTEXT_LENGTH = max(100, MAX_CONTEXT_LENGTH) # 최소 길이 보장
except ValueError: MAX_CONTEXT_LENGTH = 30000

# ChromaDB 경로 설정 (Vercel 환경 고려)
# api/index.py 파일의 위치를 기준으로 상위 폴더의 chroma_db를 가리키도록 설정
DEFAULT_CHROMA_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'chroma_db'))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", DEFAULT_CHROMA_DB_PATH)

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "doctor_duck_collection_lc")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "jhgan/ko-sbert-nli")

logging.info("--- 설정 로드 완료 ---")
logging.info(f"검색 결과 수: {N_RESULTS}, 생성 온도: {TEMPERATURE}, 최대 컨텍스트: {MAX_CONTEXT_LENGTH}")
logging.info(f"ChromaDB 경로: {CHROMA_DB_PATH}, 컬렉션: {COLLECTION_NAME}, 임베딩 모델: {EMBEDDING_MODEL_NAME}")
logging.info("----------------------")

# --- Flask 앱 초기화 ---
app = Flask(__name__)
# CORS 설정: 실제 운영 시에는 GitHub Pages 도메인만 명시적으로 허용하는 것이 안전
# 예: CORS(app, origins=["https://sunpark20.github.io"])
CORS(app)
logging.info("Flask 앱 초기화 및 CORS 설정 완료.")

# --- Google AI 및 ChromaDB 전역 변수 ---
model = None
chroma_collection = None
services_initialized = False

# --- 서비스 초기화 함수 ---
def initialize_services():
    global model, chroma_collection, services_initialized
    if services_initialized:
        logging.info("서비스는 이미 초기화되었습니다.")
        return

    logging.info("서비스 초기화 시작...")

    # Google AI 설정
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY 환경 변수를 찾을 수 없습니다.")
        genai.configure(api_key=google_api_key)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
        ]
        model = genai.GenerativeModel(GEMINI_MODEL_NAME, safety_settings=safety_settings)
        logging.info(f"Google AI 모델 '{GEMINI_MODEL_NAME}' 로드 완료.")
    except Exception as e:
        logging.error(f"Google AI 설정 중 오류 발생: {e}", exc_info=True)
        # 여기서는 앱 중단 대신, 나중에 요청 처리 시 오류를 반환하도록 함
        raise RuntimeError(f"Google AI 초기화 실패: {e}") # 앱 로딩 시 실패 전파

    # ChromaDB 로드
    try:
        logging.info(f"ChromaDB 로드 시도 중... (경로: {CHROMA_DB_PATH})")
        if not os.path.exists(CHROMA_DB_PATH):
             logging.warning(f"ChromaDB 경로({CHROMA_DB_PATH})가 존재하지 않습니다. Vercel 배포 시 이 경로에 DB 파일이 복사되었는지 확인하세요.")
             # Vercel 빌드 로그 확인 필요
             raise FileNotFoundError(f"ChromaDB 디렉토리({CHROMA_DB_PATH})를 찾을 수 없습니다.")

        # 임베딩 함수 로드 (시간이 걸릴 수 있음)
        logging.info(f"임베딩 함수 로드 중... ({EMBEDDING_MODEL_NAME})")
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
        logging.info("임베딩 함수 로드 완료.")

        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        logging.info(f"ChromaDB 클라이언트 생성 완료 (경로: {CHROMA_DB_PATH}). 컬렉션 로드 시도...")

        # 컬렉션 가져오기 시도 (존재하지 않으면 오류 발생)
        chroma_collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_function)
        collection_count = chroma_collection.count()
        logging.info(f"ChromaDB 컬렉션 '{COLLECTION_NAME}' 로드 완료. (데이터 수: {collection_count})")
        if collection_count == 0:
            logging.warning(f"주의: 컬렉션 '{COLLECTION_NAME}'에 데이터가 없습니다.")

    except Exception as e:
        logging.error(f"ChromaDB 컬렉션 '{COLLECTION_NAME}' 로드 중 오류 발생: {e}", exc_info=True)
        logging.error(f"DB 경로 '{CHROMA_DB_PATH}', 컬렉션 '{COLLECTION_NAME}', 임베딩 모델 '{EMBEDDING_MODEL_NAME}' 확인 필요.")
        raise RuntimeError(f"ChromaDB 초기화 실패: {e}") # 앱 로딩 시 실패 전파

    services_initialized = True
    logging.info("모든 서비스 초기화 성공.")

# --- 검색 함수 ---
def search_similar_chunks(query):
    if not services_initialized or chroma_collection is None:
        logging.error("오류: ChromaDB 서비스가 초기화되지 않았습니다.")
        return [] # 빈 리스트 반환

    logging.info(f"'{query}' 관련 청크 검색 시작 (최대 {N_RESULTS}개)...")
    try:
        results = chroma_collection.query(query_texts=[query], n_results=N_RESULTS, include=['metadatas', 'documents'])
        if not results or not results.get('ids') or not results['ids'][0]:
            logging.info(f"'{query}'에 대한 검색 결과 없음.")
            return []

        retrieved_chunks = []
        ids = results['ids'][0]
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]

        # 데이터 무결성 체크
        if len(documents) != len(ids) or len(metadatas) != len(ids):
            logging.warning(f"검색 결과 ID({len(ids)}), 문서({len(documents)}), 메타데이터({len(metadatas)}) 개수 불일치. 최소 개수로 맞춥니다.")
            min_len = min(len(ids), len(documents), len(metadatas))
            ids, documents, metadatas = ids[:min_len], documents[:min_len], metadatas[:min_len]

        for id_val, doc, meta in zip(ids, documents, metadatas):
            # 메타데이터가 dict 형태인지 확인 (오류 방지)
            chunk_metadata = meta if isinstance(meta, dict) else {"title": "형식 오류", "url": ""}
            retrieved_chunks.append({"text": doc or "", "metadata": chunk_metadata})

        logging.info(f"'{query}' 관련 청크 {len(retrieved_chunks)}개 검색 완료.")
        return retrieved_chunks
    except Exception as e:
        logging.error(f"ChromaDB 검색 중 오류 발생: {e}", exc_info=True)
        return [] # 오류 발생 시 빈 리스트 반환

# --- 답변 생성 함수 ---
def generate_answer(query, retrieved_chunks):
    if not services_initialized or model is None:
        logging.error("오류: Google AI 모델 서비스가 초기화되지 않았습니다.")
        return "오류: 답변 생성 모델이 준비되지 않았습니다."

    if not retrieved_chunks:
        logging.info("검색된 청크가 없어 기본 응답 반환.")
        return "닥터덕 영상 내용 중에서 현재 질문과 관련된 정보를 찾을 수 없습니다."

    context = ""
    sources = set()
    current_length = 0
    included_chunk_count = 0

    logging.info(f"검색된 청크 {len(retrieved_chunks)}개를 컨텍스트로 구성 시작...")
    for i, chunk in enumerate(retrieved_chunks):
        metadata = chunk.get('metadata', {})
        title = metadata.get('title', '알 수 없는 영상')
        url = metadata.get('url', '')
        text = chunk.get('text', '')
        if not text:
            logging.warning(f"청크 {i+1} (ID: {chunk.get('id', 'N/A')}) 텍스트가 비어있어 건너<0xEB><0x9B><0x84>니다.")
            continue

        chunk_info = f"--- 영상 발췌 (출처: '{title}') ---\n{text}\n---\n\n"
        chunk_len = len(chunk_info.encode('utf-8')) # 글자 수 대신 바이트 길이로 계산하는 것이 더 정확할 수 있음

        if current_length + chunk_len <= MAX_CONTEXT_LENGTH:
            context += chunk_info
            current_length += chunk_len
            included_chunk_count += 1
            if url and url.startswith('http'):
                 # 중복 제거를 위해 (title, url) 튜플 사용
                 sources.add((title, url))
        else:
            logging.info(f"컨텍스트 최대 길이({MAX_CONTEXT_LENGTH} 바이트) 도달, 청크 {i + 1}부터 제외.")
            break

    logging.info(f"컨텍스트 포함 청크 수: {included_chunk_count} / {len(retrieved_chunks)}")
    logging.info(f"포함된 컨텍스트 바이트 수: {current_length} / {MAX_CONTEXT_LENGTH}")
    unique_sources_count = len(sources)
    logging.info(f"참고 영상 출처 개수 (고유): {unique_sources_count}")

    if not context:
        logging.warning("유효한 검색 내용이 없어 답변 컨텍스트를 구성할 수 없습니다.")
        return "오류: 답변 생성에 사용할 유효한 정보를 찾지 못했습니다."

    # --- 프롬프트 구성 (기존 프롬프트 활용) ---
    prompt = f"""당신은 '닥터덕' 유튜브 채널 영상 내용을 기반으로 사용자의 질문에 답변하는 기능의학 전문 AI 어시스턴트입니다. ... (중략) ...

질문: "{query}"

... (중략, 답변 생성 규칙 포함) ...

이제, 위의 규칙에 따라 질문에 대한 답변을 생성해주세요:"""
    # logging.debug(f"Gemini에게 전달될 최종 프롬프트:\n{prompt[:500]}...") # 디버깅 시 프롬프트 확인

    logging.info("Gemini API 호출 시작...")
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=TEMPERATURE)
        )
        logging.info("Gemini API 응답 수신 완료.")

        # 응답 유효성 검사 및 차단 처리
        if not response.candidates:
            block_reason = "응답 후보 없음"; safety_ratings_str = "확인 불가"
            try:
                if response.prompt_feedback:
                    block_reason = response.prompt_feedback.block_reason or "차단 사유 명시되지 않음"
                    if response.prompt_feedback.block_reason_message: block_reason += f" ({response.prompt_feedback.block_reason_message})"
                    safety_ratings = response.prompt_feedback.safety_ratings
                    safety_ratings_str = ", ".join([f"{r.category.name}: {r.probability.name}" for r in safety_ratings])
            except Exception as feedback_e:
                logging.warning(f"피드백 정보 접근 중 오류: {feedback_e}")

            logging.warning(f"Gemini 응답이 비어 있거나 차단되었습니다. 이유: {block_reason}, 안전 등급: [{safety_ratings_str}]")
            if "SAFETY" in str(block_reason).upper():
                return f"죄송합니다. 답변 생성 요청이 안전상의 이유({block_reason})로 처리되지 않았습니다."
            else:
                return "죄송합니다. 답변을 생성하는 데 문제가 발생했습니다. (빈 응답)"

        # 정상 응답 텍스트 추출
        generated_text = "".join(part.text for part in response.candidates[0].content.parts)

        # 출처 정보 추가 (소스가 있는 경우)
        if sources:
            # url 기준으로 정렬
            sorted_sources = sorted(list(sources), key=lambda item: item[1])
            source_list_text = f"\n\n**참고 영상 ({unique_sources_count}개):**\n"
            source_list_text += "\n".join([f"- {title} ({url})" for title, url in sorted_sources])
            generated_text += "\n" + source_list_text # 줄바꿈 추가

        return generated_text.strip()

    except ValueError as ve:
        logging.error(f"Google AI API 호출 값 오류: {ve}", exc_info=True)
        return f"죄송합니다. 답변 생성 요청 처리 중 오류가 발생했습니다. (오류 코드: VAL)"
    except Exception as e:
        logging.error(f"Google AI API 호출 중 예상치 못한 오류 발생: {e}", exc_info=True)
        error_type = type(e).__name__
        return f"죄송합니다. 답변을 생성하는 동안 서버 오류({error_type})가 발생했습니다. 잠시 후 다시 시도해주세요."


# --- API 엔드포인트 정의 ---
@app.route('/api/ask', methods=['POST'])
def ask_question():
    logging.info(f"'/api/ask' 요청 수신: {request.remote_addr}")

    # 서비스 초기화 확인 및 시도
    if not services_initialized:
        try:
            initialize_services()
        except Exception as init_e:
            logging.critical(f"요청 처리 중 서비스 초기화 실패: {init_e}", exc_info=True)
            # 503 Service Unavailable 반환 (서버가 요청 처리 준비 안됨)
            return jsonify({"error": "서버 내부 오류: 서비스를 시작할 수 없습니다."}), 503

    if not request.is_json:
        logging.warning("잘못된 요청 형식: JSON 타입이 아님")
        return jsonify({"error": "요청 형식이 잘못되었습니다 (Content-Type: application/json 필요)."}), 400

    data = request.get_json()
    user_query = data.get('query')

    if not user_query or not isinstance(user_query, str) or not user_query.strip():
        logging.warning(f"잘못된 요청 데이터: 'query'가 없거나 비어있음 (수신 데이터: {data})")
        return jsonify({"error": "'query' 파라미터가 필요하며, 비어있지 않은 문자열이어야 합니다."}), 400

    sanitized_query = user_query.strip()
    logging.info(f"처리할 질문: '{sanitized_query}'")

    # 1. 검색
    try:
        retrieved_chunks = search_similar_chunks(sanitized_query)
    except Exception as search_e:
        logging.error(f"검색 중 예외 발생: {search_e}", exc_info=True)
        return jsonify({"error": "내부 서버 오류: 관련 정보를 검색하는 중 문제가 발생했습니다."}), 500

    # 2. 답변 생성
    try:
        answer = generate_answer(sanitized_query, retrieved_chunks)
    except Exception as gen_e:
        logging.error(f"답변 생성 중 예외 발생: {gen_e}", exc_info=True)
        return jsonify({"error": "내부 서버 오류: 답변을 생성하는 중 문제가 발생했습니다."}), 500

    logging.info(f"생성된 답변 전송 (일부): {answer[:100]}...")
    return jsonify({"answer": answer})

# --- 서버 상태 확인용 엔드포인트 (선택 사항) ---
@app.route('/api/health', methods=['GET'])
def health_check():
    status = {"status": "ok", "services_initialized": services_initialized}
    if services_initialized:
        status["chromadb_collection_count"] = chroma_collection.count() if chroma_collection else "N/A"
    return jsonify(status)

# --- 앱 시작 시 서비스 초기화 시도 ---
# Vercel 환경에서는 이 코드가 서버리스 함수 인스턴스가 처음 로드될 때 실행됨
try:
    initialize_services()
except Exception as e:
    # 초기화 실패 시 로그만 남기고 앱 자체는 로드되도록 함 (요청 시 다시 초기화 시도)
    logging.critical(f"앱 시작 시 서비스 초기화 실패: {e}. 요청 시 다시 시도됩니다.", exc_info=True)

# --- 로컬 테스트용 실행 구문 (Vercel 배포 시에는 사용되지 않음) ---
if __name__ == '__main__':
    logging.info("Flask 개발 서버 시작 (http://127.0.0.1:5000)")
    # debug=True는 Vercel에서 사용하지 마세요. 실제 서버에서는 False여야 합니다.
    app.run(host='0.0.0.0', port=5000, debug=False)
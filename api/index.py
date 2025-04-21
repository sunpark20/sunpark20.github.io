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
import shutil  # 파일/디렉토리 복사/삭제 위해 추가
import threading  # 동시 초기화 방지 위해 추가
import boto3  # R2/S3 접근 위해 추가
import botocore # boto3 예외 처리 위해 추가

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- .env 파일 로드 (로컬 개발 시 유용, Vercel에서는 환경변수 직접 설정) ---
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env') # api 폴더 기준 상위 폴더의 .env
load_dotenv(dotenv_path=dotenv_path)
logging.info(f".env 파일 로드 시도: {dotenv_path} (파일 존재 여부: {os.path.exists(dotenv_path)})")

# --- 설정 로드 (환경 변수 우선) ---
GEMINI_MODEL_NAME = os.getenv('GEMINI_MODEL_NAME', 'gemini-1.5-flash-latest')
try: N_RESULTS = int(os.getenv("CHROMA_N_RESULTS", "15")); N_RESULTS = max(1, N_RESULTS)
except ValueError: N_RESULTS = 15
try: TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.6")); TEMPERATURE = max(0.0, min(1.0, TEMPERATURE))
except ValueError: TEMPERATURE = 0.6
try: MAX_CONTEXT_LENGTH = int(os.getenv("GEMINI_MAX_CONTEXT_LENGTH", "30000")); MAX_CONTEXT_LENGTH = max(100, MAX_CONTEXT_LENGTH)
except ValueError: MAX_CONTEXT_LENGTH = 30000

# --- Vercel의 임시 저장 공간 경로 ---
RUNTIME_CHROMA_DB_PATH = "/tmp/chroma_db_runtime" # /tmp 아래 경로 사용

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "doctor_duck_collection_lc")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "jhgan/ko-sbert-nli")

# --- 임베딩 모델 캐시 경로 지정 ---
SENTENCE_TRANSFORMERS_HOME = '/tmp/sentence_transformers_cache'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = SENTENCE_TRANSFORMERS_HOME

# --- Cloudflare R2 설정 (Vercel 환경 변수로 설정 필요) ---
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
# R2 엔드포인트 URL 구성
R2_ENDPOINT_URL = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com" if R2_ACCOUNT_ID else None
# R2 내의 ChromaDB 데이터 경로 (버킷 최상위에 chroma_db 폴더가 있다고 가정)
R2_CHROMA_DB_PREFIX = os.getenv("R2_CHROMA_DB_PREFIX", "chroma_db") # 환경변수 또는 기본값 사용

logging.info("--- 설정 로드 완료 ---")
logging.info(f"모델: {GEMINI_MODEL_NAME}, 검색결과: {N_RESULTS}, 온도: {TEMPERATURE}, 최대컨텍스트: {MAX_CONTEXT_LENGTH}")
logging.info(f"런타임 ChromaDB 경로: {RUNTIME_CHROMA_DB_PATH}, 컬렉션: {COLLECTION_NAME}")
logging.info(f"임베딩 모델: {EMBEDDING_MODEL_NAME}, 캐시 경로: {SENTENCE_TRANSFORMERS_HOME}")
logging.info(f"R2 버킷: {R2_BUCKET_NAME}, 계정ID: {R2_ACCOUNT_ID}, Prefix: {R2_CHROMA_DB_PREFIX}")
logging.info("----------------------")


# --- Flask 앱 초기화 ---
app = Flask(__name__)
CORS(app)
logging.info("Flask 앱 초기화 및 CORS 설정 완료.")

# --- 전역 변수 및 초기화 상태 플래그 ---
model = None
chroma_collection = None
services_initialized = False
initialization_lock = threading.Lock() # 동시 초기화 방지용 락
initialization_error = None # 초기화 에러 저장

# --- R2 다운로드 함수 ---
def download_r2_folder(bucket_name, s3_folder, local_dir):
    """R2 버킷의 폴더 전체를 로컬 디렉토리로 다운로드 (S3 호환 API 사용)"""
    if not all([R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY]):
         logging.error("R2 관련 환경 변수(R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY)가 설정되지 않았습니다.")
         raise ValueError("R2 환경 변수 누락")

    logging.info(f"R2 엔드포인트({R2_ENDPOINT_URL})에 연결 시도...")
    try:
        s3 = boto3.client(
            's3',
            endpoint_url=R2_ENDPOINT_URL,
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            config=botocore.client.Config(signature_version='s3v4'), # R2는 v4 서명 필요
            region_name='auto' # R2는 리전 지정 불필요
        )
        logging.info(f"R2 클라이언트 생성 완료. 버킷 '{bucket_name}'의 '{s3_folder}/' 에서 객체 목록 가져오는 중...")
    except Exception as client_e:
        logging.error(f"R2 클라이언트 생성 실패: {client_e}", exc_info=True)
        raise RuntimeError(f"R2 클라이언트 생성 실패: {client_e}")

    paginator = s3.get_paginator('list_objects_v2')
    # Prefix가 빈 문자열이면 ''으로, 아니면 '/'를 붙여서 사용
    prefix_param = s3_folder + '/' if s3_folder else ''
    operation_parameters = {'Bucket': bucket_name, 'Prefix': prefix_param}

    download_count = 0
    found_objects = False
    try:
        pages = paginator.paginate(**operation_parameters)
        for page in pages:
            if "Contents" in page:
                found_objects = True
                for obj in page['Contents']:
                    s3_key = obj['Key']
                    # s3_folder가 비어있지 않다면 해당 부분을 제거하고 상대 경로 계산
                    relative_path = os.path.relpath(s3_key, s3_folder) if s3_folder else s3_key
                    local_file_path = os.path.join(local_dir, relative_path)

                    # S3 키가 디렉토리를 나타내면 로컬에서도 디렉토리 생성
                    if s3_key.endswith('/'):
                         os.makedirs(local_file_path, exist_ok=True)
                         continue

                    # 파일이면 상위 디렉토리 생성 후 다운로드
                    local_file_dir = os.path.dirname(local_file_path)
                    os.makedirs(local_file_dir, exist_ok=True)

                    logging.debug(f"R2 객체 다운로드 중: {s3_key} -> {local_file_path}")
                    s3.download_file(bucket_name, s3_key, local_file_path)
                    download_count += 1
            else:
                # 'Contents'가 없는 페이지도 있을 수 있음, 마지막에 최종 판단
                 pass

        if not found_objects and prefix_param: # Prefix를 지정했는데 객체가 없는 경우
             logging.warning(f"R2 버킷 '{bucket_name}'의 '{prefix_param}' 경로에 객체가 없습니다.")
             raise FileNotFoundError(f"R2에서 ChromaDB 데이터를 찾을 수 없습니다 (경로: {prefix_param}).")
        elif download_count == 0 and not prefix_param: # Prefix 없이 버킷 최상위를 봤는데 객체가 없는 경우
             logging.warning(f"R2 버킷 '{bucket_name}'이 비어있습니다.")
             raise FileNotFoundError(f"R2 버킷 '{bucket_name}'이 비어있습니다.")
        else:
             logging.info(f"총 {download_count}개의 객체를 {local_dir}로 다운로드 완료.")

    except botocore.exceptions.ClientError as e:
        logging.error(f"R2 접근 중 오류 발생: {e}", exc_info=True)
        error_code = e.response.get('Error', {}).get('Code')
        if error_code == 'NoSuchBucket':
             raise FileNotFoundError(f"R2 버킷 '{bucket_name}'을(를) 찾을 수 없습니다.")
        elif error_code in ['InvalidAccessKeyId', 'SignatureDoesNotMatch', 'AccessDenied']:
             raise PermissionError("R2 접근 권한 오류. API 토큰(Access Key/Secret Key) 또는 버킷 권한을 확인하세요.")
        else:
             raise RuntimeError(f"R2 작업 중 예상치 못한 오류 발생: {e}")
    except Exception as e:
        logging.error(f"R2 다운로드 중 예외 발생: {e}", exc_info=True)
        raise RuntimeError(f"R2 데이터 다운로드 실패: {e}")

# --- 서비스 초기화 함수 ---
def initialize_services():
    global model, chroma_collection, services_initialized, initialization_error

    # 락을 사용하여 동시 초기화 방지
    with initialization_lock:
        if services_initialized:
            logging.info("서비스는 이미 초기화되어 있습니다.")
            return
        # 이전에 에러가 있었다면 다시 시도하지 않고 바로 에러 발생 (선택적)
        # if initialization_error:
        #     raise initialization_error

        logging.info("서비스 초기화 시작...")
        try:
            # 1. Google AI 설정
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key: raise ValueError("GOOGLE_API_KEY 환경 변수를 찾을 수 없습니다.")
            genai.configure(api_key=google_api_key)
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
            ]
            model = genai.GenerativeModel(GEMINI_MODEL_NAME, safety_settings=safety_settings)
            logging.info(f"Google AI 모델 '{GEMINI_MODEL_NAME}' 로드 완료.")

            # 2. ChromaDB 준비 (R2에서 다운로드)
            logging.info(f"런타임 ChromaDB 준비 시작 (R2 다운로드): {RUNTIME_CHROMA_DB_PATH}")
            if os.path.exists(RUNTIME_CHROMA_DB_PATH):
                logging.info(f"기존 런타임 ChromaDB({RUNTIME_CHROMA_DB_PATH}) 삭제.")
                shutil.rmtree(RUNTIME_CHROMA_DB_PATH)
            os.makedirs(RUNTIME_CHROMA_DB_PATH, exist_ok=True) # 대상 폴더 생성

            if not R2_BUCKET_NAME: raise ValueError("R2_BUCKET_NAME 환경 변수가 설정되지 않았습니다.")

            logging.info(f"R2 버킷 '{R2_BUCKET_NAME}'의 '{R2_CHROMA_DB_PREFIX}/' 에서 다운로드 시작...")
            download_r2_folder(R2_BUCKET_NAME, R2_CHROMA_DB_PREFIX, RUNTIME_CHROMA_DB_PATH)
            logging.info("R2 데이터 다운로드 완료.")

            # 3. 임베딩 함수 로드
            logging.info(f"임베딩 함수 로드 시작 ({EMBEDDING_MODEL_NAME}). 모델 다운로드가 필요할 수 있음...")
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
            logging.info("임베딩 함수 로드 완료.")

            # 4. ChromaDB 컬렉션 로드
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
            # 여기서 에러를 다시 발생시켜 상위 호출자가 알 수 있도록 함
            raise RuntimeError(f"서비스 초기화 실패: {e}")

# --- 검색 함수 ---
def search_similar_chunks(query):
    if not services_initialized or chroma_collection is None:
        logging.error("검색 시도 실패: ChromaDB 서비스가 초기화되지 않았습니다.")
        raise RuntimeError("서비스가 준비되지 않았습니다.")

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

        min_len = min(len(ids), len(documents), len(metadatas))
        if len(ids) != min_len or len(documents) != min_len or len(metadatas) != min_len:
             logging.warning(f"검색 결과 ID({len(ids)}), 문서({len(documents)}), 메타데이터({len(metadatas)}) 개수 불일치. 최소 개수({min_len})로 맞춥니다.")
             ids, documents, metadatas = ids[:min_len], documents[:min_len], metadatas[:min_len]

        for id_val, doc, meta in zip(ids, documents, metadatas):
            chunk_metadata = meta if isinstance(meta, dict) else {"title": "형식 오류", "url": ""}
            retrieved_chunks.append({"text": doc or "", "metadata": chunk_metadata})

        logging.info(f"'{query}' 관련 청크 {len(retrieved_chunks)}개 검색 완료.")
        return retrieved_chunks
    except Exception as e:
        logging.error(f"ChromaDB 검색 중 오류 발생: {e}", exc_info=True)
        # 검색 실패 시 빈 리스트 반환 또는 에러 발생 중 선택
        raise RuntimeError(f"ChromaDB 검색 실패: {e}")


# --- 답변 생성 함수 ---
def generate_answer(query, retrieved_chunks):
    if not services_initialized or model is None:
        logging.error("답변 생성 시도 실패: Google AI 모델 서비스가 초기화되지 않았습니다.")
        raise RuntimeError("서비스가 준비되지 않았습니다.")

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
        if not text: continue

        chunk_info = f"--- 영상 발췌 (출처: '{title}') ---\n{text}\n---\n\n"
        chunk_len = len(chunk_info.encode('utf-8')) # 바이트 길이 계산

        if current_length + chunk_len <= MAX_CONTEXT_LENGTH:
            context += chunk_info
            current_length += chunk_len
            included_chunk_count += 1
            if url and url.startswith('http'): sources.add((title, url))
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

    # --- 프롬프트 구성 ---
    # 주의: 프롬프트 길이가 너무 길면 Gemini API 호출 시 오류 발생 가능
    prompt = f"""당신은 '닥터덕' 유튜브 채널 영상 내용을 기반으로 사용자의 질문에 답변하는 기능의학 전문 AI 어시스턴트입니다. 당신의 임무는 제공된 스크립트 내용을 바탕으로 질문에 대해 정확하고 유용한 정보를 제공하는 것입니다. **당신은 닥터덕 의사인 것처럼, 환자에게 명확한 이유와 함께 설명하고, 필요시 대안을 비교하며 건강상의 이점을 강조하여 답변해야 합니다.**

다음은 사용자의 질문과 관련된 닥터덕 영상 스크립트 발췌 내용입니다. **주의: 일부 발췌 내용은 질문과 직접적인 관련이 없을 수 있습니다.**

{context}
위 스크립트 내용들을 주의 깊게 살펴보고, **질문에 대한 답변에 필요한 관련 정보만 선별하여** 다음 사용자의 질문에 **한국어로** 답변해주세요.

질문: "{query}"

**<<< 답변 생성 시 반드시 다음 규칙을 엄격하게 준수하세요 >>>**
0.  **관련성 판단 및 심층 분석:** 제공된 스크립트 발췌 내용 중 질문과 관련 없는 정보는 **제외**하고, 답변에 필요한 정보만을 사용하여 내용을 **종합하고 요약**하세요. 여러 영상의 관련 내용을 자연스럽게 엮어서 환자에게 설명하듯이 답변해주세요. **만약 스크립트에서 여러 대안(예: 다른 브랜드, 다른 방법)이 언급되면, 각 장단점을 비교하고 닥터덕의 관점에서 건강상의 이점을 기준으로 어떤 것을 더 추천하는지(또는 어떤 절충점이나 가치 판단, 예를 들어 '맛보다는 건강' 등) 명확히 설명하세요.**
1.  **근거 기반 답변:** **오직 위에 제공된 관련 있는 "스크립트 발췌 내용"만을 근거로 답변해야 합니다.** 당신의 사전 지식, 추측, 또는 스크립트에 명시되지 않은 정보는 절대 답변에 포함하지 마세요. 외부 웹 검색 등도 하지 마세요.
2.  **스크립트 내용 종합/요약:** 질문과 관련된 정보가 여러 발췌 내용에 흩어져 있다면, 해당 내용들을 **종합하고 요약하여 자연스러운 문장으로 답변**해주세요. 단순히 내용을 나열하지 마세요.
3.  **정보 부재 시:** 만약 제공된 스크립트 내용 어디에도 질문과 관련된 정보를 **전혀** 찾을 수 없다면, 그때는 **"제공된 닥터덕 영상 내용 중에는 질문과 관련된 정보를 찾을 수 없습니다."** 라고만 답변하세요.
4.  **매우 중요 (의학적 조언 금지):** 답변 내용은 반드시 스크립트에서 언급된 **사실**만을 포함해야 합니다. **개인에게 특정 영양제, 치료법, 또는 건강 관련 행동을 직접적으로 추천하거나 권장하는 것은 절대 금지**입니다. 예를 들어, "어떤 영양제를 먹어야 할까요?" 라는 질문에는, 스크립트에 언급된 영양제들의 종류나 효능(언급된 경우)을 **정보 전달 목적**으로 요약할 수는 있지만, "당신은 OOO 영양제를 드시는 것이 좋겠습니다" 와 같은 개인적인 추천은 절대 해서는 안 됩니다. 항상 '닥터덕 영상 내용에 따르면...' 또는 '영상에서는 ~에 대해 언급합니다' 와 같은 객관적인 톤을 유지하세요. (단, 닥터덕의 1인칭 발언 인용은 가능)
5.  **닥터덕 1인칭 인용:** 닥터덕이 1인칭으로 말하는 내용 ("제가 볼 때...", "제 경험상...")은 그대로 인용해도 좋습니다.
6.  **화자 구분:** 스크립트 내용에 **게스트나 환자의 발언이 있다면, 그것이 게스트나 환자의 말임을 명확히 구별**해야 합니다. 닥터덕의 의견인 것처럼 전달하면 절대 안 됩니다. 예를 들어, "영상 내용에 따르면 한 환자분은 '증상이 이렇다'고 말했고, 이에 닥터덕은 '그럴 때는 이렇게 해보라'고 조언했습니다." 와 같이 명확히 구분하세요. (스크립트에 화자 정보가 명시되어 있지 않다면, 문맥으로 최대한 추론하되, 불확실하면 '영상 내용에 따르면 ~라고 언급됩니다' 와 같이 중립적으로 서술하세요.)
7.  **명확하고 간결하게:** 답변은 질문에 대해 명확하고 간결하게 작성해주세요. 불필요한 미사여구나 서론/결론은 최소화하세요.
8.  **출처 제외:** **답변 마지막에 출처 정보는 추가하지 마세요.** 출처 정보는 별도로 제공됩니다.
9.  **복용량 강조:** 사용자가 '비타민 어떻게 먹어?' 또는 특정 영양제의 복용법을 물을 경우, 스크립트에 언급된 **구체적인 복용량(예: 6g, 5000IU 등)과 복용 방식(예: 물에 타서 나눠 마시기)** 정보를 최대한 찾아서 명확하게 알려주세요. 여러 영상에서 언급된 복용량 정보를 종합하여 제시하는 것이 좋습니다.
10. **실용적 조언 포함 (스크립트 기반):** 스크립트에 **조리법, 관리법, 또는 맛/식감 개선 팁 등 실용적인 조언**이 언급되어 있다면, 질문과 관련될 경우 답변에 **포함**시켜 주세요.

이제, 위의 규칙에 따라 질문에 대한 답변을 생성해주세요:"""

    logging.info("Gemini API 호출 시작...")
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=TEMPERATURE)
        )
        logging.info("Gemini API 응답 수신 완료.")

        if not response.candidates:
            block_reason = "응답 후보 없음"; safety_ratings_str = "확인 불가"
            try:
                if response.prompt_feedback:
                    block_reason = response.prompt_feedback.block_reason or "차단 사유 명시되지 않음"
                    if response.prompt_feedback.block_reason_message: block_reason += f" ({response.prompt_feedback.block_reason_message})"
                    safety_ratings = response.prompt_feedback.safety_ratings
                    safety_ratings_str = ", ".join([f"{r.category.name}: {r.probability.name}" for r in safety_ratings])
            except Exception as feedback_e: logging.warning(f"피드백 정보 접근 중 오류: {feedback_e}")
            logging.warning(f"Gemini 응답이 비어 있거나 차단되었습니다. 이유: {block_reason}, 안전 등급: [{safety_ratings_str}]")
            if "SAFETY" in str(block_reason).upper(): return f"죄송합니다. 답변 생성 요청이 안전상의 이유({block_reason})로 처리되지 않았습니다."
            else: return "죄송합니다. 답변을 생성하는 데 문제가 발생했습니다. (빈 응답)"

        generated_text = "".join(part.text for part in response.candidates[0].content.parts)

        if sources:
            sorted_sources = sorted(list(sources), key=lambda item: item[1])
            source_list_text = f"\n\n**참고 영상 ({unique_sources_count}개):**\n"
            source_list_text += "\n".join([f"- {title} ({url})" for title, url in sorted_sources])
            generated_text += "\n" + source_list_text

        return generated_text.strip()

    except ValueError as ve:
        logging.error(f"Google AI API 호출 값 오류: {ve}", exc_info=True)
        raise RuntimeError(f"답변 생성 실패 (API 값 오류): {ve}")
    except Exception as e:
        logging.error(f"Google AI API 호출 중 예상치 못한 오류 발생: {e}", exc_info=True)
        raise RuntimeError(f"답변 생성 실패 (API 호출 오류): {e}")


# --- API 엔드포인트 정의 ---
@app.route('/api/ask', methods=['POST'])
def ask_question():
    # 매 요청 시 서비스 초기화 확인 및 시도
    if not services_initialized:
        logging.info("서비스가 초기화되지 않아 초기화 시도...")
        try:
            # initialize_services() 내부에서 락을 사용하므로 동시 호출 문제 없음
            initialize_services()
        except Exception as init_e:
            logging.critical(f"요청 처리 중 서비스 초기화 실패: {init_e}", exc_info=True)
            return jsonify({"error": f"서버 내부 오류: 서비스를 시작할 수 없습니다 ({type(init_e).__name__})."}), 503

    # 요청 처리 로직
    logging.info(f"'/api/ask' 요청 수신: {request.remote_addr}")
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

    try:
        # 1. 검색
        retrieved_chunks = search_similar_chunks(sanitized_query)
        # 2. 답변 생성
        answer = generate_answer(sanitized_query, retrieved_chunks)

        logging.info(f"생성된 답변 전송 (일부): {answer[:100]}...")
        return jsonify({"answer": answer})

    except RuntimeError as service_err: # 서비스 미준비 또는 하위 작업 실패
         logging.error(f"서비스 오류 발생: {service_err}", exc_info=True)
         # 사용자에게는 일반적인 서버 오류 메시지 표시
         return jsonify({"error": "내부 서버 오류: 요청을 처리하는 중 문제가 발생했습니다."}), 500
    except Exception as e: # 기타 예상치 못한 예외 처리
        logging.error(f"요청 처리 중 예외 발생: {e}", exc_info=True)
        return jsonify({"error": "내부 서버 오류: 알 수 없는 문제가 발생했습니다."}), 500


# --- 서버 상태 확인용 엔드포인트 ---
@app.route('/api/health', methods=['GET'])
def health_check():
    status_code = 503 # 기본적으로 서비스 미준비 상태
    status = {"status": "initializing", "services_initialized": services_initialized}
    if services_initialized:
        try:
            # 간단한 DB 접근 테스트 (선택 사항)
            count = chroma_collection.count() if chroma_collection else "N/A"
            status["status"] = "ok"
            status["chromadb_collection_count"] = count
            status_code = 200
        except Exception as db_e:
            status["status"] = "error"
            status["error"] = f"ChromaDB 접근 오류: {db_e}"
            status_code = 500
    elif initialization_error: # 초기화 중 에러 발생 시
         status["status"] = "error"
         status["error"] = f"초기화 실패: {initialization_error}"

    return jsonify(status), status_code


# --- 앱 시작 시 초기화 시도 제거 ---
# 첫 요청 시 initialize_services()가 호출되도록 함

# --- 로컬 테스트용 실행 구문 ---
if __name__ == '__main__':
    logging.info("Flask 개발 서버 시작 (http://127.0.0.1:5000)")
    # 로컬 테스트 시에도 첫 요청 시 초기화되도록 debug=False 권장
    # 실제 서비스 환경에서는 Gunicorn 등 WSGI 서버 사용 필요
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)), debug=False)
"""
GIST Rules Analyzer - Prebuilt Database Version
===============================================

사전 구축된 FAISS 데이터베이스를 로드하는 고속 시작 버전입니다.

사용 전 요구사항:
    python build_rule_database.py  # 먼저 실행

특징:
- 3초 내 앱 시작 완료
- 매 쿼리마다 FAISS 인덱스 성능 비교
- 실시간 스트리밍 응답
- 추가 PDF 업로드 지원 (선택적)
"""

import gradio as gr
import os
import json
import time
import hashlib
import html
import threading
import fitz
import openai
import faiss
import numpy as np
from pathlib import Path
from collections import OrderedDict
from datetime import datetime
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from typing import Dict, Generator, List, Optional

# Environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_API_KEY = os.getenv("HUGGING_FACE_API_KEY") or os.getenv("HF_TOKEN")
HF_ENTERPRISE = os.getenv("HUGGING_FACE_ENTERPRISE")

# Configuration
DB_PATH = Path("faiss_db")
DIMENSION = 384
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Models setup
MODELS = {
    "GPT-4": {"model_id": "gpt-4", "provider": "openai"},
    "DeepSeek-R1": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "provider": "novita",
    },
    "Gemma-3-27B": {"model_id": "google/gemma-2-27b-it", "provider": "hf_inference"},
    "Llama-3.3-70B": {
        "model_id": "meta-llama/Llama-3.3-70B-Instruct",
        "provider": "hf_inference",
    },
    "QwQ-32B": {"model_id": "Qwen/QwQ-32B-Preview", "provider": "hf_inference"},
}

# Rerank options
RERANK_OPTIONS = {
    "없음": None,
    "ms-marco-MiniLM-L-6-v2": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "ms-marco-MiniLM-L-12-v2": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "mmarco-mMiniLMv2-L12-H384-v1": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
}

# Initialize embeddings
EMBED_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# System prompt
system_prompt = """You are a GIST Rules and Regulations Expert Assistant. 
You have comprehensive knowledge of all GIST academic rules, regulations, guidelines, and policies.
Always provide accurate, detailed answers based on the provided context.
When answering questions about GIST rules, cite specific regulation numbers and titles when available."""


# --------- (A) PERFORMANCE LOGGING & FAISS INDEX COMPARISON ---------
class PerformanceLogger:
    """성능 측정 결과를 JSON 파일로 기록하는 클래스"""

    def __init__(self, log_dir: str = "performance_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = (
            self.log_dir
            / f"faiss_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )

    def log_query_performance(self, query: str, results: Dict):
        """단일 쿼리의 성능 결과를 로깅"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "query_hash": hashlib.md5(query.encode()).hexdigest()[:8],
            "results": results,
        }

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        print(f"📊 성능 결과 기록: {self.log_file}")


class ChatLogger:
    """채팅 세션과 관련된 모든 정보를 기록하는 클래스"""

    def __init__(self, log_dir: str = "chat_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = (
            self.log_dir
            / f"chat_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )

    def log_chat_interaction(
        self,
        user_query: str,
        bot_response: str,
        rerank_method: str,
        model_info: Dict,
        performance_metrics: Dict,
        retrieved_docs: List[str] = None,
        faiss_performance: Dict = None,
        session_id: str = None,
    ):
        """채팅 상호작용과 관련된 모든 정보를 로그에 기록"""

        # 리랭킹 상세 정보
        rerank_config = RERANK_OPTIONS.get(rerank_method, {})
        rerank_info = {
            "method_name": rerank_method,
            "enabled": rerank_config.get("enabled", False) if rerank_config else False,
            "model": rerank_config.get("model") if rerank_config else None,
            "top_k": rerank_config.get("top_k", 3) if rerank_config else 3,
        }

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "query_hash": hashlib.md5(user_query.encode()).hexdigest()[:8],
            "interaction": {
                "user_query": user_query,
                "bot_response": bot_response,
                "query_length": len(user_query),
                "response_length": len(bot_response),
            },
            "model_info": model_info,
            "rerank_info": rerank_info,
            "performance_metrics": performance_metrics,
            "faiss_performance": faiss_performance,
            "retrieved_documents": retrieved_docs[:3]
            if retrieved_docs
            else None,  # 처음 3개만 저장
            "metadata": {
                "total_documents": len(shared_state.get("pdfs", [])),
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "vector_dimension": DIMENSION,
            },
        }

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False, indent=None) + "\n")

        print(
            f"💬 채팅 로그 기록: {self.log_file.name} ({len(user_query)} chars query)"
        )

    def get_log_files(self) -> List[Path]:
        """사용 가능한 로그 파일 목록 반환"""
        return sorted(self.log_dir.glob("chat_session_*.jsonl"), reverse=True)


class FaissIndexComparator:
    """다양한 FAISS 인덱스 타입별 성능 비교"""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.indexes = {}
        self.performance_logger = PerformanceLogger()

    def load_indexes_from_files(self) -> Dict:
        """파일에서 사전 구축된 FAISS 인덱스들 로드"""
        print("🔧 사전 구축된 FAISS 인덱스 로드 중...")

        results = {}

        # 기본 IndexFlatL2 (vectorstore에서)
        if (DB_PATH / "index.faiss").exists():
            flat_index = faiss.read_index(str(DB_PATH / "index.faiss"))
            results["IndexFlatL2"] = {
                "index": flat_index,
                "memory_usage": flat_index.d * flat_index.ntotal * 4,  # float32
            }

        # IndexIVFFlat
        if (DB_PATH / "vectorstore_ivf.faiss").exists():
            ivf_index = faiss.read_index(str(DB_PATH / "vectorstore_ivf.faiss"))
            results["IndexIVFFlat"] = {
                "index": ivf_index,
                "memory_usage": ivf_index.d * ivf_index.ntotal * 4,
            }

        # IndexHNSWFlat
        if (DB_PATH / "vectorstore_hnsw.faiss").exists():
            hnsw_index = faiss.read_index(str(DB_PATH / "vectorstore_hnsw.faiss"))
            results["IndexHNSWFlat"] = {
                "index": hnsw_index,
                "memory_usage": hnsw_index.d * hnsw_index.ntotal * 4,
            }

        # 결과 저장
        self.indexes = {name: data["index"] for name, data in results.items()}

        print(f"✅ {len(results)}개 인덱스 타입 로드 완료")
        return results

    def compare_search_performance(self, query_vector: np.ndarray, k: int = 3) -> Dict:
        """모든 인덱스에서 검색 성능 비교"""
        results = {}

        for index_name, index in self.indexes.items():
            start_time = time.time()

            try:
                distances, indices = index.search(query_vector.reshape(1, -1), k)
                search_time = time.time() - start_time
                results[index_name] = {
                    "search_time_ms": search_time * 1000,  # 밀리초로 변환
                    "distances": distances.tolist(),
                    "indices": indices.tolist(),
                    "success": True,
                }

            except Exception as e:
                results[index_name] = {
                    "search_time_ms": 0,
                    "error": str(e),
                    "success": False,
                }
                print(f"⚠️ {index_name} 검색 실패: {e}")

        return results


class PerformanceMetrics:
    """성능 지표를 추적하는 클래스"""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.first_token_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.token_count: int = 0
        self.retrieval_time: float = 0.0

    def start_query(self):
        """쿼리 시작 시간 기록"""
        self.start_time = time.time()
        self.first_token_time = None
        self.end_time = None
        self.token_count = 0
        self.retrieval_time = 0.0

    def first_token_received(self):
        """첫 토큰 수신 시간 기록"""
        if self.first_token_time is None:
            self.first_token_time = time.time()

    def add_token(self, token_text: str):
        """토큰 추가"""
        self.token_count += len(token_text.split())

    def query_complete(self):
        """쿼리 완료 시간 기록"""
        self.end_time = time.time()

    def get_metrics(self) -> Dict:
        """현재 메트릭 반환"""
        if not self.start_time:
            return {"상태": "측정 중..."}

        metrics = {}
        current_time = self.end_time or time.time()

        # 첫 토큰까지 시간
        if self.first_token_time:
            time_to_first_token = self.first_token_time - self.start_time
            metrics["첫 토큰"] = f"{time_to_first_token:.2f}초"
        else:
            metrics["첫 토큰"] = "대기 중..."

        # 전체 응답 시간
        if self.end_time:
            total_time = self.end_time - self.start_time
            metrics["총 시간"] = f"{total_time:.2f}초"

            # 토큰/초
            if self.token_count > 0 and total_time > 0:
                tokens_per_second = self.token_count / total_time
                metrics["속도"] = f"{tokens_per_second:.1f} tokens/s"
        else:
            elapsed = current_time - self.start_time
            metrics["경과 시간"] = f"{elapsed:.1f}초"

        # 검색 시간
        if self.retrieval_time > 0:
            metrics["검색 시간"] = f"{self.retrieval_time:.2f}초"

        return metrics


# Session variables
MAX_SESSIONS = 100
sessions = OrderedDict()
session_lock = threading.Lock()

# Global shared state
shared_state: Dict = {
    "current_model": "GPT-4",
    "vectorstore": None,
    "faiss_comparator": None,
    "performance_logger": None,
    "chat_logger": None,
    "database_loaded": False,
    "database_info": {},
}
shared_state_lock = threading.Lock()

# Performance trackers
performance_trackers = {
    method: PerformanceMetrics() for method in RERANK_OPTIONS.keys()
}


def load_existing_database():
    """기존 FAISS 데이터베이스 로드"""
    print("🔍 기존 FAISS 데이터베이스 검색 중...")

    if not DB_PATH.exists():
        print(f"❌ 데이터베이스 경로가 존재하지 않습니다: {DB_PATH}")
        print("🛠️ 먼저 다음 명령어를 실행하세요: python build_rule_database.py")
        return False

    if not (DB_PATH / "index.faiss").exists() or not (DB_PATH / "index.pkl").exists():
        print("❌ FAISS 데이터베이스 파일이 없습니다!")
        print("🛠️ 먼저 다음 명령어를 실행하세요: python build_rule_database.py")
        return False

    try:
        # 메타데이터 로드
        info_file = DB_PATH / "database_info.json"
        if info_file.exists():
            with open(info_file, "r", encoding="utf-8") as f:
                database_info = json.load(f)
                shared_state["database_info"] = database_info
                print(
                    f"📊 데이터베이스 정보: {database_info['total_documents']}개 문서, {database_info['total_chunks']}개 청크"
                )

        # FAISS 벡터스토어 로드
        print("🔄 FAISS 벡터스토어 로드 중...")
        vectorstore = FAISS.load_local(
            str(DB_PATH), EMBED_MODEL, allow_dangerous_deserialization=True
        )

        with shared_state_lock:
            shared_state["vectorstore"] = vectorstore

            # 성능 비교 시스템 초기화
            print("🔧 FAISS 성능 비교 시스템 초기화 중...")
            shared_state["performance_logger"] = PerformanceLogger()
            shared_state["chat_logger"] = ChatLogger()
            shared_state["faiss_comparator"] = FaissIndexComparator(dimension=DIMENSION)

            # 사전 구축된 인덱스들 로드
            index_results = shared_state["faiss_comparator"].load_indexes_from_files()
            print(f"✅ 성능 비교 준비 완료: {len(index_results)}개 인덱스 타입")

            shared_state["database_loaded"] = True

        print("✅ 데이터베이스 로드 완료!")
        return True

    except Exception as e:
        print(f"❌ 데이터베이스 로드 실패: {e}")
        return False


# --------- (B) SESSION SETUP ---------
def get_session_id(request: gr.Request):
    raw_id = request.client.host + str(request.headers.get("user-agent"))
    return hashlib.md5(raw_id.encode()).hexdigest()


def init_session(session_id: str):
    sessions[session_id] = {
        "client": None,
        "history": {method: [] for method in RERANK_OPTIONS.keys()},
    }


def get_client(model_name: str):
    model_info = MODELS[model_name]

    if model_info["provider"] == "openai":
        return openai.OpenAI(api_key=OPENAI_API_KEY)
    elif model_info["provider"] == "novita":
        return openai.OpenAI(
            api_key=HF_API_KEY, base_url="https://api.novita.ai/v3/openai"
        )
    elif model_info["provider"] == "hf_inference":
        return InferenceClient(api_key=HF_API_KEY)
    else:
        raise ValueError(f"Unknown provider: {model_info['provider']}")


# --------- (C) RETRIEVER CREATION ---------
def create_retriever(vectorstore, rerank_method="없음"):
    if rerank_method == "없음" or not rerank_method:
        return vectorstore.as_retriever(search_kwargs={"k": 3})

    try:
        cross_encoder = HuggingFaceCrossEncoder(model_name=rerank_method)
        compressor = CrossEncoderReranker(model=cross_encoder, top_k=3)
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
        )
    except Exception as e:
        print(f"Reranker creation failed for {rerank_method}: {e}")
        return vectorstore.as_retriever(search_kwargs={"k": 3})


# --------- (D) QUERY HANDLERS ---------
def handle_query_for_rerank(
    user_query: str, rerank_method: str, request: gr.Request
) -> Generator:
    """특정 rerank 방법으로 쿼리 처리"""
    session_id = get_session_id(request)

    # Ensure session-safe access
    with session_lock:
        if session_id not in sessions:
            init_session(session_id)
        session = sessions[session_id]
        sessions.move_to_end(session_id)

    # 성능 추적 시작
    metrics = performance_trackers[rerank_method]
    metrics.start_query()

    # 히스토리 가져오기
    history = session["history"][rerank_method]
    messages = history.copy()
    client = session["client"]

    # 현재 모델 정보 가져오기
    with shared_state_lock:
        current_model = str(shared_state["current_model"])
        vectorstore = shared_state["vectorstore"]
        faiss_comparator = shared_state.get("faiss_comparator")
        performance_logger = shared_state.get("performance_logger")

    model_info = MODELS[current_model]

    # Extract relevant text data from PDFs
    context = ""
    faiss_performance_results = {}

    if vectorstore:
        print(f"🔍 [{rerank_method}] Retrieving relevant GIST rules and regulations...")
        retrieval_start = time.time()

        retriever = create_retriever(vectorstore, rerank_method)
        if retriever:
            docs = retriever.invoke(user_query)
            # 문서 소스 정보 포함
            context_parts: List[str] = []
            for doc in docs:
                source_info = doc.metadata.get(
                    "filename", doc.metadata.get("source", "")
                )
                category = doc.metadata.get("category", "")
                context_parts.append(f"[{category}] {source_info}:\n{doc.page_content}")
            context = "\n\n".join(context_parts)

        retrieval_end = time.time()
        metrics.retrieval_time = retrieval_end - retrieval_start
        print(
            f"📊 [{rerank_method}] Retrieved {len(docs) if 'docs' in locals() else 0} documents in {metrics.retrieval_time:.2f}s"
        )

        # 🚀 FAISS 인덱스 성능 비교 실행
        if faiss_comparator and performance_logger:
            print(f"🔬 [{rerank_method}] FAISS 인덱스 성능 비교 시작...")
            try:
                # 쿼리 벡터 생성
                query_vector = EMBED_MODEL.embed_query(user_query)
                query_vector_np = np.array(query_vector, dtype=np.float32)

                # 모든 인덱스에서 성능 비교
                faiss_performance_results = faiss_comparator.compare_search_performance(
                    query_vector_np, k=3
                )

                # 성능 결과 로깅
                performance_logger.log_query_performance(
                    user_query,
                    {
                        "rerank_method": rerank_method,
                        "faiss_performance": faiss_performance_results,
                        "retrieval_time": metrics.retrieval_time,
                    },
                )

                # 성능 결과 출력
                print(f"📊 [{rerank_method}] FAISS 성능 비교 결과:")
                for index_name, result in faiss_performance_results.items():
                    if result.get("success"):
                        print(f"   • {index_name}: {result['search_time_ms']:.2f}ms")
                    else:
                        print(
                            f"   • {index_name}: 실패 - {result.get('error', 'Unknown error')}"
                        )

            except Exception as e:
                print(f"⚠️ [{rerank_method}] FAISS 성능 비교 중 오류: {e}")
                faiss_performance_results = {"error": str(e)}

    messages.append(
        {
            "role": "user",
            "content": f"Context (GIST Rules & Regulations):\n{context}\n\nQuestion: {user_query}",
        }
    )

    # Add user message to history first
    history.append({"role": "user", "content": user_query})

    # Create initial assistant message placeholder
    history.append({"role": "assistant", "content": ""})

    # Yield initial state with user query
    yield history, format_metrics(metrics.get_metrics(), rerank_method)

    # Invoke client with user query using streaming
    print(f"💬 [{rerank_method}] Inquiring LLM with streaming...")

    try:
        if model_info["provider"] == "openai":
            # OpenAI 스트리밍
            completion = client.chat.completions.create(
                model=model_info["model_id"], messages=messages, stream=True
            )

            bot_response = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    if not metrics.first_token_time:
                        metrics.first_token_received()

                    chunk_content = chunk.choices[0].delta.content
                    bot_response += chunk_content
                    metrics.add_token(chunk_content)

                    # Update the last message (assistant's response)
                    history[-1]["content"] = html.escape(bot_response)
                    yield history, format_metrics(metrics.get_metrics(), rerank_method)

        else:
            # HuggingFace Inference Client 스트리밍
            completion = client.chat.completions.create(
                model=model_info["model_id"], messages=messages, stream=True
            )

            bot_response = ""
            for chunk in completion:
                if hasattr(chunk, "choices") and chunk.choices[0].delta.content:
                    if not metrics.first_token_time:
                        metrics.first_token_received()

                    chunk_content = chunk.choices[0].delta.content
                    bot_response += chunk_content
                    metrics.add_token(chunk_content)

                    # Update the last message (assistant's response)
                    history[-1]["content"] = html.escape(bot_response)
                    yield history, format_metrics(metrics.get_metrics(), rerank_method)

    except Exception as e:
        error_msg = f"오류가 발생했습니다: {str(e) if str(e) else '스트리밍 중 알 수 없는 오류가 발생했습니다.'}"
        print(
            f"❌ [{rerank_method}] Streaming error: {e if str(e) else 'Unknown streaming error'}"
        )
        history[-1]["content"] = error_msg
        yield history, format_metrics(metrics.get_metrics(), rerank_method)

    # 완료 시점 기록
    metrics.query_complete()
    final_metrics = format_metrics(metrics.get_metrics(), rerank_method)

    # 🚀 채팅 상호작용 로깅
    with shared_state_lock:
        chat_logger = shared_state.get("chat_logger")

    if chat_logger and "bot_response" in locals():
        try:
            # 검색된 문서 정보 추출
            retrieved_doc_sources = []
            if "docs" in locals() and docs:
                for doc in docs:
                    source = doc.metadata.get(
                        "filename", doc.metadata.get("source", "Unknown")
                    )
                    category = doc.metadata.get("category", "")
                    retrieved_doc_sources.append(f"[{category}] {source}")

            # 채팅 로그 기록
            chat_logger.log_chat_interaction(
                user_query=user_query,
                bot_response=bot_response,
                rerank_method=rerank_method,
                model_info=model_info,
                performance_metrics=metrics.get_metrics(),
                retrieved_docs=retrieved_doc_sources,
                faiss_performance=faiss_performance_results
                if "faiss_performance_results" in locals()
                else None,
                session_id=session_id,
            )
        except Exception as e:
            print(f"⚠️ 채팅 로그 기록 중 오류: {e}")

    yield history, final_metrics


def handle_multi_query(user_query, request: gr.Request):
    """모든 rerank 모드에서 동시에 쿼리 실행"""
    if not user_query.strip():
        return [[] for _ in RERANK_OPTIONS.keys()] + [
            format_metrics({}, method) for method in RERANK_OPTIONS.keys()
        ]

    print(f"💬 사용자 질문 처리 시작: {user_query[:50]}...")

    # 모든 rerank 모드에 대해 제너레이터 생성
    generators = {
        method: handle_query_for_rerank(user_query, method, request)
        for method in RERANK_OPTIONS.keys()
    }

    # 현재 상태 추적
    current_states = {
        method: ([], format_metrics({}, method)) for method in RERANK_OPTIONS.keys()
    }
    active_generators = set(RERANK_OPTIONS.keys())

    while active_generators:
        updated_methods = set()

        for method in list(active_generators):
            try:
                history, metrics = next(generators[method])
                current_states[method] = (history, metrics)
                updated_methods.add(method)
            except StopIteration:
                active_generators.remove(method)
                print(f"✅ {method} completed")

        if updated_methods or len(active_generators) == 0:
            # 결과를 올바른 순서로 정렬하여 반환
            results = []
            for method in RERANK_OPTIONS.keys():
                history, metrics = current_states[method]
                results.extend([history, metrics])

            yield results

    print("✅ 모든 검색 방식으로 답변 완료!")


def handle_additional_pdf_upload(pdfs, request: gr.Request):
    """추가 PDF 업로드 처리"""
    if not pdfs:
        return "업로드된 파일이 없습니다."

    print("📄 Processing additional PDF(s)...")

    try:
        # PDF에서 텍스트 추출
        additional_docs = []
        for pdf in pdfs:
            text = ""
            try:
                doc = fitz.open(pdf)
                text = "\n".join([page.get_text("text") for page in doc])
                doc.close()
            except Exception as e:
                print(f"PDF 처리 실패 {pdf}: {e}")
                continue

            if text.strip():
                document = Document(
                    page_content=text,
                    metadata={"source": pdf, "filename": os.path.basename(pdf)},
                )
                docs = TEXT_SPLITTER.split_documents([document])
                additional_docs.extend(docs)

        if not additional_docs:
            return "처리할 수 있는 텍스트가 없습니다."

        # 기존 벡터스토어에 추가
        with shared_state_lock:
            vectorstore = shared_state["vectorstore"]
            if vectorstore:
                print("🔄 Merging with existing documents...")
                new_vectorstore = FAISS.from_documents(additional_docs, EMBED_MODEL)
                vectorstore.merge_from(new_vectorstore)
                shared_state["vectorstore"] = vectorstore

        return f"✅ Added {len(pdfs)} PDFs in {time.time():.2f} seconds"

    except Exception as e:
        return f"❌ 오류 발생: {str(e)}"


# --------- (D) 로그 뷰어 기능 ---------
def get_log_files():
    """사용 가능한 로그 파일 목록 반환"""
    chat_logs_dir = Path("chat_logs")
    performance_logs_dir = Path("performance_logs")

    log_files = []

    # 채팅 로그 파일들
    if chat_logs_dir.exists():
        for log_file in sorted(
            chat_logs_dir.glob("chat_session_*.jsonl"), reverse=True
        ):
            log_files.append(f"📝 {log_file.name} (채팅 로그)")

    # 성능 로그 파일들
    if performance_logs_dir.exists():
        for log_file in sorted(
            performance_logs_dir.glob("faiss_performance_*.jsonl"), reverse=True
        ):
            log_files.append(f"⚡ {log_file.name} (성능 로그)")

    return log_files if log_files else ["📁 사용 가능한 로그 파일이 없습니다."]


def load_log_content(selected_log_file: str):
    """선택된 로그 파일의 내용을 읽어서 포맷된 텍스트로 반환"""
    if not selected_log_file or "사용 가능한 로그 파일이 없습니다" in selected_log_file:
        return "📝 로그 파일을 선택해주세요."

    try:
        # 파일명 추출 (이모지와 설명 제거)
        file_name = selected_log_file.split(" ")[1]  # 이모지 다음 첫 번째 단어

        # 파일 경로 결정
        if "채팅 로그" in selected_log_file:
            log_path = Path("chat_logs") / file_name
        elif "성능 로그" in selected_log_file:
            log_path = Path("performance_logs") / file_name
        else:
            return "❌ 알 수 없는 로그 파일 형식입니다."

        if not log_path.exists():
            return f"❌ 로그 파일을 찾을 수 없습니다: {log_path}"

        # 로그 내용 읽기
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            return "📝 로그 파일이 비어있습니다."

        # 로그 포맷팅
        formatted_content = []
        formatted_content.append(f"# 📊 로그 파일: {file_name}")
        formatted_content.append(f"📁 **경로**: `{log_path}`")
        formatted_content.append(f"📈 **총 항목 수**: {len(lines)}개")
        formatted_content.append("\n---\n")

        for i, line in enumerate(lines[-20:], 1):  # 최근 20개 항목만 표시
            try:
                log_entry = json.loads(line.strip())

                if "interaction" in log_entry:
                    # 채팅 로그 포맷팅
                    timestamp = (
                        log_entry.get("timestamp", "").replace("T", " ").split(".")[0]
                    )
                    query = (
                        log_entry["interaction"]["user_query"][:100] + "..."
                        if len(log_entry["interaction"]["user_query"]) > 100
                        else log_entry["interaction"]["user_query"]
                    )
                    response_length = log_entry["interaction"]["response_length"]
                    rerank_method = log_entry["rerank_info"]["method_name"]
                    model = log_entry["model_info"]["model_id"]

                    formatted_content.append(f"## 💬 채팅 #{len(lines) - 20 + i}")
                    formatted_content.append(f"**시간**: {timestamp}")
                    formatted_content.append(f"**질문**: {query}")
                    formatted_content.append(f"**답변 길이**: {response_length}자")
                    formatted_content.append(f"**리랭킹**: {rerank_method}")
                    formatted_content.append(f"**모델**: {model}")

                    # 성능 지표
                    if (
                        "performance_metrics" in log_entry
                        and log_entry["performance_metrics"]
                    ):
                        metrics = log_entry["performance_metrics"]
                        if "total_time" in metrics:
                            formatted_content.append(
                                f"**총 시간**: {metrics['total_time']:.2f}초"
                            )
                        if "tokens_per_second" in metrics:
                            formatted_content.append(
                                f"**속도**: {metrics['tokens_per_second']:.1f} tokens/s"
                            )

                elif "results" in log_entry:
                    # 성능 로그 포맷팅
                    timestamp = (
                        log_entry.get("timestamp", "").replace("T", " ").split(".")[0]
                    )
                    query_hash = log_entry.get("query_hash", "")
                    rerank_method = log_entry["results"]["rerank_method"]

                    formatted_content.append(f"## ⚡ 성능 측정 #{len(lines) - 20 + i}")
                    formatted_content.append(f"**시간**: {timestamp}")
                    formatted_content.append(f"**쿼리 해시**: {query_hash}")
                    formatted_content.append(f"**리랭킹**: {rerank_method}")

                    # FAISS 성능 결과
                    if "faiss_performance" in log_entry["results"]:
                        faiss_results = log_entry["results"]["faiss_performance"]
                        for index_name, result in faiss_results.items():
                            if result.get("success"):
                                formatted_content.append(
                                    f"**{index_name}**: {result['search_time_ms']:.2f}ms"
                                )

                formatted_content.append("\n---\n")

            except (json.JSONDecodeError, KeyError) as e:
                formatted_content.append(f"❌ 로그 항목 파싱 오류 (라인 {i}): {e}")
                continue

        return "\n".join(formatted_content)

    except Exception as e:
        return f"❌ 로그 파일 읽기 오류: {str(e)}"


# --------- (E) UTILITY FUNCTIONS ---------
def format_metrics(metrics: Dict, rerank_method: str) -> str:
    """메트릭을 HTML 형식으로 포맷"""
    if not metrics:
        return f"**{rerank_method}** - 측정 중..."

    # 안전한 문자열 변환
    try:
        formatted_lines = [f"**{rerank_method}**"]
        for key, value in metrics.items():
            formatted_lines.append(f"- **{key}**: {value}")

        result = "\n".join(formatted_lines)
        # 문자열 타입 확인 및 보장
        return str(result) if result else f"**{rerank_method}** - 데이터 없음"

    except Exception as e:
        print(f"❌ format_metrics 오류: {e}")
        return f"**{rerank_method}** - 형식 오류"


def copy_as_markdown(history, rerank_method):
    """대화 내용을 마크다운으로 복사"""
    if not history:
        return "복사할 내용이 없습니다."

    markdown_content = f"# GIST Rules Analyzer - {rerank_method} 검색 결과\n\n"
    markdown_content += (
        f"**생성 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )

    for i, message in enumerate(history):
        if message["role"] == "user":
            markdown_content += f"## 🙋‍♂️ 질문 {(i // 2) + 1}\n{message['content']}\n\n"
        elif message["role"] == "assistant":
            markdown_content += (
                f"## 🤖 답변 ({rerank_method})\n{message['content']}\n\n"
            )

    return markdown_content


def reset_all_chats():
    """모든 채팅 기록 초기화"""
    with session_lock:
        for session in sessions.values():
            for method in RERANK_OPTIONS.keys():
                session["history"][method] = []

    # 성능 트래커 초기화
    for tracker in performance_trackers.values():
        tracker.__init__()

    return [[] for _ in RERANK_OPTIONS.keys()] + [
        format_metrics({}, method) for method in RERANK_OPTIONS.keys()
    ]


def change_model(model_name: str):
    """모델 변경"""
    print(f"🔄 Model changed to: {model_name}")

    with shared_state_lock:
        shared_state["current_model"] = model_name

    # 모든 세션의 클라이언트 업데이트
    with session_lock:
        for session in sessions.values():
            session["client"] = get_client(model_name)

    return f"✅ {model_name} 준비완료"


# --------- (F) DATABASE STATUS ---------
def get_database_status():
    """데이터베이스 상태 반환"""
    with shared_state_lock:
        if not shared_state["database_loaded"]:
            return "❌ 데이터베이스가 로드되지 않았습니다.\n먼저 `python build_rule_database.py`를 실행하세요."

        db_info = shared_state["database_info"]
        status_lines = [
            "✅ **데이터베이스 로드 완료**",
            "",
            f"📊 **통계**:",
            f"- 총 문서: {db_info.get('total_documents', 'N/A')}개",
            f"- 총 청크: {db_info.get('total_chunks', 'N/A')}개",
            f"- 임베딩 차원: {db_info.get('dimension', 'N/A')}",
            f"- 생성 일시: {db_info.get('created_at', 'N/A')}",
            "",
            f"💾 **파일 크기**:",
            f"- FAISS 인덱스: {db_info.get('file_sizes', {}).get('vectorstore.faiss', 0) / (1024 * 1024):.1f}MB",
            f"- 메타데이터: {db_info.get('file_sizes', {}).get('vectorstore.pkl', 0) / (1024 * 1024):.1f}MB",
        ]

        additional_indexes = db_info.get("additional_indexes", {})
        if additional_indexes:
            status_lines.extend(
                ["", f"⚡ **성능 최적화 인덱스**: {len(additional_indexes)}개 준비됨"]
            )

        return "\n".join(status_lines)


# --------- (G) UI SETUP ---------
css = """
div {
    flex-wrap: nowrap !important;
}
.responsive-height {
    height: 100vh !important;
    padding-bottom: 20px !important;
}
.fill-height {
    height: 100% !important;
    flex-wrap: nowrap !important;
}
.extend-height {
    min-height: 300px !important;
    flex: 1 !important;
    overflow: auto !important;
}
.metrics-box {
    background: linear-gradient(135deg, #e3f2fd 0%, #f8f9fa 100%) !important;
    border: 2px solid #2196f3 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin: 12px 0 !important;
    color: #1565c0 !important;
    font-weight: 500 !important;
    box-shadow: 0 2px 8px rgba(33, 150, 243, 0.1) !important;
}
.metrics-box h2 {
    color: #0d47a1 !important;
    font-size: 1.1em !important;
    margin-bottom: 8px !important;
    font-weight: 600 !important;
}
.metrics-box p {
    color: #1565c0 !important;
    margin: 4px 0 !important;
    font-size: 0.95em !important;
    line-height: 1.4 !important;
}
.status-box {
    background: linear-gradient(135deg, #1a237e 0%, #283593 100%) !important;
    border: 2px solid #3f51b5 !important;
    border-radius: 12px !important;
    padding: 20px !important;
    margin: 12px 0 !important;
    color: #ffffff !important;
    font-weight: 500 !important;
    box-shadow: 0 4px 12px rgba(63, 81, 181, 0.3) !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}
.status-box h2, .status-box h3 {
    color: #e3f2fd !important;
    margin-bottom: 8px !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3) !important;
}
.status-box strong {
    color: #e8eaf6 !important;
    font-weight: 600 !important;
}
.status-box code {
    background: rgba(255, 255, 255, 0.15) !important;
    padding: 2px 8px !important;
    border-radius: 4px !important;
    font-family: 'Courier New', monospace !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}
.status-box p {
    color: #ffffff !important;
    margin: 6px 0 !important;
}
.status-box ul li {
    color: #ffffff !important;
    margin: 4px 0 !important;
}
footer {
    display: none !important;
}
"""

# 데이터베이스 로드 (앱 시작 시)
print("🚀 GIST Rules Analyzer (Prebuilt Database Version) 시작!")
if not load_existing_database():
    print("❌ 앱을 시작할 수 없습니다. 먼저 데이터베이스를 구축해주세요.")
    exit(1)

with gr.Blocks(
    title="GIST Rules Analyzer - Prebuilt DB", css=css, fill_height=True
) as demo:
    gr.Markdown(
        "<center><h1>📚 GIST Rules Analyzer (Prebuilt)</h1><p><strong>사전 구축 데이터베이스 버전</strong> | 고속 시작 및 실시간 성능 비교</p></center>"
    )

    # 데이터베이스 상태 표시
    with gr.Row():
        database_status = gr.Markdown(
            value=get_database_status(), elem_classes=["status-box"]
        )

    with gr.Row():
        with gr.Column(scale=2):
            # 공통 컨트롤
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    list(MODELS.keys()),
                    label="🧠 LLM 모델 선택",
                    value="GPT-4",
                    scale=2,
                )
                model_status = gr.Textbox(
                    label="모델 상태",
                    value="✅ GPT-4 준비완료",
                    interactive=False,
                    scale=1,
                )

            additional_pdf_upload = gr.Files(
                label="📄 추가 문서 업로드 (Vector Store 확장)", file_types=[".pdf"]
            )

            user_input = gr.Textbox(
                label="🔍 질의문 입력 (Query Input)",
                placeholder="예: 교수님이 박사과정 학생을 지도할 수 있는 기간은 언제까지인가요?",
                info="4가지 검색 방식으로 동시 테스트됩니다",
                lines=3,
                interactive=True,
            )

        with gr.Column(scale=1):
            submit_btn = gr.Button("🚀 테스트 실행", variant="primary", size="lg")
            reset_btn = gr.Button("🔄 초기화", size="lg")

    # 4개의 채팅 인터페이스 (2x2 그리드)
    with gr.Row(elem_classes=["fill-height"]):
        with gr.Column(scale=1, elem_classes=["fill-height"]):
            with gr.Group(elem_classes=["extend-height"]):
                gr.Markdown("### 🔍 기본 벡터 검색 (Baseline)")
                with gr.Row():
                    gr.Dropdown(
                        ["Vector Search Only"],
                        value="Vector Search Only",
                        interactive=False,
                        scale=3,
                        show_label=False,
                    )
                chatbot_none = gr.Chatbot(
                    elem_classes=["extend-height"],
                    show_copy_button=True,
                    type="messages",
                )
                metrics_none = gr.Markdown(
                    "**없음** - 측정 중...", elem_classes=["metrics-box"]
                )
                copy_btn_none = gr.Button("📋 결과 복사", size="sm")

            with gr.Group(elem_classes=["extend-height"]):
                gr.Markdown("### 🎯 Cross-Encoder Rerank (Basic)")
                with gr.Row():
                    gr.Dropdown(
                        ["ms-marco-MiniLM-L-6-v2"],
                        value="ms-marco-MiniLM-L-6-v2",
                        interactive=False,
                        scale=3,
                        show_label=False,
                    )
                chatbot_basic = gr.Chatbot(
                    elem_classes=["extend-height"],
                    show_copy_button=True,
                    type="messages",
                )
                metrics_basic = gr.Markdown(
                    "**Cross-Encoder (기본)** - 측정 중...",
                    elem_classes=["metrics-box"],
                )
                copy_btn_basic = gr.Button("📋 결과 복사", size="sm")

        with gr.Column(scale=1, elem_classes=["fill-height"]):
            with gr.Group(elem_classes=["extend-height"]):
                gr.Markdown("### 🚀 Cross-Encoder Rerank (Advanced)")
                with gr.Row():
                    gr.Dropdown(
                        ["ms-marco-MiniLM-L-12-v2"],
                        value="ms-marco-MiniLM-L-12-v2",
                        interactive=False,
                        scale=3,
                        show_label=False,
                    )
                chatbot_advanced = gr.Chatbot(
                    elem_classes=["extend-height"],
                    show_copy_button=True,
                    type="messages",
                )
                metrics_advanced = gr.Markdown(
                    "**Cross-Encoder (고성능)** - 측정 중...",
                    elem_classes=["metrics-box"],
                )
                copy_btn_advanced = gr.Button("📋 결과 복사", size="sm")

            with gr.Group(elem_classes=["extend-height"]):
                gr.Markdown("### 🌍 Multilingual Cross-Encoder")
                with gr.Row():
                    gr.Dropdown(
                        ["mmarco-mMiniLMv2-L12-H384-v1"],
                        value="mmarco-mMiniLMv2-L12-H384-v1",
                        interactive=False,
                        scale=3,
                        show_label=False,
                    )
                chatbot_multilingual = gr.Chatbot(
                    elem_classes=["extend-height"],
                    show_copy_button=True,
                    type="messages",
                )
                metrics_multilingual = gr.Markdown(
                    "**다국어 Cross-Encoder** - 측정 중...",
                    elem_classes=["metrics-box"],
                )
                copy_btn_multilingual = gr.Button("📋 결과 복사", size="sm")

    # 이벤트 핸들러 (Generator로 수정)
    def init_client_on_first_query(user_query, request: gr.Request):
        session_id = get_session_id(request)
        with session_lock:
            if session_id not in sessions:
                init_session(session_id)
            if sessions[session_id]["client"] is None:
                sessions[session_id]["client"] = get_client(
                    shared_state["current_model"]
                )

        # Generator를 제대로 yield
        for result in handle_multi_query(user_query, request):
            yield result

    # 모델 변경 이벤트
    model_dropdown.change(
        fn=change_model, inputs=[model_dropdown], outputs=[model_status]
    )

    # 추가 PDF 업로드
    additional_pdf_upload.upload(
        fn=handle_additional_pdf_upload,
        inputs=[additional_pdf_upload],
        outputs=[database_status],
    )

    # 멀티 쿼리 처리
    submit_btn.click(
        fn=init_client_on_first_query,
        inputs=[user_input],
        outputs=[
            chatbot_none,
            metrics_none,
            chatbot_basic,
            metrics_basic,
            chatbot_advanced,
            metrics_advanced,
            chatbot_multilingual,
            metrics_multilingual,
        ],
    )

    user_input.submit(
        fn=init_client_on_first_query,
        inputs=[user_input],
        outputs=[
            chatbot_none,
            metrics_none,
            chatbot_basic,
            metrics_basic,
            chatbot_advanced,
            metrics_advanced,
            chatbot_multilingual,
            metrics_multilingual,
        ],
    )

    # 초기화
    reset_btn.click(
        fn=reset_all_chats,
        outputs=[
            chatbot_none,
            metrics_none,
            chatbot_basic,
            metrics_basic,
            chatbot_advanced,
            metrics_advanced,
            chatbot_multilingual,
            metrics_multilingual,
        ],
    )

    # 복사 기능
    copy_btn_none.click(
        fn=lambda h: copy_as_markdown(h, "기본 벡터 검색"),
        inputs=[chatbot_none],
        outputs=[gr.Textbox(visible=False)],
        js="(result) => navigator.clipboard.writeText(result)",
    )

    copy_btn_basic.click(
        fn=lambda h: copy_as_markdown(h, "Cross-Encoder (기본)"),
        inputs=[chatbot_basic],
        outputs=[gr.Textbox(visible=False)],
        js="(result) => navigator.clipboard.writeText(result)",
    )

    copy_btn_advanced.click(
        fn=lambda h: copy_as_markdown(h, "Cross-Encoder (고성능)"),
        inputs=[chatbot_advanced],
        outputs=[gr.Textbox(visible=False)],
        js="(result) => navigator.clipboard.writeText(result)",
    )

    copy_btn_multilingual.click(
        fn=lambda h: copy_as_markdown(h, "다국어 Cross-Encoder"),
        inputs=[chatbot_multilingual],
        outputs=[gr.Textbox(visible=False)],
        js="(result) => navigator.clipboard.writeText(result)",
    )

if __name__ == "__main__":
    print("🎉 GIST Rules Analyzer (Prebuilt Database) 준비완료!")
    print("🌐 http://localhost:7860 에서 실행 중...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

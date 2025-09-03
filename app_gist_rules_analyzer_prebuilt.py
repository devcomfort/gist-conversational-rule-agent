"""
GIST Rules Analyzer - Prebuilt Database Version (LiteLLM Integrated)
==================================================================

사전 구축된 FAISS 데이터베이스를 로드하는 고속 시작 버전입니다.
LiteLLM의 자동 모델 감지와 환경변수 기반 API 키 관리를 활용합니다.

🚀 사용 전 요구사항:
    python build_rule_database.py  # 먼저 실행

✨ 주요 특징:
- ⚡ 3초 내 앱 시작 완료
- 🤖 LiteLLM 완전 통합으로 15+ LLM 프로바이더 자동 지원
- 🎯 Dynamic Knee Detection으로 적응형 문서 선택
- 📡 실시간 스트리밍 응답
- 📄 추가 PDF 업로드 지원 (선택적)
- 🔑 환경변수 기반 자동 API 키 관리

🌐 지원하는 LLM 프로바이더 (환경변수만 설정하면 자동 감지):
- OpenAI: OPENAI_API_KEY
- Anthropic: ANTHROPIC_API_KEY
- Google: GOOGLE_API_KEY (Vertex AI, Gemini, Palm)
- Azure: AZURE_API_KEY
- Fireworks AI: FIREWORKS_AI_API_KEY ⭐ (기본 모델)
- Together AI: TOGETHER_AI_API_KEY
- Groq: GROQ_API_KEY
- Cohere: COHERE_API_KEY
- DeepSeek: DEEPSEEK_API_KEY
- Perplexity: PERPLEXITY_API_KEY
- Replicate: REPLICATE_API_TOKEN
- HuggingFace: HUGGINGFACE_API_KEY
- Novita AI: NOVITA_API_KEY
- 기타 LiteLLM 지원 프로바이더

📖 사용법:
1. 원하는 프로바이더의 API 키를 환경변수에 설정
2. 앱을 실행하면 자동으로 사용 가능한 모델들 감지
3. UI에서 모델 선택하여 사용
"""

import gradio as gr
import os
import json
import time
import hashlib
import html
import threading
import fitz
import litellm
import pystache
from pathlib import Path
from collections import OrderedDict
from datetime import datetime
from dotenv import load_dotenv
from litellm import get_valid_models
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Cross-encoder reranking 제거 - pure kneed optimization만 사용
from langchain_core.retrievers import BaseRetriever
from kneed import KneeLocator
from typing import Dict, Generator, List, Optional
from pydantic import Field
import matplotlib.pyplot as plt
import matplotlib
import base64
from io import BytesIO

# matplotlib 백엔드 설정 (서버 환경 대응)
matplotlib.use("Agg")

# Environment variables - LiteLLM이 자동으로 감지하므로 최소화
load_dotenv()
# LiteLLM은 다음 환경변수들을 자동으로 감지하고 사용:
# OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY,
# TOGETHER_AI_API_KEY, DEEPSEEK_API_KEY, PERPLEXITY_API_KEY,
# REPLICATE_API_TOKEN, HUGGINGFACE_API_KEY, NOVITA_API_KEY,
# FIREWORKS_AI_API_KEY 등

# Configuration
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Template paths
LEGAL_SYSTEM_PROMPT_PATH = Path("system_prompts/legal_agent_system_prompt.mustache")
LEGAL_QUERY_TEMPLATE_PATH = Path("templates/legal_query_template.mustache")

# 지원하는 임베딩 모델 설정 (TODO.txt 기반 - build_multi_embedding_databases.py와 동일)
EMBEDDING_MODELS = {
    "Qwen/Qwen3-Embedding-0.6B": {
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "db_name": "faiss_qwen3_embedding_0.6b",
        "dimension": 1024,
        "mteb_rank": 3,
        "description": "Qwen3 Embedding 0.6B - MTEB 3위",
    },
    "jinaai/jina-embeddings-v3": {
        "model_name": "jinaai/jina-embeddings-v3",
        "db_name": "faiss_jina_embeddings_v3",
        "dimension": 1024,
        "mteb_rank": 22,
        "description": "Jina Embeddings v3 - MTEB 22위",
    },
    "BAAI/bge-m3": {
        "model_name": "BAAI/bge-m3",
        "db_name": "faiss_bge_m3",
        "dimension": 1024,
        "mteb_rank": 23,
        "description": "BGE M3 - MTEB 23위",
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "db_name": "faiss_all_minilm_l6_v2",
        "dimension": 384,
        "mteb_rank": 117,
        "description": "All MiniLM L6 v2 - MTEB 117위 (기존 기본 모델)",
    },
}

# 기본 임베딩 모델
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 레거시 지원을 위한 기본 DB 경로
LEGACY_DB_PATH = Path("faiss_db")

# Default model configuration
DEFAULT_MODEL_ID = "fireworks_ai/accounts/fireworks/models/gpt-oss-20b"


def _detect_provider(model_id: str) -> str:
    """LiteLLM이 지원하는 프로바이더들을 감지"""
    # Anthropic
    if model_id.startswith(("claude", "anthropic/")):
        return "anthropic"

    # Google providers
    if model_id.startswith(("vertex_ai/", "gemini")):
        return "vertex_ai"
    if model_id.startswith("palm/"):
        return "palm"

    # Azure
    if model_id.startswith("azure/"):
        return "azure"

    # AWS
    if model_id.startswith("bedrock/"):
        return "bedrock"
    if model_id.startswith("sagemaker/"):
        return "sagemaker"

    # Specialized providers
    if model_id.startswith("novita/"):
        return "novita"
    if model_id.startswith("fireworks_ai/"):
        return "fireworks_ai"
    if model_id.startswith("together_ai/"):
        return "together_ai"
    if model_id.startswith("groq/"):
        return "groq"
    if model_id.startswith("cohere/"):
        return "cohere"
    if model_id.startswith("deepseek/"):
        return "deepseek"
    if model_id.startswith("perplexity/"):
        return "perplexity"
    if model_id.startswith("ollama/"):
        return "ollama"
    if model_id.startswith("replicate/"):
        return "replicate"
    if model_id.startswith("huggingface/"):
        return "huggingface"

    # Default to OpenAI for unrecognized patterns
    return "openai"


def _load_dynamic_models() -> Dict[str, Dict[str, str]]:
    """LiteLLM get_valid_models()를 활용한 동적 모델 로딩"""

    print("🔍 LiteLLM을 통해 사용 가능한 모델들을 검색 중...")

    try:
        # LiteLLM이 환경변수를 자동으로 확인하여 사용 가능한 모델들 반환
        model_ids = get_valid_models(check_provider_endpoint=True)
        print(f"✅ {len(model_ids)}개의 사용 가능한 모델을 발견했습니다")
    except Exception as e:
        print(f"⚠️ 모델 검색 중 오류: {e}")
        print("📋 기본 모델 목록을 사용합니다")
        model_ids = []

    dynamic: Dict[str, Dict[str, str]] = {}
    for mid in model_ids:
        dynamic[mid] = {"model_id": mid, "provider": _detect_provider(mid)}

    # 기본 모델이 목록에 없는 경우 추가 (환경변수가 있다고 가정)
    if DEFAULT_MODEL_ID not in dynamic:
        dynamic[DEFAULT_MODEL_ID] = {
            "model_id": DEFAULT_MODEL_ID,
            "provider": "fireworks_ai",
        }
        print(f"➕ 기본 모델 추가: {DEFAULT_MODEL_ID}")

    # 최소한 하나의 모델이 있어야 함
    if not dynamic:
        # 폴백으로 일반적인 모델들 추가
        fallback_models = [
            {"model_id": "gpt-4o-mini", "provider": "openai"},
            {"model_id": "claude-3-haiku-20240307", "provider": "anthropic"},
            {"model_id": DEFAULT_MODEL_ID, "provider": "fireworks_ai"},
        ]
        for model in fallback_models:
            dynamic[model["model_id"]] = model
        print("📋 폴백 모델 목록을 사용합니다")

    print(f"🎯 총 {len(dynamic)}개 모델 준비 완료")
    return dynamic


# Models setup (동적 로딩)
MODELS = _load_dynamic_models()

# Kneed Sensitivity Testing Options (공식 문서 기반 올바른 정의)
# 작은 S = 빠른 knee 감지 = 적은 문서 선택 = 적극적
# 큰 S = 보수적 감지 = 많은 문서 선택 = 보수적
SENSITIVITY_OPTIONS = {
    "적극적 선택 (S=1)": {
        "sensitivity": 1,
        "description": "빠른 knee 감지, 적은 문서 선택",
    },
    "균형 선택 (S=3)": {"sensitivity": 3, "description": "중간 정도 문서 선택"},
    "표준 선택 (S=5)": {"sensitivity": 5, "description": "표준적 문서 선택"},
    "보수적 선택 (S=10)": {
        "sensitivity": 10,
        "description": "신중한 문서 선택, 많은 문서 포함",
    },
}

# 추가 실험을 위한 전체 sensitivity 범위 (문서 예시 기반)
FULL_SENSITIVITY_RANGE = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100]

# Initialize embeddings
EMBED_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


# Template loading functions
def load_legal_system_prompt() -> str:
    """Load legal agent system prompt from mustache template"""
    try:
        if LEGAL_SYSTEM_PROMPT_PATH.exists():
            with open(LEGAL_SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
                template_content = f.read()

            # Legal system prompt는 static template이므로 바로 반환
            return template_content
        else:
            print(f"⚠️ Legal system prompt not found: {LEGAL_SYSTEM_PROMPT_PATH}")
            # Fallback to basic system prompt
            return """You are a GIST Rules and Regulations Expert Assistant. 
You have comprehensive knowledge of all GIST academic rules, regulations, guidelines, and policies.
Always provide accurate, detailed answers based on the provided context.
When answering questions about GIST rules, cite specific regulation numbers and titles when available."""
    except Exception as e:
        print(f"❌ Error loading legal system prompt: {e}")
        return "You are a GIST legal expert assistant."


def load_legal_query_template() -> str:
    """Load legal query template from mustache file"""
    try:
        if LEGAL_QUERY_TEMPLATE_PATH.exists():
            with open(LEGAL_QUERY_TEMPLATE_PATH, "r", encoding="utf-8") as f:
                return f.read()
        else:
            print(f"⚠️ Legal query template not found: {LEGAL_QUERY_TEMPLATE_PATH}")
            return "{{user_query}}\n\nContext:\n{{context}}"
    except Exception as e:
        print(f"❌ Error loading legal query template: {e}")
        return "{{user_query}}\n\nContext:\n{{context}}"


# Load templates
LEGAL_SYSTEM_PROMPT = load_legal_system_prompt()
LEGAL_QUERY_TEMPLATE = load_legal_query_template()

print("📋 Legal templates loaded successfully")


def render_legal_query(user_query: str, context_documents: List[Document]) -> str:
    """
    Legal query template을 사용하여 사용자 쿼리를 렌더링

    Args:
        user_query: 사용자의 질문
        context_documents: 검색된 관련 문서들

    Returns:
        렌더링된 legal query 문자열
    """
    try:
        # 현재 시간
        current_time = datetime.now()

        # 문서들을 템플릿 형식으로 변환
        template_docs = []
        for idx, doc in enumerate(context_documents):
            doc_data = {
                "index": idx + 1,
                "source": doc.metadata.get(
                    "source", doc.metadata.get("filename", "Unknown")
                ),
                "category": doc.metadata.get("category", ""),
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }

            # priority 설정 (카테고리별 우선순위)
            if doc.metadata.get("category") == "학칙":
                doc_data["priority"] = "High (학칙)"
            elif doc.metadata.get("category") in ["규정", "시행세칙"]:
                doc_data["priority"] = "Medium (규정/시행세칙)"
            else:
                doc_data["priority"] = "Normal"

            template_docs.append(doc_data)

        # 복수 법령이 관련된 경우 체크 (2개 이상의 서로 다른 source)
        unique_sources = set(doc.get("source", "") for doc in template_docs)
        has_multiple_regulations = len(unique_sources) > 1

        # 템플릿 데이터 구성
        template_data = {
            "user_query": user_query,
            "query_timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "current_datetime": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "document_count": len(context_documents),
            "context_documents": template_docs,
            "multiple_regulations": has_multiple_regulations,
        }

        # Mustache template 렌더링
        renderer = pystache.Renderer()
        rendered_query = renderer.render(LEGAL_QUERY_TEMPLATE, template_data)

        return rendered_query

    except Exception as e:
        print(f"❌ Legal query rendering failed: {e}")
        # Fallback: 간단한 형식
        context_text = "\n\n".join(
            [
                f"Document {i + 1}: {doc.page_content}"
                for i, doc in enumerate(context_documents)
            ]
        )
        return f"User Query: {user_query}\n\nContext Documents:\n{context_text}"


# --------- (A) GLOBAL STATE ---------


# Session variables
MAX_SESSIONS = 100
sessions = OrderedDict()
session_lock = threading.Lock()

# Global shared state
shared_state: Dict = {
    "current_model": None,
    "vectorstore": None,
    "database_loaded": False,
    "database_info": {},
    "last_retrievers": {},  # rerank_method별 마지막 사용된 retriever 저장
}
shared_state_lock = threading.Lock()


def get_embedding_model(model_key: str = None):
    """임베딩 모델을 동적으로 로드하고 반환"""
    if model_key is None:
        model_key = shared_state["current_embedding_model"]

    if model_key not in EMBEDDING_MODELS:
        print(f"⚠️ 지원하지 않는 임베딩 모델: {model_key}")
        model_key = DEFAULT_EMBEDDING_MODEL

    model_config = EMBEDDING_MODELS[model_key]

    try:
        print(f"🔧 임베딩 모델 로딩 중: {model_config['description']}")
        embed_model = HuggingFaceEmbeddings(
            model_name=model_config["model_name"],
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print(f"✅ 임베딩 모델 로딩 완료: {model_config['description']}")
        return embed_model
    except Exception as e:
        print(f"❌ 임베딩 모델 로딩 실패: {e}")
        # 기본 모델로 폴백
        if model_key != DEFAULT_EMBEDDING_MODEL:
            print(
                f"🔄 기본 모델로 폴백: {EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL]['description']}"
            )
            return get_embedding_model(DEFAULT_EMBEDDING_MODEL)
        else:
            raise e


def get_database_path(embedding_model_key: str = None) -> Path:
    """임베딩 모델에 따른 데이터베이스 경로 반환"""
    if embedding_model_key is None:
        embedding_model_key = shared_state["current_embedding_model"]

    if embedding_model_key in EMBEDDING_MODELS:
        db_name = EMBEDDING_MODELS[embedding_model_key]["db_name"]
        db_path = Path(db_name)

        # 멀티 임베딩 데이터베이스가 존재하는지 확인
        if db_path.exists() and (db_path / "index.faiss").exists():
            return db_path

    # 폴백: 레거시 데이터베이스 사용
    if LEGACY_DB_PATH.exists() and (LEGACY_DB_PATH / "index.faiss").exists():
        print(f"⚠️ 레거시 데이터베이스 사용: {LEGACY_DB_PATH}")
        return LEGACY_DB_PATH

    # 멀티 임베딩 데이터베이스 경로 반환 (존재하지 않아도)
    return Path(EMBEDDING_MODELS[embedding_model_key]["db_name"])


def load_database_for_embedding_model(embedding_model_key: str = None):
    """특정 임베딩 모델에 대한 데이터베이스 로드"""
    if embedding_model_key is None:
        embedding_model_key = shared_state["current_embedding_model"]

    db_path = get_database_path(embedding_model_key)
    model_config = EMBEDDING_MODELS[embedding_model_key]

    print(f"🔍 데이터베이스 검색 중: {model_config['description']}")
    print(f"📁 경로: {db_path}")

    if not db_path.exists():
        print(f"❌ 데이터베이스 경로가 존재하지 않습니다: {db_path}")
        print("🛠️ 다음 명령어로 멀티 임베딩 데이터베이스를 구축하세요:")
        print(
            f"   python build_multi_embedding_databases.py --model '{embedding_model_key}'"
        )
        return False

    if not (db_path / "index.faiss").exists() or not (db_path / "index.pkl").exists():
        print("❌ FAISS 데이터베이스 파일이 없습니다!")
        print("🛠️ 다음 명령어로 멀티 임베딩 데이터베이스를 구축하세요:")
        print(
            f"   python build_multi_embedding_databases.py --model '{embedding_model_key}'"
        )
        return False

    try:
        # 임베딩 모델 로드
        embed_model = get_embedding_model(embedding_model_key)

        # 메타데이터 로드
        info_file = db_path / "database_info.json"
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
            str(db_path), embed_model, allow_dangerous_deserialization=True
        )

        with shared_state_lock:
            shared_state["vectorstore"] = vectorstore
            shared_state["embed_model"] = embed_model
            shared_state["current_embedding_model"] = embedding_model_key
            shared_state["database_loaded"] = True

        print(f"✅ {model_config['description']} 데이터베이스 로드 완료!")
        return True

    except Exception as e:
        print(f"❌ 데이터베이스 로드 실패: {e}")
        return False


def load_existing_database():
    """기존 FAISS 데이터베이스 로드 (멀티 임베딩 모델 지원)"""
    return load_database_for_embedding_model(DEFAULT_EMBEDDING_MODEL)


# --------- (B) SESSION SETUP ---------
def get_session_id(request: gr.Request):
    raw_id = request.client.host + str(request.headers.get("user-agent"))
    return hashlib.md5(raw_id.encode()).hexdigest()


def init_session(session_id: str):
    sessions[session_id] = {
        "client": None,
        "history": {method: [] for method in SENSITIVITY_OPTIONS.keys()},
    }


def get_client(model_name: str):
    """LiteLLM 자동 설정 반환 - 환경변수 기반 자동 감지"""
    model_info = MODELS[model_name]

    # LiteLLM이 환경변수에서 자동으로 API 키를 찾아서 사용
    # 수동 설정 최소화하고 LiteLLM의 자동 감지 기능 활용
    config = {"type": "litellm", "model_id": model_info["model_id"]}

    return config


# --------- (C) DYNAMIC KNEE RETRIEVER CLASS ---------
class DynamicKneeRetriever(BaseRetriever):
    """
    Knee Point Detection을 사용하여 동적으로 관련 문서 개수를 결정하는 Retriever

    고정된 k개 대신 유사도 곡선의 knee point를 찾아서
    자연스러운 cutoff 지점까지의 모든 관련 문서를 반환합니다.
    """

    # Pydantic v2 모델 필드 정의
    vectorstore: FAISS
    min_docs: int = 2
    max_docs: int = 30  # 보수적 설정을 위해 증가
    sensitivity: float = 5.0  # 표준 선택 (문서 기반)
    direction: str = "decreasing"
    curve: str = "convex"
    last_knee_info: Dict = Field(default_factory=dict)

    # Pydantic 모델 구성을 위한 설정
    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        vectorstore: FAISS,
        min_docs: int = 2,
        max_docs: int = 30,
        sensitivity: float = 5.0,
        direction: str = "decreasing",
        curve: str = "convex",
        **data,
    ):
        """
        Args:
            vectorstore: FAISS 벡터스토어
            min_docs: 최소 반환 문서 수
            max_docs: 최대 검색 문서 수 (knee 찾기용)
            sensitivity: knee detection 민감도 (기본 5.0, 작을수록 적극적, 클수록 보수적)
            direction: "decreasing" (거리 기준) 또는 "increasing" (유사도 기준)
            curve: "convex" 또는 "concave"
        """
        # Pydantic v2 초기화
        super().__init__(
            vectorstore=vectorstore,
            min_docs=min_docs,
            max_docs=max_docs,
            sensitivity=sensitivity,
            direction=direction,
            curve=curve,
            **data,
        )

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """쿼리에 대해 knee point 기반으로 관련 문서들을 반환"""
        try:
            # 1. 최대 개수로 문서와 점수 가져오기
            docs_and_scores = self.vectorstore.similarity_search_with_score(
                query, k=self.max_docs
            )

            if len(docs_and_scores) <= self.min_docs:
                # 문서가 너무 적으면 모든 문서 반환
                self.last_knee_info = {
                    "total_docs": len(docs_and_scores),
                    "selected_docs": len(docs_and_scores),
                    "knee_point": None,
                    "reason": "Too few documents",
                }
                return [doc for doc, _ in docs_and_scores]

            # 2. 점수 추출 및 정렬 (FAISS는 거리를 반환하므로 작을수록 유사)
            scores = [score for _, score in docs_and_scores]
            documents = [doc for doc, _ in docs_and_scores]

            # 3. Knee point 찾기
            knee_idx = self._find_knee_point(scores)

            # 4. 최종 문서 선택
            if knee_idx is None or knee_idx < self.min_docs:
                # Knee를 찾지 못했거나 너무 적으면 최소 개수 반환
                selected_docs = documents[: self.min_docs]
                knee_reason = "No clear knee found, using min_docs"
            else:
                # Knee point까지 선택 (inclusive)
                selected_docs = documents[: knee_idx + 1]
                knee_reason = f"Knee point detected at index {knee_idx}"

            # 5. 분석 결과 저장 (시각화를 위해 전체 점수 저장)
            self.last_knee_info = {
                "total_docs": len(docs_and_scores),
                "selected_docs": len(selected_docs),
                "knee_point": knee_idx,
                "scores": scores,  # 시각화를 위해 모든 점수 저장
                "scores_preview": scores[:10],  # 로그용 처음 10개
                "selected_scores": [
                    score for _, score in docs_and_scores[: len(selected_docs)]
                ],
                "reason": knee_reason,
                "sensitivity": self.sensitivity,
            }

            print(
                f"🔍 Dynamic Retrieval: {len(selected_docs)}/{len(docs_and_scores)} docs selected (knee at {knee_idx})"
            )
            return selected_docs

        except Exception as e:
            print(f"❌ DynamicKneeRetriever error: {e}")
            # 에러 시 fallback으로 기본 검색
            return self.vectorstore.similarity_search(query, k=self.min_docs)

    def _find_knee_point(self, scores: List[float]) -> Optional[int]:
        """점수 리스트에서 knee point 찾기"""
        if len(scores) < 3:  # 최소 3개는 있어야 knee 찾기 가능
            return None

        try:
            # x축은 문서 인덱스, y축은 거리/점수
            x = list(range(len(scores)))
            y = scores

            # KneeLocator로 knee point 찾기
            kl = KneeLocator(
                x=x,
                y=y,
                curve=self.curve,
                direction=self.direction,
                S=self.sensitivity,
                online=True,  # 온라인 모드로 더 정확한 탐지
            )

            return kl.knee

        except Exception as e:
            print(f"⚠️ Knee detection failed: {e}")
            return None

    def get_knee_info(self) -> Dict:
        """마지막 knee 분석 결과 반환"""
        return self.last_knee_info.copy()

    def visualize_knee_detection(self, save_path: Optional[str] = None) -> str:
        """
        Knee Point Detection 결과를 시각화하고 base64 인코딩된 이미지 반환

        Returns:
            base64로 인코딩된 PNG 이미지 데이터 또는 에러 메시지
        """
        if not self.last_knee_info:
            return "⚠️ No knee detection data available. Please run a query first."

        try:
            scores = self.last_knee_info.get("scores", [])
            knee_point = self.last_knee_info.get("knee_point")
            selected_docs = self.last_knee_info.get("selected_docs", 0)
            total_docs = self.last_knee_info.get("total_docs", len(scores))
            reason = self.last_knee_info.get("reason", "Unknown")
            sensitivity = self.last_knee_info.get("sensitivity", 1.0)

            if not scores:
                return "⚠️ No similarity scores available for visualization."

            # 영어 폰트만 사용 (한글 폰트 오류 방지)
            plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
            plt.rcParams["axes.unicode_minus"] = False

            # 시각화 생성
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle(
                f"Dynamic Knee Point Detection Analysis\nSensitivity: {sensitivity} | Selected: {selected_docs}/{total_docs} docs",
                fontsize=14,
                fontweight="bold",
            )

            # 문서 인덱스 (x축)
            x = list(range(len(scores)))

            # 상위 플롯: 전체 유사도 곡선
            ax1.plot(
                x,
                scores,
                "b-o",
                linewidth=2,
                markersize=5,
                label="Document Similarity Distance",
                alpha=0.7,
            )

            # Knee point 표시
            if knee_point is not None and knee_point < len(scores):
                ax1.axvline(
                    x=knee_point,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Knee Point (idx={knee_point})",
                )
                ax1.plot(
                    knee_point,
                    scores[knee_point],
                    "ro",
                    markersize=10,
                    label=f"Knee: {scores[knee_point]:.4f}",
                )

            # 선택된 문서 영역 표시
            if selected_docs > 0:
                selected_x = x[:selected_docs]
                selected_scores = scores[:selected_docs]
                ax1.fill_between(
                    selected_x,
                    0,
                    selected_scores,
                    alpha=0.3,
                    color="green",
                    label=f"Selected Documents ({selected_docs})",
                )

            ax1.set_xlabel("Document Index (ranked by similarity)")
            ax1.set_ylabel("Similarity Distance (lower = more similar)")
            ax1.set_title(f"Document Similarity Curve\nReason: {reason}")
            ax1.legend(loc="upper right")
            ax1.grid(True, alpha=0.3)

            # 하위 플롯: Knee Detection 세부사항 (확대)
            if knee_point is not None and len(scores) > 3:
                # Knee 주변 데이터 확대 표시
                start_idx = max(0, knee_point - 3)
                end_idx = min(len(scores), knee_point + 4)
                zoom_x = x[start_idx:end_idx]
                zoom_scores = scores[start_idx:end_idx]

                ax2.plot(
                    zoom_x, zoom_scores, "b-o", linewidth=3, markersize=8, alpha=0.8
                )
                ax2.axvline(x=knee_point, color="red", linestyle="--", linewidth=2)
                ax2.plot(knee_point, scores[knee_point], "ro", markersize=12)

                # 데이터 포인트 라벨링
                for i, (xi, yi) in enumerate(zip(zoom_x, zoom_scores)):
                    ax2.annotate(
                        f"{yi:.4f}",
                        (xi, yi),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        fontsize=9,
                    )

                ax2.set_xlabel("Document Index (zoomed around knee)")
                ax2.set_ylabel("Similarity Distance")
                ax2.set_title("Knee Point Detail View")
                ax2.grid(True, alpha=0.3)
            else:
                # Knee point가 없는 경우
                ax2.plot(x, scores, "b-", linewidth=2, alpha=0.5)
                ax2.set_xlabel("Document Index")
                ax2.set_ylabel("Similarity Distance")
                ax2.set_title("No Clear Knee Point Detected")
                ax2.text(
                    0.5,
                    0.5,
                    reason,
                    transform=ax2.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                )
                ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # 이미지를 base64로 인코딩
            buffer = BytesIO()
            plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            buffer.seek(0)

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"Knee detection graph saved to: {save_path}")

            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)  # 메모리 정리

            return f"data:image/png;base64,{image_base64}"

        except Exception as e:
            print(f"Visualization error: {e}")
            return f"Visualization failed: {str(e)}"


# DynamicKneeCompressionRetriever 제거 - pure kneed detection만 사용


# --------- (D) RETRIEVER CREATION ---------
def create_retriever_with_sensitivity(
    vectorstore,
    sensitivity_config: dict,
    min_docs: int = 2,
    max_docs: int = 30,  # 더 많은 범위 검색으로 증가
    direction: str = "decreasing",
    curve: str = "convex",
):
    """
    Sensitivity 기반 Dynamic Knee Point Detection Retriever 생성

    Args:
        vectorstore: FAISS 벡터스토어
        sensitivity_config: sensitivity 설정 (SENSITIVITY_OPTIONS에서 가져온 dict)
        min_docs: 최소 반환 문서 수
        max_docs: 최대 검색 문서 수 (knee 찾기용, 보수적 설정을 위해 증가)
        direction: knee detection 방향 ("decreasing" 또는 "increasing")
        curve: knee detection 곡선 타입 ("convex" 또는 "concave")

    Note:
        - 작은 sensitivity (S=1) = 빠른 knee 감지 = 적은 문서 선택 = 적극적
        - 큰 sensitivity (S=10) = 보수적 감지 = 많은 문서 선택 = 보수적
    """
    sensitivity = sensitivity_config["sensitivity"]
    description = sensitivity_config["description"]

    print(
        f"🎯 Creating DynamicKneeRetriever with sensitivity={sensitivity} ({description})"
    )

    return DynamicKneeRetriever(
        vectorstore=vectorstore,
        min_docs=min_docs,
        max_docs=max_docs,
        sensitivity=sensitivity,
        direction=direction,
        curve=curve,
    )


# --------- (D) QUERY HANDLERS ---------
def handle_query_for_sensitivity(
    user_query: str, sensitivity_key: str, request: gr.Request
) -> Generator:
    """특정 sensitivity 설정으로 쿼리 처리"""
    session_id = get_session_id(request)

    # Ensure session-safe access
    with session_lock:
        if session_id not in sessions:
            init_session(session_id)
        session = sessions[session_id]
        sessions.move_to_end(session_id)

    # 히스토리 가져오기
    history = session["history"][sensitivity_key]
    messages = history.copy()
    client = session["client"]

    # 현재 모델 정보 가져오기
    with shared_state_lock:
        current_model = str(shared_state["current_model"])
        vectorstore = shared_state["vectorstore"]

    model_info = MODELS[current_model]
    sensitivity_config = SENSITIVITY_OPTIONS[sensitivity_key]

    # Extract relevant text data from PDFs with Dynamic Knee Detection
    docs = []
    context_documents = []

    if vectorstore:
        print(f"🔍 [{sensitivity_key}] Retrieving relevant GIST rules...")

        retriever = create_retriever_with_sensitivity(vectorstore, sensitivity_config)
        if retriever:
            # Global state에 retriever 저장 (시각화용)
            with shared_state_lock:
                shared_state["last_retrievers"][sensitivity_key] = retriever
            docs = retriever.invoke(user_query)
            context_documents = docs

        print(f"📊 [{sensitivity_key}] Retrieved {len(docs)} documents")

        # Knee detection 결과 출력
        if hasattr(retriever, "get_knee_info"):
            knee_info = retriever.get_knee_info()
            if knee_info:
                print(
                    f"🎯 [{sensitivity_key}] Knee Detection: {knee_info.get('selected_docs', 0)}/{knee_info.get('total_docs', 0)} docs, reason: {knee_info.get('reason', 'N/A')}"
                )

    # Legal query template을 사용하여 프롬프트 생성
    if context_documents:
        rendered_query = render_legal_query(user_query, context_documents)
    else:
        rendered_query = (
            f"User Query: {user_query}\n\nNo relevant documents found in the database."
        )

    # Legal system prompt와 rendered query 사용
    messages = [
        {"role": "system", "content": LEGAL_SYSTEM_PROMPT},
        {"role": "user", "content": rendered_query},
    ]

    # Add user message to history first
    history.append({"role": "user", "content": user_query})

    # Create initial assistant message placeholder
    history.append({"role": "assistant", "content": ""})

    # Yield initial state with user query
    yield history

    # Invoke client with user query using streaming
    print(f"💬 [{sensitivity_key}] Inquiring LLM with streaming...")

    try:
        # LiteLLM 자동 감지를 사용한 통합 스트리밍
        # 환경변수에 설정된 API 키를 자동으로 사용
        completion = litellm.completion(
            model=model_info["model_id"],
            messages=messages,
            stream=True,
        )

        bot_response = ""
        for chunk in completion:
            if (
                hasattr(chunk, "choices")
                and chunk.choices
                and chunk.choices[0].delta
                and chunk.choices[0].delta.content
            ):
                chunk_content = chunk.choices[0].delta.content
                bot_response += chunk_content

                # Update the last message (assistant's response)
                history[-1]["content"] = html.escape(bot_response)
                yield history

    except Exception as e:
        error_msg = f"오류가 발생했습니다: {str(e) if str(e) else '스트리밍 중 알 수 없는 오류가 발생했습니다.'}"
        print(
            f"❌ [{sensitivity_key}] Streaming error: {e if str(e) else 'Unknown streaming error'}"
        )
        history[-1]["content"] = error_msg
        yield history

    yield history


def handle_multi_query(user_query, request: gr.Request):
    """모든 sensitivity 모드에서 동시에 쿼리 실행"""
    if not user_query.strip():
        return [[] for _ in SENSITIVITY_OPTIONS.keys()]

    print(f"💬 사용자 질문 처리 시작: {user_query[:50]}...")

    # 모든 sensitivity 모드에 대해 제너레이터 생성
    generators = {
        method: handle_query_for_sensitivity(user_query, method, request)
        for method in SENSITIVITY_OPTIONS.keys()
    }

    # 현재 상태 추적
    current_states = {method: [] for method in SENSITIVITY_OPTIONS.keys()}
    active_generators = set(SENSITIVITY_OPTIONS.keys())

    while active_generators:
        updated_methods = set()

        for method in list(active_generators):
            try:
                history = next(generators[method])
                current_states[method] = history
                updated_methods.add(method)
            except StopIteration:
                active_generators.remove(method)
                print(f"✅ {method} completed")

        if updated_methods or len(active_generators) == 0:
            # 결과를 올바른 순서로 정렬하여 반환
            results = []
            for method in SENSITIVITY_OPTIONS.keys():
                history = current_states[method]
                results.append(history)

            yield results

    print("✅ 모든 sensitivity 설정으로 답변 완료!")


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
            embed_model = shared_state["embed_model"]
            if vectorstore and embed_model:
                print("🔄 Merging with existing documents...")
                new_vectorstore = FAISS.from_documents(additional_docs, embed_model)
                vectorstore.merge_from(new_vectorstore)
                shared_state["vectorstore"] = vectorstore

        return f"✅ Added {len(pdfs)} PDFs in {time.time():.2f} seconds"

    except Exception as e:
        return f"❌ 오류 발생: {str(e)}"


# --------- (E) VISUALIZATION FUNCTIONS ---------
def generate_knee_visualization(sensitivity_key: str):
    """특정 sensitivity 설정의 knee detection 결과를 시각화"""
    try:
        with shared_state_lock:
            retrievers = shared_state.get("last_retrievers", {})

        if sensitivity_key not in retrievers:
            return "No data available for this sensitivity setting. Please run a query first."

        retriever = retrievers[sensitivity_key]

        # DynamicKneeRetriever의 시각화 호출
        if hasattr(retriever, "visualize_knee_detection"):
            return retriever.visualize_knee_detection()
        else:
            return f"'{sensitivity_key}' setting does not use knee detection."

    except Exception as e:
        return f"Visualization generation failed: {str(e)}"


def get_all_knee_visualizations():
    """모든 sensitivity 설정의 knee detection 결과를 시각화"""
    results = {}
    try:
        with shared_state_lock:
            retrievers = shared_state.get("last_retrievers", {})

        for method in SENSITIVITY_OPTIONS.keys():
            if method in retrievers:
                results[method] = generate_knee_visualization(method)
            else:
                results[method] = "No data available"

        return results
    except Exception as e:
        return {method: f"Error: {str(e)}" for method in SENSITIVITY_OPTIONS.keys()}


# --------- (F) UTILITY FUNCTIONS ---------
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
            for method in SENSITIVITY_OPTIONS.keys():
                session["history"][method] = []

    return [[] for _ in SENSITIVITY_OPTIONS.keys()]


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


def change_embedding_model(embedding_model_key: str):
    """임베딩 모델 변경 및 해당 데이터베이스 로드"""
    print(f"🔄 임베딩 모델 변경: {embedding_model_key}")

    if embedding_model_key not in EMBEDDING_MODELS:
        status_msg = f"❌ 지원하지 않는 임베딩 모델: {embedding_model_key}"
        return status_msg, get_database_status()

    model_config = EMBEDDING_MODELS[embedding_model_key]

    # 데이터베이스 로드 시도
    if load_database_for_embedding_model(embedding_model_key):
        status_msg = f"✅ {model_config['description']} 로딩 완료"
        db_status = get_database_status()
        return status_msg, db_status
    else:
        status_msg = f"❌ {model_config['description']} 로딩 실패"
        db_status = f"❌ 임베딩 모델 '{model_config['description']}' 데이터베이스를 찾을 수 없습니다.\n🛠️ 다음 명령어로 데이터베이스를 구축하세요:\n   python build_multi_embedding_databases.py --model \"{embedding_model_key}\""
        return status_msg, db_status


# --------- (F) DATABASE STATUS ---------
def get_database_status():
    """데이터베이스 상태 반환"""
    with shared_state_lock:
        if not shared_state["database_loaded"]:
            return "❌ 데이터베이스가 로드되지 않았습니다.\n🛠️ 다음 명령어로 멀티 임베딩 데이터베이스를 구축하세요:\n   `python build_multi_embedding_databases.py`"

        db_info = shared_state["database_info"]
        current_embedding = shared_state.get("current_embedding_model", "Unknown")

        # 현재 임베딩 모델 정보
        embedding_info = EMBEDDING_MODELS.get(current_embedding, {})
        embedding_desc = embedding_info.get("description", current_embedding)
        mteb_rank = embedding_info.get("mteb_rank", "N/A")
        dimension = embedding_info.get("dimension", "N/A")

        status_lines = [
            "✅ **데이터베이스 로드 완료**",
            "",
            "🤖 **현재 임베딩 모델**:",
            f"- 모델: {embedding_desc}",
            f"- MTEB 순위: {mteb_rank}위",
            f"- 차원: {dimension}",
            "",
            "📊 **통계**:",
            f"- 총 문서: {db_info.get('total_documents', 'N/A')}개",
            f"- 총 청크: {db_info.get('total_chunks', 'N/A')}개",
            f"- 생성 일시: {db_info.get('created_at', 'N/A').split('T')[0] if db_info.get('created_at') else 'N/A'}",
        ]

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
.status-box p {
    color: #ffffff !important;
    margin: 6px 0 !important;
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
        "<center><h1>⚖️ GIST Legal Rules Analyzer (Professional Legal Assistant)</h1><p><strong>🎯 전문 법률 분석</strong> | 체계적 법령해석과 Knee Point Detection | <strong>📚 법학적 해석방법론 적용</strong> | <strong>⚡ LiteLLM 통합</strong></p></center>"
    )

    # 데이터베이스 상태 표시
    with gr.Row():
        database_status = gr.Markdown(
            value=get_database_status(), elem_classes=["status-box"]
        )

    # 공통 컨트롤
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                # 동적 모델 목록 생성 및 기본값 설정
                model_choices = list(MODELS.keys())
                default_value = (
                    DEFAULT_MODEL_ID
                    if DEFAULT_MODEL_ID in MODELS
                    else (model_choices[0] if model_choices else "")
                )
                with shared_state_lock:
                    shared_state["current_model"] = default_value

                model_dropdown = gr.Dropdown(
                    model_choices,
                    label="🧠 LLM 모델 선택",
                    value=default_value,
                    scale=2,
                    allow_custom_value=True,
                )
                model_status = gr.Textbox(
                    label="모델 상태",
                    value=f"✅ {DEFAULT_MODEL_ID.split('/')[-1]} 준비완료",
                    interactive=False,
                    scale=1,
                )

            # 임베딩 모델 선택
            with gr.Row():
                embedding_choices = list(EMBEDDING_MODELS.keys())
                embedding_dropdown = gr.Dropdown(
                    embedding_choices,
                    label="📊 임베딩 모델 선택 (MTEB 순위 기준)",
                    value=DEFAULT_EMBEDDING_MODEL,
                    scale=2,
                    info="임베딩 모델 변경 시 해당 데이터베이스가 자동 로드됩니다",
                )
                embedding_status = gr.Textbox(
                    label="임베딩 상태",
                    value=f"✅ {EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL]['description']} 준비완료",
                    interactive=False,
                    scale=1,
                )

            additional_pdf_upload = gr.Files(
                label="📄 추가 문서 업로드 (Vector Store 확장)", file_types=[".pdf"]
            )

            user_input = gr.Textbox(
                label="⚖️ 법률 질의문 입력 (Legal Query Input)",
                placeholder="예: 교수님이 박사과정 학생을 지도할 수 있는 기간은 언제까지인가요? (학칙 제○조 관련)",
                info="🏛️ 법학적 해석방법론 기반 전문 분석 | 📚 체계적 법령 해석 및 조문 인용 | 🎯 Dynamic Knee Detection 문서 선택",
                lines=3,
                interactive=True,
            )

        with gr.Column(scale=1):
            submit_btn = gr.Button("🚀 테스트 실행", variant="primary", size="lg")
            reset_btn = gr.Button("🔄 초기화", size="lg")

    # 탭으로 채팅과 시각화 분리
    with gr.Tabs() as main_tabs:
        with gr.TabItem("⚖️ Legal Analysis Results", id="chat_tab"):
            gr.Markdown("### 🏛️ 전문 법률 분석 및 Sensitivity 비교")
            gr.Markdown(
                "**법학적 해석방법론** 기반 체계적 분석 | **조문 인용 및 법리적 해석** | **작은 S값 = 적극적 문서선택**, **큰 S값 = 보수적 문서선택**"
            )

            # 4개의 sensitivity 설정별 채팅 인터페이스 (2x2 그리드)
            sensitivity_keys = list(SENSITIVITY_OPTIONS.keys())
            with gr.Row(elem_classes=["fill-height"]):
                with gr.Column(scale=1, elem_classes=["fill-height"]):
                    # 첫 번째 sensitivity 설정
                    with gr.Group(elem_classes=["extend-height"]):
                        config1 = SENSITIVITY_OPTIONS[sensitivity_keys[0]]
                        gr.Markdown(f"### 🔹 {sensitivity_keys[0]}")
                        with gr.Row():
                            gr.Dropdown(
                                [
                                    f"Sensitivity: {config1['sensitivity']} | {config1['description']}"
                                ],
                                value=f"Sensitivity: {config1['sensitivity']} | {config1['description']}",
                                interactive=False,
                                scale=3,
                                show_label=False,
                            )
                        chatbot_sens1 = gr.Chatbot(
                            elem_classes=["extend-height"],
                            show_copy_button=True,
                            type="messages",
                        )
                        copy_btn_sens1 = gr.Button("📋 결과 복사", size="sm")

                    # 두 번째 sensitivity 설정
                    with gr.Group(elem_classes=["extend-height"]):
                        config2 = SENSITIVITY_OPTIONS[sensitivity_keys[1]]
                        gr.Markdown(f"### 🔸 {sensitivity_keys[1]}")
                        with gr.Row():
                            gr.Dropdown(
                                [
                                    f"Sensitivity: {config2['sensitivity']} | {config2['description']}"
                                ],
                                value=f"Sensitivity: {config2['sensitivity']} | {config2['description']}",
                                interactive=False,
                                scale=3,
                                show_label=False,
                            )
                        chatbot_sens2 = gr.Chatbot(
                            elem_classes=["extend-height"],
                            show_copy_button=True,
                            type="messages",
                        )
                        copy_btn_sens2 = gr.Button("📋 결과 복사", size="sm")

                with gr.Column(scale=1, elem_classes=["fill-height"]):
                    # 세 번째 sensitivity 설정
                    with gr.Group(elem_classes=["extend-height"]):
                        config3 = SENSITIVITY_OPTIONS[sensitivity_keys[2]]
                        gr.Markdown(f"### 🔶 {sensitivity_keys[2]}")
                        with gr.Row():
                            gr.Dropdown(
                                [
                                    f"Sensitivity: {config3['sensitivity']} | {config3['description']}"
                                ],
                                value=f"Sensitivity: {config3['sensitivity']} | {config3['description']}",
                                interactive=False,
                                scale=3,
                                show_label=False,
                            )
                        chatbot_sens3 = gr.Chatbot(
                            elem_classes=["extend-height"],
                            show_copy_button=True,
                            type="messages",
                        )
                        copy_btn_sens3 = gr.Button("📋 결과 복사", size="sm")

                    # 네 번째 sensitivity 설정
                    with gr.Group(elem_classes=["extend-height"]):
                        config4 = SENSITIVITY_OPTIONS[sensitivity_keys[3]]
                        gr.Markdown(f"### 🔥 {sensitivity_keys[3]}")
                        with gr.Row():
                            gr.Dropdown(
                                [
                                    f"Sensitivity: {config4['sensitivity']} | {config4['description']}"
                                ],
                                value=f"Sensitivity: {config4['sensitivity']} | {config4['description']}",
                                interactive=False,
                                scale=3,
                                show_label=False,
                            )
                        chatbot_sens4 = gr.Chatbot(
                            elem_classes=["extend-height"],
                            show_copy_button=True,
                            type="messages",
                        )
                        copy_btn_sens4 = gr.Button("📋 결과 복사", size="sm")

        with gr.TabItem("📊 Legal Document Analysis Visualization", id="viz_tab"):
            gr.Markdown("### 📈 법률 문서 선택 패턴 시각화")
            gr.Markdown(
                "**법령별 문서 검색 결과 시각화** | **S=1 (적극적) → S=10 (보수적)** 순으로 각 sensitivity의 knee point 감지 패턴과 **법적 근거 문서 선택 과정**을 시각적으로 분석합니다."
            )

            with gr.Row():
                viz_refresh_btn = gr.Button("🔄 그래프 새로고침", variant="secondary")
                viz_save_btn = gr.Button("💾 그래프 저장", variant="secondary")

            # 4개의 sensitivity별 시각화 결과 (2x2 그리드)
            with gr.Row():
                with gr.Column(scale=1):
                    config1 = SENSITIVITY_OPTIONS[sensitivity_keys[0]]
                    gr.Markdown(f"#### 🔹 {sensitivity_keys[0]}")
                    viz_image_sens1 = gr.Image(
                        label=f"Sensitivity: {config1['sensitivity']} | {config1['description']}",
                        show_label=True,
                        interactive=False,
                        height=400,
                    )

                    config2 = SENSITIVITY_OPTIONS[sensitivity_keys[1]]
                    gr.Markdown(f"#### 🔸 {sensitivity_keys[1]}")
                    viz_image_sens2 = gr.Image(
                        label=f"Sensitivity: {config2['sensitivity']} | {config2['description']}",
                        show_label=True,
                        interactive=False,
                        height=400,
                    )

                with gr.Column(scale=1):
                    config3 = SENSITIVITY_OPTIONS[sensitivity_keys[2]]
                    gr.Markdown(f"#### 🔶 {sensitivity_keys[2]}")
                    viz_image_sens3 = gr.Image(
                        label=f"Sensitivity: {config3['sensitivity']} | {config3['description']}",
                        show_label=True,
                        interactive=False,
                        height=400,
                    )

                    config4 = SENSITIVITY_OPTIONS[sensitivity_keys[3]]
                    gr.Markdown(f"#### 🔥 {sensitivity_keys[3]}")
                    viz_image_sens4 = gr.Image(
                        label=f"Sensitivity: {config4['sensitivity']} | {config4['description']}",
                        show_label=True,
                        interactive=False,
                        height=400,
                    )

    # 시각화 관련 함수들
    def refresh_visualizations():
        """모든 sensitivity 설정의 knee detection 결과를 시각화"""
        results = get_all_knee_visualizations()
        sensitivity_keys = list(SENSITIVITY_OPTIONS.keys())
        return (
            results.get(sensitivity_keys[0], "No data available"),
            results.get(sensitivity_keys[1], "No data available"),
            results.get(sensitivity_keys[2], "No data available"),
            results.get(sensitivity_keys[3], "No data available"),
        )

    def save_visualizations():
        """시각화 결과를 파일로 저장"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = Path(f"sensitivity_visualizations_{timestamp}")
            save_dir.mkdir(exist_ok=True)

            # Sensitivity 기반 파일명 매핑
            sensitivity_keys = list(SENSITIVITY_OPTIONS.keys())
            methods_mapping = {}
            for key in sensitivity_keys:
                config = SENSITIVITY_OPTIONS[key]
                clean_name = f"sensitivity_{config['sensitivity']}"
                methods_mapping[key] = clean_name

            saved_files = []
            with shared_state_lock:
                retrievers = shared_state.get("last_retrievers", {})

            for method, retriever in retrievers.items():
                try:
                    save_name = methods_mapping.get(
                        method,
                        method.replace(" ", "_").replace("(", "").replace(")", ""),
                    )
                    save_path = save_dir / f"{save_name}.png"

                    if hasattr(retriever, "visualize_knee_detection"):
                        retriever.visualize_knee_detection(str(save_path))
                        saved_files.append(str(save_path))
                except Exception as e:
                    print(f"Save failed - {method}: {e}")

            if saved_files:
                return f"{len(saved_files)} graphs have been saved:\n" + "\n".join(
                    saved_files
                )
            else:
                return "No graphs available to save. Please run a query first."
        except Exception as e:
            return f"Save failed: {str(e)}"

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

    # 임베딩 모델 변경 이벤트
    embedding_dropdown.change(
        fn=change_embedding_model,
        inputs=[embedding_dropdown],
        outputs=[embedding_status, database_status],
    )

    # 추가 PDF 업로드
    additional_pdf_upload.upload(
        fn=handle_additional_pdf_upload,
        inputs=[additional_pdf_upload],
        outputs=[database_status],
    )

    # 멀티 쿼리 처리 (Sensitivity 기반)
    submit_btn.click(
        fn=init_client_on_first_query,
        inputs=[user_input],
        outputs=[
            chatbot_sens1,
            chatbot_sens2,
            chatbot_sens3,
            chatbot_sens4,
        ],
    )

    user_input.submit(
        fn=init_client_on_first_query,
        inputs=[user_input],
        outputs=[
            chatbot_sens1,
            chatbot_sens2,
            chatbot_sens3,
            chatbot_sens4,
        ],
    )

    # 초기화
    reset_btn.click(
        fn=reset_all_chats,
        outputs=[
            chatbot_sens1,
            chatbot_sens2,
            chatbot_sens3,
            chatbot_sens4,
        ],
    )

    # 시각화 이벤트 연결
    viz_refresh_btn.click(
        fn=refresh_visualizations,
        inputs=[],
        outputs=[viz_image_sens1, viz_image_sens2, viz_image_sens3, viz_image_sens4],
    )

    viz_save_btn.click(
        fn=save_visualizations,
        inputs=[],
        outputs=[gr.Textbox(visible=False)],  # 결과를 콘솔에만 표시
    )

    # 복사 기능 (Sensitivity 기반)
    sensitivity_keys = list(SENSITIVITY_OPTIONS.keys())

    copy_btn_sens1.click(
        fn=lambda h: copy_as_markdown(h, sensitivity_keys[0]),
        inputs=[chatbot_sens1],
        outputs=[gr.Textbox(visible=False)],
        js="(result) => navigator.clipboard.writeText(result)",
    )

    copy_btn_sens2.click(
        fn=lambda h: copy_as_markdown(h, sensitivity_keys[1]),
        inputs=[chatbot_sens2],
        outputs=[gr.Textbox(visible=False)],
        js="(result) => navigator.clipboard.writeText(result)",
    )

    copy_btn_sens3.click(
        fn=lambda h: copy_as_markdown(h, sensitivity_keys[2]),
        inputs=[chatbot_sens3],
        outputs=[gr.Textbox(visible=False)],
        js="(result) => navigator.clipboard.writeText(result)",
    )

    copy_btn_sens4.click(
        fn=lambda h: copy_as_markdown(h, sensitivity_keys[3]),
        inputs=[chatbot_sens4],
        outputs=[gr.Textbox(visible=False)],
        js="(result) => navigator.clipboard.writeText(result)",
    )

if __name__ == "__main__":
    print("⚖️ GIST Legal Rules Analyzer (Professional Legal Assistant) 준비완료!")
    print("🏛️ 법학적 해석방법론 기반 전문 법률 분석 시스템")
    print("📚 체계적 법령 해석: 문리해석 → 체계적해석 → 목적론적해석 → 우선순위적용")
    print("📋 Legal Template System: 전문 시스템프롬프트 + 쿼리템플릿 적용")
    print("🎯 S=1,3,5,10 sensitivity로 법적 근거 문서 선택 최적화")
    print("⚡ LiteLLM 통합으로 다양한 LLM 프로바이더 지원")
    print("🌐 http://localhost:7860 에서 실행 중...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

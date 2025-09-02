"""
LiberVance RAG (Retrieval-Augmented Generation) 시스템

이 모듈은 PDF 문서를 기반으로 하는 지능형 질의응답 시스템입니다.
사용자가 업로드한 PDF 문서들을 벡터 데이터베이스로 색인화하고,
질문에 가장 관련성이 높은 문서 내용을 검색하여 컨텍스트로 활용해
정확하고 상세한 답변을 생성합니다.

주요 기능:
- PDF 문서 자동 텍스트 추출 및 청킹
- FAISS 벡터 스토어를 통한 의미론적 검색
- 다양한 LLM 모델 지원 (GPT-4, DeepSeek-R1, Gemma-3, LLaMA-3.3, QwQ-32B)
- 실시간 모델 전환 기능
- 세션별 독립적인 문서 관리
- 대화 기록 저장 및 관리

기술 스택:
- 임베딩: sentence-transformers/all-MiniLM-L6-v2
- 벡터 스토어: FAISS
- 텍스트 분할: RecursiveCharacterTextSplitter
- PDF 처리: PyMuPDF (fitz)
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
import numpy as np
from numpy.typing import NDArray
from collections import OrderedDict
from datetime import datetime
from dotenv import load_dotenv
from litellm import get_valid_models
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

try:
    from kneed import KneeLocator
except Exception:
    KneeLocator = None

# Environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NOVITA_API_KEY = os.getenv("NOVITA_API_KEY")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_AI_API_KEY") or os.getenv("FIREWORKS_API_KEY")

# Session variables
MAX_SESSIONS = 100
sessions = OrderedDict()
session_lock = threading.Lock()


# Model setup (dynamic using LiteLLM get_valid_models)
def _detect_provider(model_id: str) -> str:
    if model_id.startswith("novita/"):
        return "novita"
    if model_id.startswith("fireworks_ai/"):
        return "fireworks"
    return "openai"


def _load_dynamic_models() -> dict:
    # Map FIREWORKS_API_KEY -> FIREWORKS_AI_API_KEY if only legacy var is set
    if os.getenv("FIREWORKS_API_KEY") and not os.getenv("FIREWORKS_AI_API_KEY"):
        os.environ["FIREWORKS_AI_API_KEY"] = os.getenv("FIREWORKS_API_KEY") or ""

    try:
        model_ids = get_valid_models(check_provider_endpoint=True)
    except Exception:
        model_ids = []

    dynamic: dict = {}
    for mid in model_ids:
        dynamic[mid] = {"model_id": mid, "provider": _detect_provider(mid)}

    # Ensure Fireworks default appears when key present
    default_fw = "fireworks_ai/accounts/fireworks/models/gpt-oss-20b"
    if (
        os.getenv("FIREWORKS_AI_API_KEY") or os.getenv("FIREWORKS_API_KEY")
    ) and default_fw not in dynamic:
        dynamic[default_fw] = {"model_id": default_fw, "provider": "fireworks"}

    # Ensure at least one OpenAI option when OPENAI_API_KEY is set
    if OPENAI_API_KEY and not any(
        p.get("provider") == "openai" for p in dynamic.values()
    ):
        dynamic.setdefault(
            "gpt-4o-mini", {"model_id": "gpt-4o-mini", "provider": "openai"}
        )

    return dynamic


MODELS = _load_dynamic_models()
DEFAULT_MODEL = (
    "fireworks_ai/accounts/fireworks/models/gpt-oss-20b"
    if "fireworks_ai/accounts/fireworks/models/gpt-oss-20b" in MODELS
    else (list(MODELS.keys())[0] if MODELS else "")
)

# Retrieval 모드 설정
RETRIEVAL_MODES = {
    "적응형 (Knee Detection)": {"use_knee": True, "default_k": 30},
    "고정 Top-K": {"use_knee": False, "default_k": 30},
}

# Rerank 설정
RERANK_OPTIONS = {
    "없음": {"enabled": False, "model": None, "top_k": 30},
    "Cross-Encoder (기본)": {
        "enabled": True,
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "top_k": 30,
    },
    "Cross-Encoder (고성능)": {
        "enabled": True,
        "model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "top_k": 30,
    },
    "다국어 Cross-Encoder": {
        "enabled": True,
        "model": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "top_k": 30,
    },
}

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_MODEL = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_ID,
    encode_kwargs={"normalize_embeddings": True},
)
DIMENSION = 384
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

system_prompt = "You are a helpful assistant that can answer questions based on the context when provided."


def _compute_knee_cutoff(sorted_scores, min_k: int = 1, default_k: int = 30) -> int:
    """내림차순 정렬된 점수 시퀀스에서 무릎점을 찾아 반환 개수를 결정.

    - sorted_scores: 높은 점수 → 낮은 점수 순으로 정렬된 리스트
    - min_k: 최소 반환 개수 보장
    - default_k: 무릎점을 찾지 못했을 때 기본 반환 개수
    """
    try:
        n = len(sorted_scores)
        if n <= 0:
            return 0
        if n <= min_k:
            return n

        if KneeLocator is not None:
            try:
                x = list(range(1, n + 1))
                kl = KneeLocator(
                    x, sorted_scores, curve="convex", direction="decreasing"
                )
                if kl.knee is not None:
                    return max(min_k, min(int(kl.knee), n))
            except Exception:
                pass

        # 단순 휴리스틱: 이전 대비 상대 하락폭이 임계치 이상인 첫 지점에서 절단
        threshold = 0.15
        for i in range(1, n):
            prev_score = sorted_scores[i - 1]
            curr_score = sorted_scores[i]
            if prev_score == 0:
                continue
            drop_ratio = (prev_score - curr_score) / (abs(prev_score) + 1e-9)
            if drop_ratio >= threshold:
                return max(min_k, i)

        return max(min_k, min(default_k, n))
    except Exception:
        return max(min_k, default_k)


def _get_total_docs_in_vectorstore(vectorstore) -> int:
    """Vectorstore 내 총 문서 수 추정 (FAISS 우선, 그 외 안전한 폴백).

    - FAISS: index.ntotal 사용
    - 기타: index_to_docstore_id 길이, docstore._dict 길이 순서로 시도
    """
    try:
        index = getattr(vectorstore, "index", None)
        if index is not None:
            ntotal = getattr(index, "ntotal", None)
            if isinstance(ntotal, (int, float)) and int(ntotal) > 0:
                return int(ntotal)
    except Exception:
        pass
    try:
        ids = getattr(vectorstore, "index_to_docstore_id", None)
        if ids is not None:
            return len(ids)
    except Exception:
        pass
    try:
        docstore = getattr(vectorstore, "docstore", None)
        if docstore is not None:
            d = getattr(docstore, "_dict", None)
            if d is not None:
                return len(d)
    except Exception:
        pass
    return 0


# --------- (A) SESSION SETUP ---------
def get_session_id(request: gr.Request):
    raw_id = request.client.host + str(request.headers.get("user-agent"))
    return hashlib.sha256(raw_id.encode()).hexdigest()


def init_session(
    session_id: str,
    model_name=DEFAULT_MODEL,
    rerank_method="없음",
    retrieval_mode="적응형 (Knee Detection)",
):
    if len(sessions) >= MAX_SESSIONS:
        evicted_id, _ = sessions.popitem(last=False)
        print(f"🧹 Removed LRU session: {evicted_id[:8]}...")
    model_info = MODELS[model_name]
    sessions[session_id] = {
        "history": [{"role": "system", "content": system_prompt}],
        "vectorstore": None,
        "retriever": None,
        "pdfs": [],
        "model_id": model_info["model_id"],
        "client": create_client(model_info["provider"]),
        "rerank_method": rerank_method,
        "retrieval_mode": retrieval_mode,
    }


# --------- (B) DOCUMENT TEXT EXTRACTION ---------
def extract_text_from_pdf(pdf):
    doc = fitz.open(pdf)
    return "\n".join([page.get_text("text") for page in doc])


def create_vectorstore_from_pdfs(pdfs):
    all_docs = []
    for pdf in pdfs:
        text = extract_text_from_pdf(pdf)
        doc = Document(page_content=text, metadata={"source": pdf})
        docs = TEXT_SPLITTER.split_documents([doc])
        all_docs.extend(docs)
    vs = FAISS.from_documents(all_docs, EMBED_MODEL)
    # 메타: 벡터스토어에 현재 임베딩 모델 식별자 기록
    try:
        setattr(vs, "_embedding_model_id", EMBED_MODEL_ID)
    except Exception:
        pass
    return vs


def create_retriever(
    vectorstore, rerank_method="없음", retrieval_mode="적응형 (Knee Detection)"
):
    """벡터스토어에서 retriever 생성 - rerank 옵션 지원"""
    if not vectorstore:
        return None

    rerank_config = RERANK_OPTIONS.get(rerank_method, RERANK_OPTIONS["없음"])
    retrieval_config = RETRIEVAL_MODES.get(
        retrieval_mode, RETRIEVAL_MODES["적응형 (Knee Detection)"]
    )
    use_knee = retrieval_config["use_knee"]
    default_k = retrieval_config["default_k"]

    if not rerank_config["enabled"]:
        if use_knee:
            # 적응형 검색: kneedle 기반 적응형 선택
            print("🔍 Using basic vector retrieval (no rerank) with kneedle cutoff")

            class AdaptiveKneeRetriever:
                def __init__(self, vectorstore, default_k: int = 30):
                    self.vectorstore = vectorstore
                    self.default_k = default_k

                def invoke(self, query, **kwargs):
                    try:
                        total = _get_total_docs_in_vectorstore(self.vectorstore)
                        k_all = max(total, 1000) if total <= 0 else total
                        results = self.vectorstore.similarity_search_with_score(
                            query, k=k_all
                        )
                    except Exception:
                        # 일부 백엔드가 _with_score를 지원하지 않을 경우의 폴백
                        total = _get_total_docs_in_vectorstore(self.vectorstore)
                        k_all = max(total, 1000) if total <= 0 else total
                        docs = self.vectorstore.similarity_search(query, k=k_all)
                        # 점수 없으므로 기본값으로 자르고 rank 메타데이터 부여
                        for idx, d in enumerate(docs[: self.default_k]):
                            d.metadata = dict(
                                d.metadata or {}, rank=idx + 1, selected=True
                            )
                        return docs[: self.default_k]

                    # results: List[Tuple[Document, distance]] (작을수록 유사)
                    docs = [doc for doc, _ in results]
                    distances: NDArray[np.float64] = np.asarray(
                        [float(distance) for _, distance in results], dtype=np.float64
                    )
                    # 거리 → 유사도 변환 (높을수록 유사)
                    scores: NDArray[np.float64] = np.divide(1.0, distances + 1.0)
                    # 점수 기준 내림차순 인덱스
                    order: NDArray[np.int64] = np.argsort(-scores)
                    scores_sorted: NDArray[np.float64] = scores[order]
                    cutoff_k = _compute_knee_cutoff(
                        scores_sorted.tolist(), min_k=1, default_k=self.default_k
                    )
                    selected_idx: NDArray[np.int64] = order[:cutoff_k]
                    # 선택된 문서에 rank/score/index 메타데이터 부여
                    selected_docs = []
                    for rank, idx in enumerate(selected_idx, start=1):
                        d = docs[int(idx)]
                        d.metadata = dict(
                            d.metadata or {},
                            rank=rank,
                            score=float(scores[int(idx)]),
                            original_index=int(idx),
                            selected=True,
                        )
                        selected_docs.append(d)
                    return selected_docs

            return AdaptiveKneeRetriever(vectorstore, default_k=default_k)
        else:
            # 고정 Top-K 검색
            print(f"🔢 Using fixed top-{default_k} retrieval (no rerank)")
            return vectorstore.as_retriever(search_kwargs={"k": default_k})

    try:
        # Rerank 기능 활성화
        print(f"🔄 Setting up reranker: {rerank_method}")
        print(f"   Model: {rerank_config['model']}")

        # 더 많은 문서를 초기 검색 (rerank를 위해)
        top_k = rerank_config.get("top_k", 30)
        if not isinstance(top_k, int) or top_k <= 0:
            top_k = 30
        # 전체 문서 대상으로 1차 검색
        total = _get_total_docs_in_vectorstore(vectorstore)
        initial_k = max(total, 1000) if total <= 0 else total
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": initial_k})

        # Cross-encoder 모델 설정
        cross_encoder = HuggingFaceCrossEncoder(model_name=rerank_config["model"])
        compressor = CrossEncoderReranker(model=cross_encoder)

        # ContextualCompressionRetriever로 래핑
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )

        if use_knee:
            # Kneedle 기반 적응형 제한을 위한 래퍼 클래스
            class AdaptiveKneeLimitedRetriever:
                def __init__(self, retriever, cross_encoder, default_k):
                    self.retriever = retriever
                    self.cross_encoder = cross_encoder
                    self.default_k = default_k

                def _predict_scores(self, query, docs):
                    pairs = [(query, d.page_content) for d in docs]

                    def _to_score_list(result, num_docs: int):
                        try:
                            arr = np.asarray(result, dtype=float)
                            if arr.ndim == 0:
                                arr = np.full(num_docs, float(arr))
                            else:
                                arr = arr.reshape(-1)
                            if arr.size < num_docs:
                                pad_value = float(arr[-1]) if arr.size > 0 else 1.0
                                arr = np.pad(
                                    arr,
                                    (0, num_docs - arr.size),
                                    constant_values=pad_value,
                                )
                            elif arr.size > num_docs:
                                arr = arr[:num_docs]
                            return arr.tolist()
                        except Exception:
                            return [1.0 for _ in range(num_docs)]

                    # 다양한 구현 차이를 고려한 안전한 예측 호출
                    for attr in ("score", "predict"):
                        method = getattr(self.cross_encoder, attr, None)
                        if callable(method):
                            try:
                                return _to_score_list(method(pairs), len(docs))
                            except Exception:
                                pass
                    for attr in ("model", "_model"):
                        underlying = getattr(self.cross_encoder, attr, None)
                        if underlying is not None:
                            method = getattr(underlying, "predict", None)
                            if callable(method):
                                try:
                                    return _to_score_list(method(pairs), len(docs))
                                except Exception:
                                    pass
                    # 점수 산출 불가 시 균등 점수 반환 → 기본 k로 절단됨
                    return [1.0 for _ in docs]

                def invoke(self, query, **kwargs):
                    docs = self.retriever.invoke(query, **kwargs)
                    if not docs:
                        return []
                    scores: NDArray[np.float64] = np.asarray(
                        self._predict_scores(query, docs), dtype=np.float64
                    )
                    order: NDArray[np.int64] = np.argsort(-scores)
                    scores_sorted: NDArray[np.float64] = scores[order]
                    cutoff_k = _compute_knee_cutoff(
                        scores_sorted.tolist(), min_k=1, default_k=self.default_k
                    )
                    selected_idx: NDArray[np.int64] = order[:cutoff_k]
                    selected_docs = []
                    for rank, idx in enumerate(selected_idx, start=1):
                        d = docs[int(idx)]
                        d.metadata = dict(
                            d.metadata or {},
                            rank=rank,
                            score=float(scores[int(idx)]),
                            original_index=int(idx),
                            selected=True,
                        )
                        selected_docs.append(d)
                    return selected_docs

            adaptive_retriever = AdaptiveKneeLimitedRetriever(
                compression_retriever, cross_encoder, top_k
            )
        else:
            # 고정 Top-K로 제한된 Compression Retriever
            class TopKLimitedRetriever:
                def __init__(self, retriever, default_k):
                    self.retriever = retriever
                    self.default_k = default_k

                def invoke(self, query, **kwargs):
                    docs = self.retriever.invoke(query, **kwargs)
                    return (
                        docs[: self.default_k] if len(docs) > self.default_k else docs
                    )

            adaptive_retriever = TopKLimitedRetriever(compression_retriever, top_k)

        print(f"✅ Reranker setup complete - adaptive cutoff with default {top_k}")
        return adaptive_retriever

    except Exception as e:
        print(f"❌ Reranker setup failed: {e}")
        if use_knee:
            print("   Falling back to basic vector retrieval (adaptive kneedle)")

            class AdaptiveKneeRetriever:
                def __init__(self, vectorstore, default_k: int = 30):
                    self.vectorstore = vectorstore
                    self.default_k = default_k

                def invoke(self, query, **kwargs):
                    try:
                        total = _get_total_docs_in_vectorstore(self.vectorstore)
                        k_all = max(total, 1000) if total <= 0 else total
                        results = self.vectorstore.similarity_search_with_score(
                            query, k=k_all
                        )
                    except Exception:
                        total = _get_total_docs_in_vectorstore(self.vectorstore)
                        k_all = max(total, 1000) if total <= 0 else total
                        docs = self.vectorstore.similarity_search(query, k=k_all)
                        return docs[: self.default_k]

                    docs = []
                    scores = []
                    for doc, distance in results:
                        docs.append(doc)
                        sim = 1.0 / (1.0 + float(distance))
                        scores.append(sim)

                    cutoff_k = _compute_knee_cutoff(
                        scores, min_k=1, default_k=self.default_k
                    )
                    return docs[:cutoff_k]

            return AdaptiveKneeRetriever(vectorstore, default_k=default_k)
        else:
            print(f"   Falling back to basic vector retrieval (fixed top-{default_k})")
            return vectorstore.as_retriever(search_kwargs={"k": default_k})


def handle_pdf_upload(pdfs, request: gr.Request):
    start_time = time.time()  # ⏱️ TIMER
    session_id = get_session_id(request)
    # Ensure session-safe access
    with session_lock:
        if session_id not in sessions:
            init_session(session_id)
            print(
                f"✅ New session created: {session_id[:8]}... | Total sessions: {len(sessions)}"
            )
        # Generate vectorstore from PDFs
        print("Processing PDF(s)...")
        vectorstore = create_vectorstore_from_pdfs(pdfs) if pdfs else None
        sessions[session_id]["vectorstore"] = vectorstore

        # Create retriever with current rerank method and retrieval mode
        rerank_method = sessions[session_id].get("rerank_method", "없음")
        retrieval_mode = sessions[session_id].get(
            "retrieval_mode", "적응형 (Knee Detection)"
        )
        sessions[session_id]["retriever"] = create_retriever(
            vectorstore, rerank_method, retrieval_mode
        )
        sessions[session_id]["pdfs"] = pdfs
        sessions.move_to_end(session_id)
    end_time = time.time()  # ⏱️ TIMER
    elapsed_time = end_time - start_time  # ⏱️ TIMER
    print(f"Processed {len(pdfs)} PDFs in {elapsed_time:.2f} seconds")  # ⏱️ TIMER


# --------- (C) PRIMARY CHAT FUNCTION ---------
def handle_query(user_query, request: gr.Request):
    session_id = get_session_id(request)
    # Ensure session-safe access
    with session_lock:
        if session_id not in sessions:
            init_session(session_id)
            print(
                f"✅ New session created: {session_id[:8]}... | Total sessions: {len(sessions)}"
            )
        session = sessions[session_id]
        sessions.move_to_end(session_id)

    history = session["history"]
    messages = history.copy()
    # client = session["client"]  # not used
    model_info = None
    for name, info in MODELS.items():
        if info["model_id"] == session["model_id"]:
            model_info = info
            break

    start_time = time.time()  # ⏱️ TIMER

    # Extract relevant text data from PDFs
    context = ""
    retriever = session["retriever"]
    # 벡터스토어-임베딩 모델 불일치 감지 시 재구축
    vectorstore = session.get("vectorstore")
    if vectorstore is not None:
        vs_model_id = getattr(vectorstore, "_embedding_model_id", None)
        if vs_model_id != EMBED_MODEL_ID:
            print("⚠️ Rebuilding vectorstore due to embedding model change...")
            pdfs = session.get("pdfs", [])
            vectorstore = create_vectorstore_from_pdfs(pdfs) if pdfs else None
            session["vectorstore"] = vectorstore
            retriever = (
                create_retriever(
                    vectorstore,
                    session.get("rerank_method", "없음"),
                    session.get("retrieval_mode", "적응형 (Knee Detection)"),
                )
                if vectorstore
                else None
            )
            session["retriever"] = retriever
    if retriever:
        rerank_method = session.get("rerank_method", "없음")
        print(f"Retrieving relevant data using: {rerank_method}")

        retrieval_start = time.time()
        docs = retriever.invoke(user_query)
        retrieval_end = time.time()

        context = "\n".join([doc.page_content for doc in docs])
        print(
            f"📊 Retrieved {len(docs)} documents in {retrieval_end - retrieval_start:.2f}s"
        )
    messages.append(
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
    )

    # Add user message to history first
    history.append({"role": "user", "content": user_query})

    # Create initial assistant message placeholder
    history.append({"role": "assistant", "content": ""})

    # Yield initial state with user query
    yield history

    # Invoke client with user query using streaming
    print("Inquiring LLM with streaming...")

    # Resolve LiteLLM endpoint & credentials
    api_key = OPENAI_API_KEY
    base_url = None
    if model_info and model_info["provider"] == "novita":
        api_key = NOVITA_API_KEY
        base_url = "https://api.novita.ai/v3/openai"
    elif model_info and model_info["provider"] == "fireworks":
        api_key = FIREWORKS_API_KEY
        base_url = "https://api.fireworks.ai/inference/v1"

    try:
        if model_info and model_info["provider"] in ("openai", "novita", "fireworks"):
            # LiteLLM 스트리밍
            completion_fn = getattr(litellm, "completion")
            completion = completion_fn(
                model=session["model_id"],
                messages=messages,
                stream=True,
                api_key=api_key,
                base_url=base_url,
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
                    history[-1]["content"] = html.escape(bot_response)
                    yield history

        else:
            # 기타 프로바이더: 기본 LiteLLM 스트리밍 호출 (base_url 없음)
            completion_fn = getattr(litellm, "completion")
            completion = completion_fn(
                model=session["model_id"], messages=messages, stream=True
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
                    history[-1]["content"] = html.escape(bot_response)
                    yield history

    except Exception as e:
        print(f"❌ Streaming error: {e}")
        print("Falling back to non-streaming mode...")

        # Fallback to non-streaming via LiteLLM
        try:
            completion_fn = getattr(litellm, "completion")
            completion = completion_fn(
                model=session["model_id"],
                messages=messages,
                api_key=api_key,
                base_url=base_url,
            )
            if (
                hasattr(completion, "choices")
                and completion.choices
                and hasattr(completion.choices[0], "message")
            ):
                bot_response = completion.choices[0].message.content
            else:
                bot_response = str(getattr(completion, "output_text", "")) or str(
                    completion
                )
            history[-1]["content"] = html.escape(bot_response)
            yield history
        except Exception as fallback_error:
            print(f"❌ Fallback error: {fallback_error}")
            error_message = "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다."
            history[-1]["content"] = error_message
            yield history
            return

    # Save final history
    save_history(history, session_id)

    end_time = time.time()  # ⏱️ TIMER
    elapsed_time = end_time - start_time  # ⏱️ TIMER
    print(f"Responded to user query in {elapsed_time:.2f} seconds")  # ⏱️ TIMER


# --------- (D) ADDITIONAL FUNCTIONS ---------
def create_client(provider):
    """모델 제공자에 따라 클라이언트/설정 생성"""
    if provider in ("openai", "novita", "fireworks"):
        if provider == "novita":
            return {
                "type": "litellm",
                "api_key": NOVITA_API_KEY,
                "base_url": "https://api.novita.ai/v3/openai",
            }
        if provider == "fireworks":
            return {
                "type": "litellm",
                "api_key": FIREWORKS_API_KEY,
                "base_url": "https://api.fireworks.ai/inference/v1",
            }
        return {"type": "litellm", "api_key": OPENAI_API_KEY, "base_url": None}

    raise ValueError(f"Unsupported provider: {provider}")


def change_model(model_name, request: gr.Request):
    """사용자 선택에 따라 모델 변경"""
    session_id = get_session_id(request)
    with session_lock:
        if session_id not in sessions:
            init_session(session_id)
            print(
                f"✅ New session created: {session_id[:8]}... | Total sessions: {len(sessions)}"
            )
        model_info = MODELS[model_name]
        sessions[session_id]["model_id"] = model_info["model_id"]
        sessions[session_id]["client"] = create_client(model_info["provider"])
        sessions.move_to_end(session_id)
    # print(f"🔄 Now using: {model_name}")


def change_rerank_method(rerank_method, request: gr.Request):
    """사용자 선택에 따라 rerank 방법 변경"""
    session_id = get_session_id(request)
    with session_lock:
        if session_id not in sessions:
            init_session(session_id, rerank_method=rerank_method)
            print(
                f"✅ New session created: {session_id[:8]}... | Total sessions: {len(sessions)}"
            )
        else:
            # 기존 세션에서 rerank 방법 변경
            sessions[session_id]["rerank_method"] = rerank_method

            # 벡터스토어가 있으면 retriever 재생성
            vectorstore = sessions[session_id]["vectorstore"]
            if vectorstore:
                retrieval_mode = sessions[session_id].get(
                    "retrieval_mode", "적응형 (Knee Detection)"
                )
                print(f"🔄 Updating retriever with: {rerank_method}")
                sessions[session_id]["retriever"] = create_retriever(
                    vectorstore, rerank_method, retrieval_mode
                )

            sessions.move_to_end(session_id)


def change_retrieval_mode(retrieval_mode, request: gr.Request):
    """사용자 선택에 따라 retrieval 모드 변경"""
    session_id = get_session_id(request)
    with session_lock:
        if session_id not in sessions:
            init_session(session_id, retrieval_mode=retrieval_mode)
            print(
                f"✅ New session created: {session_id[:8]}... | Total sessions: {len(sessions)}"
            )
        else:
            # 기존 세션에서 retrieval 모드 변경
            sessions[session_id]["retrieval_mode"] = retrieval_mode

            # 벡터스토어가 있으면 retriever 재생성
            vectorstore = sessions[session_id]["vectorstore"]
            if vectorstore:
                rerank_method = sessions[session_id].get("rerank_method", "없음")
                print(f"🔄 Updating retriever with: {retrieval_mode}")
                sessions[session_id]["retriever"] = create_retriever(
                    vectorstore, rerank_method, retrieval_mode
                )

            sessions.move_to_end(session_id)


def save_history(history, session_id):
    """대화 기록(history)을 JSON 파일로 저장"""
    folder = "./chat_logs_lvrag"
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d")
    filename = os.path.join(folder, f"{timestamp}_{session_id}.json")
    counter = 1
    while os.path.exists(filename):
        filename = os.path.join(folder, f"{timestamp}_{session_id}_{counter}.json")
        counter += 1
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def reset_session(request: gr.Request):
    """대화 및 파일 업로드 내역 삭제"""
    session_id = get_session_id(request)
    with session_lock:
        current_rerank_method = "없음"
        current_retrieval_mode = "적응형 (Knee Detection)"
        if session_id in sessions:
            current_rerank_method = sessions[session_id].get("rerank_method", "없음")
            current_retrieval_mode = sessions[session_id].get(
                "retrieval_mode", "적응형 (Knee Detection)"
            )

        init_session(
            session_id,
            rerank_method=current_rerank_method,
            retrieval_mode=current_retrieval_mode,
        )
        sessions.move_to_end(session_id)
        print(f"♻️ Session {session_id[:8]}... reset.")
    return "", []


# --------- (E) Gradio UI ---------
css = """
div {
    flex-wrap: nowrap !important;
}
.responsive-height {
    height: 768px !important;
    padding-bottom: 64px !important;
}
.fill-height {
    height: 100% !important;
    flex-wrap: nowrap !important;
}
.extend-height {
    min-height: 260px !important;
    flex: 1 !important;
    overflow: auto !important;
}
footer {
    display: none !important;
}
"""

with gr.Blocks(title="LiberVance RAG", css=css, fill_height=True) as demo:
    gr.Markdown("<center><h1>📄 LiberVance RAG</h1></center>")
    with gr.Row(elem_classes=["responsive-height"]):
        # Output column
        with gr.Column(elem_classes=["fill-height"]):
            chatbot = gr.Chatbot(
                label="Chatbot", type="messages", elem_classes=["extend-height"]
            )
        # Input column
        with gr.Column(elem_classes=["fill-height"]):
            model_dropdown = gr.Dropdown(
                list(MODELS.keys()),
                label="Select Model",
                value=DEFAULT_MODEL,
                allow_custom_value=True,
            )
            rerank_dropdown = gr.Dropdown(
                list(RERANK_OPTIONS.keys()),
                label="Rerank Method",
                value="없음",
                info="Cross-Encoder 모델을 사용해 검색 결과를 재정렬합니다",
            )
            retrieval_mode_dropdown = gr.Dropdown(
                list(RETRIEVAL_MODES.keys()),
                label="Retrieval Mode",
                value="적응형 (Knee Detection)",
                info="문서 선택 방식: 적응형(동적 개수) vs 고정 Top-K",
            )
            pdf_upload = gr.Files(
                label="Upload file(s) (PDF only)",
                file_types=[".pdf"],
                elem_classes=["extend-height"],
            )
            user_input = gr.Textbox(
                label="Enter your query here",
                placeholder="e.g., Summarize the key points from this document.",
                lines=3,
            )
            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary")
                reset_btn = gr.Button("Reset", variant="secondary")
    # Event listeners
    pdf_upload.change(fn=handle_pdf_upload, inputs=[pdf_upload], outputs=[])
    model_dropdown.input(fn=change_model, inputs=[model_dropdown], outputs=[])
    rerank_dropdown.input(fn=change_rerank_method, inputs=[rerank_dropdown], outputs=[])
    retrieval_mode_dropdown.input(
        fn=change_retrieval_mode, inputs=[retrieval_mode_dropdown], outputs=[]
    )
    user_input.submit(handle_query, inputs=[user_input], outputs=[chatbot])
    submit_btn.click(handle_query, inputs=[user_input], outputs=[chatbot])
    reset_btn.click(reset_session, inputs=[], outputs=[user_input, chatbot])


def main():
    """메인 실행 함수 - CLI entry point"""
    demo.launch(share=True, favicon_path="")


def main_dev():
    """개발용 실행 함수 - 로컬 서버만"""
    demo.launch(share=False, server_name="localhost", server_port=7860)


def main_prod():
    """프로덕션 실행 함수 - 외부 접근 가능"""
    import os

    port = int(os.getenv("PORT", 7860))
    demo.launch(share=False, server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    main()

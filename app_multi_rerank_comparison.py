"""
LiberVance RAG (Retrieval-Augmented Generation) 시스템 - Enhanced Version

이 모듈은 PDF 문서를 기반으로 하는 지능형 질의응답 시스템의 향상된 버전입니다.
기존 기능에 추가로 다음 기능들이 포함되어 있습니다:
- 성능 지표 모니터링 (첫 토큰 시간, 완료 시간, TPS)
- 모든 rerank 모드 동시 비교 테스트
- 모델 선택 동기화
- Markdown 형태 응답 복사 기능

주요 기능:
- PDF 문서 자동 텍스트 추출 및 청킹
- FAISS 벡터 스토어를 통한 의미론적 검색
- 다양한 LLM 모델 지원 (GPT-4, DeepSeek-R1, Gemma-3, LLaMA-3.3, QwQ-32B)
- 실시간 스트리밍 응답
- 다중 rerank 모드 병렬 테스트
- 성능 지표 실시간 모니터링
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
import openai
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
from typing import Dict, Generator, Tuple, List, Optional

# Environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_API_KEY = os.getenv("HUGGING_FACE_API_KEY") or os.getenv("HF_TOKEN")
HF_ENTERPRISE = os.getenv("HUGGING_FACE_ENTERPRISE")

# Session variables
MAX_SESSIONS = 100
sessions = OrderedDict()
session_lock = threading.Lock()

# Global shared state
shared_state: Dict = {"current_model": "GPT-4", "vectorstore": None, "pdfs": []}
shared_state_lock = threading.Lock()

# Model setup
MODELS = {
    "GPT-4": {"model_id": "gpt-4", "provider": "openai"},
    "DeepSeek-R1": {"model_id": "deepseek-ai/DeepSeek-R1", "provider": "novita"},
    "Gemma-3-27B": {"model_id": "google/gemma-3-27b-it", "provider": "hf-inference"},
    "Llama-3.3-70B": {
        "model_id": "meta-llama/Llama-3.3-70B-Instruct",
        "provider": "hf-inference",
    },
    "QwQ-32B": {"model_id": "Qwen/QwQ-32B", "provider": "hf-inference"},
}

# Rerank 설정
RERANK_OPTIONS = {
    "없음": {"enabled": False, "model": None, "top_k": 3},
    "Cross-Encoder (기본)": {
        "enabled": True,
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "top_k": 3,
    },
    "Cross-Encoder (고성능)": {
        "enabled": True,
        "model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "top_k": 3,
    },
    "다국어 Cross-Encoder": {
        "enabled": True,
        "model": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "top_k": 3,
    },
}

EMBED_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
DIMENSION = 384
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

system_prompt = "You are a helpful assistant that can answer questions based on the context when provided."


class PerformanceMetrics:
    """성능 지표를 추적하는 클래스"""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.first_token_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.token_count: int = 0
        self.retrieval_time: float = 0.0

    def reset(self):
        self.start_time = None
        self.first_token_time = None
        self.end_time = None
        self.token_count = 0
        self.retrieval_time = 0.0

    def start_query(self):
        self.reset()
        self.start_time = time.time()

    def first_token_received(self):
        if self.first_token_time is None:
            self.first_token_time = time.time()

    def add_token(self, token_text: str = ""):
        self.token_count += 1

    def finish_query(self):
        self.end_time = time.time()

    def get_metrics(self) -> Dict[str, float]:
        if not self.start_time:
            return {}

        metrics: Dict[str, float] = {}
        current_time = self.end_time or time.time()

        # 전체 소요 시간
        metrics["total_time"] = current_time - self.start_time

        # 첫 토큰까지의 시간
        if self.first_token_time:
            metrics["time_to_first_token"] = self.first_token_time - self.start_time

        # Tokens per second
        if self.token_count > 0 and self.first_token_time and self.end_time:
            generation_time = self.end_time - self.first_token_time
            if generation_time > 0:
                metrics["tokens_per_second"] = self.token_count / generation_time

        # 검색 시간
        metrics["retrieval_time"] = self.retrieval_time

        return metrics


# 각 rerank 모드별 성능 지표 추적
performance_trackers = {mode: PerformanceMetrics() for mode in RERANK_OPTIONS.keys()}


# --------- (A) SESSION SETUP ---------
def get_session_id(request: gr.Request):
    raw_id = request.client.host + str(request.headers.get("user-agent"))
    return hashlib.sha256(raw_id.encode()).hexdigest()


def init_session(session_id: str, rerank_method="없음"):
    with shared_state_lock:
        current_model = str(shared_state["current_model"])

    if len(sessions) >= MAX_SESSIONS:
        evicted_id, _ = sessions.popitem(last=False)
        print(f"🧹 Removed LRU session: {evicted_id[:8]}...")

    model_info = MODELS[current_model]
    sessions[session_id] = {
        "history": {
            mode: [{"role": "system", "content": system_prompt}]
            for mode in RERANK_OPTIONS.keys()
        },
        "model_id": model_info["model_id"],
        "client": create_client(model_info["provider"]),
        "rerank_method": rerank_method,
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
    return FAISS.from_documents(all_docs, EMBED_MODEL)


def create_retriever(vectorstore, rerank_method="없음"):
    """벡터스토어에서 retriever 생성 - rerank 옵션 지원"""
    if not vectorstore:
        return None

    rerank_config = RERANK_OPTIONS.get(rerank_method, RERANK_OPTIONS["없음"])

    if not rerank_config["enabled"]:
        # 기본 벡터 검색만 사용
        return vectorstore.as_retriever(search_kwargs={"k": 3})

    try:
        # Rerank 기능 활성화
        top_k = rerank_config.get("top_k", 3)
        if not isinstance(top_k, int) or top_k <= 0:
            top_k = 3
        initial_k = max(10, top_k * 3)  # 최종 결과의 3배 정도 검색
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": initial_k})

        # Cross-encoder 모델 설정
        cross_encoder = HuggingFaceCrossEncoder(model_name=rerank_config["model"])
        compressor = CrossEncoderReranker(model=cross_encoder)

        # ContextualCompressionRetriever로 래핑
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )

        # Top-K 제한을 위한 래퍼 클래스
        class TopKLimitedRetriever:
            def __init__(self, retriever, top_k):
                self.retriever = retriever
                self.top_k = top_k

            def invoke(self, query, **kwargs):
                docs = self.retriever.invoke(query, **kwargs)
                return docs[: self.top_k]

        limited_retriever = TopKLimitedRetriever(compression_retriever, top_k)
        return limited_retriever

    except Exception as e:
        print(f"❌ Reranker setup failed for {rerank_method}: {e}")
        return vectorstore.as_retriever(search_kwargs={"k": 3})


def handle_pdf_upload(pdfs, request: gr.Request):
    start_time = time.time()
    session_id = get_session_id(request)

    # Ensure session-safe access
    with session_lock:
        if session_id not in sessions:
            init_session(session_id)
            print(
                f"✅ New session created: {session_id[:8]}... | Total sessions: {len(sessions)}"
            )

        sessions.move_to_end(session_id)

    # Update shared vectorstore
    with shared_state_lock:
        print("Processing PDF(s)...")
        vectorstore = create_vectorstore_from_pdfs(pdfs) if pdfs else None
        shared_state["vectorstore"] = vectorstore
        shared_state["pdfs"] = pdfs

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processed {len(pdfs)} PDFs in {elapsed_time:.2f} seconds")

    return f"✅ {len(pdfs)}개 PDF 파일 처리 완료 ({elapsed_time:.2f}초)"


# --------- (C) PRIMARY CHAT FUNCTION ---------
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

    model_info = MODELS[current_model]

    # Extract relevant text data from PDFs
    context = ""
    if vectorstore:
        print(f"🔍 [{rerank_method}] Retrieving relevant data...")
        retrieval_start = time.time()

        retriever = create_retriever(vectorstore, rerank_method)
        if retriever:
            docs = retriever.invoke(user_query)
            context = "\n".join([doc.page_content for doc in docs])

        retrieval_end = time.time()
        metrics.retrieval_time = retrieval_end - retrieval_start
        print(
            f"📊 [{rerank_method}] Retrieved {len(docs) if 'docs' in locals() else 0} documents in {metrics.retrieval_time:.2f}s"
        )

    messages.append(
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
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
                content = None
                if (
                    hasattr(chunk, "choices")
                    and chunk.choices
                    and chunk.choices[0].delta.content
                ):
                    content = chunk.choices[0].delta.content
                elif hasattr(chunk, "token"):
                    content = (
                        chunk.token.text
                        if hasattr(chunk.token, "text")
                        else str(chunk.token)
                    )

                if content:
                    if not metrics.first_token_time:
                        metrics.first_token_received()

                    bot_response += content
                    metrics.add_token(content)

                    # Update the last message (assistant's response)
                    history[-1]["content"] = html.escape(bot_response)
                    yield history, format_metrics(metrics.get_metrics(), rerank_method)

    except Exception as e:
        print(f"❌ [{rerank_method}] Streaming error: {e}")
        print("Falling back to non-streaming mode...")

        # Fallback to non-streaming
        try:
            completion = client.chat.completions.create(
                model=model_info["model_id"],
                messages=messages,
            )
            bot_response = completion.choices[0].message.content
            metrics.add_token(bot_response)
            history[-1]["content"] = html.escape(bot_response)
            yield history, format_metrics(metrics.get_metrics(), rerank_method)
        except Exception as fallback_error:
            print(f"❌ [{rerank_method}] Fallback error: {fallback_error}")
            error_message = f"[{rerank_method}] 죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다."
            history[-1]["content"] = error_message
            yield history, format_metrics(metrics.get_metrics(), rerank_method)
            return

    # 완료 시간 기록
    metrics.finish_query()

    # Save final history
    save_history(history, session_id, rerank_method)

    final_metrics = metrics.get_metrics()
    print(
        f"✅ [{rerank_method}] Query completed in {final_metrics.get('total_time', 0):.2f}s"
    )
    yield history, format_metrics(final_metrics, rerank_method)


def format_metrics(metrics: Dict[str, float], rerank_method: str) -> str:
    """성능 지표를 포맷된 문자열로 변환"""
    if not metrics:
        return f"**{rerank_method}** - 측정 중..."

    lines = [f"## 📊 {rerank_method} 성능 지표"]

    if "time_to_first_token" in metrics:
        lines.append(f"⏱️ **첫 토큰까지**: {metrics['time_to_first_token']:.2f}초")

    if "total_time" in metrics:
        lines.append(f"🏁 **총 소요 시간**: {metrics['total_time']:.2f}초")

    if "tokens_per_second" in metrics:
        lines.append(f"🚀 **속도**: {metrics['tokens_per_second']:.1f} tokens/sec")

    if "retrieval_time" in metrics:
        lines.append(f"🔍 **검색 시간**: {metrics['retrieval_time']:.2f}초")

    return "\n".join(lines)


# --------- (D) ADDITIONAL FUNCTIONS ---------
def create_client(provider):
    """모델 정보에 따라 InferenceClient 객체 생성"""
    if provider == "openai":
        return openai.Client(api_key=OPENAI_API_KEY)

    headers = {}
    if HF_ENTERPRISE:
        headers["X-HF-Bill-To"] = HF_ENTERPRISE

    return InferenceClient(
        provider=provider,
        api_key=HF_API_KEY,
        headers=headers if headers else None,
    )


def change_model(model_name, request: gr.Request):
    """사용자 선택에 따라 모델 변경 - 모든 세션에 동기화"""
    session_id = get_session_id(request)

    # 공유 상태 업데이트
    with shared_state_lock:
        shared_state["current_model"] = model_name

    # 현재 세션 업데이트
    with session_lock:
        if session_id not in sessions:
            init_session(session_id)
        else:
            model_info = MODELS[model_name]
            sessions[session_id]["model_id"] = model_info["model_id"]
            sessions[session_id]["client"] = create_client(model_info["provider"])
            sessions.move_to_end(session_id)

    print(f"🔄 Model changed to: {model_name}")
    return f"✅ 모델이 {model_name}로 변경되었습니다"


def save_history(history, session_id, rerank_method):
    """대화 기록(history)을 JSON 파일로 저장"""
    folder = "./chat_logs_lvrag_enhanced"
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder, f"{timestamp}_{session_id}_{rerank_method}.json")

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def reset_session(request: gr.Request):
    """대화 및 파일 업로드 내역 삭제"""
    session_id = get_session_id(request)

    with session_lock:
        init_session(session_id)
        sessions.move_to_end(session_id)
        print(f"♻️ Session {session_id[:8]}... reset.")

    # 성능 지표도 리셋
    for tracker in performance_trackers.values():
        tracker.reset()

    # 모든 채팅창을 빈 상태로 리셋
    empty_histories = [[] for _ in RERANK_OPTIONS.keys()]
    empty_metrics = [format_metrics({}, method) for method in RERANK_OPTIONS.keys()]

    return "", *empty_histories, *empty_metrics


def copy_as_markdown(history, rerank_method):
    """채팅 내역을 마크다운 형식으로 변환"""
    if not history:
        return "복사할 내용이 없습니다."

    markdown_content = [f"# {rerank_method} 채팅 기록\n"]

    for i, message in enumerate(history):
        if message["role"] == "system":
            continue

        role = "사용자" if message["role"] == "user" else "AI 어시스턴트"
        content = (
            html.unescape(message["content"])
            if isinstance(message["content"], str)
            else str(message["content"])
        )

        markdown_content.append(f"## {role}\n")
        markdown_content.append(f"{content}\n")

    result = "\n".join(markdown_content)
    print(f"📋 Markdown content prepared for {rerank_method} ({len(result)} chars)")
    return result


# --------- (E) Multi-Chat Interface ---------
def create_rerank_interface():
    """각 rerank 모드별 채팅 인터페이스 생성"""
    interfaces = {}

    for rerank_method in RERANK_OPTIONS.keys():
        with gr.Column():
            gr.Markdown(f"### 🔄 {rerank_method}")

            chatbot = gr.Chatbot(
                label=f"{rerank_method} 채팅", type="messages", height=400
            )

            metrics_display = gr.Markdown(
                value=format_metrics({}, rerank_method), label="성능 지표"
            )

            copy_btn = gr.Button(f"📋 {rerank_method} Markdown 복사", size="sm")
            copy_output = gr.Textbox(
                label="Markdown 내용 (Ctrl+A, Ctrl+C로 복사)", lines=3, visible=False
            )

            # 복사 버튼 이벤트
            copy_btn.click(
                fn=lambda hist, method=rerank_method: (
                    copy_as_markdown(hist, method),
                    gr.update(visible=True),
                ),
                inputs=[chatbot],
                outputs=[copy_output, copy_output],
            )

            interfaces[rerank_method] = {
                "chatbot": chatbot,
                "metrics": metrics_display,
                "copy_btn": copy_btn,
                "copy_output": copy_output,
            }

    return interfaces


def handle_multi_query(user_query, request: gr.Request):
    """모든 rerank 모드에서 동시에 쿼리 실행"""
    if not user_query.strip():
        return [[] for _ in RERANK_OPTIONS.keys()] + [
            format_metrics({}, method) for method in RERANK_OPTIONS.keys()
        ]

    print(f"🚀 Starting multi-query test with: {user_query[:50]}...")

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

    print("🎉 All rerank modes completed!")


# --------- (F) Gradio UI ---------
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
footer {
    display: none !important;
}
"""

with gr.Blocks(title="LiberVance RAG Enhanced", css=css, fill_height=True) as demo:
    gr.Markdown(
        "<center><h1>📄 LiberVance RAG Enhanced - Multi-Rerank Comparison</h1></center>"
    )

    with gr.Row():
        with gr.Column(scale=2):
            # 공통 컨트롤
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    list(MODELS.keys()),
                    label="🤖 Model Selection (모든 채팅창 동기화)",
                    value="GPT-4",
                    scale=2,
                )
                model_status = gr.Textbox(
                    label="모델 상태",
                    value="✅ GPT-4 준비됨",
                    interactive=False,
                    scale=1,
                )

            pdf_upload = gr.Files(
                label="📁 Upload PDF files (모든 채팅창 공유)", file_types=[".pdf"]
            )

            pdf_status = gr.Textbox(
                label="파일 상태", value="📁 PDF 파일을 업로드하세요", interactive=False
            )

            user_input = gr.Textbox(
                label="🔍 Query (모든 Rerank 모드에서 동시 테스트)",
                placeholder="예: 이 문서의 주요 내용을 요약해주세요",
                lines=3,
            )

            with gr.Row():
                submit_btn = gr.Button("🚀 Submit to All", variant="primary", scale=2)
                reset_btn = gr.Button("♻️ Reset All", variant="secondary", scale=1)

        with gr.Column(scale=3):
            # Rerank 모드별 채팅 인터페이스
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🔄 없음")
                    chatbot_none = gr.Chatbot(label="없음", type="messages", height=350)
                    metrics_none = gr.Markdown(
                        value=format_metrics({}, "없음"), elem_classes=["metrics-box"]
                    )
                    copy_btn_none = gr.Button("📋 Markdown 복사", size="sm")
                    copy_output_none = gr.Textbox(
                        label="Markdown (Ctrl+A, Ctrl+C)", lines=2, visible=False
                    )

                with gr.Column():
                    gr.Markdown("### 🔄 Cross-Encoder (기본)")
                    chatbot_basic = gr.Chatbot(
                        label="Cross-Encoder (기본)", type="messages", height=350
                    )
                    metrics_basic = gr.Markdown(
                        value=format_metrics({}, "Cross-Encoder (기본)"),
                        elem_classes=["metrics-box"],
                    )
                    copy_btn_basic = gr.Button("📋 Markdown 복사", size="sm")
                    copy_output_basic = gr.Textbox(
                        label="Markdown (Ctrl+A, Ctrl+C)", lines=2, visible=False
                    )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🔄 Cross-Encoder (고성능)")
                    chatbot_advanced = gr.Chatbot(
                        label="Cross-Encoder (고성능)", type="messages", height=350
                    )
                    metrics_advanced = gr.Markdown(
                        value=format_metrics({}, "Cross-Encoder (고성능)"),
                        elem_classes=["metrics-box"],
                    )
                    copy_btn_advanced = gr.Button("📋 Markdown 복사", size="sm")
                    copy_output_advanced = gr.Textbox(
                        label="Markdown (Ctrl+A, Ctrl+C)", lines=2, visible=False
                    )

                with gr.Column():
                    gr.Markdown("### 🔄 다국어 Cross-Encoder")
                    chatbot_multilang = gr.Chatbot(
                        label="다국어 Cross-Encoder", type="messages", height=350
                    )
                    metrics_multilang = gr.Markdown(
                        value=format_metrics({}, "다국어 Cross-Encoder"),
                        elem_classes=["metrics-box"],
                    )
                    copy_btn_multilang = gr.Button("📋 Markdown 복사", size="sm")
                    copy_output_multilang = gr.Textbox(
                        label="Markdown (Ctrl+A, Ctrl+C)", lines=2, visible=False
                    )

    # 이벤트 리스너 설정
    pdf_upload.change(fn=handle_pdf_upload, inputs=[pdf_upload], outputs=[pdf_status])

    model_dropdown.change(
        fn=change_model, inputs=[model_dropdown], outputs=[model_status]
    )

    # 멀티 쿼리 처리
    submit_btn.click(
        fn=handle_multi_query,
        inputs=[user_input],
        outputs=[
            chatbot_none,
            metrics_none,
            chatbot_basic,
            metrics_basic,
            chatbot_advanced,
            metrics_advanced,
            chatbot_multilang,
            metrics_multilang,
        ],
    )

    user_input.submit(
        fn=handle_multi_query,
        inputs=[user_input],
        outputs=[
            chatbot_none,
            metrics_none,
            chatbot_basic,
            metrics_basic,
            chatbot_advanced,
            metrics_advanced,
            chatbot_multilang,
            metrics_multilang,
        ],
    )

    # 리셋 기능
    reset_btn.click(
        fn=reset_session,
        inputs=[],
        outputs=[
            user_input,
            chatbot_none,
            chatbot_basic,
            chatbot_advanced,
            chatbot_multilang,
            metrics_none,
            metrics_basic,
            metrics_advanced,
            metrics_multilang,
        ],
    )

    # 복사 버튼 이벤트
    copy_btn_none.click(
        fn=lambda hist: (copy_as_markdown(hist, "없음"), gr.update(visible=True)),
        inputs=[chatbot_none],
        outputs=[copy_output_none, copy_output_none],
    )

    copy_btn_basic.click(
        fn=lambda hist: (
            copy_as_markdown(hist, "Cross-Encoder (기본)"),
            gr.update(visible=True),
        ),
        inputs=[chatbot_basic],
        outputs=[copy_output_basic, copy_output_basic],
    )

    copy_btn_advanced.click(
        fn=lambda hist: (
            copy_as_markdown(hist, "Cross-Encoder (고성능)"),
            gr.update(visible=True),
        ),
        inputs=[chatbot_advanced],
        outputs=[copy_output_advanced, copy_output_advanced],
    )

    copy_btn_multilang.click(
        fn=lambda hist: (
            copy_as_markdown(hist, "다국어 Cross-Encoder"),
            gr.update(visible=True),
        ),
        inputs=[chatbot_multilang],
        outputs=[copy_output_multilang, copy_output_multilang],
    )


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
    main_dev()

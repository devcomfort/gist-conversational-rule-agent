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

# Environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
HF_ENTERPRISE = os.getenv("HUGGING_FACE_ENTERPRISE")

# Session variables
MAX_SESSIONS = 100
sessions = OrderedDict()
session_lock = threading.Lock()

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


# --------- (A) SESSION SETUP ---------
def get_session_id(request: gr.Request):
    raw_id = request.client.host + str(request.headers.get("user-agent"))
    return hashlib.sha256(raw_id.encode()).hexdigest()


def init_session(session_id: str, model_name="GPT-4", rerank_method="없음"):
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
        print("🔍 Using basic vector retrieval (no rerank)")
        return vectorstore.as_retriever(search_kwargs={"k": 3})

    try:
        # Rerank 기능 활성화
        print(f"🔄 Setting up reranker: {rerank_method}")
        print(f"   Model: {rerank_config['model']}")

        # 더 많은 문서를 초기 검색 (rerank를 위해)
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

        print(f"✅ Reranker setup complete - will return top {top_k} documents")
        return limited_retriever

    except Exception as e:
        print(f"❌ Reranker setup failed: {e}")
        print("   Falling back to basic vector retrieval")
        return vectorstore.as_retriever(search_kwargs={"k": 3})


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

        # Create retriever with current rerank method
        rerank_method = sessions[session_id].get("rerank_method", "없음")
        sessions[session_id]["retriever"] = create_retriever(vectorstore, rerank_method)
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
    client = session["client"]
    model_info = None
    for name, info in MODELS.items():
        if info["model_id"] == session["model_id"]:
            model_info = info
            break

    start_time = time.time()  # ⏱️ TIMER

    # Extract relevant text data from PDFs
    context = ""
    retriever = session["retriever"]
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

    try:
        if model_info and model_info["provider"] == "openai":
            # OpenAI 스트리밍
            completion = client.chat.completions.create(
                model=session["model_id"], messages=messages, stream=True
            )

            bot_response = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    chunk_content = chunk.choices[0].delta.content
                    bot_response += chunk_content
                    # Update the last message (assistant's response)
                    history[-1]["content"] = html.escape(bot_response)
                    yield history

        else:
            # HuggingFace Inference Client 스트리밍
            completion = client.chat.completions.create(
                model=session["model_id"], messages=messages, stream=True
            )

            bot_response = ""
            for chunk in completion:
                if (
                    hasattr(chunk, "choices")
                    and chunk.choices
                    and chunk.choices[0].delta.content
                ):
                    chunk_content = chunk.choices[0].delta.content
                    bot_response += chunk_content
                    # Update the last message (assistant's response)
                    history[-1]["content"] = html.escape(bot_response)
                    yield history
                elif hasattr(chunk, "token"):
                    # HuggingFace 스트리밍의 경우 token 필드를 사용할 수 있음
                    bot_response += (
                        chunk.token.text
                        if hasattr(chunk.token, "text")
                        else str(chunk.token)
                    )
                    history[-1]["content"] = html.escape(bot_response)
                    yield history

    except Exception as e:
        print(f"❌ Streaming error: {e}")
        print("Falling back to non-streaming mode...")

        # Fallback to non-streaming
        try:
            completion = client.chat.completions.create(
                model=session["model_id"],
                messages=messages,
            )
            bot_response = completion.choices[0].message.content
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
                print(f"🔄 Updating retriever with: {rerank_method}")
                sessions[session_id]["retriever"] = create_retriever(
                    vectorstore, rerank_method
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
        if session_id in sessions:
            current_rerank_method = sessions[session_id].get("rerank_method", "없음")

        init_session(session_id, rerank_method=current_rerank_method)
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
                list(MODELS.keys()), label="Select Model", value="GPT-4"
            )
            rerank_dropdown = gr.Dropdown(
                list(RERANK_OPTIONS.keys()),
                label="Rerank Method",
                value="없음",
                info="Cross-Encoder 모델을 사용해 검색 결과를 재정렬합니다",
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

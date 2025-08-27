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
import os, json, time, hashlib, html, threading, fitz, openai
from collections import OrderedDict
from datetime import datetime
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
EMBED_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
DIMENSION = 384
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

system_prompt = "You are a helpful assistant that can answer questions based on the context when provided."


# --------- (A) SESSION SETUP ---------
def get_session_id(request: gr.Request):
    raw_id = request.client.host + str(request.headers.get("user-agent"))
    return hashlib.sha256(raw_id.encode()).hexdigest()


def init_session(session_id: str, model_name="GPT-4"):
    if len(sessions) >= MAX_SESSIONS:
        evicted_id, _ = sessions.popitem(last=False)
        print(f"🧹 Removed LRU session: {evicted_id[:8]}...")
    model_info = MODELS[model_name]
    sessions[session_id] = {
        "history": [{"role": "system", "content": system_prompt}],
        "vectorstore": None,
        "pdfs": [],
        "model_id": model_info["model_id"],
        "client": create_client(model_info["provider"]),
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
        print(f"Processing PDF(s)...")
        sessions[session_id]["vectorstore"] = (
            create_vectorstore_from_pdfs(pdfs) if pdfs else None
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
    vectorstore = session["vectorstore"]
    client = session["client"]
    start_time = time.time()  # ⏱️ TIMER

    # Extract relevant text data from PDFs
    context = ""
    if vectorstore:
        print("Retrieving relevant data...")
        retriever = vectorstore.as_retriever()
        docs = retriever.invoke(user_query)
        context = "\n".join([doc.page_content for doc in docs])
    messages.append(
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
    )

    # Invoke client with user query
    print("Inquiring LLM...")
    completion = client.chat.completions.create(
        model=session["model_id"],
        messages=messages,
    )
    # Update history
    print("Processing response...")
    bot_response = completion.choices[0].message.content
    bot_response = html.escape(bot_response)
    history.append({"role": "user", "content": user_query})
    history.append({"role": "assistant", "content": bot_response})
    save_history(history, session_id)

    end_time = time.time()  # ⏱️ TIMER
    elapsed_time = end_time - start_time  # ⏱️ TIMER
    print(f"Responded to user query in {elapsed_time:.2f} seconds")  # ⏱️ TIMER
    return history


# --------- (D) ADDITIONAL FUNCTIONS ---------
def create_client(provider):
    """모델 정보에 따라 InferenceClient 객체 생성"""
    if provider == "openai":
        return openai.Client(api_key=OPENAI_API_KEY)
    return InferenceClient(
        provider=provider,
        api_key=HF_API_KEY,
        headers={"X-HF-Bill-To": HF_ENTERPRISE},
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
        init_session(session_id)
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

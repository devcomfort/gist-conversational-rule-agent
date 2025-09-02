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
import matplotlib.pyplot as plt  # type: ignore
import matplotlib  # type: ignore
import io
import base64
import tempfile
from collections import OrderedDict
from datetime import datetime
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Dict, List, Optional, Tuple

# matplotlib 백엔드 설정 (서버 환경에서 GUI 없이 사용)
matplotlib.use("Agg")

# matplotlib 한글 폰트 설정 (폰트가 없어도 에러나지 않도록)
try:
    plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass  # 폰트 설정 실패해도 무시

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


# --------- (A) RETRIEVAL VISUALIZATION SYSTEM ---------


class RetrievalVisualizer:
    """
    Top-k Retrieval과 Knee Detection을 위한 시각화 시스템

    LiberVance RAG 시스템에서 검색된 문서들의 유사도를 시각화하여
    사용자가 검색 품질을 평가할 수 있도록 돕습니다.
    """

    def __init__(self):
        self.last_retrieval_info: Dict = {}

    def record_retrieval(
        self, query: str, docs_and_scores: List[Tuple], k: int, method: str = "top-k"
    ):
        """검색 결과 기록"""
        scores = [score for _, score in docs_and_scores]
        documents = [doc for doc, _ in docs_and_scores]

        self.last_retrieval_info = {
            "query": query,
            "method": method,
            "k": k,
            "total_docs": len(docs_and_scores),
            "scores": scores,
            "documents": documents,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        print(
            f"📊 Recorded {method} retrieval: {len(docs_and_scores)} docs for query: {query[:50]}..."
        )

    def visualize_retrieval(self, save_path: Optional[str] = None) -> str:
        """
        검색 결과를 시각화하고 base64 인코딩된 이미지 반환

        Returns:
            base64로 인코딩된 PNG 이미지 데이터 또는 에러 메시지
        """
        if not self.last_retrieval_info:
            return "⚠️ No retrieval data available. Please run a query first."

        try:
            scores = self.last_retrieval_info.get("scores", [])
            method = self.last_retrieval_info.get("method", "top-k")
            k = self.last_retrieval_info.get("k", 0)
            total_docs = self.last_retrieval_info.get("total_docs", 0)
            query = self.last_retrieval_info.get("query", "Unknown Query")
            timestamp = self.last_retrieval_info.get("timestamp", "")

            if not scores:
                return "⚠️ No similarity scores available."

            # 시각화 생성
            plt.figure(figsize=(14, 10))

            # 문서 인덱스 (x축)
            x = list(range(len(scores)))

            # 메인 플롯 - 유사도 거리 곡선
            plt.subplot(2, 1, 1)
            plt.plot(
                x,
                scores,
                "b-o",
                linewidth=2,
                markersize=6,
                label="Document Similarity Distance",
            )

            # Top-k 표시 (k번째 문서 강조)
            if method == "top-k" and k > 0 and k <= len(scores):
                plt.axvline(
                    x=k - 1,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Top-{k} Cutoff",
                )
                plt.plot(
                    k - 1,
                    scores[k - 1],
                    "ro",
                    markersize=10,
                    label=f"K-th Document (Score: {scores[k - 1]:.4f})",
                )

                # 선택된 문서 영역 강조
                selected_x = x[:k]
                selected_scores = scores[:k]
                plt.fill_between(
                    selected_x,
                    selected_scores,
                    alpha=0.3,
                    color="green",
                    label=f"Selected Top-{k} Documents",
                )

            plt.xlabel("Document Index (Similarity Rank)", fontsize=11)
            plt.ylabel("FAISS Distance (Lower = More Similar)", fontsize=11)
            plt.title(
                f'Document Retrieval Analysis ({method.upper()})\nQuery: "{query[:60]}..."',
                fontsize=12,
                fontweight="bold",
            )
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 두 번째 플롯 - 차이 분석 (기울기)
            plt.subplot(2, 1, 2)
            if len(scores) > 1:
                # 점수 간 차이 (기울기)
                diffs = [scores[i + 1] - scores[i] for i in range(len(scores) - 1)]
                diff_x = list(range(len(diffs)))

                plt.plot(
                    diff_x,
                    diffs,
                    "g-s",
                    linewidth=2,
                    markersize=4,
                    label="Score Delta (Slope)",
                )

                # Top-k cutoff 표시
                if method == "top-k" and k > 1 and k <= len(diffs) + 1:
                    plt.axvline(
                        x=k - 2,
                        color="red",
                        linestyle="--",
                        linewidth=2,
                        label="Top-K Cutoff",
                    )
                    if k - 2 < len(diffs):
                        plt.plot(k - 2, diffs[k - 2], "ro", markersize=8)

                plt.xlabel("Document Index Gap", fontsize=11)
                plt.ylabel("Distance Increase", fontsize=11)
                plt.title("Document Similarity Change Analysis", fontsize=11)
                plt.legend()
                plt.grid(True, alpha=0.3)

            # 결과 정보 텍스트 (영어로 작성)
            info_text = (
                f"📊 Analysis Results:\n"
                f"• Total documents: {total_docs}\n"
                f"• Retrieval method: {method.upper()}\n"
                f"• Retrieved documents: {k if method == 'top-k' else total_docs}\n"
                f"• Analysis time: {timestamp}"
            )

            plt.figtext(
                0.02,
                0.02,
                info_text,
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
            )

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)

            # 이미지를 메모리에 저장하고 base64로 인코딩
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
            img_buffer.seek(0)

            # base64 인코딩
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

            plt.close()  # 메모리 해제

            return f"data:image/png;base64,{img_base64}"

        except Exception as e:
            plt.close()  # 에러 시에도 메모리 해제
            return f"❌ Visualization generation error: {str(e)}"

    def get_retrieval_info(self) -> Dict:
        """마지막 검색 결과 정보 반환"""
        return self.last_retrieval_info.copy()


# 전역 시각화 객체
retrieval_visualizer = RetrievalVisualizer()


# --------- (B) SESSION SETUP ---------
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
    doc = fitz.open(pdf)  # type: ignore
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
        print("Processing PDF(s)...")
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
    visualization_image = None
    viz_status = "No documents uploaded yet"

    if vectorstore:
        print("Retrieving relevant data...")
        # 시각화를 위해 직접 similarity_search_with_score 사용
        k = 4  # 기본 검색 개수
        docs_and_scores = vectorstore.similarity_search_with_score(user_query, k=k)
        docs = [doc for doc, _ in docs_and_scores]
        context = "\n".join([doc.page_content for doc in docs])

        # 시각화를 위한 정보 기록
        global retrieval_visualizer
        retrieval_visualizer.record_retrieval(
            query=user_query, docs_and_scores=docs_and_scores, k=k, method="top-k"
        )

        # 자동으로 시각화 생성
        print("🎨 Auto-generating retrieval visualization...")
        visualization_image, viz_status = generate_retrieval_visualization()

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

    # Return history, visualization image, status, and image visibility
    image_visible = (
        gr.update(visible=True) if visualization_image else gr.update(visible=False)
    )
    return history, visualization_image, viz_status, image_visible


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

    # 시각화 상태도 초기화
    global retrieval_visualizer
    retrieval_visualizer = RetrievalVisualizer()

    return (
        "",
        [],
        None,
        "Session reset. Upload documents and run a query to see visualization.",
        gr.update(visible=False),
    )


def generate_retrieval_visualization():
    """검색 결과 시각화 생성 - 임시 파일로 저장 (자동 생성용)"""
    global retrieval_visualizer

    try:
        result = retrieval_visualizer.visualize_retrieval()

        if result.startswith("data:image/png;base64,"):
            # base64 이미지를 임시 파일로 저장
            img_data = result.replace("data:image/png;base64,", "")
            img_bytes = base64.b64decode(img_data)

            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                temp_file.write(img_bytes)
                temp_path = temp_file.name

            return temp_path, "✅ Auto-generated after retrieval"
        else:
            # 에러 메시지인 경우
            return None, result

    except Exception as e:
        return None, f"❌ Visualization error: {str(e)}"


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
    gr.Markdown(
        "<center><h1>📄 LiberVance RAG with Auto-Generated Visualization</h1><p>🎨 Automatic Top-K retrieval analysis after each query</p></center>"
    )
    with gr.Row(elem_classes=["responsive-height"]):
        # Output column
        with gr.Column(scale=2, elem_classes=["fill-height"]):
            chatbot = gr.Chatbot(
                label="Chatbot", type="messages", elem_classes=["extend-height"]
            )
        # Input column
        with gr.Column(scale=1, elem_classes=["fill-height"]):
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

            # Retrieval Visualization Section (Auto-Generated)
            with gr.Group():
                gr.Markdown("### 📊 Auto-Generated Retrieval Analysis")
                viz_status = gr.Textbox(
                    label="Visualization Status",
                    value="Upload documents and run a query to see automatic visualization",
                    interactive=False,
                    lines=2,
                )
                retrieval_viz_image = gr.Image(
                    label="Top-K Retrieval Analysis (Auto-Generated)",
                    height=400,
                    type="filepath",
                    visible=False,
                )
    # Event listeners
    pdf_upload.change(fn=handle_pdf_upload, inputs=[pdf_upload], outputs=[])
    model_dropdown.input(fn=change_model, inputs=[model_dropdown], outputs=[])

    # Query submission with automatic visualization
    user_input.submit(
        fn=handle_query,
        inputs=[user_input],
        outputs=[chatbot, retrieval_viz_image, viz_status, retrieval_viz_image],
    )
    submit_btn.click(
        fn=handle_query,
        inputs=[user_input],
        outputs=[chatbot, retrieval_viz_image, viz_status, retrieval_viz_image],
    )

    # Reset with visualization cleanup
    reset_btn.click(
        fn=reset_session,
        inputs=[],
        outputs=[
            user_input,
            chatbot,
            retrieval_viz_image,
            viz_status,
            retrieval_viz_image,
        ],
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
    main()

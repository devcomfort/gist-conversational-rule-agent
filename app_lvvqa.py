"""
LiberVance VQA (Visual Question Answering) 시스템

이 모듈은 이미지와 텍스트를 함께 처리할 수 있는 멀티모달 AI 시스템을 제공합니다.
사용자가 이미지를 업로드하고 해당 이미지에 대해 자연어로 질문하면,
Vision-Language 모델이 이미지의 내용을 분석하여 답변을 생성합니다.

주요 기능:
- 다양한 VL 모델 지원 (o3, LLaMA-4-Scout, Gemma-3, Qwen-2.5-VL, LLaVA 등)
- 실시간 모델 전환 기능
- 세션 기반 대화 관리
- 이미지를 포함한 대화 기록 저장
- 안전한 멀티세션 처리

사용 예시:
- 문서 이미지 분석 및 요약
- 차트/그래프 데이터 해석
- 의료/과학 이미지 분석
- 일반 사진에 대한 질의응답
"""

import gradio as gr
import os, json, time, hashlib, html, threading, openai, base64
from collections import OrderedDict
from datetime import datetime
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from io import BytesIO
from PIL import Image

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
    # "GPT-4o-mini": {"model_id": "gpt-4o-mini", "provider": "openai"},
    "o3": {"model_id": "o3", "provider": "openai"},
    "LLaMA-4-Scout-17B-16E": {"model_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct", "provider": "novita"},
    "Gemma-3-27B": {"model_id": "google/gemma-3-27b-it", "provider": "nebius"},
    # "Phi-4-Multimodal": {"model_id": "microsoft/Phi-4-multimodal-instruct", "provider": ""},
    "Qwen-2.5-VL-7B": {"model_id": "Qwen/Qwen2.5-VL-7B-Instruct", "provider": "hyperbolic"},
    # "LLaMA-3.2-11B-Vision": {"model_id": "meta-llama/LLaMA-3.2-11B-Vision-Instruct", "provider": "novita"},
    "LLaVA-1.5-13B": {"model_id": "llava-hf/llava-1.5-13b-hf", "provider": "nebius"},
}
default_model = "o3"  # Default model to use if not specified

system_prompt = (
    "You are a helpful assistant that can answer questions based on the context when provided."
)


# --------- (A) SESSION SETUP ---------
def get_session_id(request: gr.Request):
    raw_id = request.client.host + str(request.headers.get("user-agent"))
    return hashlib.sha256(raw_id.encode()).hexdigest()

def init_session(session_id: str, model_name=default_model):
    model_info = MODELS[model_name]
    sessions[session_id] = {
        "model_history": [{"role": "system", "content": system_prompt}],
        "chat_history": [],
        "model_id": model_info["model_id"],
        "client": create_client(model_info["provider"])
    }

# --------- (B) IMAGE CONVERSION ---------
def encode_ndarray_to_base64(image):
    pil_img = Image.fromarray(image.astype("uint8"))
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

# --------- (C) PRIMARY CHAT FUNCTION ---------
def handle_query(user_query, user_img, request: gr.Request):
    session_id = get_session_id(request)
    # Ensure session-safe access
    with session_lock:
        if session_id not in sessions:
            if len(sessions) >= MAX_SESSIONS:
                evicted_id, _ = sessions.popitem(last=False)
                print(f"🧹 Removed LRU session: {evicted_id[:8]}...")
            init_session(session_id)
            print(f"✅ New session created: {session_id[:8]}... | Total sessions: {len(sessions)}")
        session = sessions[session_id]
        sessions.move_to_end(session_id)
    
    model_history = session["model_history"]
    chat_history = session["chat_history"]
    if user_query == "":
        return chat_history
    
    start_time = time.time()                                                    # ⏱️ TIMER
    
    # Extract relevant text data from PDFs
    content = [{"type": "text", "text": user_query}]
    image_url = None
    if user_img is not None:
        image_url = encode_ndarray_to_base64(user_img)
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    messages = model_history.copy()
    messages.append({"role": "user", "content": content})
   
    # Invoke client with user query
    print("Inquiring LLM...")
    client = session["client"]
    completion = client.chat.completions.create(
        model=session["model_id"],
        messages=messages,
    )
    # Update history
    print("Processing response...")
    bot_response = completion.choices[0].message.content
    bot_response = html.escape(bot_response)
    model_history.append({"role": "user", "content": user_query})
    model_history.append({"role": "assistant", "content": bot_response})
    save_history(model_history, session_id)
   
    end_time = time.time()                                                      # ⏱️ TIMER
    elapsed_time = end_time - start_time                                        # ⏱️ TIMER
    print(f"Responded to user query in {elapsed_time:.2f} seconds")             # ⏱️ TIMER

    if user_img is not None:
        user_html = f"{html.escape(user_query)}<br><img src='{image_url}' width='200'>"
        chat_history.append({"role": "user", "content": user_html})
    else:
        chat_history.append({"role": "user", "content": html.escape(user_query)})
    chat_history.append({"role": "assistant", "content": bot_response})

    return chat_history

# --------- (D) ADDITIONAL FUNCTIONS ---------
def create_client(provider):
    """모델 정보에 따라 InferenceClient 객체 생성"""
    if provider == "openai":
        return openai.Client(
            api_key=OPENAI_API_KEY
        )
    return InferenceClient(
        provider=provider, 
        api_key=HF_API_KEY, 
        headers={"X-HF-Bill-To": HF_ENTERPRISE},
    )

def change_model(model_name, request:gr.Request):
    """사용자 선택에 따라 모델 변경"""
    session_id = get_session_id(request)
    with session_lock:
        if session_id not in sessions:
            if len(sessions) >= MAX_SESSIONS:
                evicted_id, _ = sessions.popitem(last=False)
                print(f"🧹 Removed LRU session: {evicted_id[:8]}...")
            init_session(session_id)
            print(f"✅ New session created: {session_id[:8]}... | Total sessions: {len(sessions)}")
        model_info = MODELS[model_name]
        sessions[session_id]["model_id"] = model_info["model_id"]
        sessions[session_id]["client"] = create_client(model_info["provider"])
        sessions.move_to_end(session_id)
    # print(f"🔄 Now using: {model_name}")

def save_history(history, session_id):
    """대화 기록(history)을 JSON 파일로 저장"""
    folder = "./chat_logs_lvvqa"
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

with gr.Blocks(title="LiberVance VQA", css=css, fill_height=True) as demo:
    gr.Markdown("<center><h1>🖼️ LiberVance VQA</h1></center>")
    with gr.Row(elem_classes=["responsive-height"]):
        # Output column
        with gr.Column(elem_classes=["fill-height"]):
            chatbot = gr.Chatbot(label="Chatbot", type="messages", elem_classes=["extend-height"])
        # Input column
        with gr.Column(elem_classes=["fill-height"]):
            model_dropdown = gr.Dropdown(list(MODELS.keys()), label="Select Model", value=default_model)
            img_upload = gr.Image(label="Upload image", elem_classes=["extend-height"])
            user_input = gr.Textbox(label="Enter your query", placeholder="e.g., Summarize the key points from this document.", lines=3)
            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary")
                reset_btn = gr.Button("Reset", variant="secondary")
    # Event listeners
    model_dropdown.input(fn=change_model, inputs=[model_dropdown], outputs=[])
    user_input.submit(handle_query, inputs=[user_input, img_upload], outputs=[chatbot])
    submit_btn.click(handle_query, inputs=[user_input, img_upload], outputs=[chatbot])
    reset_btn.click(reset_session, inputs=[], outputs=[user_input, chatbot])

demo.launch(share=True, favicon_path="")
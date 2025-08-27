"""
LiberVance Search v2 (LV-Search2) 시스템

이 모듈은 Google Custom Search Engine과 OpenAI GPT-4를 결합한
고급 웹 검색 어시스턴트입니다. LangGraph를 사용하여 구조화된
검색 → 요약 → 응답 워크플로우를 구현하며, 검색 쿼리 최적화와
다국어 대화 지원을 제공합니다.

주요 기능:
- Google Custom Search API를 통한 정확한 웹 검색
- GPT-4 기반 쿼리 최적화 및 답변 생성
- LangGraph 상태 기반 워크플로우
- 대화 컨텍스트를 고려한 검색 쿼리 개선
- 다국어 지원 (사용자 질문 언어로 자동 응답)
- 세션별 대화 기록 관리

워크플로우:
1. Search Node: 대화 맥락을 고려한 검색 쿼리 생성
2. Summary Node: 검색 결과와 컨텍스트를 종합하여 답변 생성
3. Response Node: 최종 응답을 대화 기록에 저장

기술 스택:
- 검색: Google Custom Search Engine API
- LLM: OpenAI GPT-4
- 워크플로우: LangGraph StateGraph
- 프롬프트: LangChain ChatPromptTemplate
"""

import gradio as gr
import os, json, time, hashlib, requests, threading
from collections import OrderedDict
from datetime import datetime
from dotenv import load_dotenv

# from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
# from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# Environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# HF_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
# HF_ENTERPRISE = os.getenv("HUGGING_FACE_ENTERPRISE")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# Session variables
MAX_SESSIONS = 100
sessions = OrderedDict()
session_lock = threading.Lock()

# Model setup
MODEL = "gpt-4"
CLIENT = ChatOpenAI(model=MODEL)

# --------- (A) SESSION SETUP ---------
def get_session_id(request: gr.Request):
    raw_id = request.client.host + str(request.headers.get("user-agent"))
    return hashlib.sha256(raw_id.encode()).hexdigest()

def init_session(session_id: str):
    sessions[session_id] = {
        "history": [],
        "graph": create_graph(),
    }

# --------- (B) SEARCH FUNCTION ---------
def google_search(search_item, api_key, cse_id, search_depth=10, site_filter=None):
    url = 'https://www.googleapis.com/customsearch/v1'
    params = {
        'q': search_item,
        'key': api_key,
        'cx': cse_id,
        'num': search_depth
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    results = response.json()
    return [item['snippet'] for item in results.get('items', [])]

# --------- (C) LANGGRAPH CREATION ---------
class State(TypedDict):
    messages: list
    search_results: list
    summary: str

def search_node(state: State):
    messages = state["messages"]
    recent_msgs = [msg["content"] for msg in messages][-5:]
    context = "\n".join(recent_msgs)

    prompt = ChatPromptTemplate.from_template("""
        You are a search assistant.  
        Based on the recent conversation below, generate a concise web search query that will return relevant, up-to-date results.  
        Only output the query itself—no explanation.

        Conversation context:
        {context}

        Refined query:
    """)
    chain = prompt | CLIENT
    refined_query = chain.invoke({"context": context}).content.strip()
    print(f"Refined query: {refined_query}")

    search_results = google_search(refined_query, GOOGLE_API_KEY, GOOGLE_CSE_ID)
    return {"search_results": search_results}

def summary_node(state: State):
    messages = state["messages"]
    recent_msgs = [msg["content"] for msg in messages][-5:]
    context = "\n".join(recent_msgs)

    user_query = state["messages"][-1]["content"]
    search_results = "\n".join(state["search_results"])

    prompt = ChatPromptTemplate.from_template("""
        You are a helpful and conversational assistant in an ongoing multi-turn dialogue. 
        Use the full context of the conversation and the search results below to respond naturally, as if continuing the conversation.
        Make sure to respond in the same language as the user's question—do not switch languages.

        Conversation history:
        {context}

        Most recent question:
        {question}

        Search results:
        {search_results}
    """)
    chain = prompt | CLIENT
    summary = chain.invoke({"context": context, "question": user_query, "search_results": search_results})
    return {"summary": summary.content}

def response_node(state: State):
    assistant_msg = {"role": "assistant", "content": state["summary"]}
    return {"messages": state["messages"] + [assistant_msg]}

def create_graph():
    builder = StateGraph(State)
    memory = MemorySaver()
    # Graph build
    builder.add_node("search", search_node)
    builder.add_node("summarize", summary_node)
    builder.add_node("respond", response_node)
    builder.set_entry_point("search")
    builder.add_edge("search", "summarize")
    builder.add_edge("summarize", "respond")
    builder.set_finish_point("respond")
    return builder.compile(checkpointer=memory)

# --------- (D) PRIMARY CHAT FUNCTION ---------
def handle_query(user_query, request: gr.Request):
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
    
    history = session["history"]
    history.append({"role": "user", "content": user_query})
    start_time = time.time()                                                    # ⏱️ TIMER
    
    # Invoke graph with user query
    print("Inquiring LLM...")
    state = session["graph"].invoke(
        {"messages": history},
        {"configurable": {"thread_id": session_id}}
    )
    # Update history with bot response
    print("Processing response...")
    bot_response = state["messages"][-1]["content"]
    history.append({"role": "assistant", "content": bot_response})
    save_history(history, session_id)

    # Get time
    end_time = time.time()                                                      # ⏱️ TIMER
    elapsed_time = end_time - start_time                                        # ⏱️ TIMER
    print(f"Responded to user query in {elapsed_time:.2f} seconds")             # ⏱️ TIMER
    return history

# --------- (E) ADDITIONAL FUNCTIONS ---------
def save_history(history, session_id):
    folder = "./chat_logs_lvsearch"
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
    session_id = get_session_id(request)
    with session_lock:
        init_session(session_id)
        print(f"♻️ Session {session_id[:8]}... reset.")
    return "", []


# --------- (E) Gradio UI ---------
css = """
.responsive-height {
    height: 768px !important;
    padding-bottom: 64px !important;
}
.extend-height {
    min-height: 260px !important;
    flex: 1 !important;
    overflow: auto !important;
}
footer {
    visibility: hidden !important;
}
"""

with gr.Blocks(title="LiberVance Search", css=css) as demo:
    gr.Markdown("<center><h1>🔎 LiberVance Search</h1></center>")
    # Input and Output components
    with gr.Column(elem_classes=["responsive-height"]):
        chatbot = gr.Chatbot(label="Chatbot", type="messages", elem_classes=["extend-height"])
        user_input = gr.Textbox(label="Enter your query here", placeholder="e.g., What is the capital of Japan?", lines=3)
        submit_btn = gr.Button("Submit", variant="primary")
        reset_btn = gr.Button("Reset", variant="secondary")
    # Event listeners
    user_input.submit(handle_query, inputs=[user_input], outputs=[chatbot], preprocess=False)
    submit_btn.click(handle_query, inputs=[user_input], outputs=[chatbot], preprocess=False)
    reset_btn.click(reset_session, inputs=[], outputs=[user_input, chatbot], preprocess=False)

demo.launch(share=True, favicon_path="")
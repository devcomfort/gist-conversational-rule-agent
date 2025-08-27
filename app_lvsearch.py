"""
LiberVance Search (LV-Search) 시스템

이 모듈은 LangGraph와 Tavily 검색 API를 활용한 지능형 웹 검색 어시스턴트입니다.
사용자의 질문에 대해 실시간 웹 검색을 수행하고, 최신 정보를 바탕으로
정확하고 포괄적인 답변을 제공합니다.

주요 기능:
- LangGraph 기반 에이전트 워크플로우
- Tavily API를 통한 고품질 웹 검색
- GROQ LLaMA-3.1-8B 모델 활용
- 실시간 정보 검색 및 종합
- 세션 기반 대화 관리
- 한국 시간대 기반 정보 처리

기술 스택:
- 에이전트 프레임워크: LangGraph
- 검색 엔진: Tavily Search API
- LLM: GROQ LLaMA-3.1-8B-instant
- 상태 관리: MemorySaver
- 멀티세션 지원

사용 사례:
- 최신 뉴스 및 현재 이벤트 질의
- 실시간 데이터가 필요한 질문
- 날짜/시간 관련 정보 검색
- 트렌드 및 시장 정보 조회
"""

import gradio as gr
import os, json, time, hashlib, threading
from collections import OrderedDict
from datetime import datetime
from dotenv import load_dotenv
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.agents import AgentAction
from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq

from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from tavily import TavilyClient

# Environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Variables for session management
MAX_SESSIONS = 100
sessions = OrderedDict()
session_lock = threading.Lock()

system_prompt = (
    "You're an assistant connected to a search tool. "
    "For any questions involving dates, times, current events, or live information, "
    "you should call the search tool to look up the latest data, "
    "assuming that the user is in South Korea."
)

# --------- (A) SESSION SETUP ---------
def get_session_id(request: gr.Request):
    raw_id = request.client.host + str(request.headers.get("user-agent"))
    return hashlib.sha256(raw_id.encode()).hexdigest()

def init_session(session_id: str):
    sessions[session_id] = {
        "history": [{"role": "system", "content": system_prompt}],
        "graph": create_graph(),
    }

# --------- (B) LANGGRAPH CREATION ---------
class State(TypedDict):
    messages: Annotated[list, add_messages]

class CustomTavilyRetriever:
    def __init__(self, api_key: str, k: int = 5):
        self.client = TavilyClient(api_key=api_key)
        self.k = k

    def get_relevant_documents(self, query: str) -> List[Document]:
        client = TavilyClient(api_key=self.api_key)
        response = client.search(
            query=query,
            search_depth="advanced",
            time_range="year",
            include_raw_content=True,
            max_results=self.k
        )
        return [
            Document(page_content=result["content"], metadata={"url": result["url"]})
            for result in response.get("results", [])
        ]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)

def web_search_tool(query: str) -> str:
    retriever = CustomTavilyRetriever(api_key=TAVILY_API_KEY, k=5)
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join(doc.page_content for doc in docs)

def create_graph():
    # Setup LLM and tools
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.5,
        max_tokens=512
    )
    # tool = TavilySearchResults(max_results=2)
    tool = StructuredTool.from_function(
        func=web_search_tool,
        name="web_search_tool",
        description="Searches the web via Tavily for up-to-date information.",
    )
    llm_with_tools = llm.bind_tools([tool])
    # Setup agent
    def chatbot(state: State):
        response = llm_with_tools.invoke(state["messages"])
        if isinstance(response, AgentAction):
            print("Tool node triggered")
            return response  # Triggers tool node
        print("Tool node not triggered")
        return {"messages": [response]}
    # Initialize state graph
    builder = StateGraph(State)
    memory = MemorySaver()
    # Register nodes
    builder.add_node("chatbot", chatbot)
    builder.add_node("tools", ToolNode(tools=[tool]))
    # Register edges
    builder.set_entry_point("chatbot")
    builder.add_conditional_edges("chatbot", tools_condition)
    builder.add_edge("tools", "chatbot")
    builder.set_finish_point("chatbot")
    # Compile graph
    return builder.compile(checkpointer=memory)

# --------- (C) PRIMARY CHAT FUNCTION ---------
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
    start_time = time.time()                                                    # ⏱️ TIMER
    
    # Invoke graph with user query
    print("Inquiring LLM...")
    state = session["graph"].invoke(
        {"messages": [{"role": "user", "content": user_query}]},
        {"configurable": {"thread_id": session_id}}
    )
    # Update history with bot response
    print("Processing response...")
    bot_response = state["messages"][-1].content
    history.append({"role": "user", "content": user_query})
    history.append({"role": "assistant", "content": bot_response})
    save_history(history, session_id)

    # Get time
    end_time = time.time()                                                      # ⏱️ TIMER
    elapsed_time = end_time - start_time                                        # ⏱️ TIMER
    print(f"Responded to user query in {elapsed_time:.2f} seconds")             # ⏱️ TIMER
    return history

# --------- (D) ADDITIONAL FUNCTIONS ---------
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


# # --------- (E) Gradio UI ---------
# css = """
# .responsive-height {
#     height: 768px !important;
#     padding-bottom: 64px !important;
# }
# .extend-height {
#     min-height: 260px !important;
#     flex: 1 !important;
#     overflow: auto !important;
# }
# footer {
#     visibility: hidden !important;
# }
# """

# with gr.Blocks(title="LiberVance Search", css=css) as demo:
#     gr.Markdown("<center><h1>🔎 LiberVance Search</h1></center>")
#     # Input and Output components
#     with gr.Column(elem_classes=["responsive-height"]):
#         chatbot = gr.Chatbot(label="Chatbot", type="messages", elem_classes=["extend-height"])
#         user_input = gr.Textbox(label="Enter your query here", placeholder="e.g., What is the capital of Japan?", lines=3)
#         submit_btn = gr.Button("Submit", variant="primary")
#         reset_btn = gr.Button("Reset", variant="secondary")
#     # Event listeners
#     user_input.submit(handle_query, inputs=[user_input], outputs=[chatbot], preprocess=False)
#     submit_btn.click(handle_query, inputs=[user_input], outputs=[chatbot], preprocess=False)
#     reset_btn.click(reset_session, inputs=[], outputs=[user_input, chatbot], preprocess=False)

# demo.launch(share=True, favicon_path="")
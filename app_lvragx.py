"""
LiberVance RAG-X (확장형 RAG) 시스템

이 모듈은 Excel과 PDF 파일을 동시에 처리할 수 있는 확장된 RAG 시스템입니다.
OpenAI의 고성능 o1 모델을 활용하여 구조화된 데이터(Excel)와 
비구조화된 데이터(PDF)를 통합적으로 분석하고, 
마크다운 테이블 생성 및 Excel 다운로드 기능을 지원합니다.

주요 기능:
- Excel 파일을 마크다운 테이블로 자동 변환
- PDF 파일을 OpenAI Files API로 직접 처리
- 마크다운 테이블 자동 추출 및 Excel 변환
- 엑셀 다운로드 요청 자동 감지
- OpenAI o1 모델을 통한 고급 분석
- 대용량 파일 처리 최적화

특화 기능:
- 다중 시트 Excel 파일 지원
- 테이블 크기 제한 (최대 10,000행 × 10,000열)
- 자동 특수문자 이스케이프 처리
- 실시간 Excel 파일 생성 및 다운로드
- JSON 데이터 추출 및 처리

사용 사례:
- 비즈니스 데이터 분석 및 보고서 생성
- 재고 관리 및 예측 분석
- 예산 계획 및 리소스 할당
- 데이터 시각화를 위한 테이블 생성
- 복합 문서 (Excel + PDF) 통합 분석
"""

import gradio as gr
import os, json, time, hashlib, threading
from collections import OrderedDict
from datetime import datetime
from dotenv import load_dotenv
# from gradio_excelupload_test import make_pdf
# from langchain_community.document_loaders import PyPDFLoader
from openai import OpenAI
import pandas as pd
import re
# import tiktoken

# Environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Session variables
MAX_SESSIONS = 100
sessions = OrderedDict()
session_lock = threading.Lock()

# Model setup
MODEL = "o1"
CLIENT = OpenAI(api_key=OPENAI_API_KEY)

system_prompt = (
    "You are a helpful assistant that can answer questions based on the context when provided."
)


# --------- (A) SESSION SETUP ---------
def get_session_id(request: gr.Request):
    raw_id = request.client.host + str(request.headers.get("user-agent"))
    return hashlib.sha256(raw_id.encode()).hexdigest()

def init_session(session_id: str):
    sessions[session_id] = {
        "history": [{"role": "system", "content": system_prompt}],
        "files": [],
    }

# --------- (B) TABLE EXTRACTION ---------
def extract_tables_from_markdown(markdown_text):
    tables = []
    # 하이픈(-)을 이스케이프 처리
    table_pattern = r'\|(.+)\|[\r\n]\|([\s:\-]+\|)+'
    table_bodies = re.findall(r'\|(.+)\|[\r\n]\|([\s:\-]+\|)+[\r\n]((.*\|[\r\n])+)', markdown_text)

    for table_body in table_bodies:
        try:
            header_line = f"|{table_body[0]}|"
            separator_line = f"|{table_body[1]}"
            content_lines = table_body[2]

            headers = [cell.strip() for cell in header_line.split('|')[1:-1]]
            rows = []
            for line in content_lines.split('\n'):
                if '|' in line:
                    row_cells = [cell.strip() for cell in line.split('|')[1:-1]]
                    if row_cells:
                        rows.append(row_cells)

            if len(headers) > 0 and len(rows) > 0:
                max_cols = max(len(headers), max(len(row) for row in rows))
                padded_headers = headers + [''] * (max_cols - len(headers))
                padded_rows = [row + [''] * (max_cols - len(row)) for row in rows]

                df = pd.DataFrame(padded_rows, columns=padded_headers)
                tables.append(df)
        except Exception as e:
            print(f"테이블 추출 오류: {e}")
            continue

    return tables

# -----------------------------------------------------------------------------
# 응답을 엑셀로 변환하는 함수
# -----------------------------------------------------------------------------
def convert_response_to_excel(bot_response: str, filename=None) -> tuple:
    dfs = extract_tables_from_markdown(bot_response)
    if not dfs:
        data = {"Content": [bot_response]}
        dfs = [pd.DataFrame(data)]
    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"table_data_{ts}.xlsx"

    try:
        with pd.ExcelWriter(filename) as writer:
            for i, df in enumerate(dfs):
                sheet_name = f"Table_{i+1}" if i > 0 else "Data"
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        return filename, "엑셀 저장 성공"
    except Exception as e:
        return None, f"엑셀 변환 오류: {str(e)}"

# -----------------------------------------------------------------------------
# 토큰 수 제한(간단 예시)
# -----------------------------------------------------------------------------
def trim_history(history, max_tokens=128000):
    # 너무 길면 최근 10개만 유지한다는 예시
    if len(history) > 10:
        return history[-10:]
    return history

# -----------------------------------------------------------------------------
# 엑셀 다운로드 요청 감지 함수
# -----------------------------------------------------------------------------
def detect_excel_download_request(text):
    """사용자 입력에서 엑셀 다운로드 요청을 감지하는 함수"""
    patterns = [
        r'엑셀로\s*다운로드',
        r'엑셀\s*파일',
        r'excel\s*download',
        r'download\s*excel',
        r'엑셀\s*내려받',
        r'엑셀\s*저장',
        r'표\s*다운로드',
        r'표\s*저장',
        r'table\s*download',
        r'csv\s*다운로드',
        r'xlsx',
        r'xls',
        r'엑셀로\s*보내',
        r'엑셀로\s*변환'
    ]
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

# -----------------------------------------------------------------------------
# JSON 데이터 추출 함수
# -----------------------------------------------------------------------------
def extract_json(text):
    try:
        # 특정 코드 블록(```) 안에서 JSON 찾기 등 원하는 방식으로 구현
        # 예시는 단순화
        array_pattern = r'\[\s*{[\s\S]*?}\s*\]'
        matches = re.findall(array_pattern, text)
        if matches:
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        return json.loads(text)  # 최후의 수단
    except json.JSONDecodeError as e:
        print(f"JSON 추출 오류: {e}")
        return None

# --------- (B) DOCUMENT TEXT EXTRACTION ---------
def excel_to_markdown(file_path, max_rows=10000, max_cols=10000):
    """
    엑셀 파일을 마크다운 테이블 형식으로 변환하는 함수
    
    Args:
        file_path: 엑셀 파일 경로
        max_rows: 변환할 최대 행 수 (기본값: 100)
        max_cols: 변환할 최대 열 수 (기본값: 20)
    
    Returns:
        마크다운 테이블 형식의 문자열
    """
    try:
        # 엑셀 파일의 모든 시트 읽기
        xl = pd.ExcelFile(file_path)
        sheet_names = xl.sheet_names
        
        result = []
        
        for sheet_name in sheet_names:
            # 각 시트를 DataFrame으로 읽기
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # 데이터프레임 크기 제한
            if df.shape[0] > max_rows:
                df = df.iloc[:max_rows, :]
                
            if df.shape[1] > max_cols:
                df = df.iloc[:, :max_cols]
            
            # NaN 값을 빈 문자열로 변환
            df = df.fillna('')
            
            # 열 이름에 있는 특수 문자 처리
            df.columns = [str(col).replace('|', '\\|').strip() for col in df.columns]
            
            # 데이터 값에 있는 특수 문자 처리
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).apply(lambda x: x.replace('|', '\\|').replace('\n', ' '))
            
            # 테이블 헤더와 구분선 수동 생성
            header = "| " + " | ".join(str(col) for col in df.columns) + " |"
            separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
            
            result.append(header)
            result.append(separator)
            
            # 테이블 내용 생성
            for _, row in df.iterrows():
                row_values = [str(val) if len(str(val)) < 50 else str(val)[:47] + "..." for val in row]
                result.append("| " + " | ".join(row_values) + " |")
            
            # 시트 사이에 빈 줄 추가
            if sheet_names.index(sheet_name) < len(sheet_names) - 1:
                result.append("\n")
        
        return "\n".join(result)
    
    except Exception as e:
        return f"엑셀 파일 변환 중 오류 발생: {str(e)}"

# --------- (C) PRIMARY CHAT FUNCTION ---------
def handle_query(user_query: str, user_pdfs: list, request: gr.Request):
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
    messages = history.copy()
    files = session["files"]
    start_time = time.time()                                                    # ⏱️ TIMER

    # PDF/Excel 파일 처리
    if files:
        content = []
        for file in files:
            # PDF 파일을 OpenAI API에 업로드
            if file.name.endswith('.pdf'):
                try:
                    file_obj = CLIENT.files.create(
                        file=open(file.name, "rb"),
                        purpose="user_data"
                    )
                    content.append({
                        "type": "file",
                        "file": {"file_id": file_obj.id}})
                except Exception as e:
                    print(f"PDF 처리 오류: {e}")
            # Excel 파일을 messages에 추가
            elif file.name.endswith('.xlsx' or '.xls'):
                try:
                    file_contents = excel_to_markdown(file.name)
                    messages.append({
                        "role": "user",
                        "content": f"엑셀 내용 ({file.name}):\n{file_contents}"
                    })
                except Exception as e:
                    print(f"엑셀 처리 오류: {e}")
    # PDF + query를 messages에 추가
    content.append({"type": "text", "text": user_query})
    messages.append({"role": "user", "content": content})

    try:
        completion = CLIENT.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_completion_tokens=16384,
        )
        bot_response = completion.choices[0].message.content

        # 엑셀 다운로드 요청 처리
        excel_file = None
        if detect_excel_download_request(user_query):
            tables = extract_tables_from_markdown(bot_response)
            if tables:
                try:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    excel_file = f"table_data_{ts}.xlsx"
                    with pd.ExcelWriter(excel_file) as writer:
                        for i, df in enumerate(tables):
                            sheet_name = f"Table_{i+1}" if i > 0 else "Data"
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                    bot_response += "\n\n✅ 마크다운 테이블을 엑셀 파일로 변환했습니다. 아래에서 다운로드할 수 있습니다."
                except Exception as e:
                    bot_response += f"\n\n❌ 엑셀 변환 중 오류가 발생했습니다: {str(e)}"
            else:
                bot_response += "\n\n❓ 변환할 테이블을 찾지 못했습니다. 테이블 형식의 데이터를 요청해 주세요."
        
        # 대화 기록 저장
        history.append({"role": "user", "content": user_query})
        history.append({"role": "assistant", "content": bot_response})
        save_history(history, session_id)
   
        end_time = time.time()                                                      # ⏱️ TIMER
        elapsed_time = end_time - start_time                                        # ⏱️ TIMER
        print(f"Responded to user query in {elapsed_time:.2f} seconds")             # ⏱️ TIMER
        return history, gr.update(value=excel_file, visible=excel_file is not None)

    except Exception as e:
        error_message = f"오류가 발생했습니다: {str(e)}"
        print(error_message)  # 에러 메시지 출력
        
        # 에러 메시지 추가
        history.append({"role": "user", "content": user_query})
        history.append({"role": "assistant", "content": error_message})
        
        return history, gr.update(value=None, visible=False)

# --------- (D) ADDITIONAL FUNCTIONS ---------
def save_history(history, session_id):
    """대화 기록(history)을 JSON 파일로 저장"""
    folder = "./chat_logs_lvragx"
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


# # --------- (E) Gradio UI ---------
# css = """
# div {
#     flex-wrap: nowrap !important;
# }
# .responsive-height {
#     height: 768px !important;
#     padding-bottom: 64px !important;
# }
# .fill-height {
#     height: 100% !important;
#     flex-wrap: nowrap !important;
# }
# .extend-height {
#     min-height: 260px !important;
#     flex: 1 !important;
#     overflow: auto !important;
# }
# footer {
#     display: none !important;
# }
# """

# with gr.Blocks(title="LiberVance RAG-X", css=css, fill_height=True) as demo:
#     gr.Markdown("<center><h1>📈 LiberVance RAG-X</h1></center>")
#     with gr.Row(elem_classes=["responsive-height"]):
#         # Input column
#         with gr.Column(elem_classes=["fill-height"]):
#             chatbot = gr.Chatbot(elem_classes=["extend-height"], type="messages")
#             file_download = gr.File(label="Download file", visible=False, elem_classes=["extend-height"])
#         # Output column
#         with gr.Column(elem_classes=["fill-height"]):
#             file_upload = gr.Files(label="Upload file(s) (PDF or Excel)", file_types=[".pdf", ".xlsx"], elem_classes=["extend-height"])
#             user_input = gr.Textbox(label="Enter your query here", placeholder=(
#                 "e.g.,\n"
#                 "1. Show the table of contents, data summary, or specific sheet/table from the uploaded files.\n"
#                 "2. Identify when a key metric (e.g., inventory level, budget, or capacity) will reach a threshold or run out.\n"
#                 "3. Recommend an action or timeline (e.g., reorder date, resource allocation, deadline) based on trends in the data."
#             ), lines=8)
#             with gr.Row():
#                 send_btn = gr.Button("Submit", variant="primary")
#                 reset_btn = gr.Button("Clear", variant="secondary", elem_classes="clear-btn")
#     # Event listeners
#     user_input.submit(handle_query, inputs=[user_input, file_upload], 
#                       outputs=[chatbot, file_download])
#     send_btn.click(handle_query, inputs=[user_input, file_upload], 
#                    outputs=[chatbot, file_download])
#     reset_btn.click(reset_session, inputs=[], outputs=[user_input, chatbot])

# demo.launch(share=True, favicon_path="")
"""
LiberVance AI 통합 웹 애플리케이션

이 모듈은 여러 AI 서비스를 하나의 Gradio 웹 인터페이스로 통합합니다:
- LV-Search: 웹 검색 기반 AI 어시스턴트 (GROQ, Tavily)
- LV-RAG: PDF 문서 기반 질의응답 시스템 (OpenAI, HuggingFace)  
- LV-RAG-X: Excel/PDF 파일 처리 확장 RAG 시스템 (OpenAI)
- LV-VQA: 이미지 질의응답 시스템 (Vision-Language 모델)

현재 LV-RAG 탭만 활성화되어 있으며, 사용자가 PDF 파일을 업로드하여
문서 내용에 대해 질문할 수 있는 기능을 제공합니다.
"""

import gradio as gr
import app_lvsearch as lvsearch
import app_lvrag as lvrag
import app_lvragx as lvragx
import app_lvvqa as lvvqa

"""
Tab 1: LiberVance Search (LV-Search)    -> GROQ, TAVILY
       LiberVance Search (LV-Search)    -> HUGGING_FACE, GOOGLE_CSE
Tab 2: LiberVance RAG (LV-RAG)          -> OPENAI, HUGGING_FACE
Tab 3: LiberVance RAG X (LV-RAG-X)      -> OPENAI
Tab 4: LiberVance VQA (LV-VQA)          -> OPENAI, HUGGING_FACE
"""

# --------- Gradio UI ---------
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
.title {
    overflow: visible !important;
    max-height: none !important;
    max-width: 960px !important;
    margin: 0 auto !important;
    margin-bottom: var(--spacing-xl) !important;
    text-align: justify !important;
    text-align-last: center !important;
}
.instructions {
    text-align: center;
    font-size: var(--text-md);
    line-height: 1.6;
    padding: var(--spacing-xxl);
    background-color: var(--block-background-fill);
    border-radius: var(--block-radius);
    border: 1px solid var(--block-border-color);
    color: var(--block-label-text-color);
}
button {
    min-width: 0 !important;
}
footer {
    display: none !important;
}
@media (max-width: 1024px) {
    .responsive-height {
        height: auto !important;
        flex-direction: column !important;
    }
    .fill-height {
        width: 100% !important;
        flex-direction: column !important;
    }
    .divider {
        display: block;
    }
}
"""

with gr.Blocks(title="LiberVance AI", css=css, fill_height=True, 
               theme=gr.themes.Default(primary_hue=gr.themes.colors.red, secondary_hue=gr.themes.colors.pink)
               ) as demo:
    
    # # Tab 1: LV-Search
    # with gr.Tab("🔎 LV-Search"):
    #     gr.Markdown("""
    #         <center><h1>🔎 LiberVance Search</h1></center>
    #         <h4>
    #             LiberVance Search (LV-Search) is an advanced AI assistant powered by large language model (LLM) agents with internet search capabilities. 
    #             It can understand your questions and fetch up-to-date information from the web, 
    #             helping you stay informed and get accurate answers in real time.
    #         </h4>
    #         """,
    #         elem_classes=["title"])
    #     # Input/Output components
    #     with gr.Column(elem_classes=["responsive-height"]):
    #         lvsearch_chatbot = gr.Chatbot(label="Chatbot", type="messages", elem_classes=["extend-height"])
    #         lvsearch_user_input = gr.Textbox(label="Enter your query here", placeholder="e.g., What is the capital of Japan?", lines=3)
    #         lvsearch_submit_btn = gr.Button("Submit", variant="primary")
    #         lvsearch_reset_btn = gr.Button("Reset", variant="secondary")
    #     # Event listeners
    #     lvsearch_user_input.submit(lvsearch.handle_query, inputs=[lvsearch_user_input], outputs=[lvsearch_chatbot], preprocess=False)
    #     lvsearch_submit_btn.click(lvsearch.handle_query, inputs=[lvsearch_user_input], outputs=[lvsearch_chatbot], preprocess=False)
    #     lvsearch_reset_btn.click(lvsearch.reset_session, inputs=[], outputs=[lvsearch_user_input, lvsearch_chatbot], preprocess=False)

    # Tab 2: LV-RAG
    with gr.Tab("📄 LV-RAG"):
        gr.Markdown("""
            <center><h1>📄 LiberVance RAG</h1></center>
            <h4>
                LiberVance RAG (LV-RAG) is a general-purpose Retrieval-Augmented Generation (RAG) system designed for dynamic and flexible use. 
                Upload one or more PDFs, and the system will answer your queries based on their content. 
                You can also choose your preferred LLM in real time to customize your experience.
            </h4>
            """,
            elem_classes=["title"])
        with gr.Row(elem_classes=["responsive-height"]):
            # Input column
            with gr.Column(elem_classes=["fill-height"]):
                gr.Markdown(
                    "<div class='instructions'>"
                    "📊 <b>Upload your PDFs and ask questions</b><br>"
                    "1. Upload one or more PDF files containing your documents (e.g., reports, manuals, contracts, presentations).<br>"
                    "2. Ask any question in natural language—no technical terms required.<br>"
                    "3. The AI will read through the content and provide accurate answers, summaries, or extracted information."
                    "</div>")
                lvrag_pdf_upload = gr.Files(label="Upload file(s) (PDF only)", file_types=[".pdf"], elem_classes=["extend-height"])
                lvrag_user_input = gr.Textbox(label="Enter your query here", placeholder="e.g., Summarize the key points from this document.", lines=3)
                with gr.Row():
                    lvrag_submit_btn = gr.Button("Submit", variant="primary")
                    lvrag_reset_btn = gr.Button("Reset", variant="secondary")
            # Output column
            with gr.Column(elem_classes=["fill-height"]):
                lvrag_dropdown = gr.Dropdown(list(lvrag.MODELS.keys()), label="Select Model", value="GPT-4")
                lvrag_chatbot = gr.Chatbot(label="Chatbot", type="messages", elem_classes=["extend-height"])
        # Event listeners
        lvrag_pdf_upload.change(fn=lvrag.handle_pdf_upload, inputs=[lvrag_pdf_upload], outputs=[])
        lvrag_dropdown.input(fn=lvrag.change_model, inputs=[lvrag_dropdown], outputs=[])
        lvrag_user_input.submit(lvrag.handle_query, inputs=[lvrag_user_input], outputs=[lvrag_chatbot])
        lvrag_submit_btn.click(lvrag.handle_query, inputs=[lvrag_user_input], outputs=[lvrag_chatbot])
        lvrag_reset_btn.click(lvrag.reset_session, inputs=[], outputs=[lvrag_chatbot, lvrag_user_input])
        
    # # Tab 3: LV-RAG-X
    # with gr.Tab("📈 LV-RAG-X"):
    #     gr.Markdown("""
    #         <center><h1>📈 LiberVance RAG-X</h1></center>
    #         <h4>
    #             LiberVance RAG-X (LV-RAG-X) is a general-purpose Retrieval-Augmented Generation (RAG) system designed to handle structured data from Excel and PDF files. 
    #             Users can upload one or more documents and interact with the system using natural language. 
    #             LV-RAG-X parses and integrates tabular and textual information to generate context-aware responses, extract structured outputs, and support data-driven analysis across various domains.
    #         </h4>
    #         """,
    #         elem_classes=["title"])
    #     with gr.Row(elem_classes=["responsive-height"]):
    #         # Input column
    #         with gr.Column(elem_classes=["fill-height"]):
    #             gr.Markdown(
    #                 "<div class='instructions'>"
    #                 "📊 <b>Upload your documents and ask questions</b><br>"
    #                 "1. Upload Excel or PDF files containing your data (e.g., reports, logs, tables, records).<br>"
    #                 "2. Ask any question in natural language—no technical jargon needed.<br>"
    #                 "3. The AI will analyze your data and respond with relevant insights, summaries, or structured outputs."
    #                 "</div>")
    #             lvragx_file_upload = gr.Files(label="Upload file(s) (PDF or Excel)", file_types=[".pdf", ".xlsx"], elem_classes=["extend-height"])
    #             lvragx_user_input = gr.Textbox(label="Enter your query here", placeholder=(
    #                 "e.g.,\n"
    #                 "1. Show the table of contents, data summary, or specific sheet/table from the uploaded files.\n"
    #                 "2. Identify when a key metric (e.g., inventory level, budget, or capacity) will reach a threshold or run out.\n"
    #                 "3. Recommend an action or timeline (e.g., reorder date, resource allocation, deadline) based on trends in the data."
    #             ), lines=8)
    #             with gr.Row():
    #                 lvragx_send_btn = gr.Button("Submit", variant="primary")
    #                 lvragx_reset_btn = gr.Button("Reset", variant="secondary")
    #         # Output column
    #         with gr.Column(elem_classes=["fill-height"]):
    #             lvragx_chatbot = gr.Chatbot(elem_classes=["extend-height"], type="messages")
    #             lvragx_file_download = gr.File(label="Download file", visible=False, elem_classes=["extend-height"])
    #     # Event listeners
    #     lvragx_user_input.submit(lvragx.handle_query, inputs=[lvragx_user_input, lvragx_file_upload], 
    #                     outputs=[lvragx_chatbot, lvragx_file_download])
    #     lvragx_send_btn.click(lvragx.handle_query, inputs=[lvragx_user_input, lvragx_file_upload], 
    #                 outputs=[lvragx_chatbot, lvragx_file_download])
    #     lvragx_reset_btn.click(lvragx.reset_session, inputs=[], outputs=[lvragx_user_input, lvragx_chatbot])
        
    # # Tab 4: LV-VQA
    # with gr.Tab("🖼️ LV-VQA"):
    #     gr.Markdown("""
    #         <center><h1>🖼️ LiberVance VQA</h1></center>
    #         <h4>
    #             LiberVance VQA (LV-VQA) is a ...
    #         </h4>
    #         """,
    #         elem_classes=["title"])
    #     with gr.Row(elem_classes=["responsive-height"]):
    #         # Output column
    #         with gr.Column(elem_classes=["fill-height"]):
    #             lvvqa_chatbot = gr.Chatbot(label="Chatbot", type="messages", elem_classes=["extend-height"])
    #         # Input column
    #         with gr.Column(elem_classes=["fill-height"]):
    #             lvvqa_dropdown = gr.Dropdown(list(lvvqa.MODELS.keys()), label="Select Model", value="GPT-4")
    #             lvvqa_img_upload = gr.Image(label="Upload image", elem_classes=["extend-height"])
    #             lvvqa_user_input = gr.Textbox(label="Enter your query", placeholder="e.g., Summarize the key points from this document.", lines=3)
    #             with gr.Row():
    #                 lvvqa_submit_btn = gr.Button("Submit", variant="primary")
    #                 lvvqa_reset_btn = gr.Button("Reset", variant="secondary")
    #     # Event listeners
    #     lvvqa_dropdown.input(fn=lvvqa.change_model, inputs=[lvvqa_dropdown], outputs=[])
    #     lvvqa_user_input.submit(lvvqa.handle_query, inputs=[lvvqa_user_input, lvvqa_img_upload], outputs=[lvvqa_chatbot])
    #     lvvqa_submit_btn.click(lvvqa.handle_query, inputs=[lvvqa_user_input, lvvqa_img_upload], outputs=[lvvqa_chatbot])
    #     lvvqa_reset_btn.click(lvvqa.reset_session, inputs=[], outputs=[lvvqa_user_input, lvvqa_chatbot])

demo.launch(
    # server_name="0.0.0.0",
    # server_port=7860,
    share=True,
    share_server_address="ai.libervance.com:7000",
    favicon_path="",
)
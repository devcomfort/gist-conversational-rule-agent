#!/usr/bin/env python3
"""
🔍 GIST Rules Analyzer - 로그 뷰어 (독립 실행)
채팅 로그와 성능 로그를 분석하고 시각화하는 전용 앱
"""

import gradio as gr
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# CSS 스타일링
css = """
/* 로그 뷰어 전용 스타일 */
.log-viewer-container {
    max-height: 600px;
    overflow-y: auto;
    padding: 16px;
    border: 2px solid #e0e0e0;
    border-radius: 12px;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
}

.log-entry-box {
    background: linear-gradient(135deg, #ffffff 0%, #f1f3f4 100%);
    border: 1px solid #dadce0;
    border-radius: 8px;
    padding: 12px;
    margin: 8px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.log-stats-box {
    background: linear-gradient(135deg, #1a237e 0%, #283593 100%) !important;
    border: 2px solid #3f51b5 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    color: #ffffff !important;
    font-weight: 500 !important;
    box-shadow: 0 4px 12px rgba(63, 81, 181, 0.3) !important;
}

.log-stats-box h1, .log-stats-box h2, .log-stats-box h3 {
    color: #e3f2fd !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3) !important;
}

.log-stats-box p, .log-stats-box strong {
    color: #ffffff !important;
}

.metric-highlight {
    background: rgba(255,255,255,0.2);
    padding: 4px 8px;
    border-radius: 4px;
    display: inline-block;
    margin: 2px;
}

.file-selector {
    background: #f8f9fa;
    border: 2px solid #dee2e6;
    border-radius: 8px;
    padding: 16px;
}
"""


# --------- 로그 분석 함수들 ---------
def get_log_files():
    """사용 가능한 로그 파일 목록 반환"""
    chat_logs_dir = Path("chat_logs")
    performance_logs_dir = Path("performance_logs")

    log_files = []

    # 채팅 로그 파일들
    if chat_logs_dir.exists():
        for log_file in sorted(
            chat_logs_dir.glob("chat_session_*.jsonl"), reverse=True
        ):
            log_files.append(f"📝 {log_file.name} (채팅 로그)")

    # 성능 로그 파일들
    if performance_logs_dir.exists():
        for log_file in sorted(
            performance_logs_dir.glob("faiss_performance_*.jsonl"), reverse=True
        ):
            log_files.append(f"⚡ {log_file.name} (성능 로그)")

    return log_files if log_files else ["📁 사용 가능한 로그 파일이 없습니다."]


def analyze_log_stats(log_path: Path):
    """로그 파일의 통계 정보를 분석"""
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        stats = {
            "total_entries": len(lines),
            "file_size_mb": log_path.stat().st_size / (1024 * 1024),
            "created": datetime.fromtimestamp(log_path.stat().st_ctime),
            "modified": datetime.fromtimestamp(log_path.stat().st_mtime),
        }

        # 채팅 로그 특화 분석
        if "chat_session" in log_path.name:
            total_queries = 0
            total_response_length = 0
            rerank_methods = {}
            models_used = {}

            for line in lines:
                try:
                    entry = json.loads(line.strip())
                    if "interaction" in entry:
                        total_queries += 1
                        total_response_length += entry["interaction"]["response_length"]

                        rerank = entry["rerank_info"]["method_name"]
                        rerank_methods[rerank] = rerank_methods.get(rerank, 0) + 1

                        model = entry["model_info"]["model_id"]
                        models_used[model] = models_used.get(model, 0) + 1
                except:
                    continue

            stats.update(
                {
                    "total_queries": total_queries,
                    "avg_response_length": total_response_length
                    // max(total_queries, 1),
                    "rerank_methods": rerank_methods,
                    "models_used": models_used,
                }
            )

        return stats
    except Exception as e:
        return {"error": str(e)}


def load_log_content(selected_log_file: str, show_entries: int = 20):
    """선택된 로그 파일의 내용을 읽어서 포맷된 텍스트로 반환"""
    if not selected_log_file or "사용 가능한 로그 파일이 없습니다" in selected_log_file:
        return "📝 로그 파일을 선택해주세요."

    try:
        # 파일명 추출
        file_name = selected_log_file.split(" ")[1]

        # 파일 경로 결정
        if "채팅 로그" in selected_log_file:
            log_path = Path("chat_logs") / file_name
        elif "성능 로그" in selected_log_file:
            log_path = Path("performance_logs") / file_name
        else:
            return "❌ 알 수 없는 로그 파일 형식입니다."

        if not log_path.exists():
            return f"❌ 로그 파일을 찾을 수 없습니다: {log_path}"

        # 통계 정보 분석
        stats = analyze_log_stats(log_path)

        # 로그 내용 읽기
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            return "📝 로그 파일이 비어있습니다."

        # 헤더 생성
        formatted_content = []
        formatted_content.append(f"# 📊 로그 분석: {file_name}")
        formatted_content.append("## 📈 **파일 정보**")
        formatted_content.append(f"- 📁 **경로**: `{log_path}`")
        formatted_content.append(f"- 📊 **총 항목 수**: {len(lines)}개")
        formatted_content.append(
            f"- 💾 **파일 크기**: {stats.get('file_size_mb', 0):.2f}MB"
        )
        formatted_content.append(
            f"- 📅 **생성**: {stats.get('created', 'Unknown').strftime('%Y-%m-%d %H:%M:%S') if 'created' in stats else 'Unknown'}"
        )

        # 채팅 로그 특화 통계
        if "chat_session" in file_name and "total_queries" in stats:
            formatted_content.append("## 🔍 **채팅 통계**")
            formatted_content.append(f"- 💬 **총 대화 수**: {stats['total_queries']}회")
            formatted_content.append(
                f"- 📝 **평균 응답 길이**: {stats['avg_response_length']}자"
            )

            if stats["rerank_methods"]:
                formatted_content.append("- 🔄 **리랭킹 방법 사용 빈도**:")
                for method, count in stats["rerank_methods"].items():
                    formatted_content.append(f"  - {method}: {count}회")

            if stats["models_used"]:
                formatted_content.append("- 🧠 **사용된 모델**:")
                for model, count in stats["models_used"].items():
                    formatted_content.append(f"  - {model}: {count}회")

        formatted_content.append(
            f"\n---\n## 📋 **최근 {min(show_entries, len(lines))}개 항목**\n"
        )

        # 로그 항목들 표시
        for i, line in enumerate(lines[-show_entries:], 1):
            try:
                log_entry = json.loads(line.strip())

                if "interaction" in log_entry:
                    # 채팅 로그 포맷팅
                    timestamp = (
                        log_entry.get("timestamp", "").replace("T", " ").split(".")[0]
                    )
                    query = (
                        log_entry["interaction"]["user_query"][:100] + "..."
                        if len(log_entry["interaction"]["user_query"]) > 100
                        else log_entry["interaction"]["user_query"]
                    )
                    response_length = log_entry["interaction"]["response_length"]
                    rerank_method = log_entry["rerank_info"]["method_name"]
                    model = log_entry["model_info"]["model_id"]

                    formatted_content.append(
                        f"### 💬 **채팅 #{len(lines) - show_entries + i}**"
                    )
                    formatted_content.append(f"🕐 **시간**: {timestamp}")
                    formatted_content.append(f"❓ **질문**: {query}")
                    formatted_content.append(f"📏 **답변 길이**: {response_length}자")
                    formatted_content.append(f"🔄 **리랭킹**: {rerank_method}")
                    formatted_content.append(f"🧠 **모델**: {model}")

                    # 성능 지표
                    if (
                        "performance_metrics" in log_entry
                        and log_entry["performance_metrics"]
                    ):
                        metrics = log_entry["performance_metrics"]
                        perf_info = []
                        if "total_time" in metrics:
                            perf_info.append(f"⏱️ {metrics['total_time']:.2f}초")
                        if "tokens_per_second" in metrics:
                            perf_info.append(
                                f"🚀 {metrics['tokens_per_second']:.1f} tokens/s"
                            )
                        if perf_info:
                            formatted_content.append(
                                f"📊 **성능**: {' | '.join(perf_info)}"
                            )

                elif "results" in log_entry:
                    # 성능 로그 포맷팅
                    timestamp = (
                        log_entry.get("timestamp", "").replace("T", " ").split(".")[0]
                    )
                    query_hash = log_entry.get("query_hash", "")
                    rerank_method = log_entry["results"]["rerank_method"]

                    formatted_content.append(
                        f"### ⚡ **성능 측정 #{len(lines) - show_entries + i}**"
                    )
                    formatted_content.append(f"🕐 **시간**: {timestamp}")
                    formatted_content.append(f"🔑 **쿼리 해시**: {query_hash}")
                    formatted_content.append(f"🔄 **리랭킹**: {rerank_method}")

                    # FAISS 성능 결과
                    if "faiss_performance" in log_entry["results"]:
                        formatted_content.append("🏃 **FAISS 성능**:")
                        faiss_results = log_entry["results"]["faiss_performance"]
                        for index_name, result in faiss_results.items():
                            if result.get("success"):
                                formatted_content.append(
                                    f"  - **{index_name}**: {result['search_time_ms']:.2f}ms"
                                )

                formatted_content.append("---")

            except (json.JSONDecodeError, KeyError) as e:
                formatted_content.append(
                    f"❌ **로그 항목 파싱 오류** (라인 {len(lines) - show_entries + i}): {e}"
                )
                continue

        return "\n".join(formatted_content)

    except Exception as e:
        return f"❌ 로그 파일 읽기 오류: {str(e)}"


def refresh_log_list():
    """로그 파일 목록을 새로고침"""
    return gr.Dropdown(choices=get_log_files())


# --------- Gradio UI ---------
def create_log_viewer_app():
    """로그 뷰어 앱 생성"""

    with gr.Blocks(
        title="GIST Rules Analyzer - Log Viewer", css=css, fill_height=True
    ) as app:
        # 헤더
        gr.Markdown("""
        <center>
        <h1>📊 GIST Rules Analyzer - 로그 뷰어</h1>
        <p><strong>채팅 로그와 성능 로그를 분석하고 시각화</strong></p>
        <p style='color: #666;'>메인 RAG 시스템의 모든 로깅 데이터를 한 곳에서 확인하세요</p>
        </center>
        """)

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["file-selector"]):
                gr.Markdown("### 🔍 **로그 파일 선택**")

                log_file_dropdown = gr.Dropdown(
                    choices=get_log_files(),
                    label="📁 로그 파일",
                    value=None,
                    allow_custom_value=False,
                    info="채팅 로그 또는 성능 로그를 선택하세요",
                )

                with gr.Row():
                    refresh_btn = gr.Button(
                        "🔄 목록 새로고침", variant="secondary", size="sm"
                    )

                show_entries_slider = gr.Slider(
                    minimum=5,
                    maximum=50,
                    value=20,
                    step=5,
                    label="표시할 항목 수",
                    info="최근 몇 개의 로그 항목을 표시할지 선택",
                )

                # 퀴 실행 버튼들
                with gr.Row():
                    chat_logs_btn = gr.Button("💬 채팅 로그만", size="sm")
                    perf_logs_btn = gr.Button("⚡ 성능 로그만", size="sm")

            with gr.Column(scale=3):
                gr.Markdown("### 📋 **로그 내용**")

                log_content_display = gr.Markdown(
                    value="📝 왼쪽에서 로그 파일을 선택해주세요.",
                    elem_classes=["log-stats-box"],
                )

        # 푸터 정보
        with gr.Row():
            gr.Markdown("""
            ---
            <center>
            <p style='color: #888; font-size: 0.9em;'>
            💡 <strong>팁</strong>: 로그 파일이 업데이트되면 '🔄 목록 새로고침' 버튼을 눌러주세요<br/>
            🔗 메인 RAG 시스템은 <a href="http://localhost:7860" target="_blank">localhost:7860</a>에서 실행 중입니다
            </p>
            </center>
            """)

        # 이벤트 핸들러
        log_file_dropdown.change(
            fn=load_log_content,
            inputs=[log_file_dropdown, show_entries_slider],
            outputs=[log_content_display],
        )

        show_entries_slider.change(
            fn=load_log_content,
            inputs=[log_file_dropdown, show_entries_slider],
            outputs=[log_content_display],
        )

        refresh_btn.click(
            fn=refresh_log_list,
            outputs=[log_file_dropdown],
        )

        # 퀵 필터 버튼들
        def filter_chat_logs():
            files = get_log_files()
            chat_files = [f for f in files if "(채팅 로그)" in f]
            if chat_files:
                return gr.Dropdown(choices=files, value=chat_files[0])
            return gr.Dropdown(choices=files)

        def filter_perf_logs():
            files = get_log_files()
            perf_files = [f for f in files if "(성능 로그)" in f]
            if perf_files:
                return gr.Dropdown(choices=files, value=perf_files[0])
            return gr.Dropdown(choices=files)

        chat_logs_btn.click(
            fn=filter_chat_logs,
            outputs=[log_file_dropdown],
        )

        perf_logs_btn.click(
            fn=filter_perf_logs,
            outputs=[log_file_dropdown],
        )

    return app


# --------- 메인 실행 ---------
if __name__ == "__main__":
    print("🔍 GIST Rules Analyzer - 로그 뷰어 시작!")
    print("📊 채팅 로그와 성능 로그를 분석합니다...")

    # 로그 디렉토리 확인
    chat_logs_dir = Path("chat_logs")
    performance_logs_dir = Path("performance_logs")

    if not chat_logs_dir.exists():
        chat_logs_dir.mkdir(exist_ok=True)
        print(f"📁 채팅 로그 디렉토리 생성: {chat_logs_dir}")

    if not performance_logs_dir.exists():
        performance_logs_dir.mkdir(exist_ok=True)
        print(f"📁 성능 로그 디렉토리 생성: {performance_logs_dir}")

    # 로그 파일 개수 확인
    chat_files = list(chat_logs_dir.glob("chat_session_*.jsonl"))
    perf_files = list(performance_logs_dir.glob("faiss_performance_*.jsonl"))

    print(f"📝 발견된 채팅 로그: {len(chat_files)}개")
    print(f"⚡ 발견된 성능 로그: {len(perf_files)}개")

    if len(chat_files) == 0 and len(perf_files) == 0:
        print("⚠️  로그 파일이 없습니다. 메인 RAG 앱에서 채팅을 해보세요!")
        print("🔗 메인 앱: http://localhost:7860")

    app = create_log_viewer_app()

    print("🎉 로그 뷰어 준비완료!")
    print("🌐 http://localhost:7862 에서 실행 중...")
    print("💡 메인 RAG 앱과 함께 사용하세요 (localhost:7860)")

    app.launch(server_name="0.0.0.0", server_port=7862, share=False, show_error=True)

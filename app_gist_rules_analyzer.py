"""
GIST Rules Analyzer - 지스트 학칙 규정 자동 분석 시스템

이 모듈은 GIST의 모든 학칙과 규정 문서를 자동으로 처리하여
지능형 질의응답을 제공하는 시스템입니다.

주요 특징:
- 프로젝트 내 모든 PDF 파일 자동 스캔 및 처리
- 다양한 rerank 방법 동시 비교 분석
- 실시간 성능 지표 모니터링
- 스트리밍 방식 실시간 응답
- Markdown 형태 결과 내보내기

자동 처리 기능:
- 시스템 시작 시 모든 PDF 파일 자동 탐지
- 백그라운드에서 자동 벡터화 처리
- 진행 상황 실시간 모니터링
- 처리 완료 후 즉시 사용 가능

기술 스택:
- 임베딩: sentence-transformers/all-MiniLM-L6-v2
- 벡터 스토어: FAISS
- 텍스트 분할: RecursiveCharacterTextSplitter
- PDF 처리: PyMuPDF (fitz)
- UI: Gradio with custom CSS
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
import glob
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
from typing import Dict, Generator, List, Optional
import faiss
import numpy as np
from pathlib import Path

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
shared_state: Dict = {
    "current_model": "GPT-4",
    "vectorstore": None,
    "pdfs": [],
    "auto_processing_complete": False,
    "processing_status": "📚 RAG 시스템 초기화 중... (GIST 규정집 로딩)",
    "processed_count": 0,
    "total_count": 0,
    "faiss_comparator": None,
    "performance_logger": None,  # 시작 시 초기화
}
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

system_prompt = """You are a GIST Rules and Regulations Expert Assistant. 
You have comprehensive knowledge of all GIST academic rules, regulations, guidelines, and policies.
Always provide accurate, detailed answers based on the provided context.
When answering questions about GIST rules, cite specific regulation numbers and titles when available."""


# --------- (A) PERFORMANCE LOGGING & FAISS INDEX COMPARISON ---------
class PerformanceLogger:
    """성능 측정 결과를 JSON 파일로 기록하는 클래스"""

    def __init__(self, log_dir: str = "performance_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = (
            self.log_dir
            / f"faiss_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )

    def log_query_performance(self, query: str, results: Dict):
        """단일 쿼리의 성능 결과를 로깅"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "query_hash": hashlib.md5(query.encode()).hexdigest()[:8],
            "results": results,
        }

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        print(f"📊 성능 결과 기록: {self.log_file}")


class FaissIndexComparator:
    """다양한 FAISS 인덱스 타입별 성능 비교"""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.indexes = {}
        self.performance_logger = PerformanceLogger()

    def create_indexes_from_vectorstore(self, vectorstore: FAISS) -> Dict:
        """기존 벡터스토어에서 다양한 인덱스 타입 생성"""
        print("🔧 다양한 FAISS 인덱스 타입 생성 중...")

        # 기존 벡터와 문서 추출
        vectors = vectorstore.index.reconstruct_n(0, vectorstore.index.ntotal)

        results = {}

        # 1. IndexFlatL2 (기본값 - 이미 존재)
        start_time = time.time()
        flat_index = faiss.IndexFlatL2(self.dimension)
        flat_index.add(vectors)
        results["IndexFlatL2"] = {
            "creation_time": time.time() - start_time,
            "index": flat_index,
            "memory_usage": flat_index.d * flat_index.ntotal * 4,  # float32
        }

        # 2. IndexIVFFlat (클러스터 기반 근사 검색)
        if vectorstore.index.ntotal > 100:  # 최소 데이터 필요
            start_time = time.time()
            nlist = min(100, int(np.sqrt(vectorstore.index.ntotal)))  # 클러스터 수
            quantizer = faiss.IndexFlatL2(self.dimension)
            ivf_index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            ivf_index.train(vectors)
            ivf_index.add(vectors)
            ivf_index.nprobe = min(10, nlist)  # 검색할 클러스터 수
            results["IndexIVFFlat"] = {
                "creation_time": time.time() - start_time,
                "index": ivf_index,
                "nlist": nlist,
                "nprobe": ivf_index.nprobe,
                "memory_usage": ivf_index.d * ivf_index.ntotal * 4
                + nlist * ivf_index.d * 4,
            }

        # 3. IndexHNSWFlat (그래프 기반 고속 검색)
        if vectorstore.index.ntotal > 50:
            start_time = time.time()
            hnsw_index = faiss.IndexHNSWFlat(self.dimension, 32)  # M=32
            hnsw_index.hnsw.efConstruction = 200
            hnsw_index.hnsw.efSearch = 64
            hnsw_index.add(vectors)
            results["IndexHNSWFlat"] = {
                "creation_time": time.time() - start_time,
                "index": hnsw_index,
                "M": 32,
                "efConstruction": 200,
                "efSearch": 64,
                "memory_usage": hnsw_index.d * hnsw_index.ntotal * 4
                + hnsw_index.ntotal * 200,  # 추정
            }

        # 결과 저장
        self.indexes = {name: data["index"] for name, data in results.items()}

        print(f"✅ {len(results)}개 인덱스 타입 준비 완료")
        return results

    def compare_search_performance(self, query_vector: np.ndarray, k: int = 3) -> Dict:
        """모든 인덱스에서 검색 성능 비교"""
        results = {}

        for index_name, index in self.indexes.items():
            start_time = time.time()

            try:
                distances, indices = index.search(query_vector.reshape(1, -1), k)
                search_time = time.time() - start_time
                results[index_name] = {
                    "search_time_ms": search_time * 1000,  # 밀리초로 변환
                    "distances": distances.tolist(),
                    "indices": indices.tolist(),
                    "success": True,
                }

            except Exception as e:
                results[index_name] = {
                    "search_time_ms": 0,
                    "error": str(e),
                    "success": False,
                }
                print(f"⚠️ {index_name} 검색 실패: {e}")

        return results


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


def scan_pdf_files() -> List[str]:
    """프로젝트 디렉토리에서 모든 PDF 파일을 스캔"""
    print("🔍 Scanning for PDF files in project directory...")

    # 현재 디렉토리와 하위 디렉토리에서 모든 PDF 파일 찾기
    pdf_files = []

    # glob을 사용해서 재귀적으로 PDF 파일 검색
    pdf_patterns = [
        "*.pdf",
        "**/*.pdf",
        "rules/**/*.pdf",
        "documents/**/*.pdf",
        "data/**/*.pdf",
    ]

    for pattern in pdf_patterns:
        found_files = glob.glob(pattern, recursive=True)
        for file in found_files:
            if file not in pdf_files and os.path.exists(file):
                pdf_files.append(file)

    print(f"📄 Found {len(pdf_files)} PDF files")

    # 파일들을 크기순으로 정렬 (작은 파일부터 처리)
    pdf_files.sort(key=lambda x: os.path.getsize(x) if os.path.exists(x) else 0)

    # 대용량 파일 필터링 (50MB 이상 파일은 경고)
    large_files = []
    filtered_files = []

    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):
            file_size = os.path.getsize(pdf_file)
            if file_size > 50 * 1024 * 1024:  # 50MB
                large_files.append((pdf_file, file_size))
            else:
                filtered_files.append(pdf_file)
        else:
            print(f"⚠️ File not found during scan: {pdf_file}")

    if large_files:
        print(f"📊 Found {len(large_files)} large files (>50MB):")
        for file_path, size in large_files[:5]:  # 처음 5개만 표시
            size_mb = size / (1024 * 1024)
            print(f"   - {os.path.basename(file_path)}: {size_mb:.1f}MB")
        if len(large_files) > 5:
            print(f"   ... and {len(large_files) - 5} more large files")

        # 큰 파일들도 포함하되, 나중에 처리
        filtered_files.extend([f[0] for f in large_files])

    print(f"📄 Total files to process: {len(filtered_files)}")
    return filtered_files


def auto_process_pdfs():
    """백그라운드에서 모든 PDF 파일을 자동 처리"""
    print("🚀 Starting automatic PDF processing...")

    try:
        # PDF 파일 스캔 - 모든 파일 처리 (제한 없음)
        pdf_files = scan_pdf_files()

        print(f"📚 전체 GIST 규정집 처리: {len(pdf_files)}개 PDF 파일")
        print("🎯 완전한 RAG 시스템 구축을 위해 모든 문서를 처리합니다")

        with shared_state_lock:
            shared_state["total_count"] = len(pdf_files)
            shared_state["processed_count"] = 0
            shared_state["processing_status"] = (
                f"📖 **전체 Document Processing 시작**\n\n"
                f"📚 **처리 대상**: {len(pdf_files)}개 모든 PDF 파일\n"
                f"🎯 완전한 GIST 규정집 데이터베이스 구축 중"
            )

        if not pdf_files:
            with shared_state_lock:
                shared_state["processing_status"] = (
                    "⚠️ 규정집을 찾을 수 없어요. 파일이 있는지 확인해주세요"
                )
                shared_state["auto_processing_complete"] = True
            return

        # 문서 처리
        print("📝 Processing PDF documents...")
        all_docs = []

        successfully_processed = 0
        failed_files = []

        for i, pdf_file in enumerate(pdf_files):
            try:
                with shared_state_lock:
                    shared_state["processing_status"] = (
                        f"📄 {os.path.basename(pdf_file)} 문서를 읽는 중... ({i + 1}/{len(pdf_files)})"
                    )

                print(
                    f"\r📄 처리 중 [{i + 1}/{len(pdf_files)}]: {os.path.basename(pdf_file)}",
                    end="",
                    flush=True,
                )

                # 파일 존재 확인
                if not os.path.exists(pdf_file):
                    print(f"⚠️ File not found: {pdf_file}")
                    failed_files.append(f"{os.path.basename(pdf_file)} (파일 없음)")
                    continue

                # 파일 크기에 따른 처리 방식 결정
                file_size = os.path.getsize(pdf_file)
                is_large_file = file_size > 10 * 1024 * 1024  # 10MB
                is_very_large_file = file_size > 20 * 1024 * 1024  # 20MB

                # PDF 텍스트 추출
                doc = fitz.open(pdf_file)
                text_pages = []

                # 📈 파일 크기별 최적화된 처리
                page_count = doc.page_count
                if is_very_large_file:
                    batch_size = 3  # 매우 큰 파일: 3페이지씩
                elif is_large_file:
                    batch_size = 8  # 큰 파일: 8페이지씩
                else:
                    batch_size = min(page_count, 20)  # 작은 파일: 최대 20페이지씩

                for batch_start in range(0, page_count, batch_size):
                    batch_end = min(batch_start + batch_size, page_count)
                    batch_pages = []

                    for page_num in range(batch_start, batch_end):
                        try:
                            page = doc[page_num]
                            page_text = page.get_text("text")
                            if page_text.strip():
                                batch_pages.append(page_text)

                            # 메모리 절약을 위해 페이지 객체 정리
                            del page

                        except Exception as page_error:
                            print(
                                f"⚠️ Error reading page {page_num + 1} of {pdf_file}: {page_error}"
                            )
                            continue

                    if batch_pages:
                        text_pages.extend(batch_pages)

                    # 대용량 파일 처리 시 진행 상황 업데이트
                    if is_large_file and batch_end < page_count:
                        progress = (batch_end / page_count) * 100
                        with shared_state_lock:
                            shared_state["processing_status"] = (
                                f"📘 큰 문서를 차근차근 읽는 중: {os.path.basename(pdf_file)} "
                                f"({i + 1}/{len(pdf_files)}) - 진행률: {progress:.1f}%"
                            )

                doc.close()
                text = "\n".join(text_pages)

                # 메모리 정리
                del text_pages

                # 명시적 메모리 정리 (gc.collect() 없이)
                if (i + 1) % 20 == 0:
                    print(
                        f"\r🧹 [{i + 1}/{len(pdf_files)}] 메모리 정리 완료    "
                    )  # 공백으로 이전 텍스트 지움

                if text.strip():  # 텍스트가 있는 경우만 처리
                    # 파일 크기에 따른 청크 크기 조정
                    if is_large_file:
                        # 대용량 파일은 더 큰 청크 사용 (메모리 효율성)
                        dynamic_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=800, chunk_overlap=100
                        )
                    else:
                        # 일반 파일은 기본 설정 사용
                        dynamic_splitter = TEXT_SPLITTER

                    document = Document(
                        page_content=text,
                        metadata={
                            "source": pdf_file,
                            "filename": os.path.basename(pdf_file),
                            "category": get_document_category(pdf_file),
                            "page_count": page_count,
                            "file_size": file_size,
                            "is_large_file": is_large_file,
                        },
                    )

                    # 텍스트 분할
                    docs = dynamic_splitter.split_documents([document])
                    all_docs.extend(docs)
                    successfully_processed += 1

                    # 메모리 정리
                    del text, document

                    file_size_mb = file_size / (1024 * 1024)
                    print(
                        f"\r✅ [{i + 1}/{len(pdf_files)}] {os.path.basename(pdf_file)} ({len(docs)} chunks, {file_size_mb:.1f}MB)"
                    )
                else:
                    print(f"⚠️ No text content found in: {pdf_file}")
                    failed_files.append(f"{os.path.basename(pdf_file)} (텍스트 없음)")

                with shared_state_lock:
                    shared_state["processed_count"] = i + 1

            except Exception as e:
                print(f"❌ Error processing {pdf_file}: {e}")
                failed_files.append(
                    f"{os.path.basename(pdf_file)} (처리 오류: {str(e)[:50]}...)"
                )
                continue

        if all_docs:
            # 벡터스토어 생성
            total_chunks = len(all_docs)
            print(f"🔄 Creating vector store from {total_chunks:,} document chunks...")

            # 📈 전체 데이터 처리를 위한 배치 크기 최적화
            if total_chunks > 30000:
                batch_size = 200  # 초대용량: 매우 작은 배치로 안전하게
            elif total_chunks > 20000:
                batch_size = 300  # 대용량: 작은 배치
            elif total_chunks > 15000:
                batch_size = 400  # 중대용량: 중간 배치
            elif total_chunks > 10000:
                batch_size = 500  # 많은 청크: 중간 배치
            elif total_chunks > 5000:
                batch_size = 800  # 중간 청크: 큰 배치
            else:
                batch_size = min(
                    1000, max(200, total_chunks // 10)
                )  # 적은 청크: 최대 배치

            num_batches = (total_chunks + batch_size - 1) // batch_size

            print(
                f"📊 Processing in {num_batches} batches of ~{batch_size} chunks each"
            )

            vectorstore: Optional[FAISS] = None
            vectorization_start = time.time()

            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, total_chunks)
                batch_docs = all_docs[batch_start:batch_end]

                batch_start_time = time.time()

                # 진행 상황 업데이트
                progress_percent = (batch_idx / num_batches) * 100
                elapsed_time = time.time() - vectorization_start

                if batch_idx > 0:  # 첫 배치 이후부터 예상 시간 계산
                    avg_time_per_batch = elapsed_time / batch_idx
                    remaining_batches = num_batches - batch_idx
                    estimated_remaining = avg_time_per_batch * remaining_batches

                    eta_minutes = int(estimated_remaining // 60)
                    eta_seconds = int(estimated_remaining % 60)

                    if eta_minutes > 0:
                        eta_text = f"약 {eta_minutes}분 {eta_seconds}초 남음"
                    else:
                        eta_text = f"약 {eta_seconds}초 남음"

                    # 처리 속도 계산
                    total_processed_chunks = batch_idx * batch_size
                    chunks_per_second = (
                        total_processed_chunks / elapsed_time if elapsed_time > 0 else 0
                    )
                    speed_text = f"{chunks_per_second:.1f} 청크/초"
                else:
                    eta_text = "예상 시간 계산 중..."
                    speed_text = "속도 계산 중..."

                # 시각적 진행률 바 생성
                progress_bar_length = 20
                filled_length = int(progress_percent / 5)  # 5%씩 표시
                progress_bar = "█" * filled_length + "░" * (
                    progress_bar_length - filled_length
                )

                elapsed_minutes = int(elapsed_time // 60)
                elapsed_seconds = int(elapsed_time % 60)
                elapsed_text = (
                    f"{elapsed_minutes}분 {elapsed_seconds}초"
                    if elapsed_minutes > 0
                    else f"{elapsed_seconds}초"
                )

                with shared_state_lock:
                    shared_state["processing_status"] = (
                        f"🧠 **Vector Store 구축 중** (RAG 시스템 준비)\n\n"
                        f"**배치 진행률**: {batch_idx + 1}/{num_batches} ({progress_percent:.1f}%)\n"
                        f"`{progress_bar}` {progress_percent:.1f}%\n\n"
                        f"**처리 중**: {batch_start:,} ~ {batch_end:,} / {total_chunks:,} chunks\n"
                        f"**소요 시간**: {elapsed_text}\n"
                        f"**처리 속도**: {speed_text}\n"
                        f"**예상 완료**: {eta_text}"
                    )

                try:
                    if vectorstore is None:
                        # 첫 번째 배치로 벡터스토어 생성
                        print(
                            f"\r🚀 벡터 스토어 초기화: 배치 1/{num_batches} ({len(batch_docs)} chunks)",
                            end="",
                            flush=True,
                        )

                        # 첫 번째 배치는 시간이 오래 걸릴 수 있으므로 중간 상태 업데이트
                        with shared_state_lock:
                            shared_state["processing_status"] = (
                                f"🚀 **FAISS Index 초기화 중**\n\n"
                                f"**단계**: Initial Vector Store 생성\n"
                                f"**처리 중**: {len(batch_docs)}개 chunks → embedding 변환\n"
                                f"⏳ 첫 번째 배치는 인덱스 초기화로 시간이 더 소요됩니다"
                            )

                        vectorstore = FAISS.from_documents(batch_docs, EMBED_MODEL)
                    else:
                        # 기존 벡터스토어에 배치 추가
                        print(
                            f"\r➕ 벡터 스토어 확장: 배치 {batch_idx + 1}/{num_batches} ({len(batch_docs)} chunks)",
                            end="",
                            flush=True,
                        )
                        batch_vectorstore = FAISS.from_documents(
                            batch_docs, EMBED_MODEL
                        )
                        vectorstore.merge_from(batch_vectorstore)
                        # 메모리 정리
                        del batch_vectorstore

                    batch_time = time.time() - batch_start_time
                    chunks_per_second = (
                        len(batch_docs) / batch_time if batch_time > 0 else 0
                    )
                    print(
                        f"\r✅ 배치 [{batch_idx + 1}/{num_batches}] 완료: {len(batch_docs)} chunks in {batch_time:.1f}s ({chunks_per_second:.1f} chunks/s)"
                    )

                    # 배치 완료 후 진행 상황 업데이트
                    completed_percent = ((batch_idx + 1) / num_batches) * 100
                    completed_chunks = (batch_idx + 1) * batch_size

                    if batch_idx < num_batches - 1:  # 마지막 배치가 아닌 경우
                        # 시각적 진행률 바 업데이트
                        progress_bar = "█" * int(completed_percent / 5) + "░" * (
                            20 - int(completed_percent / 5)
                        )

                        with shared_state_lock:
                            shared_state["processing_status"] = (
                                f"✨ **Vector Store 확장 중**\n\n"
                                f"**완료 배치**: {batch_idx + 1}/{num_batches} ({completed_percent:.1f}%)\n"
                                f"`{progress_bar}` {completed_percent:.1f}%\n\n"
                                f"✅ **처리 완료**: {min(completed_chunks, total_chunks):,} / {total_chunks:,} chunks\n"
                                f"🚀 **벡터화 속도**: {chunks_per_second:.1f} chunks/sec"
                            )

                except Exception as e:
                    print(f"❌ Error in batch {batch_idx + 1}/{num_batches}: {e}")
                    # 배치 크기를 줄여서 재시도
                    if len(batch_docs) > 50:
                        print("🔄 Retrying with smaller batch size...")
                        smaller_batch_size = len(batch_docs) // 2
                        for sub_batch_start in range(
                            0, len(batch_docs), smaller_batch_size
                        ):
                            sub_batch_end = min(
                                sub_batch_start + smaller_batch_size, len(batch_docs)
                            )
                            sub_batch_docs = batch_docs[sub_batch_start:sub_batch_end]

                            try:
                                if vectorstore is None:
                                    vectorstore = FAISS.from_documents(
                                        sub_batch_docs, EMBED_MODEL
                                    )
                                else:
                                    sub_vectorstore = FAISS.from_documents(
                                        sub_batch_docs, EMBED_MODEL
                                    )
                                    vectorstore.merge_from(sub_vectorstore)
                                    del sub_vectorstore
                            except Exception as sub_e:
                                print(f"❌ Sub-batch failed: {sub_e}")
                                continue
                    else:
                        print(
                            f"❌ Skipping batch {batch_idx + 1} due to persistent errors"
                        )
                        continue

                # 메모리 정리
                del batch_docs

                # 5개 배치마다 메모리 정리 알림 (명시적 del만 사용)
                if (batch_idx + 1) % 5 == 0:
                    print(
                        f"\r🧹 배치 [{batch_idx + 1}] 메모리 정리 완료",
                        end="",
                        flush=True,
                    )

            total_vectorization_time = time.time() - vectorization_start

            if vectorstore is None:
                print("❌ Failed to create vector store - all batches failed")
                with shared_state_lock:
                    shared_state["processing_status"] = (
                        "😞 **준비 과정에서 문제가 발생했어요**\n\n"
                        "문서들을 학습하는 중에 오류가 발생했습니다.\n"
                        "메모리가 부족하거나 시스템에 문제가 있을 수 있어요."
                    )
                    shared_state["auto_processing_complete"] = True
                return

            print(
                f"✅ Vector store creation complete! Total time: {total_vectorization_time:.1f}s"
            )
            print(
                f"📊 Average speed: {total_chunks / total_vectorization_time:.1f} chunks/second"
            )

            # 상세한 처리 결과 생성
            success_rate = (
                (successfully_processed / len(pdf_files)) * 100 if pdf_files else 0
            )
            failed_count = len(failed_files)

            # 벡터화 시간 포맷팅
            vectorization_minutes = int(total_vectorization_time // 60)
            vectorization_seconds = int(total_vectorization_time % 60)

            if vectorization_minutes > 0:
                vectorization_time_text = (
                    f"{vectorization_minutes}분 {vectorization_seconds}초"
                )
            else:
                vectorization_time_text = f"{vectorization_seconds}초"

            result_summary = [
                "🎉 **완전한 GIST 규정집 RAG 시스템 구축 완료!**",
                f"📚 **처리된 문서**: {successfully_processed}/{len(pdf_files)}개 ({success_rate:.1f}%)",
                f"🧠 **생성된 Chunks**: {len(all_docs):,}개 (전체 규정집 포함)",
                f"⏱️ **Vector Store 생성 시간**: {vectorization_time_text}",
                "🔍 **모든 GIST 규정에 대해 4가지 Rerank 방식으로 테스트 가능**",
            ]

            if failed_count > 0:
                result_summary.append(
                    f"⚠️ **일부 문제**: {failed_count}개 파일을 읽지 못했어요"
                )
                if failed_count <= 5:  # 실패한 파일이 5개 이하면 모두 표시
                    result_summary.append(
                        f"**문제 파일**: {', '.join(failed_files[:5])}"
                    )
                else:  # 많으면 일부만 표시
                    result_summary.append(
                        f"**문제 파일** (일부): {', '.join(failed_files[:3])}... 외 {failed_count - 3}개"
                    )

            final_status = "\n".join(result_summary)

            with shared_state_lock:
                shared_state["vectorstore"] = vectorstore
                shared_state["pdfs"] = pdf_files

                # ✨ FAISS 인덱스 비교기 & 성능 로거 초기화
                print("🔧 FAISS 성능 비교 시스템 초기화 중...")
                shared_state["performance_logger"] = PerformanceLogger()
                shared_state["faiss_comparator"] = FaissIndexComparator(
                    dimension=DIMENSION
                )

                # 다양한 FAISS 인덱스 타입 생성
                try:
                    index_creation_results = shared_state[
                        "faiss_comparator"
                    ].create_indexes_from_vectorstore(vectorstore)
                    print(
                        f"✅ 성능 비교 준비 완료: {len(index_creation_results)}개 인덱스 타입"
                    )
                except Exception as e:
                    print(f"⚠️ FAISS 인덱스 생성 중 오류: {e}")
                    shared_state["faiss_comparator"] = None

                shared_state["processing_status"] = final_status
                shared_state["auto_processing_complete"] = True

            print("✅ Auto-processing complete!")
            print(
                f"   Successfully processed: {successfully_processed}/{len(pdf_files)} files"
            )
            print(f"   Generated chunks: {len(all_docs)}")
            if failed_count > 0:
                print(f"   Failed files: {failed_count}")
                for failed_file in failed_files[:3]:  # 처음 3개만 로그에 출력
                    print(f"     - {failed_file}")
                if failed_count > 3:
                    print(f"     ... and {failed_count - 3} more")
        else:
            with shared_state_lock:
                shared_state["processing_status"] = (
                    "😔 읽을 수 있는 문서가 없어요. 파일을 확인해주세요"
                )
                shared_state["auto_processing_complete"] = True

    except Exception as e:
        print(f"❌ Auto-processing failed: {e}")
        with shared_state_lock:
            shared_state["processing_status"] = f"😞 문제가 발생했어요: {str(e)}"
            shared_state["auto_processing_complete"] = True


def get_document_category(file_path: str) -> str:
    """파일 경로를 기반으로 문서 카테고리 결정"""
    path_lower = file_path.lower()

    if "학칙" in path_lower or "규정" in path_lower:
        return "학칙·규정"
    elif "지침" in path_lower:
        return "지침·기준"
    elif "운영" in path_lower:
        return "운영·관리"
    elif "연구" in path_lower:
        return "연구·학술"
    elif "학생" in path_lower:
        return "학생·교육"
    else:
        return "기타"


def get_processing_status():
    """현재 처리 상태를 반환"""
    with shared_state_lock:
        if shared_state["auto_processing_complete"]:
            return shared_state["processing_status"]
        else:
            processed = shared_state["processed_count"]
            total = shared_state["total_count"]
            status = shared_state["processing_status"]
            if total > 0:
                progress_percent = processed / total * 100
                progress_bar = "█" * int(progress_percent / 5) + "░" * (
                    20 - int(progress_percent / 5)
                )
                progress_text = f"{processed}/{total} ({progress_percent:.1f}%)"
                return f"{status}\n\n📈 **진행률**: {progress_text}\n`{progress_bar}`"
            return status


def get_processing_status_with_complete_check():
    """처리 상태와 완료 여부를 함께 반환"""
    with shared_state_lock:
        is_complete = shared_state["auto_processing_complete"]
        status = get_processing_status()
        return status, is_complete


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
        doc = Document(
            page_content=text,
            metadata={
                "source": pdf,
                "filename": os.path.basename(pdf) if hasattr(pdf, "name") else str(pdf),
                "category": "업로드된 문서",
            },
        )
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
        return vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )  # GIST 규정이 많으므로 더 많은 문서 검색

    try:
        # Rerank 기능 활성화
        top_k = rerank_config.get("top_k", 3)
        if not isinstance(top_k, int) or top_k <= 0:
            top_k = 3
        initial_k = max(15, top_k * 5)  # 더 많은 초기 검색 결과
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
        return vectorstore.as_retriever(search_kwargs={"k": 5})


def handle_additional_pdf_upload(pdfs, request: gr.Request):
    """추가 PDF 파일 업로드 처리"""
    if not pdfs:
        return get_processing_status()

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

    # 기존 벡터스토어와 병합
    with shared_state_lock:
        existing_vectorstore = shared_state["vectorstore"]
        existing_pdfs = shared_state["pdfs"].copy()

    try:
        print("📄 Processing additional PDF(s)...")
        new_vectorstore = create_vectorstore_from_pdfs(pdfs)

        if existing_vectorstore and new_vectorstore:
            # 기존 벡터스토어와 병합
            print("🔄 Merging with existing documents...")
            existing_vectorstore.merge_from(new_vectorstore)
            merged_vectorstore = existing_vectorstore
            all_pdfs = existing_pdfs + list(pdfs)
        else:
            merged_vectorstore = new_vectorstore or existing_vectorstore
            all_pdfs = list(pdfs) if not existing_vectorstore else existing_pdfs

        with shared_state_lock:
            shared_state["vectorstore"] = merged_vectorstore
            shared_state["pdfs"] = all_pdfs

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"✅ Added {len(pdfs)} PDFs in {elapsed_time:.2f} seconds")

        return f"✅ {len(pdfs)}개 문서 추가 완료! ({elapsed_time:.2f}초) | 총 {len(all_pdfs)}개 문서"

    except Exception as e:
        print(f"❌ Additional PDF upload failed: {e}")
        return f"😞 파일 추가에 실패했어요: {str(e)}"


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
        faiss_comparator = shared_state.get("faiss_comparator")
        performance_logger = shared_state.get("performance_logger")

    model_info = MODELS[current_model]

    # Extract relevant text data from PDFs
    context = ""
    faiss_performance_results = {}

    if vectorstore:
        print(f"🔍 [{rerank_method}] Retrieving relevant GIST rules and regulations...")
        retrieval_start = time.time()

        retriever = create_retriever(vectorstore, rerank_method)
        if retriever:
            docs = retriever.invoke(user_query)
            # 문서 소스 정보 포함
            context_parts: List[str] = []
            for doc in docs:
                source_info = doc.metadata.get(
                    "filename", doc.metadata.get("source", "")
                )
                category = doc.metadata.get("category", "")
                context_parts.append(f"[{category}] {source_info}:\n{doc.page_content}")
            context = "\n\n".join(context_parts)

        retrieval_end = time.time()
        metrics.retrieval_time = retrieval_end - retrieval_start
        print(
            f"📊 [{rerank_method}] Retrieved {len(docs) if 'docs' in locals() else 0} documents in {metrics.retrieval_time:.2f}s"
        )

        # 🚀 FAISS 인덱스 성능 비교 실행
        if faiss_comparator and performance_logger:
            print(f"🔬 [{rerank_method}] FAISS 인덱스 성능 비교 시작...")
            try:
                # 쿼리 벡터 생성
                query_vector = EMBED_MODEL.embed_query(user_query)
                query_vector_np = np.array(query_vector, dtype=np.float32)

                # 모든 인덱스에서 성능 비교
                faiss_performance_results = faiss_comparator.compare_search_performance(
                    query_vector_np, k=3
                )

                # 성능 결과 로깅
                performance_logger.log_query_performance(
                    user_query,
                    {
                        "rerank_method": rerank_method,
                        "faiss_performance": faiss_performance_results,
                        "retrieval_time": metrics.retrieval_time,
                    },
                )

                # 성능 결과 출력
                print(f"📊 [{rerank_method}] FAISS 성능 비교 결과:")
                for index_name, result in faiss_performance_results.items():
                    if result.get("success"):
                        print(f"   • {index_name}: {result['search_time_ms']:.2f}ms")
                    else:
                        print(
                            f"   • {index_name}: 실패 - {result.get('error', 'Unknown error')}"
                        )

            except Exception as e:
                print(f"⚠️ [{rerank_method}] FAISS 성능 비교 중 오류: {e}")
                faiss_performance_results = {"error": str(e)}

    messages.append(
        {
            "role": "user",
            "content": f"Context (GIST Rules & Regulations):\n{context}\n\nQuestion: {user_query}",
        }
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

    # 안전한 문자열 변환
    try:
        lines = [f"**{rerank_method}**"]

        if "time_to_first_token" in metrics:
            lines.append(f"- **첫 토큰**: {metrics['time_to_first_token']:.2f}초")

        if "total_time" in metrics:
            lines.append(f"- **총 시간**: {metrics['total_time']:.2f}초")

        if "tokens_per_second" in metrics:
            lines.append(f"- **속도**: {metrics['tokens_per_second']:.1f} tokens/s")

        if "retrieval_time" in metrics:
            lines.append(f"- **검색 시간**: {metrics['retrieval_time']:.2f}초")

        result = "\n".join(lines)
        # 문자열 타입 확인 및 보장
        return str(result) if result else f"**{rerank_method}** - 데이터 없음"

    except Exception as e:
        print(f"❌ format_metrics 오류: {e}")
        return f"**{rerank_method}** - 형식 오류"


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
    return f"✅ {model_name} 준비완료"


def save_history(history, session_id, rerank_method):
    """대화 기록(history)을 JSON 파일로 저장"""
    folder = "./chat_logs_gist_analyzer"
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

    markdown_content = [f"# GIST Rules Analyzer - {rerank_method} 검색 결과\n"]

    for i, message in enumerate(history):
        if message["role"] == "system":
            continue

        role = "❓ 질문" if message["role"] == "user" else "💡 답변"
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


def handle_multi_query(user_query, request: gr.Request):
    """모든 rerank 모드에서 동시에 쿼리 실행"""
    if not user_query.strip():
        return [[] for _ in RERANK_OPTIONS.keys()] + [
            format_metrics({}, method) for method in RERANK_OPTIONS.keys()
        ]

    print(f"💬 사용자 질문 처리 시작: {user_query[:50]}...")

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

    print("✅ 모든 검색 방식으로 답변 완료!")


# --------- (E) Gradio UI ---------
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
    background: linear-gradient(135deg, #1a237e 0%, #283593 100%) !important;
    border: 2px solid #3f51b5 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin: 12px 0 !important;
    color: #ffffff !important;
    font-weight: 500 !important;
    box-shadow: 0 4px 12px rgba(63, 81, 181, 0.3) !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}
.metrics-box h2 {
    color: #e3f2fd !important;
    font-size: 1.1em !important;
    margin-bottom: 8px !important;
    font-weight: 600 !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3) !important;
}
.metrics-box p {
    color: #ffffff !important;
    margin: 4px 0 !important;
    font-size: 0.95em !important;
    line-height: 1.4 !important;
}
.metrics-box strong {
    color: #e8eaf6 !important;
    font-weight: 600 !important;
}
.metrics-box ul li {
    color: #ffffff !important;
    margin: 4px 0 !important;
}
.status-box {
    background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%) !important;
    border: 2px solid #4caf50 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin: 12px 0 !important;
    color: #2e7d32 !important;
    font-weight: 500 !important;
    box-shadow: 0 2px 8px rgba(76, 175, 80, 0.1) !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}
.status-box h2, .status-box h3 {
    color: #1b5e20 !important;
    margin-bottom: 8px !important;
}
.status-box strong {
    color: #1b5e20 !important;
}
.status-box code {
    background: rgba(76, 175, 80, 0.1) !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-family: 'Courier New', monospace !important;
    color: #2e7d32 !important;
}
.progress-container {
    background: #f5f5f5 !important;
    border-radius: 8px !important;
    padding: 8px !important;
    margin: 8px 0 !important;
}
.auto-refresh-controls {
    background: rgba(33, 150, 243, 0.05) !important;
    border-radius: 8px !important;
    padding: 8px !important;
    margin: 4px 0 !important;
}
footer {
    display: none !important;
}
"""

# 자동 PDF 처리 시작 (백그라운드)
auto_processing_thread = threading.Thread(target=auto_process_pdfs, daemon=True)
auto_processing_thread.start()

with gr.Blocks(title="GIST Rules Analyzer", css=css, fill_height=True) as demo:
    gr.Markdown(
        "<center><h1>📚 GIST Rules Analyzer</h1><p><strong>학술용 RAG 시스템</strong> | 광주과학기술원 학칙·규정 검색 연구</p></center>"
    )

    # 처리 상태 표시
    with gr.Row():
        processing_status = gr.Markdown(
            value=get_processing_status(), elem_classes=["status-box"]
        )

    # 상태 업데이트 함수
    def update_status():
        return get_processing_status()

    # 처리 완료 확인 및 상태 업데이트
    def check_and_update_status():
        status, is_complete = get_processing_status_with_complete_check()
        return (
            status,
            gr.update(visible=not is_complete),  # 자동 새로고침 체크박스
            gr.update(interactive=is_complete),  # 전송 버튼만 제어
        )

    # 새로고침 버튼 추가
    with gr.Row(elem_classes=["auto-refresh-controls"]):
        refresh_btn = gr.Button("🔄 시스템 상태 확인", size="sm")
        auto_refresh_checkbox = gr.Checkbox(
            label="실시간 모니터링 (3초 간격)", value=True, visible=True
        )

    # 자동 새로고침을 위한 숨겨진 버튼
    auto_refresh_trigger = gr.Button(visible=False)

    # 새로고침 이벤트는 나중에 정의

    # JavaScript로 자동 새로고침 구현
    demo.load(
        None,
        None,
        None,
        js="""
        function autoRefresh() {
            if (document.querySelector('input[aria-label="실시간 모니터링 (3초 간격)"]').checked) {
                setTimeout(() => {
                    document.querySelector('button[style*="display: none"]').click();
                    autoRefresh();
                }, 3000);
            } else {
                setTimeout(autoRefresh, 1000);
            }
        }
        setTimeout(autoRefresh, 1000);
        """,
    )

    with gr.Row():
        with gr.Column(scale=2):
            # 공통 컨트롤
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    list(MODELS.keys()),
                    label="🧠 LLM 모델 선택",
                    value="GPT-4",
                    scale=2,
                )
                model_status = gr.Textbox(
                    label="모델 상태",
                    value="✅ GPT-4 준비완료",
                    interactive=False,
                    scale=1,
                )

            additional_pdf_upload = gr.Files(
                label="📄 추가 문서 업로드 (Vector Store 확장)", file_types=[".pdf"]
            )

            pdf_status = gr.Textbox(
                label="📊 시스템 상태", value=get_processing_status(), interactive=False
            )

            user_input = gr.Textbox(
                label="🔍 질의문 입력 (Query Input)",
                placeholder="예: 교수님이 박사과정 학생을 지도할 수 있는 기간은 언제까지인가요?",
                info="4가지 검색 방식으로 동시 테스트됩니다",
                lines=3,
                interactive=True,  # 항상 활성화 - 미리 입력 가능
            )

            with gr.Row():
                submit_btn = gr.Button(
                    "🚀 테스트 실행", variant="primary", scale=2, interactive=False
                )
                reset_btn = gr.Button("🔄 초기화", variant="secondary", scale=1)

        with gr.Column(scale=3):
            # Rerank 방법별 성능 비교 테스트
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🔍 기본 벡터 검색 (Baseline)")
                    chatbot_none = gr.Chatbot(
                        label="Vector Search Only", type="messages", height=350
                    )
                    metrics_none = gr.Markdown(
                        value=format_metrics({}, "없음"), elem_classes=["metrics-box"]
                    )
                    copy_btn_none = gr.Button("📋 결과 복사", size="sm")
                    copy_output_none = gr.Textbox(
                        label="Markdown 형태 (Ctrl+A, Ctrl+C)", lines=2, visible=False
                    )

                with gr.Column():
                    gr.Markdown("### 🎯 Cross-Encoder Rerank (Basic)")
                    chatbot_basic = gr.Chatbot(
                        label="ms-marco-MiniLM-L-6-v2", type="messages", height=350
                    )
                    metrics_basic = gr.Markdown(
                        value=format_metrics({}, "Cross-Encoder (기본)"),
                        elem_classes=["metrics-box"],
                    )
                    copy_btn_basic = gr.Button("📋 결과 복사", size="sm")
                    copy_output_basic = gr.Textbox(
                        label="Markdown 형태 (Ctrl+A, Ctrl+C)", lines=2, visible=False
                    )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🚀 Cross-Encoder Rerank (Advanced)")
                    chatbot_advanced = gr.Chatbot(
                        label="ms-marco-MiniLM-L-12-v2", type="messages", height=350
                    )
                    metrics_advanced = gr.Markdown(
                        value=format_metrics({}, "Cross-Encoder (고성능)"),
                        elem_classes=["metrics-box"],
                    )
                    copy_btn_advanced = gr.Button("📋 결과 복사", size="sm")
                    copy_output_advanced = gr.Textbox(
                        label="Markdown 형태 (Ctrl+A, Ctrl+C)", lines=2, visible=False
                    )

                with gr.Column():
                    gr.Markdown("### 🌍 Multilingual Cross-Encoder")
                    chatbot_multilang = gr.Chatbot(
                        label="mmarco-mMiniLMv2-L12-H384-v1",
                        type="messages",
                        height=350,
                    )
                    metrics_multilang = gr.Markdown(
                        value=format_metrics({}, "다국어 Cross-Encoder"),
                        elem_classes=["metrics-box"],
                    )
                    copy_btn_multilang = gr.Button("📋 결과 복사", size="sm")
                    copy_output_multilang = gr.Textbox(
                        label="Markdown 형태 (Ctrl+A, Ctrl+C)", lines=2, visible=False
                    )

    # 이벤트 리스너 설정
    additional_pdf_upload.change(
        fn=handle_additional_pdf_upload,
        inputs=[additional_pdf_upload],
        outputs=[pdf_status],
    )

    model_dropdown.change(
        fn=change_model, inputs=[model_dropdown], outputs=[model_status]
    )

    # 상태 업데이트 이벤트 리스너 추가
    refresh_btn.click(
        fn=check_and_update_status,
        outputs=[processing_status, auto_refresh_checkbox, submit_btn],
    )
    auto_refresh_trigger.click(
        fn=check_and_update_status,
        outputs=[processing_status, auto_refresh_checkbox, submit_btn],
    )

    # 초기 로드 시 상태 업데이트
    demo.load(
        fn=check_and_update_status,
        outputs=[processing_status, auto_refresh_checkbox, submit_btn],
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
    print("🚀 Starting GIST Rules Analyzer...")
    print("📄 Auto-processing PDF files in background...")
    main_dev()

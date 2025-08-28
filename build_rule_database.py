#!/usr/bin/env python3
"""
GIST Rules Database Builder
===========================

이 스크립트는 GIST 학칙 및 규정 PDF 파일들을 전처리하여
완성된 FAISS 벡터 데이터베이스를 생성합니다.

사용법:
    python build_rule_database.py

출력:
    - faiss_db/vectorstore.faiss (FAISS 인덱스)
    - faiss_db/vectorstore.pkl (메타데이터)
    - faiss_db/database_info.json (통계 정보)
"""

import os
import json
import time
import glob
import fitz
import faiss
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Configuration
OUTPUT_DIR = Path("faiss_db")
DIMENSION = 384
PDF_PATTERN = "rules/**/*.pdf"

# Initialize embeddings model
print("🔧 임베딩 모델 초기화 중...")
EMBED_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Text splitter
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


def create_output_directory():
    """출력 디렉토리 생성"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"📁 출력 디렉토리: {OUTPUT_DIR.absolute()}")


def scan_pdf_files():
    """모든 PDF 파일 스캔"""
    print("🔍 PDF 파일 검색 중...")
    pdf_files = glob.glob(PDF_PATTERN, recursive=True)
    pdf_files.sort()

    print(f"📄 발견된 PDF 파일: {len(pdf_files)}개")
    return pdf_files


def get_document_category(file_path: str) -> str:
    """파일 경로를 기반으로 문서 카테고리 결정"""
    path_lower = file_path.lower()

    if "학칙" in path_lower or "규정" in path_lower:
        return "학칙/규정"
    elif "매뉴얼" in path_lower or "manual" in path_lower:
        return "사용자매뉴얼"
    elif "지침" in path_lower:
        return "운영지침"
    elif "규칙" in path_lower:
        return "관리규칙"
    else:
        return "기타"


def extract_text_from_pdf(pdf_path: str) -> str:
    """PDF에서 텍스트 추출"""
    try:
        doc = fitz.open(pdf_path)
        text_pages = []

        for page_num, page in enumerate(doc):
            try:
                page_text = page.get_text("text")
                if page_text.strip():
                    text_pages.append(page_text)
            except Exception as page_error:
                print(f"⚠️ 페이지 {page_num + 1} 처리 실패: {page_error}")
                continue

        doc.close()

        if text_pages:
            return "\n".join(text_pages)
        else:
            return ""

    except Exception as e:
        print(f"❌ PDF 처리 실패 ({pdf_path}): {e}")
        return ""


def process_pdfs(pdf_files):
    """모든 PDF 파일 처리"""
    print("\n🔄 PDF 파일 텍스트 추출 시작...")

    all_docs = []
    successfully_processed = 0
    failed_files = []

    start_time = time.time()

    for i, pdf_file in enumerate(pdf_files):
        print(
            f"\r📄 처리 중 [{i + 1}/{len(pdf_files)}]: {os.path.basename(pdf_file)}",
            end="",
            flush=True,
        )

        # 파일 존재 확인
        if not os.path.exists(pdf_file):
            print(f"\n⚠️ 파일 없음: {pdf_file}")
            failed_files.append(pdf_file)
            continue

        # 텍스트 추출
        text = extract_text_from_pdf(pdf_file)

        if text.strip():
            # 파일 크기에 따른 청크 크기 조정
            file_size = os.path.getsize(pdf_file)
            is_large_file = file_size > 5 * 1024 * 1024  # 5MB 이상

            if is_large_file:
                dynamic_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800, chunk_overlap=100
                )
            else:
                dynamic_splitter = TEXT_SPLITTER

            # 문서 생성 및 분할
            doc = Document(
                page_content=text,
                metadata={
                    "source": pdf_file,
                    "filename": os.path.basename(pdf_file),
                    "category": get_document_category(pdf_file),
                    "file_size": file_size,
                    "processed_at": datetime.now().isoformat(),
                },
            )

            docs = dynamic_splitter.split_documents([doc])
            all_docs.extend(docs)
            successfully_processed += 1

            file_size_mb = file_size / (1024 * 1024)
            print(
                f"\r✅ [{i + 1}/{len(pdf_files)}] {os.path.basename(pdf_file)} ({len(docs)} chunks, {file_size_mb:.1f}MB)"
            )

        else:
            print(f"\n⚠️ 텍스트 없음: {pdf_file}")
            failed_files.append(pdf_file)

        # 메모리 정리 (20개마다)
        if (i + 1) % 20 == 0:
            print(f"\r🧹 [{i + 1}/{len(pdf_files)}] 메모리 정리 완료    ")

    processing_time = time.time() - start_time

    print(f"\n✅ 텍스트 추출 완료!")
    print(f"   성공: {successfully_processed}/{len(pdf_files)} 파일")
    print(f"   실패: {len(failed_files)} 파일")
    print(f"   청크: {len(all_docs)}개")
    print(f"   소요시간: {processing_time:.1f}초")

    return all_docs, successfully_processed, failed_files


def create_faiss_vectorstore(all_docs):
    """FAISS 벡터스토어 생성"""
    print(f"\n🔄 FAISS 벡터스토어 생성 중... ({len(all_docs)} 청크)")

    if not all_docs:
        raise ValueError("처리된 문서가 없습니다!")

    start_time = time.time()

    # 배치 크기 결정
    total_chunks = len(all_docs)
    if total_chunks <= 1000:
        batch_size = 100
    elif total_chunks <= 5000:
        batch_size = 200
    elif total_chunks <= 15000:
        batch_size = 300
    else:
        batch_size = 500

    num_batches = (total_chunks + batch_size - 1) // batch_size
    print(f"📊 배치 처리: {num_batches}개 배치, 배치당 ~{batch_size}개 청크")

    vectorstore = None

    for batch_idx in range(num_batches):
        batch_start_time = time.time()
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_chunks)
        batch_docs = all_docs[start_idx:end_idx]

        print(
            f"\r🚀 배치 [{batch_idx + 1}/{num_batches}] 처리 중... ({len(batch_docs)} chunks)",
            end="",
            flush=True,
        )

        try:
            if vectorstore is None:
                # 첫 번째 배치로 벡터스토어 생성
                vectorstore = FAISS.from_documents(batch_docs, EMBED_MODEL)
            else:
                # 기존 벡터스토어에 배치 추가
                batch_vectorstore = FAISS.from_documents(batch_docs, EMBED_MODEL)
                vectorstore.merge_from(batch_vectorstore)
                del batch_vectorstore

            batch_time = time.time() - batch_start_time
            chunks_per_second = len(batch_docs) / batch_time if batch_time > 0 else 0
            print(
                f"\r✅ 배치 [{batch_idx + 1}/{num_batches}] 완료: {len(batch_docs)} chunks in {batch_time:.1f}s ({chunks_per_second:.1f} chunks/s)"
            )

        except Exception as e:
            print(f"\n❌ 배치 {batch_idx + 1} 처리 실패: {e}")
            continue

    total_time = time.time() - start_time
    avg_speed = total_chunks / total_time if total_time > 0 else 0

    print(f"✅ 벡터스토어 생성 완료!")
    print(f"   총 소요시간: {total_time:.1f}초")
    print(f"   평균 속도: {avg_speed:.1f} chunks/second")

    return vectorstore


def create_additional_indexes(vectorstore):
    """추가 FAISS 인덱스 생성 (성능 비교용)"""
    print("\n🔧 추가 FAISS 인덱스 생성 중...")

    try:
        # 기존 벡터 추출
        vectors = vectorstore.index.reconstruct_n(0, vectorstore.index.ntotal)
        results = {}

        # IndexIVFFlat
        if vectorstore.index.ntotal > 100:
            print("  🔹 IndexIVFFlat 생성...")
            nlist = min(100, int(np.sqrt(vectorstore.index.ntotal)))
            quantizer = faiss.IndexFlatL2(DIMENSION)
            ivf_index = faiss.IndexIVFFlat(quantizer, DIMENSION, nlist)
            ivf_index.train(vectors)
            ivf_index.add(vectors)
            ivf_index.nprobe = min(10, nlist)

            faiss.write_index(ivf_index, str(OUTPUT_DIR / "vectorstore_ivf.faiss"))
            results["IndexIVFFlat"] = {"nlist": nlist, "nprobe": ivf_index.nprobe}

        # IndexHNSWFlat
        if vectorstore.index.ntotal > 50:
            print("  🔹 IndexHNSWFlat 생성...")
            hnsw_index = faiss.IndexHNSWFlat(DIMENSION, 32)
            hnsw_index.hnsw.efConstruction = 200
            hnsw_index.hnsw.efSearch = 64
            hnsw_index.add(vectors)

            faiss.write_index(hnsw_index, str(OUTPUT_DIR / "vectorstore_hnsw.faiss"))
            results["IndexHNSWFlat"] = {"M": 32, "efConstruction": 200, "efSearch": 64}

        print(f"✅ {len(results)}개 추가 인덱스 생성 완료")
        return results

    except Exception as e:
        print(f"⚠️ 추가 인덱스 생성 실패: {e}")
        return {}


def save_vectorstore(vectorstore, additional_indexes, stats):
    """벡터스토어 및 메타데이터 저장"""
    print("\n💾 데이터베이스 저장 중...")

    # FAISS 인덱스 저장
    vectorstore.save_local(str(OUTPUT_DIR))
    print(f"✅ FAISS 데이터베이스 저장: {OUTPUT_DIR}/")

    # 통계 정보 저장
    database_info = {
        "created_at": datetime.now().isoformat(),
        "total_documents": stats["successfully_processed"],
        "total_chunks": len(vectorstore.docstore._dict),
        "failed_files": len(stats["failed_files"]),
        "processing_time_seconds": stats["processing_time"],
        "vectorization_time_seconds": stats.get("vectorization_time", 0),
        "dimension": DIMENSION,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "additional_indexes": additional_indexes,
        "file_sizes": {
            "vectorstore.faiss": os.path.getsize(OUTPUT_DIR / "index.faiss"),
            "vectorstore.pkl": os.path.getsize(OUTPUT_DIR / "index.pkl"),
        },
    }

    with open(OUTPUT_DIR / "database_info.json", "w", encoding="utf-8") as f:
        json.dump(database_info, f, ensure_ascii=False, indent=2)

    print("✅ 메타데이터 저장: database_info.json")

    return database_info


def main():
    """메인 실행 함수"""
    print("🚀 GIST Rules Database Builder 시작!")
    print("=" * 50)

    overall_start = time.time()

    try:
        # 1. 출력 디렉토리 생성
        create_output_directory()

        # 2. PDF 파일 스캔
        pdf_files = scan_pdf_files()
        if not pdf_files:
            print("❌ PDF 파일을 찾을 수 없습니다!")
            return

        # 3. PDF 처리
        all_docs, successfully_processed, failed_files = process_pdfs(pdf_files)

        if not all_docs:
            print("❌ 처리된 문서가 없습니다!")
            return

        # 4. 벡터스토어 생성
        vectorization_start = time.time()
        vectorstore = create_faiss_vectorstore(all_docs)
        vectorization_time = time.time() - vectorization_start

        # 5. 추가 인덱스 생성
        additional_indexes = create_additional_indexes(vectorstore)

        # 6. 데이터베이스 저장
        stats = {
            "successfully_processed": successfully_processed,
            "failed_files": failed_files,
            "processing_time": time.time() - overall_start,
            "vectorization_time": vectorization_time,
        }

        database_info = save_vectorstore(vectorstore, additional_indexes, stats)

        # 7. 최종 결과 출력
        total_time = time.time() - overall_start
        print("\n" + "=" * 50)
        print("🎉 데이터베이스 구축 완료!")
        print(f"📁 위치: {OUTPUT_DIR.absolute()}")
        print(f"📊 총 문서: {successfully_processed}개")
        print(f"📝 총 청크: {len(all_docs)}개")
        print(f"⏱️ 소요시간: {total_time:.1f}초")
        print(
            f"💾 DB 크기: {database_info['file_sizes']['vectorstore.faiss'] / (1024 * 1024):.1f}MB"
        )

        if failed_files:
            print(f"⚠️ 실패 파일: {len(failed_files)}개")

        print("\n🚀 이제 다음 명령어로 앱을 빠르게 시작할 수 있습니다:")
        print("   python app_gist_rules_analyzer_prebuilt.py")

    except KeyboardInterrupt:
        print("\n❌ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()

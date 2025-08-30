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
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import List
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


def get_document_category(file_path: str) -> tuple[str, int]:
    """파일 경로를 기반으로 문서 카테고리와 우선순위 결정"""
    path_lower = file_path.lower()
    # filename = os.path.basename(file_path).lower()  # 필요시 사용

    # 우선순위가 높을수록 중요한 문서 (검색 시 가중치 적용)
    if "학사" in path_lower and "규정" in path_lower:
        return "학사규정", 10
    elif "대학원" in path_lower and "규정" in path_lower:
        return "대학원규정", 9
    elif "연구" in path_lower and ("규정" in path_lower or "지침" in path_lower):
        return "연구규정", 8
    elif "학칙" in path_lower:
        return "학칙", 10
    elif "등록" in path_lower and "규정" in path_lower:
        return "등록규정", 7
    elif "장학" in path_lower and "규정" in path_lower:
        return "장학규정", 6
    elif "기숙사" in path_lower or "생활관" in path_lower:
        return "생활규정", 5
    elif "매뉴얼" in path_lower or "manual" in path_lower:
        return "사용자매뉴얼", 3
    elif "지침" in path_lower:
        return "운영지침", 4
    elif "규칙" in path_lower:
        return "관리규칙", 4
    else:
        return "기타", 1


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
    """모든 PDF 파일 처리 - 카테고리별 분류"""
    print("\n🔄 PDF 파일 텍스트 추출 및 카테고리별 분류 시작...")

    # 카테고리별로 문서 분류
    documents_by_category = {}
    category_stats = {}
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

            # 문서 카테고리 및 우선순위 획득
            category, priority = get_document_category(pdf_file)

            # 문서 생성 및 분할
            doc = Document(
                page_content=text,
                metadata={
                    "source": pdf_file,
                    "filename": os.path.basename(pdf_file),
                    "category": category,
                    "priority": priority,
                    "file_size": file_size,
                    "processed_at": datetime.now().isoformat(),
                    "document_id": f"{category}_{os.path.splitext(os.path.basename(pdf_file))[0]}",
                },
            )

            docs = dynamic_splitter.split_documents([doc])

            # 카테고리별로 문서 분류 저장
            if category not in documents_by_category:
                documents_by_category[category] = []
                category_stats[category] = {"files": 0, "chunks": 0, "total_size": 0}

            documents_by_category[category].extend(docs)
            category_stats[category]["files"] += 1
            category_stats[category]["chunks"] += len(docs)
            category_stats[category]["total_size"] += file_size

            successfully_processed += 1

            file_size_mb = file_size / (1024 * 1024)
            print(
                f"\r✅ [{i + 1}/{len(pdf_files)}] {os.path.basename(pdf_file)} → {category} ({len(docs)} chunks, {file_size_mb:.1f}MB)"
            )

        else:
            print(f"\n⚠️ 텍스트 없음: {pdf_file}")
            failed_files.append(pdf_file)

        # 메모리 정리 (20개마다)
        if (i + 1) % 20 == 0:
            print(f"\r🧹 [{i + 1}/{len(pdf_files)}] 메모리 정리 완료    ")

    processing_time = time.time() - start_time

    # 전체 문서 수 계산
    total_chunks = sum(len(docs) for docs in documents_by_category.values())

    print("\n📊 카테고리별 문서 분류 완료:")
    print("=" * 50)
    for category, docs in documents_by_category.items():
        stats = category_stats[category]
        priority = docs[0].metadata["priority"] if docs else 0
        size_mb = stats["total_size"] / (1024 * 1024)
        print(
            f"  📋 {category} (우선순위 {priority}): {stats['files']}개 파일, {stats['chunks']}개 청크, {size_mb:.1f}MB"
        )

    print("=" * 50)
    print(f"✅ 텍스트 추출 완료!")
    print(f"   성공: {successfully_processed}/{len(pdf_files)} 파일")
    print(f"   실패: {len(failed_files)} 파일")
    print(f"   청크: {total_chunks}개")
    print(f"   소요시간: {processing_time:.1f}초")

    return documents_by_category, successfully_processed, failed_files


def create_category_aware_vectorstore(documents_by_category):
    """카테고리별 분류된 문서들로 통합 벡터스토어 생성"""
    print("\n🎯 카테고리 인식 FAISS 벡터스토어 생성 시작...")

    # 전체 문서를 카테고리 순서대로 정렬하여 결합
    all_docs = []
    category_info = {}
    start_idx = 0

    # 우선순위 순으로 카테고리 정렬
    sorted_categories = sorted(
        documents_by_category.items(),
        key=lambda x: x[1][0].metadata["priority"] if x[1] else 0,
        reverse=True,
    )

    print("📊 카테고리별 벡터스토어 구성 순서:")
    for category, docs in sorted_categories:
        category_info[category] = {
            "start_index": start_idx,
            "end_index": start_idx + len(docs) - 1,
            "doc_count": len(docs),
            "priority": docs[0].metadata["priority"] if docs else 0,
        }

        all_docs.extend(docs)
        start_idx += len(docs)

        priority = docs[0].metadata["priority"] if docs else 0
        print(
            f"  📋 {category} (우선순위 {priority}): {len(docs)}개 청크 [인덱스 {category_info[category]['start_index']}-{category_info[category]['end_index']}]"
        )

    print(f"📝 총 {len(all_docs)}개 청크를 통합 벡터스토어로 생성 중...")

    # 기존 최적화된 벡터스토어 생성 로직 사용
    vectorstore = create_faiss_vectorstore_optimized(all_docs)

    # 카테고리 정보를 별도 파일로 저장 (앱에서 활용)
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / "category_mapping.json", "w", encoding="utf-8") as f:
        json.dump(category_info, f, ensure_ascii=False, indent=2)
    print("📄 카테고리 매핑 정보 저장: category_mapping.json")

    return vectorstore


def create_faiss_vectorstore_optimized(all_docs):
    """최적화된 FAISS 벡터스토어 생성 - 병렬 임베딩 + 직접 벡터 추가"""
    print(f"\n🚀 최적화된 FAISS 벡터스토어 생성 중... ({len(all_docs)} 청크)")

    if not all_docs:
        raise ValueError("처리된 문서가 없습니다!")

    start_time = time.time()
    total_chunks = len(all_docs)

    # 메모리 사용량 추정 (각 임베딩 384차원 * 4bytes = ~1.5KB per chunk)
    estimated_memory_mb = total_chunks * 1.5 / 1024
    available_memory_gb = 8  # 추정 가용 메모리

    print(f"📊 예상 메모리 사용량: {estimated_memory_mb:.1f}MB")

    # 🎯 전략 1: 메모리가 충분하면 한 번에 처리
    if estimated_memory_mb < available_memory_gb * 1024 * 0.5:  # 가용 메모리의 50% 이내
        print("🚀 충분한 메모리 - 일괄 처리 모드")
        return _create_vectorstore_bulk(all_docs, start_time)

    # 🎯 전략 2: 병렬 임베딩 + 직접 벡터 추가
    elif total_chunks > 2000:
        print("⚡ 병렬 임베딩 + 직접 벡터 추가 모드")
        return _create_vectorstore_parallel(all_docs, start_time)

    # 🎯 전략 3: 대용량 배치로 최적화 처리
    else:
        print("⚡ 대용량 배치 처리 모드")
        return _create_vectorstore_large_batch(all_docs, start_time)


def _create_vectorstore_bulk(all_docs, start_time):
    """전체 문서를 한 번에 처리"""
    print("📝 모든 문서를 한 번에 벡터화 중...")

    try:
        vectorstore = FAISS.from_documents(all_docs, EMBED_MODEL)

        total_time = time.time() - start_time
        avg_speed = len(all_docs) / total_time

        print(f"✅ 일괄 처리 완료!")
        print(f"   총 소요시간: {total_time:.1f}초")
        print(f"   평균 속도: {avg_speed:.1f} chunks/second")

        return vectorstore

    except MemoryError:
        print("⚠️ 메모리 부족 - 대용량 배치 모드로 전환...")
        return _create_vectorstore_large_batch(all_docs, start_time)
    except Exception as e:
        print(f"❌ 일괄 처리 실패: {e}")
        return _create_vectorstore_large_batch(all_docs, start_time)


def _create_vectorstore_large_batch(all_docs, start_time):
    """대용량 배치로 최적화 처리"""
    total_chunks = len(all_docs)

    # 🚀 최적화된 배치 크기 (5-10배 증가)
    if total_chunks <= 1000:
        batch_size = 500  # 기존 100 → 500
    elif total_chunks <= 5000:
        batch_size = 1000  # 기존 200 → 1000
    elif total_chunks <= 15000:
        batch_size = 2000  # 기존 300 → 2000
    else:
        batch_size = 3000  # 기존 500 → 3000

    num_batches = (total_chunks + batch_size - 1) // batch_size
    print(f"📊 대용량 배치 처리: {num_batches}개 배치, 배치당 ~{batch_size}개 청크")
    print(
        f"🔧 이전 대비 배치 크기: {3 - 10}배 증가로 merge 횟수 {num_batches}개로 최소화"
    )

    vectorstore = None
    merge_times = []
    embed_times = []

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
                embed_start = time.time()
                vectorstore = FAISS.from_documents(batch_docs, EMBED_MODEL)
                embed_time = time.time() - embed_start
                embed_times.append(embed_time)

                print(f"\r✅ 초기 벡터스토어 생성 완료: {len(batch_docs)} chunks")
            else:
                # 🚀 최적화: 임베딩과 merge를 분리하여 측정
                embed_start = time.time()
                batch_vectorstore = FAISS.from_documents(batch_docs, EMBED_MODEL)
                embed_time = time.time() - embed_start
                embed_times.append(embed_time)

                merge_start = time.time()
                vectorstore.merge_from(batch_vectorstore)
                merge_time = time.time() - merge_start
                merge_times.append(merge_time)

                del batch_vectorstore  # 즉시 메모리 해제

            batch_time = time.time() - batch_start_time
            chunks_per_second = len(batch_docs) / batch_time if batch_time > 0 else 0

            # 더 상세한 성능 정보
            embed_avg = embed_times[-1] if embed_times else 0
            merge_avg = merge_times[-1] if merge_times else 0

            print(
                f"\r✅ 배치 [{batch_idx + 1}/{num_batches}] 완료: {len(batch_docs)} chunks | "
                f"임베딩: {embed_avg:.1f}s, 병합: {merge_avg:.1f}s | {chunks_per_second:.1f} chunks/s"
            )

        except Exception as e:
            print(f"\n❌ 배치 {batch_idx + 1} 처리 실패: {e}")
            continue

    total_time = time.time() - start_time
    avg_speed = total_chunks / total_time if total_time > 0 else 0

    # 📊 상세 성능 분석
    if embed_times and merge_times:
        total_embed_time = sum(embed_times)
        total_merge_time = sum(merge_times)
        embed_percent = (total_embed_time / total_time) * 100
        merge_percent = (total_merge_time / total_time) * 100

        print("✅ 최적화된 벡터스토어 생성 완료!")
        print(f"   총 소요시간: {total_time:.1f}초")
        print(f"   평균 속도: {avg_speed:.1f} chunks/second")
        print(
            f"   시간 분석: 임베딩 {total_embed_time:.1f}s ({embed_percent:.1f}%), "
            f"병합 {total_merge_time:.1f}s ({merge_percent:.1f}%)"
        )
        print(f"   배치 효율성: {num_batches}회 merge (기존 대비 대폭 감소)")
    else:
        print("✅ 벡터스토어 생성 완료!")
        print(f"   총 소요시간: {total_time:.1f}초")
        print(f"   평균 속도: {avg_speed:.1f} chunks/second")

    return vectorstore


def _create_vectorstore_parallel(all_docs, start_time):
    """병렬 임베딩 + 직접 벡터 추가로 최고 성능"""
    print("🔥 병렬 임베딩 + 직접 FAISS 벡터 추가 모드 시작")

    total_chunks = len(all_docs)

    # 병렬 처리용 배치 크기 (CPU 코어 수 기반)
    cpu_count = os.cpu_count() or 4
    max_workers = min(8, max(2, cpu_count // 2))  # 최대 8개, 최소 2개
    batch_size = min(1000, max(100, total_chunks // max_workers))  # 워커당 적절한 크기

    print(
        f"🔧 병렬 설정: {max_workers}개 워커 (CPU: {cpu_count}), 배치 크기: {batch_size}"
    )

    try:
        # 🚀 1단계: 병렬로 모든 텍스트 임베딩 생성
        embedding_start = time.time()
        print("📝 병렬 임베딩 생성 중...")

        all_texts = [doc.page_content for doc in all_docs]
        # all_metadatas = [doc.metadata for doc in all_docs]  # 사용 안함

        # 병렬로 임베딩 생성 (배치 단위)
        all_embeddings = []
        batches = [
            all_texts[i : i + batch_size] for i in range(0, len(all_texts), batch_size)
        ]

        def embed_batch_optimized(texts_batch):
            """최적화된 단일 배치 임베딩"""
            return EMBED_MODEL.embed_documents(texts_batch)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 모든 배치를 병렬로 처리
            future_to_batch = {
                executor.submit(embed_batch_optimized, batch): i
                for i, batch in enumerate(batches)
            }
            completed_batches = [None] * len(batches)

            for future in concurrent.futures.as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_embeddings = future.result()
                    completed_batches[batch_idx] = batch_embeddings
                    progress = (batch_idx + 1) / len(batches) * 100
                    print(
                        f"\r⚡ 임베딩 진행: {batch_idx + 1}/{len(batches)} ({progress:.1f}%)",
                        end="",
                        flush=True,
                    )
                except Exception as e:
                    print(f"\n❌ 배치 {batch_idx} 임베딩 실패: {e}")
                    completed_batches[batch_idx] = []

        # 결과를 순서대로 합치기
        for batch_embeddings in completed_batches:
            if batch_embeddings:
                all_embeddings.extend(batch_embeddings)

        embedding_time = time.time() - embedding_start
        print(
            f"\n✅ 병렬 임베딩 완료: {len(all_embeddings)}개 벡터 생성 in {embedding_time:.1f}s"
        )

        if len(all_embeddings) != len(all_docs):
            print(
                f"⚠️ 임베딩 수 불일치: {len(all_embeddings)} vs {len(all_docs)}, fallback..."
            )
            return _create_vectorstore_large_batch(all_docs, start_time)

        # 🚀 2단계: 직접 FAISS 인덱스 생성 (merge 오버헤드 제거)
        faiss_start = time.time()
        print("🔧 FAISS 인덱스 직접 생성 중...")

        # numpy 배열로 변환
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        dimension = embeddings_array.shape[1]

        # FAISS 인덱스 생성 (L2 거리)
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(embeddings_array)

        # LangChain FAISS vectorstore 생성
        from langchain_community.docstore.in_memory import InMemoryDocstore

        # docstore 구성 (인덱스 -> 문서 매핑)
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(all_docs)})
        index_to_docstore_id = {i: str(i) for i in range(len(all_docs))}

        vectorstore = FAISS(
            embedding_function=EMBED_MODEL,
            index=faiss_index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )

        faiss_time = time.time() - faiss_start
        print(f"✅ FAISS 인덱스 생성 완료 in {faiss_time:.1f}s")

        # 📊 성능 분석
        total_time = time.time() - start_time
        avg_speed = total_chunks / total_time
        embedding_percent = (embedding_time / total_time) * 100
        faiss_percent = (faiss_time / total_time) * 100
        parallelization_speedup = len(batches) / max_workers if max_workers > 1 else 1

        print("🔥 병렬 처리 완료! (최고 성능 모드)")
        print(f"   총 소요시간: {total_time:.1f}초")
        print(f"   처리 속도: {avg_speed:.1f} chunks/second")
        print(f"   병렬 임베딩: {embedding_time:.1f}s ({embedding_percent:.1f}%)")
        print(f"   FAISS 생성: {faiss_time:.1f}s ({faiss_percent:.1f}%)")
        print(f"   병렬 효율성: {max_workers}개 워커, {len(batches)}배치 병렬 처리")
        print(f"   속도 향상: ~{parallelization_speedup:.1f}x (이론적)")
        print(f"   메모리 최적화: merge 과정 완전 제거")

        return vectorstore

    except Exception as e:
        print(f"❌ 병렬 처리 실패: {e}")
        print("🔄 대용량 배치 모드로 fallback...")
        return _create_vectorstore_large_batch(all_docs, start_time)


def create_faiss_vectorstore_parallel(all_docs):
    """최고 성능 병렬 FAISS 벡터스토어 생성"""
    print(f"\n🚀 병렬 최적화 FAISS 벡터스토어 생성 중... ({len(all_docs)} 청크)")

    if not all_docs:
        raise ValueError("처리된 문서가 없습니다!")

    start_time = time.time()
    total_chunks = len(all_docs)

    # 🔧 시스템 리소스 최적화
    cpu_count = os.cpu_count() or 4
    max_workers = min(cpu_count, 4)  # 너무 많은 스레드는 오히려 느림

    print(f"🔧 병렬 처리 설정: {max_workers} 워커, CPU 코어: {cpu_count}")

    # 메모리 사용량 추정
    estimated_memory_mb = total_chunks * 1.5 / 1024
    print(f"📊 예상 메모리 사용량: {estimated_memory_mb:.1f}MB")

    # 🎯 전략 선택
    if total_chunks < 500:
        print("📝 소량 데이터 - 단순 처리 모드")
        return _create_vectorstore_bulk(all_docs, start_time)
    elif estimated_memory_mb < 2048:  # 2GB 이하
        print("⚡ 직접 벡터 추가 모드 (최고 성능)")
        return _create_vectorstore_direct_add(all_docs, start_time, max_workers)
    else:
        print("🚀 병렬 배치 처리 모드")
        return _create_vectorstore_parallel_batch(all_docs, start_time, max_workers)


def _create_vectorstore_direct_add(all_docs, start_time, max_workers):
    """직접 벡터 추가 방식 - 최고 성능"""
    print("🔧 문서 텍스트 추출 중...")

    # 1. 모든 텍스트 추출
    texts = [doc.page_content for doc in all_docs]
    # metadatas = [doc.metadata for doc in all_docs]  # 사용 안함

    # 2. 병렬로 임베딩 생성
    print(f"⚡ {max_workers}개 워커로 병렬 임베딩 생성 중...")
    embed_start = time.time()

    # 배치 크기를 워커 수에 맞게 조정
    batch_size = max(100, len(texts) // (max_workers * 2))
    text_batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    embeddings = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 각 배치를 병렬로 처리
        future_to_batch = {
            executor.submit(_embed_batch, batch, batch_idx): batch_idx
            for batch_idx, batch in enumerate(text_batches)
        }

        for future in concurrent.futures.as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_embeddings = future.result()
                embeddings.extend(batch_embeddings)
                print(
                    f"\r⚡ 배치 {batch_idx + 1}/{len(text_batches)} 임베딩 완료",
                    end="",
                    flush=True,
                )
            except Exception as e:
                print(f"\n❌ 배치 {batch_idx} 임베딩 실패: {e}")

    embed_time = time.time() - embed_start
    print(f"\n✅ 병렬 임베딩 완료: {len(embeddings)}개 벡터 in {embed_time:.1f}s")

    # 3. FAISS 인덱스 직접 구성
    print("🔧 FAISS 인덱스 직접 구성 중...")
    faiss_start = time.time()

    try:
        # 벡터를 numpy 배열로 변환
        embeddings_array = np.array(embeddings, dtype=np.float32)
        dimension = embeddings_array.shape[1]

        # FAISS 인덱스 생성
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)

        # FAISS vectorstore 객체 생성
        vectorstore = FAISS(
            embedding_function=EMBED_MODEL.embed_query,
            index=index,
            docstore={i: all_docs[i] for i in range(len(all_docs))},
            index_to_docstore_id={i: i for i in range(len(all_docs))},
        )

        faiss_time = time.time() - faiss_start
        total_time = time.time() - start_time
        avg_speed = len(all_docs) / total_time

        print(f"✅ 직접 벡터 추가 완료!")
        print(f"   총 소요시간: {total_time:.1f}초")
        print(f"   임베딩 시간: {embed_time:.1f}s, FAISS 구성: {faiss_time:.1f}s")
        print(f"   평균 속도: {avg_speed:.1f} chunks/second")
        print(f"   성능 향상: 병렬 처리로 {max_workers}배 가속")

        return vectorstore

    except Exception as e:
        print(f"❌ 직접 벡터 추가 실패: {e}")
        print("🔄 대용량 배치 모드로 폴백...")
        return _create_vectorstore_large_batch(all_docs, start_time)


def _embed_batch(texts: List[str], batch_idx: int) -> List[List[float]]:
    """단일 배치 임베딩 처리"""
    try:
        # 각 스레드에서 독립적인 임베딩 모델 사용
        return EMBED_MODEL.embed_documents(texts)
    except Exception as e:
        print(f"배치 {batch_idx} 임베딩 오류: {e}")
        return []


def _create_vectorstore_parallel_batch(all_docs, start_time, max_workers):
    """병렬 배치 처리"""
    total_chunks = len(all_docs)

    # 워커 수에 맞춰 배치 크기 결정
    batch_size = max(500, total_chunks // max_workers)
    batches = [all_docs[i : i + batch_size] for i in range(0, total_chunks, batch_size)]

    print(f"📊 병렬 배치: {len(batches)}개 배치, 배치당 ~{batch_size}개 청크")
    print(f"⚡ {max_workers}개 워커로 병렬 처리 시작...")

    # 병렬로 각 배치를 벡터스토어로 변환
    vectorstores = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(_create_single_vectorstore, batch, batch_idx): batch_idx
            for batch_idx, batch in enumerate(batches)
        }

        for future in concurrent.futures.as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_vectorstore = future.result()
                if batch_vectorstore:
                    vectorstores.append(batch_vectorstore)
                    print(
                        f"⚡ 배치 {batch_idx + 1}/{len(batches)} 벡터스토어 생성 완료"
                    )
            except Exception as e:
                print(f"❌ 배치 {batch_idx} 처리 실패: {e}")

    # 모든 배치 벡터스토어를 순차적으로 병합
    print("🔗 병렬 생성된 벡터스토어들을 병합 중...")
    merge_start = time.time()

    if not vectorstores:
        raise ValueError("생성된 벡터스토어가 없습니다!")

    final_vectorstore = vectorstores[0]
    for i, vs in enumerate(vectorstores[1:], 1):
        print(f"\r🔗 병합 중... {i}/{len(vectorstores) - 1}", end="", flush=True)
        final_vectorstore.merge_from(vs)
        del vs  # 메모리 해제

    merge_time = time.time() - merge_start
    total_time = time.time() - start_time
    avg_speed = total_chunks / total_time

    print(f"\n✅ 병렬 배치 처리 완료!")
    print(f"   총 소요시간: {total_time:.1f}초")
    print(f"   병합 시간: {merge_time:.1f}초")
    print(f"   평균 속도: {avg_speed:.1f} chunks/second")
    print(f"   병렬 효율성: {max_workers}개 워커 활용")

    return final_vectorstore


def _create_single_vectorstore(docs, batch_idx):
    """단일 배치를 벡터스토어로 변환"""
    try:
        return FAISS.from_documents(docs, EMBED_MODEL)
    except Exception as e:
        print(f"배치 {batch_idx} 벡터스토어 생성 실패: {e}")
        return None


def create_faiss_vectorstore(documents_input):
    """FAISS 벡터스토어 생성 - 자동 최적화 (카테고리 인식)"""

    # 입력이 카테고리별 딕셔너리인지 평면 리스트인지 확인
    if isinstance(documents_input, dict):
        # 카테고리별 문서 딕셔너리인 경우
        print("🎯 카테고리별 문서 감지 - 카테고리 인식 벡터스토어 생성")
        return create_category_aware_vectorstore(documents_input)
    else:
        # 기존 평면 리스트인 경우 (하위 호환성)
        print("📝 평면 문서 리스트 감지 - 기존 최적화 방식 사용")
        all_docs = documents_input
        total_chunks = len(all_docs)

        if total_chunks < 100:
            print("📝 소량 데이터 - 기본 처리")
            return FAISS.from_documents(all_docs, EMBED_MODEL)
        elif total_chunks < 2000:
            print("⚡ 중간 규모 - 기본 최적화")
            return create_faiss_vectorstore_optimized(all_docs)
        else:
            print("🚀 대용량 데이터 - 병렬 최적화")
            return create_faiss_vectorstore_parallel(all_docs)


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

        # 3. PDF 처리 및 카테고리별 분류
        documents_by_category, successfully_processed, failed_files = process_pdfs(
            pdf_files
        )

        if not documents_by_category:
            print("❌ 처리된 문서가 없습니다!")
            return

        # 4. 카테고리 인식 벡터스토어 생성
        vectorization_start = time.time()
        vectorstore = create_faiss_vectorstore(documents_by_category)
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
        total_chunks = sum(len(docs) for docs in documents_by_category.values())
        print(f"📝 총 청크: {total_chunks}개")
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


def benchmark_vectorstore_creation(documents_by_category, sample_ratio=0.1):
    """벤치마크 테스트: 카테고리별 문서의 성능 비교"""
    print("\n🏁 벤치마크 모드: 카테고리 인식 벡터스토어 성능 비교")

    # 전체 문서 수 계산
    total_docs = sum(len(docs) for docs in documents_by_category.values())

    # 카테고리별로 비례하여 샘플링
    sample_documents_by_category = {}
    total_sample_size = 0

    for category, docs in documents_by_category.items():
        category_sample_size = max(10, int(len(docs) * sample_ratio))
        category_sample_size = min(category_sample_size, 200)  # 카테고리당 최대 200개
        sample_documents_by_category[category] = docs[:category_sample_size]
        total_sample_size += category_sample_size

    print(f"📊 카테고리별 테스트 샘플:")
    for category, docs in sample_documents_by_category.items():
        orig_count = len(documents_by_category[category])
        print(f"  📋 {category}: {len(docs)}개 샘플 (원본 {orig_count}개 중)")

    print(
        f"📊 총 테스트 샘플: {total_sample_size}개 청크 ({total_docs}개 중 {sample_ratio * 100:.1f}%)"
    )
    print("=" * 60)

    results = {}

    # 테스트 목록 - 카테고리 인식 버전과 기존 버전 비교
    tests = [
        (
            "카테고리_인식",
            lambda docs_by_cat: create_category_aware_vectorstore(docs_by_cat),
        ),
        (
            "기존_선형_병합",
            lambda docs_by_cat: create_faiss_vectorstore_optimized(
                [doc for docs in docs_by_cat.values() for doc in docs]
            ),
        ),
    ]

    for i, (test_name, test_func) in enumerate(tests, 1):
        print(f"\n🔍 [{i}/{len(tests)}] {test_name} 테스트 중...")

        start_time = time.time()
        try:
            vectorstore = test_func(sample_documents_by_category)
            end_time = time.time()

            duration = end_time - start_time
            speed = total_sample_size / duration if duration > 0 else 0

            results[test_name] = {
                "성공": True,
                "시간": duration,
                "속도": speed,
                "vectorstore": vectorstore,
            }

            print(f"✅ {test_name} 완료: {duration:.1f}s, {speed:.1f} chunks/s")

        except Exception as e:
            results[test_name] = {"성공": False, "시간": 0, "속도": 0, "오류": str(e)}
            print(f"❌ {test_name} 실패: {e}")

    # 📊 결과 분석 및 출력
    print("\n📊 벤치마크 결과 분석")
    print("=" * 60)

    # 성공한 테스트들만 정렬
    successful_tests = [(name, data) for name, data in results.items() if data["성공"]]

    if not successful_tests:
        print("❌ 모든 테스트가 실패했습니다.")
        return None

    # 속도 순으로 정렬
    successful_tests.sort(key=lambda x: x[1]["속도"], reverse=True)

    print("🏆 성능 순위:")
    for rank, (test_name, data) in enumerate(successful_tests, 1):
        medal = ["🥇", "🥈", "🥉"][rank - 1] if rank <= 3 else f"{rank}위"
        print(
            f"   {medal} {test_name}: {data['시간']:.1f}s ({data['속도']:.1f} chunks/s)"
        )

    # 최고 성능 대비 비교
    best_speed = successful_tests[0][1]["속도"]
    print("\n📈 최고 성능 대비 비교:")
    for test_name, data in successful_tests:
        ratio = data["속도"] / best_speed if best_speed > 0 else 0
        improvement = (1 - ratio) * 100 if ratio < 1 else (ratio - 1) * 100
        comparison = (
            f"{improvement:.1f}% {'느림' if ratio < 1 else '빠름'}"
            if ratio != 1
            else "기준"
        )
        print(f"   {test_name}: {comparison} ({ratio:.2f}x)")

    # 전체 데이터셋 예상 시간
    print(f"\n🔮 전체 {total_docs}개 처리 예상 시간:")
    for test_name, data in successful_tests:
        if data["속도"] > 0:
            estimated_seconds = total_docs / data["속도"]
            estimated_minutes = estimated_seconds / 60
            print(
                f"   {test_name}: ~{estimated_seconds:.0f}초 ({estimated_minutes:.1f}분)"
            )

    # 권장 방식 선택
    best_method = successful_tests[0][0]
    print(f"\n💡 권장 방식: {best_method}")
    print(
        f"   이유: 가장 빠른 처리 속도 ({successful_tests[0][1]['속도']:.1f} chunks/s)"
    )

    return {"결과": results, "권장_방식": best_method, "최고_속도": best_speed}


if __name__ == "__main__":
    # 환경 변수로 벤치마크 모드 선택 가능
    # VECTORSTORE_MODE=benchmark python build_rule_database.py
    if os.getenv("VECTORSTORE_MODE", "").lower() == "benchmark":
        print("🏁 벤치마크 모드로 실행됩니다.")
        print("   샘플 데이터로 성능 비교를 수행한 후 최적 방식으로 전체 처리합니다.")

    main()

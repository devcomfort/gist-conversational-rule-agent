#!/usr/bin/env python3
"""
Multi-Embedding GIST Rules Database Builder
==========================================

TODO.txt에 명시된 여러 임베딩 모델로 각각 FAISS 벡터 데이터베이스를 생성합니다.

지원하는 임베딩 모델:
- Qwen/Qwen3-Embedding-0.6B (3위)
- jinaai/jina-embeddings-v3 (22위)
- BAAI/bge-m3 (23위)
- sentence-transformers/all-MiniLM-L6-v2 (117위)

사용법:
    python build_multi_embedding_databases.py
    python build_multi_embedding_databases.py --model Qwen/Qwen3-Embedding-0.6B

출력:
    - faiss_qwen3_embedding_0.6b/ (Qwen 모델용)
    - faiss_jina_embeddings_v3/ (Jina 모델용)
    - faiss_bge_m3/ (BGE 모델용)
    - faiss_all_minilm_l6_v2/ (기본 모델용)
"""

import os
import json
import time
import glob
import argparse
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv

try:
    import fitz  # PyMuPDF
except ImportError:
    print("❌ PyMuPDF가 설치되지 않았습니다. 다음 명령어로 설치하세요:")
    print("pip install PyMuPDF")
    exit(1)

try:
    import faiss
    import numpy as np
except ImportError:
    print("❌ FAISS가 설치되지 않았습니다. 다음 명령어로 설치하세요:")
    print("pip install faiss-cpu  # 또는 faiss-gpu")
    exit(1)

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Configuration
PDF_PATTERN = "rules/**/*.pdf"

# 지원하는 임베딩 모델 설정 (TODO.txt 기반)
EMBEDDING_MODELS = {
    "Qwen/Qwen3-Embedding-0.6B": {
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "db_name": "faiss_qwen3_embedding_0.6b",
        "dimension": 1024,
        "mteb_rank": 3,
        "description": "Qwen3 Embedding 0.6B - MTEB 3위",
    },
    "jinaai/jina-embeddings-v3": {
        "model_name": "jinaai/jina-embeddings-v3",
        "db_name": "faiss_jina_embeddings_v3",
        "dimension": 1024,
        "mteb_rank": 22,
        "description": "Jina Embeddings v3 - MTEB 22위",
    },
    "BAAI/bge-m3": {
        "model_name": "BAAI/bge-m3",
        "db_name": "faiss_bge_m3",
        "dimension": 1024,
        "mteb_rank": 23,
        "description": "BGE M3 - MTEB 23위",
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "db_name": "faiss_all_minilm_l6_v2",
        "dimension": 384,
        "mteb_rank": 117,
        "description": "All MiniLM L6 v2 - MTEB 117위 (기존 기본 모델)",
    },
}

# Text splitter
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


def get_document_category(file_path: str) -> tuple[str, int]:
    """파일 경로를 기반으로 문서 카테고리와 우선순위 결정"""
    path_lower = file_path.lower()

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


def scan_pdf_files():
    """모든 PDF 파일 스캔"""
    print("🔍 PDF 파일 검색 중...")
    pdf_files = glob.glob(PDF_PATTERN, recursive=True)
    pdf_files.sort()

    print(f"📄 발견된 PDF 파일: {len(pdf_files)}개")
    return pdf_files


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


def create_faiss_vectorstore_optimized(all_docs, embed_model):
    """최적화된 FAISS 벡터스토어 생성"""
    print(f"\n🚀 최적화된 FAISS 벡터스토어 생성 중... ({len(all_docs)} 청크)")

    if not all_docs:
        raise ValueError("처리된 문서가 없습니다!")

    start_time = time.time()
    total_chunks = len(all_docs)

    # 메모리 사용량 추정
    estimated_memory_mb = total_chunks * 1.5 / 1024
    print(f"📊 예상 메모리 사용량: {estimated_memory_mb:.1f}MB")

    # 일괄 처리로 시도
    try:
        print("📝 모든 문서를 한 번에 벡터화 중...")
        vectorstore = FAISS.from_documents(all_docs, embed_model)

        total_time = time.time() - start_time
        avg_speed = len(all_docs) / total_time

        print(f"✅ 일괄 처리 완료!")
        print(f"   총 소요시간: {total_time:.1f}초")
        print(f"   평균 속도: {avg_speed:.1f} chunks/second")

        return vectorstore

    except MemoryError:
        print("⚠️ 메모리 부족 - 배치 모드로 전환...")
        return _create_vectorstore_batch(all_docs, embed_model, start_time)
    except Exception as e:
        print(f"❌ 일괄 처리 실패: {e}")
        return _create_vectorstore_batch(all_docs, embed_model, start_time)


def _create_vectorstore_batch(all_docs, embed_model, start_time):
    """배치 처리로 벡터스토어 생성"""
    total_chunks = len(all_docs)
    batch_size = min(1000, max(100, total_chunks // 5))
    num_batches = (total_chunks + batch_size - 1) // batch_size

    print(f"📊 배치 처리: {num_batches}개 배치, 배치당 ~{batch_size}개 청크")

    vectorstore = None

    for batch_idx in range(num_batches):
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
                vectorstore = FAISS.from_documents(batch_docs, embed_model)
                print(f"\r✅ 초기 벡터스토어 생성 완료: {len(batch_docs)} chunks")
            else:
                # 후속 배치들을 병합
                batch_vectorstore = FAISS.from_documents(batch_docs, embed_model)
                vectorstore.merge_from(batch_vectorstore)
                del batch_vectorstore  # 즉시 메모리 해제

                print(f"\r✅ 배치 [{batch_idx + 1}/{num_batches}] 병합 완료")

        except Exception as e:
            print(f"\n❌ 배치 {batch_idx + 1} 처리 실패: {e}")
            continue

    total_time = time.time() - start_time
    avg_speed = total_chunks / total_time if total_time > 0 else 0

    print(f"\n✅ 배치 처리 완료!")
    print(f"   총 소요시간: {total_time:.1f}초")
    print(f"   평균 속도: {avg_speed:.1f} chunks/second")

    return vectorstore


def create_category_aware_vectorstore(documents_by_category, embed_model):
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

    # 벡터스토어 생성
    vectorstore = create_faiss_vectorstore_optimized(all_docs, embed_model)

    return vectorstore, category_info


def save_vectorstore(vectorstore, category_info, db_path, model_config, stats):
    """벡터스토어 및 메타데이터 저장"""
    print(f"\n💾 데이터베이스 저장 중... ({db_path})")

    # 디렉토리 생성
    db_path.mkdir(exist_ok=True)

    # FAISS 인덱스 저장
    vectorstore.save_local(str(db_path))
    print(f"✅ FAISS 데이터베이스 저장: {db_path}/")

    # 카테고리 정보 저장
    with open(db_path / "category_mapping.json", "w", encoding="utf-8") as f:
        json.dump(category_info, f, ensure_ascii=False, indent=2)

    # 통계 정보 저장
    database_info = {
        "created_at": datetime.now().isoformat(),
        "embedding_model": model_config["model_name"],
        "embedding_dimension": model_config["dimension"],
        "mteb_rank": model_config["mteb_rank"],
        "description": model_config["description"],
        "total_documents": stats["successfully_processed"],
        "total_chunks": len(vectorstore.docstore._dict),
        "failed_files": len(stats["failed_files"]),
        "processing_time_seconds": stats["processing_time"],
        "vectorization_time_seconds": stats.get("vectorization_time", 0),
        "file_sizes": {
            "index.faiss": os.path.getsize(db_path / "index.faiss"),
            "index.pkl": os.path.getsize(db_path / "index.pkl"),
        },
        "categories": category_info,
    }

    with open(db_path / "database_info.json", "w", encoding="utf-8") as f:
        json.dump(database_info, f, ensure_ascii=False, indent=2)

    print("✅ 메타데이터 저장: database_info.json, category_mapping.json")

    return database_info


def build_database_for_model(model_key: str, documents_by_category, stats):
    """특정 임베딩 모델로 데이터베이스 구축"""
    model_config = EMBEDDING_MODELS[model_key]
    print(f"\n{'=' * 60}")
    print(f"🤖 임베딩 모델: {model_config['description']}")
    print(f"📊 MTEB 순위: {model_config['mteb_rank']}위")
    print(f"📐 차원: {model_config['dimension']}")
    print(f"💾 저장 경로: {model_config['db_name']}/")
    print(f"{'=' * 60}")

    try:
        # 임베딩 모델 초기화
        print(f"🔧 임베딩 모델 로딩 중: {model_config['model_name']}")
        embed_model = HuggingFaceEmbeddings(
            model_name=model_config["model_name"],
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("✅ 임베딩 모델 로딩 완료!")

        # 벡터스토어 생성
        vectorization_start = time.time()
        vectorstore, category_info = create_category_aware_vectorstore(
            documents_by_category, embed_model
        )
        vectorization_time = time.time() - vectorization_start

        # 데이터베이스 저장
        db_path = Path(model_config["db_name"])
        updated_stats = stats.copy()
        updated_stats["vectorization_time"] = vectorization_time

        database_info = save_vectorstore(
            vectorstore, category_info, db_path, model_config, updated_stats
        )

        # 결과 출력
        total_chunks = len(vectorstore.docstore._dict)
        db_size_mb = database_info["file_sizes"]["index.faiss"] / (1024 * 1024)

        print(f"\n🎉 {model_config['description']} 데이터베이스 구축 완료!")
        print(f"📁 위치: {db_path.absolute()}")
        print(f"📝 총 청크: {total_chunks}개")
        print(f"⏱️ 벡터화 시간: {vectorization_time:.1f}초")
        print(f"💾 DB 크기: {db_size_mb:.1f}MB")

        return True

    except Exception as e:
        print(f"❌ {model_config['description']} 데이터베이스 구축 실패: {e}")
        return False


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description="Multi-Embedding GIST Rules Database Builder"
    )
    parser.add_argument(
        "--model",
        choices=list(EMBEDDING_MODELS.keys()),
        help="특정 임베딩 모델만 처리 (미지정시 모든 모델 처리)",
    )
    parser.add_argument(
        "--list-models", action="store_true", help="지원하는 임베딩 모델 목록 출력"
    )

    args = parser.parse_args()

    if args.list_models:
        print("🤖 지원하는 임베딩 모델:")
        print("=" * 70)
        for key, config in EMBEDDING_MODELS.items():
            print(f"  📊 MTEB {config['mteb_rank']:3d}위 | {config['description']}")
            print(f"  🏷️  모델명: {config['model_name']}")
            print(f"  📐 차원: {config['dimension']} | 💾 DB명: {config['db_name']}")
            print()
        return

    print("🚀 Multi-Embedding GIST Rules Database Builder 시작!")
    print("=" * 60)

    overall_start = time.time()

    try:
        # 1. PDF 파일 스캔
        pdf_files = scan_pdf_files()
        if not pdf_files:
            print("❌ PDF 파일을 찾을 수 없습니다!")
            return

        # 2. PDF 처리 및 카테고리별 분류 (한 번만 수행)
        documents_by_category, successfully_processed, failed_files = process_pdfs(
            pdf_files
        )

        if not documents_by_category:
            print("❌ 처리된 문서가 없습니다!")
            return

        # 3. 통계 정보 준비
        stats = {
            "successfully_processed": successfully_processed,
            "failed_files": failed_files,
            "processing_time": time.time() - overall_start,
        }

        # 4. 임베딩 모델별 데이터베이스 생성
        if args.model:
            # 특정 모델만 처리
            models_to_process = [args.model]
            print(f"🎯 선택된 모델: {EMBEDDING_MODELS[args.model]['description']}")
        else:
            # 모든 모델 처리
            models_to_process = list(EMBEDDING_MODELS.keys())
            print(f"🎯 모든 임베딩 모델 처리: {len(models_to_process)}개")

        successful_builds = 0
        failed_builds = 0

        for model_key in models_to_process:
            try:
                if build_database_for_model(model_key, documents_by_category, stats):
                    successful_builds += 1
                else:
                    failed_builds += 1
            except KeyboardInterrupt:
                print("\n❌ 사용자에 의해 중단되었습니다.")
                break
            except Exception as e:
                print(f"❌ {model_key} 처리 중 오류: {e}")
                failed_builds += 1

        # 5. 최종 결과 출력
        total_time = time.time() - overall_start
        print("\n" + "=" * 60)
        print("🎉 Multi-Embedding Database 구축 완료!")
        print(f"✅ 성공: {successful_builds}개 모델")
        print(f"❌ 실패: {failed_builds}개 모델")
        print(f"📊 총 문서: {successfully_processed}개")
        total_chunks = sum(len(docs) for docs in documents_by_category.values())
        print(f"📝 총 청크: {total_chunks}개")
        print(f"⏱️ 총 소요시간: {total_time:.1f}초")

        if failed_files:
            print(f"⚠️ 실패 파일: {len(failed_files)}개")

        print(f"\n🚀 이제 다음 명령어로 원하는 임베딩 모델로 앱을 실행할 수 있습니다:")
        print("   python app_gist_rules_analyzer_prebuilt.py")

    except KeyboardInterrupt:
        print("\n❌ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()

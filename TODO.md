# 📋 Project TODO - Academic Rule Agent

## 🎯 **Project Scope**

**목표**: 임베딩 모델과 청킹 기법의 차이가 RAG 성능에 미치는 영향 확인 및 최적 Retriever 구성 검증

---

## 📊 **Project Status**

### ✅ **Phase 1: Infrastructure & Framework (COMPLETED)**

- [x] **실험 프레임워크 구축**
  - [x] 28,560개 조합의 전면적 실험 시스템
  - [x] 6개 청커 × 6개 임베딩 모델 × 다양한 파라미터 조합
  - [x] 토큰화 → 청킹 → 임베딩 3단계 파이프라인
  - [x] Multi-GPU 자동 감지 및 자원 최적화

- [x] **실험 설정 및 관리 시스템**
  - [x] `experiment_configurations.py`: 통합 설정 관리
  - [x] `multi_embedding_database_builder.py`: 메인 실험 실행기
  - [x] 데이터베이스 네이밍 컨벤션 설계
  - [x] 청킹 결과 저장 시스템 (중간 결과 보존)

- [x] **임베딩 모델 선정 및 최적화**
  - [x] MTEB 기반 모델 선정 (2025년 8월 기준)
  - [x] Qwen3-8B (2위), Qwen3-0.6B (4위), Jina-v3 (22위)
  - [x] BGE-M3 (23위), all-MiniLM-L6 (117위, 베이스라인)

---

## 🏗️ **Phase 2: Data Generation (IN PROGRESS)**

### 🔄 **Priority 1: 청킹 데이터 생성**

- [ ] **대규모 청킹 실험**
  - [ ] 4,760개 청킹 조합 실행 (RTX 3090 Multi-GPU)
  - [ ] 청킹 결과 저장 및 검증 (pickle 형태)
  - [ ] 청킹 품질 메트릭 수집 (청크 수, 평균 길이, 분산)
  - [ ] 청킹 실패율 모니터링 및 개선

### 📊 **Priority 2: QA 데이터셋 생성**

- [ ] **청킹 기반 QA 생성 시스템**
  - [ ] 저장된 청킹 결과 기반 QA 쌍 생성
  - [ ] 청크별 질문-답변 쌍 자동 생성
  - [ ] SmolAgents vs LlamaIndex 성능 비교
  - [ ] 질문 품질 자동 평가 시스템
  - [ ] 도메인별 QA 데이터셋 구축

### 🔮 **Priority 3: 임베딩 데이터셋 생성**

- [ ] **청킹 결과 기반 임베딩 생성**
  - [ ] 28,560개 조합 임베딩 실행 (청킹 결과 × 임베딩 모델)
  - [ ] FAISS 벡터 데이터베이스 구축
  - [ ] 임베딩 메트릭 수집 (생성 시간, DB 크기)
  - [ ] 실험 실패율 모니터링 및 개선

---

## 🔍 **Phase 3: RAG Performance Evaluation (PLANNED)**

> 📋 **상세 평가 방법론**: [`docs/RAG_EVALUATION_METHODS.md`](docs/RAG_EVALUATION_METHODS.md) 참조

### 🎯 **Priority 1: 기본 검색 성능 평가**

- [ ] **핵심 메트릭 구현 및 측정**
  - [ ] Recall@K, Precision@K, MRR, NDCG@10 계산
  - [ ] 28,560개 조합 자동 평가 파이프라인 구축
  - [ ] RetrievalEvaluator 클래스 구현
  - [ ] 결과 저장 및 CSV 출력 시스템

- [ ] **법적 문서 특화 평가**
  - [ ] 법적 개념 검색 정확도 (위약금, 손해배상 등)
  - [ ] 조문 참조 검색 정확도 ("제3조에 따라" 등)
  - [ ] 계약 조건 검색 정확도 평가
  - [ ] 예외 조건 처리 검색 정확도

### 📈 **Priority 2: 파라미터 분석 및 최적화**

- [ ] **파라미터 민감도 분석**
  - [ ] 청크 크기별 성능 영향 분석
  - [ ] 청크 겹침 비율 영향 분석
  - [ ] 청커 유형별 성능 비교 및 순위
  - [ ] 임베딩 모델별 성능 비교 및 순위

- [ ] **검색 전략 최적화**
  - [ ] Knee Retrieval vs Top-K 비교 연구
  - [ ] 동적 검색 문서 수 결정 시스템
  - [ ] 법적 문서 특성 고려한 검색 전략

### 📊 **Priority 3: 시각화 및 대시보드**

- [ ] **성능 분석 시각화**
  - [ ] 청커-임베딩 조합 성능 히트맵
  - [ ] 파라미터 민감도 시각화 차트
  - [ ] Top 10% 고성능 조합 식별 및 분석
  - [ ] 통계적 유의성 검증 결과

- [ ] **인터랙티브 분석 도구**
  - [ ] t-SNE 임베딩 공간 시각화 (Gradio)
  - [ ] 체크박스 기반 조합 선택 및 비교
  - [ ] 실험 결과 종합 대시보드
  - [ ] 최적 조합 추천 시스템

### 🎯 **데이터 품질 평가 (Phase 2 연계)**

- [ ] **청킹 품질 자동 평가**
  - [ ] 청크 크기 분포, 개수 효율성 측정
  - [ ] 문장 경계 보존률, 법적 용어 분할 오류율
  - [ ] 조문 구조 보존률, 참조 관계 무결성

- [ ] **QA 데이터셋 품질 평가**
  - [ ] 질문 복잡도, 답변 완전성 자동 분석
  - [ ] 법적 용어 포함률, QA 일치도 검증
  - [ ] 다양성 점수 및 품질 메트릭 수집

---

## 🔬 **Phase 4: RAG System Integration (PLANNED)**

### 🎯 **Priority 1: 최적 Retriever 통합**

- [ ] **최고 성능 조합 식별**
  - [ ] 성능 기반 Top-K 청커-임베딩 조합 선정
  - [ ] 도메인별 최적 조합 분석
  - [ ] 효율성 vs 정확도 trade-off 분석

- [ ] **통합 RAG 시스템 구축**
  - [ ] 최적 조합 기반 RAG 파이프라인
  - [ ] Re-ranking 통합 (필요 시)
  - [ ] 성능 검증 및 최종 평가

### 📝 **Priority 2: Documentation & Reproducibility**

- [ ] **연구 결과 문서화**
  - [ ] 실험 결과 종합 보고서
  - [ ] 최적 설정 가이드라인
  - [ ] 재현성 확보를 위한 상세 문서화

---

## 📊 **Phase 5: Academic Contribution**

### 🏆 **Priority 1: 연구 논문 작성**

- [ ] **핵심 논문 작성**
  - [ ] "포괄적 청킹-임베딩 조합의 RAG 성능 영향 분석"
  - [ ] 실험 설계 및 방법론 상세화
  - [ ] 통계적 유의성 검증 및 결과 분석
  - [ ] 학술지 투고 준비

### 📈 **Priority 2: 추가 연구 방향**

- [ ] **확장 연구 계획**
  - [ ] Knee Retrieval 효과성 분석
  - [ ] 다국어 문서에서의 성능 비교
  - [ ] 도메인 특화 최적화 전략

---

## ⚠️ **Known Issues & Technical Considerations**

### 🐛 **Current Issues**

- [ ] **대량 문서 처리 최적화**
  - [ ] 민준 선배 이슈: 전체 매뉴얼 처리 시 성능 저하
  - [ ] 메모리 효율성 개선 (청킹 + 임베딩 동시 처리)
  - [ ] GPU 메모리 관리 최적화

### 🔧 **Technical Improvements**

- [ ] **실험 안정성 향상**
  - [ ] 실험 중단 시 재개 기능 강화
  - [ ] 에러 핸들링 및 로깅 개선
  - [ ] 실험 진행률 모니터링 시스템

---

## 🎯 **Success Metrics**

### 📊 **Phase별 정량적 목표**

#### **Phase 2: Data Generation**
- **청킹 완료율**: 4,760개 조합 중 95% 이상 성공
- **QA 생성 품질**: 청크당 평균 3개 이상의 고품질 QA 쌍
- **임베딩 완료율**: 28,560개 조합 중 95% 이상 성공
- **처리 효율성**: Multi-GPU 환경에서 7배 이상 속도 향상

#### **Phase 3: RAG Performance Evaluation**
- **성능 측정 완료**: 모든 조합에 대한 Recall@K, Precision@K 측정
- **최적 조합 식별**: Top 10% 성능 조합 명확히 식별
- **베이스라인 개선**: all-MiniLM-L6 대비 20% 이상 성능 향상 조합 발견

### 🏆 **최종 목표**

- **학술적 기여**: 청킹-임베딩 조합 효과에 대한 포괄적 분석 논문 1편
- **실용적 가치**: 법적 문서용 최적 Retriever 구성 가이드라인
- **재현성**: 완전한 실험 재현 가능한 오픈소스 프레임워크

---

## 📅 **Revised Timeline**

- **Q1 2025**: Phase 2 완료 (데이터 생성 - 청킹, QA, 임베딩)
- **Q2 2025**: Phase 3 완료 (RAG 성능 평가)
- **Q3 2025**: Phase 4 완료 (최적 시스템 통합)
- **Q4 2025**: Phase 5 완료 (논문 작성 및 발표)

---

## 📚 **References & Resources**

### 🔗 **Key Links**

- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Comprehensive RAG Guide](https://newsletter.armand.so/p/comprehensive-guide-rag-implementations)
- [Agentic RAG Research](https://g.co/gemini/share/a20f9645bbc7)

### 📖 **Technical Documentation**

- `experiment_configurations.py`: 실험 설정 중앙 관리
- `multi_embedding_database_builder.py`: 메인 실험 실행기
- `docs/RAG_EVALUATION_METHODS.md`: 포괄적 RAG 평가 방법론 가이드
- `DATABASE_NAMING_CONVENTION.md`: 데이터베이스 네이밍 규칙
- `templates/ACADEMIC_ANALYSIS.md`: 학술 분석 템플릿

---

*Last Updated: 2025-09-03*
*Current Phase: Phase 2 (Data Generation)*
*Focus: 청킹 → QA 생성 → 임베딩 데이터셋 구축*

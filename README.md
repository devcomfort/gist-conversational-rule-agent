# LiberVance AI - Multi-modal Conversational AI System

## 📚 Documentation

### 📊 **CURRENT_CODEBASE_ANALYSIS.md**
현재 코드베이스의 상세 기술 스택 분석서
- 5개 모듈별 라이브러리 및 기능 분석 (용어 각주 포함)
- 장/단기 메모리 시스템 분석
- 시스템 복잡도 및 의존성 평가
- LangChain 생태계 구현 현황

### 🔄 **LANGCHAIN_REPLACEMENT_STRATEGY_ANALYSIS.md**
LangChain 생태계 대체 전략 및 구현 방안
- 3가지 주요 전략별 상세 구현 방법
- LangGraph, LangChain-Core, Community 패키지별 대체 방안
- 모듈별 우선순위와 단계적 전환 계획
- SmolAgents + LlamaIndex 하이브리드 권장

### 🗂 **DVC_OPERATIONS.md**
DVC 운영 가이드 (Cloudflare R2 원격)
- 초기 설정(uv, dvc, dvc-s3), 원격(R2) 구성
- 워크플로우(dvc add/commit/push, pull), 보안/운영 권장사항

## 🎯 **핵심 대체 전략**
1. **LlamaIndex Workflows**: 구조화된 이벤트 기반 워크플로우 
2. **SmolAgents**: 경량 코드 중심 워크플로우 (권장)
3. **직접 LiteLLM**: 최소 의존성 구현

## 🌟 **핵심 모듈**
- 🔍 **LV-Search**: LangGraph 기반 복합 에이전트 검색 (최고 난이도)
- 📄 **LV-RAG**: FAISS 기반 PDF 문서 질의응답 (중간 난이도) 
- 📊 **LV-RAG-X**: Excel/PDF 확장 처리 시스템 (쉬움)
- 🖼️ **LV-VQA**: 멀티모달 비전-언어 질의응답 (쉬움)
- 🔎 **LV-Search2**: Google Search 기반 웹 검색 (보통)

## 📈 **예상 개선 효과**
- **🔥 50% 코드 단순화**: 복잡한 LangGraph → 직관적 도구 체인
- **⚡ 30% 성능 향상**: 불필요한 추상화 계층 제거  
- **🧩 90% 의존성 감소**: 통합된 라이브러리 사용
- **💰 비용 절감**: LiteLLM 다중 제공자 최적화

---
*LiberVance AI - 복잡성을 줄이고 기능성을 향상시킨 현대적 AI 플랫폼* 🚀✨

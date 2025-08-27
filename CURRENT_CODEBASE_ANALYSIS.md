# LiberVance AI 코드베이스 기술 스택 분석

## 📋 전체 시스템 구조

LiberVance AI는 5개의 핵심 모듈로 구성된 멀티모달 AI 플랫폼입니다.

```
📦 LiberVance AI
├── 🔍 app_lvsearch.py      - 복합 에이전트 검색 시스템
├── 🔍 app_lvsearch2.py     - 단순 검색 시스템  
├── 📄 app_lvrag.py         - PDF RAG¹ 시스템
├── 📊 app_lvragx.py        - Excel/PDF 확장 처리 시스템
└── 🖼️ app_lvvqa.py         - 비전-언어 질의응답 시스템
```

---

## 🔍 **app_lvsearch.py** - 복합 에이전트 검색 시스템

### **핵심 기술 스택**
| 계층                    | 라이브러리            | 용도                                         |
| ----------------------- | --------------------- | -------------------------------------------- |
| **에이전트 프레임워크** | LangGraph²            | StateGraph, MemorySaver 기반 복합 워크플로우 |
| **LLM³**                | langchain-groq⁴       | ChatGroq (LLaMA-3.1-8B-instant)              |
| **검색 엔진**           | tavily-python⁵        | TavilyClient 고품질 웹 검색                  |
| **도구 시스템**         | langchain-core⁶       | StructuredTool, ToolNode, AgentAction        |
| **상태 관리**           | langgraph.checkpoint⁷ | MemorySaver를 통한 세션 메모리               |

### **아키텍처 패턴**
```python
# 복잡한 에이전트 워크플로우
StateGraph(State) → ChatGroq + Tools → ToolNode → Response
     ↓
MemorySaver (장기 메모리)
```

### **메모리 관리 분석**
| 메모리 유형     | 구현 기술                    | 생명주기    | 저장 범위             |
| --------------- | ---------------------------- | ----------- | --------------------- |
| **단기 메모리** | TypedDict State              | 단일 요청   | 워크플로우 노드 간    |
| **장기 메모리** | MemorySaver                  | 세션 지속   | thread_id 기반 지속적 |
| **세션 메모리** | OrderedDict + threading.Lock | 사용자 세션 | LRU 캐시 (최대 100개) |

### **주요 기능**
- 🔄 **복잡한 상태 관리**: TypedDict 기반 State 관리
- 🛠️ **도구 통합**: StructuredTool을 통한 Tavily 검색 통합
- 🧠 **메모리 시스템**: MemorySaver를 통한 대화 컨텍스트 유지
- ⚡ **조건부 실행**: tools_condition을 통한 동적 플로우 제어

---

## 🔍 **app_lvsearch2.py** - 단순 검색 시스템

### **핵심 기술 스택**
| 계층           | 라이브러리         | 용도                                                  |
| -------------- | ------------------ | ----------------------------------------------------- |
| **워크플로우** | LangGraph²         | 선형적 3단계 워크플로우 (Search → Summary → Response) |
| **LLM³**       | langchain-openai⁸  | ChatOpenAI (GPT-4)                                    |
| **프롬프트**   | langchain.prompts⁹ | ChatPromptTemplate                                    |
| **검색 API**   | requests           | Google CSE¹⁰ API                                      |

### **아키텍처 패턴**
```python
# 선형 워크플로우
Search Node → Summary Node → Response Node
     ↓              ↓            ↓
Google CSE¹⁰    ChatPromptTemplate  응답 생성
```

### **주요 기능**
- 🎯 **단순한 파이프라인**: 검색 → 요약 → 응답의 3단계
- 💬 **프롬프트 최적화**: ChatPromptTemplate을 통한 쿼리 개선
- 🌍 **다국어 지원**: 사용자 언어 자동 감지 및 응답
- 📊 **Google 검색**: CSE¹⁰ API 활용

### **메모리 관리 분석**
| 메모리 유형     | 구현 기술                    | 생명주기    | 저장 범위               |
| --------------- | ---------------------------- | ----------- | ----------------------- |
| **단기 메모리** | StateGraph State             | 단일 요청   | 3개 노드 간 데이터 전달 |
| **장기 메모리** | MemorySaver                  | 세션 지속   | thread_id 기반 지속적   |
| **세션 메모리** | OrderedDict + threading.Lock | 사용자 세션 | LRU 캐시 (최대 100개)   |

---

## 📄 **app_lvrag.py** - PDF RAG 시스템

### **핵심 기술 스택**
| 계층                | 라이브러리                 | 용도                                     |
| ------------------- | -------------------------- | ---------------------------------------- |
| **RAG¹ 프레임워크** | langchain-community¹¹      | FAISS¹² VectorStore                      |
| **임베딩**          | langchain-huggingface¹³    | HuggingFaceEmbeddings (all-MiniLM-L6-v2) |
| **텍스트 분할**     | langchain-text-splitters¹⁴ | RecursiveCharacterTextSplitter           |
| **PDF 처리**        | PyMuPDF¹⁵ (fitz)           | PDF 텍스트 추출                          |
| **LLM³**            | openai + huggingface-hub   | 다중 모델 지원 (GPT-4, DeepSeek-R1 등)   |

### **아키텍처 패턴**
```python
# RAG 파이프라인
PDF Upload → PyMuPDF → Text Splitting → FAISS Embedding
                                              ↓
User Query → FAISS¹² Retrieval → Context + Query → LLM³ → Response
```

### **주요 기능**
- 📚 **벡터 검색**: FAISS¹² 기반 의미론적 문서 검색
- 🔄 **다중 모델**: OpenAI, HuggingFace 모델 실시간 전환
- 📄 **PDF 처리**: PyMuPDF¹⁵를 통한 안정적 텍스트 추출
- 💾 **세션 관리**: 사용자별 독립적 벡터스토어 관리

### **메모리 관리 분석**
| 메모리 유형     | 구현 기술                    | 생명주기    | 저장 범위                      |
| --------------- | ---------------------------- | ----------- | ------------------------------ |
| **단기 메모리** | 함수 로컬 변수               | 단일 요청   | 요청 처리 중에만               |
| **장기 메모리** | FAISS¹² VectorStore          | 세션 지속   | PDF 문서 임베딩 영구 저장      |
| **세션 메모리** | OrderedDict + threading.Lock | 사용자 세션 | 대화 기록, 벡터스토어 LRU 관리 |

---

## 📊 **app_lvragx.py** - Excel/PDF 확장 처리 시스템

### **핵심 기술 스택**
| 계층            | 라이브러리       | 용도                                |
| --------------- | ---------------- | ----------------------------------- |
| **LLM**         | openai           | OpenAI o1 모델 (고성능 추론)        |
| **Excel 처리**  | pandas           | Excel 파일 읽기/쓰기, 마크다운 변환 |
| **PDF 처리**    | OpenAI Files API | 직접 파일 업로드 및 처리            |
| **데이터 변환** | re (정규표현식)  | 마크다운 테이블 추출/변환           |

### **아키텍처 패턴**
```python
# 파일 처리 파이프라인
Excel/PDF Upload → pandas/Files API → OpenAI o1 → Markdown Tables
                                                        ↓
                                           Excel Download (선택적)
```

### **주요 기능**
- 🧮 **고급 추론**: OpenAI o1 모델의 강력한 분석 능력
- 📊 **Excel 처리**: pandas 기반 대용량 Excel 파일 처리
- 🔄 **양방향 변환**: Excel ↔ Markdown Table 변환
- ⬇️ **다운로드 기능**: 처리 결과를 Excel 파일로 제공

### **메모리 관리 분석**
| 메모리 유형     | 구현 기술                    | 생명주기    | 저장 범위                       |
| --------------- | ---------------------------- | ----------- | ------------------------------- |
| **단기 메모리** | 함수 로컬 변수               | 단일 요청   | 파일 처리 중에만                |
| **장기 메모리** | 파일 시스템                  | 세션 지속   | Excel 다운로드 파일 임시 저장   |
| **세션 메모리** | OrderedDict + threading.Lock | 사용자 세션 | 대화 기록, 업로드 파일 LRU 관리 |

---

## 🖼️ **app_lvvqa.py** - 비전-언어 질의응답 시스템

### **핵심 기술 스택**
| 계층                     | 라이브러리               | 용도                                    |
| ------------------------ | ------------------------ | --------------------------------------- |
| **Vision-Language 모델** | openai + huggingface-hub | o3, LLaMA-4-Scout, Qwen-2.5-VL 등       |
| **이미지 처리**          | PIL (Pillow)             | 이미지 포맷 변환 및 처리                |
| **인코딩**               | base64                   | 이미지를 Base64 문자열로 변환           |
| **다중 제공자**          | InferenceClient          | HuggingFace, Novita, Nebius, Hyperbolic |

### **아키텍처 패턴**
```python
# 멀티모달 처리
Image Upload → PIL Processing → Base64 Encoding → VL Model → Response
     +                                               ↑
Text Query ────────────────────────────────────────┘
```

### **주요 기능**
- 👁️ **멀티모달 입력**: 이미지 + 텍스트 동시 처리
- 🔄 **다중 VL¹⁶ 모델**: 실시간 모델 전환 지원
- 🖼️ **이미지 최적화**: PIL¹⁷ 기반 효율적 이미지 처리
- 🌐 **다중 제공자**: 다양한 VL¹⁶ 모델 제공자 지원

### **메모리 관리 분석**
| 메모리 유형     | 구현 기술                    | 생명주기    | 저장 범위              |
| --------------- | ---------------------------- | ----------- | ---------------------- |
| **단기 메모리** | Base64 인코딩                | 단일 요청   | 이미지 처리 중에만     |
| **장기 메모리** | 없음                         | -           | 이미지는 저장하지 않음 |
| **세션 메모리** | OrderedDict + threading.Lock | 사용자 세션 | 대화 기록만 LRU 관리   |

---

## 📊 **공통 인프라 스택**

### **UI/UX 계층**
| 기술           | 용도                     |
| -------------- | ------------------------ |
| **Gradio**     | 웹 인터페이스 프레임워크 |
| **Custom CSS** | 반응형 UI 디자인         |

### **세션 관리**
| 기술               | 용도                    |
| ------------------ | ----------------------- |
| **OrderedDict**    | LRU 캐시 기반 세션 관리 |
| **threading.Lock** | 동시성 제어             |
| **hashlib.sha256** | 세션 ID 생성            |

### **데이터 저장**
| 기술            | 용도                   |
| --------------- | ---------------------- |
| **JSON**        | 대화 기록 저장         |
| **파일 시스템** | 로그 및 임시 파일 저장 |

### **환경 관리**
| 기술              | 용도                                |
| ----------------- | ----------------------------------- |
| **python-dotenv** | 환경 변수 관리                      |
| **API 키 관리**   | OpenAI, HuggingFace, Google, Tavily |

---

## 🎯 **시스템별 복잡도 분석**

| 모듈                 | 워크플로우 복잡도 | 메모리 관리          | 주요 기술 의존성        |
| -------------------- | ----------------- | -------------------- | ----------------------- |
| **app_lvsearch.py**  | ⭐⭐⭐⭐⭐ (매우 복잡) | MemorySaver          | LangGraph, Tavily       |
| **app_lvsearch2.py** | ⭐⭐⭐ (보통)        | 세션 기반            | LangGraph, Google CSE¹⁰ |
| **app_lvrag.py**     | ⭐⭐⭐ (보통)        | FAISS¹² Vector Store | LangChain¹⁸, FAISS¹²    |
| **app_lvragx.py**    | ⭐⭐ (단순)         | 기본 세션            | OpenAI, pandas          |
| **app_lvvqa.py**     | ⭐⭐ (단순)         | 기본 세션            | OpenAI, HuggingFace     |

---

## 📈 **현재 시스템의 강점 및 한계**

### **✅ 강점**
- **다양한 모달리티**: 텍스트, 이미지, PDF, Excel 처리
- **유연한 LLM 지원**: 다중 모델 및 제공자 지원
- **안정적 세션 관리**: 사용자별 독립적 상태 관리
- **확장 가능한 아키텍처**: 모듈별 독립적 개발

### **⚠️ 한계**
- **기술 스택 분산**: LangChain¹⁸, OpenAI, HuggingFace 등 다양한 의존성
- **복잡한 워크플로우**: 특히 LangGraph² 기반 시스템의 높은 복잡도  
- **메모리 시스템 불일치**: 모듈별로 다른 메모리 관리 방식
- **중복된 기능**: 세션 관리, 모델 로딩 등 공통 코드 중복

---

## 🔄 **전환 준비도 평가**

각 모듈의 현재 구현을 바탕으로 다음 단계에서 제시될 전환 전략의 적용 가능성을 평가합니다:

| 모듈                 | Agent Workflow 전환      | Memory System 전환 | 우선순위  |
| -------------------- | ------------------------ | ------------------ | --------- |
| **app_lvragx.py**    | 🟢 쉬움 (단순 구조)       | 🟢 쉬움             | **1순위** |
| **app_lvvqa.py**     | 🟢 쉬움 (단순 구조)       | 🟢 쉬움             | **1순위** |
| **app_lvrag.py**     | 🟡 보통 (RAG 복잡성)      | 🟢 쉬움             | **2순위** |
| **app_lvsearch2.py** | 🟡 보통 (워크플로우)      | 🟢 쉬움             | **2순위** |
| **app_lvsearch.py**  | 🔴 어려움 (복잡 에이전트) | 🟡 보통             | **3순위** |

---

## 📚 **용어 및 기술 각주**

1. **RAG**: Retrieval-Augmented Generation. 외부 지식베이스에서 관련 정보를 검색하여 답변 생성 품질을 향상시키는 기법
2. **LangGraph**: LangChain 생태계의 상태 기반 그래프 워크플로우 프레임워크. StateGraph와 노드 간 연결을 통해 복잡한 AI 에이전트 워크플로우 구현
3. **LLM**: Large Language Model. GPT, LLaMA, Gemma 등의 대규모 언어 모델
4. **langchain-groq**: GROQ 하드웨어 가속기를 활용한 초고속 LLM 추론을 위한 LangChain 통합 라이브러리
5. **tavily-python**: Tavily Search API의 Python 클라이언트. AI 최적화된 웹 검색 서비스
6. **langchain-core**: LangChain의 핵심 추상화 계층. Document, Tool, Agent 등의 기본 인터페이스 제공
7. **langgraph.checkpoint**: LangGraph의 체크포인트 시스템. MemorySaver를 통한 워크플로우 상태 영속화
8. **langchain-openai**: OpenAI API를 LangChain 인터페이스로 래핑한 통합 라이브러리
9. **langchain.prompts**: LangChain의 프롬프트 템플릿 시스템. ChatPromptTemplate 등 제공
10. **CSE**: Custom Search Engine. Google의 맞춤형 검색 엔진 서비스
11. **langchain-community**: LangChain 커뮤니티 통합 라이브러리. FAISS, Chroma 등 외부 도구 연결
12. **FAISS**: Facebook AI Similarity Search. Meta에서 개발한 고성능 벡터 유사도 검색 라이브러리
13. **langchain-huggingface**: HuggingFace 모델과 LangChain 연동을 위한 통합 라이브러리
14. **langchain-text-splitters**: LangChain의 텍스트 분할 도구 모음. RecursiveCharacterTextSplitter 등 제공
15. **PyMuPDF**: Python용 PDF 처리 라이브러리. fitz 모듈명으로 import
16. **VL**: Vision-Language. 이미지와 텍스트를 함께 처리하는 멀티모달 AI 모델
17. **PIL**: Python Imaging Library (Pillow). Python의 표준 이미지 처리 라이브러리
18. **LangChain**: LLM 애플리케이션 개발을 위한 종합 프레임워크. 체인, 에이전트, 메모리 등의 추상화 제공

# 📊 RAG 평가 방법론 - 법적 문서 기반 청킹-임베딩 조합 연구

## 🎯 **프로젝트 맞춤 평가 전략**

**데이터**: 법적 문서 (rules 디렉토리)  
**조합**: 28,560개 (6개 청커 × 다양한 파라미터 × 6개 임베딩 모델)  
**목표**: 최적 Retriever 구성을 위한 체계적 성능 분석

---

## 📋 **Phase 2: 데이터 품질 평가 (현재 단계)**

### 🔍 **1. 청킹 품질 평가**

#### **메트릭 정의**
```python
class ChunkQualityEvaluator:
    """청킹 결과 품질 평가"""
    
    def evaluate_chunking_quality(self, chunks, original_docs):
        metrics = {}
        
        # 1) 청크 크기 분포
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        metrics['avg_chunk_size'] = np.mean(chunk_sizes)
        metrics['std_chunk_size'] = np.std(chunk_sizes)
        metrics['chunk_size_cv'] = metrics['std_chunk_size'] / metrics['avg_chunk_size']
        
        # 2) 청크 개수 효율성
        total_content = sum(len(doc.page_content) for doc in original_docs)
        metrics['chunks_per_1k_chars'] = len(chunks) / (total_content / 1000)
        
        # 3) 문장 경계 보존률 (법적 문서 중요)
        sentence_boundary_preserved = self.check_sentence_boundaries(chunks)
        metrics['sentence_boundary_rate'] = sentence_boundary_preserved
        
        # 4) 법적 용어 분할 오류률
        legal_term_split_errors = self.check_legal_term_integrity(chunks)
        metrics['legal_term_error_rate'] = legal_term_split_errors
        
        return metrics
    
    def check_sentence_boundaries(self, chunks):
        """문장이 중간에 끊어진 청크 비율 확인"""
        boundary_preserved = 0
        for chunk in chunks:
            text = chunk.page_content.strip()
            if text.endswith(('.', '!', '?', '다', '음')):  # 한국어 문장 종료
                boundary_preserved += 1
        return boundary_preserved / len(chunks)
    
    def check_legal_term_integrity(self, chunks):
        """법적 용어가 중간에 분할된 경우 확인"""
        legal_terms = ['계약서', '위약금', '손해배상', '법정손해금', '계약해지']
        split_errors = 0
        
        for chunk in chunks:
            content = chunk.page_content
            for term in legal_terms:
                # 용어가 청크 경계에서 분할되었는지 확인
                if self.is_term_split_at_boundary(content, term):
                    split_errors += 1
        
        return split_errors / len(chunks)
```

#### **법적 문서 특화 평가**
```python
def evaluate_legal_document_chunking(chunks):
    """법적 문서 청킹 특화 평가"""
    
    # 1) 조문/항목 구조 보존률
    article_preservation = check_article_structure_preservation(chunks)
    
    # 2) 참조 관계 유지률 (예: "제3조에 따라")
    reference_integrity = check_legal_reference_integrity(chunks)
    
    # 3) 계약 조건 완전성
    contract_clause_completeness = check_contract_clause_completeness(chunks)
    
    return {
        'article_preservation_rate': article_preservation,
        'reference_integrity_rate': reference_integrity,  
        'clause_completeness_rate': contract_clause_completeness
    }
```

### 📝 **2. QA 데이터셋 품질 평가**

#### **자동 품질 평가**
```python
class QAQualityEvaluator:
    """생성된 QA 쌍의 품질 자동 평가"""
    
    def __init__(self, korean_nlp_model="klue/roberta-large"):
        self.nlp_model = korean_nlp_model
        self.legal_keywords = self.load_legal_keywords()
    
    def evaluate_qa_quality(self, qa_pairs, source_chunks):
        metrics = {}
        
        # 1) 질문 복잡도 분석
        question_complexity = self.analyze_question_complexity(qa_pairs)
        metrics['avg_question_complexity'] = np.mean(question_complexity)
        
        # 2) 답변 완전성 (청크 내용 기반)
        answer_completeness = self.check_answer_completeness(qa_pairs, source_chunks)
        metrics['answer_completeness_rate'] = np.mean(answer_completeness)
        
        # 3) 법적 용어 포함률
        legal_term_coverage = self.check_legal_term_coverage(qa_pairs)
        metrics['legal_term_coverage'] = legal_term_coverage
        
        # 4) 질문-답변 일치도
        qa_consistency = self.measure_qa_consistency(qa_pairs)
        metrics['qa_consistency_score'] = np.mean(qa_consistency)
        
        # 5) 다양성 점수
        diversity_score = self.calculate_qa_diversity(qa_pairs)
        metrics['qa_diversity_score'] = diversity_score
        
        return metrics
    
    def analyze_question_complexity(self, qa_pairs):
        """질문 복잡도 분석 (문장 길이, 의문사 유형, 법적 개념 포함)"""
        complexity_scores = []
        
        for qa in qa_pairs:
            question = qa['question']
            
            # 문장 길이 점수
            length_score = min(len(question) / 100, 1.0)
            
            # 의문사 복잡도 (누가, 언제 < 어떻게, 왜)
            complexity_words = ['어떻게', '왜', '어떤 조건', '어떤 경우']
            complexity_score = sum(1 for word in complexity_words if word in question)
            
            # 법적 개념 포함 점수
            legal_concept_score = sum(1 for term in self.legal_keywords if term in question) / len(self.legal_keywords)
            
            total_score = (length_score + complexity_score + legal_concept_score) / 3
            complexity_scores.append(total_score)
        
        return complexity_scores
```

---

## 🔍 **Phase 3: Retrieval 성능 평가 (핵심 단계)**

### 📊 **1. 기본 검색 성능 메트릭**

#### **구현 가능한 핵심 메트릭**
```python
class RetrievalEvaluator:
    """28,560개 조합 대상 검색 성능 평가"""
    
    def __init__(self, qa_dataset, vector_databases_dir):
        self.qa_dataset = qa_dataset  # Phase 2에서 생성된 QA 쌍
        self.db_dir = vector_databases_dir
        
    def evaluate_all_combinations(self):
        """모든 청킹-임베딩 조합 평가"""
        results = {}
        
        # 모든 FAISS 데이터베이스 로드
        db_paths = self.get_all_database_paths()
        
        for db_path in tqdm(db_paths, desc="Evaluating combinations"):
            combination_id = self.extract_combination_id(db_path)
            
            # 벡터스토어 로드
            vectorstore = self.load_vectorstore(db_path)
            
            # 검색 성능 평가
            metrics = self.evaluate_single_combination(vectorstore)
            
            results[combination_id] = metrics
            
            # 메모리 정리
            del vectorstore
            gc.collect()
        
        return self.analyze_results(results)
    
    def evaluate_single_combination(self, vectorstore):
        """단일 조합 평가"""
        metrics = {
            'recall_at_1': [], 'recall_at_5': [], 'recall_at_10': [],
            'precision_at_1': [], 'precision_at_5': [], 'precision_at_10': [],
            'mrr': [], 'ndcg_at_10': [], 'map': []
        }
        
        for qa_item in self.qa_dataset:
            query = qa_item['question']
            relevant_chunk_ids = qa_item['relevant_chunks']
            
            # 검색 수행 (다양한 K값)
            search_results = {
                k: vectorstore.similarity_search_with_score(query, k=k)
                for k in [1, 5, 10]
            }
            
            # 메트릭 계산
            for k in [1, 5, 10]:
                retrieved_ids = [doc.metadata.get('chunk_id') for doc, _ in search_results[k]]
                
                # Recall@K
                recall_k = len(set(retrieved_ids) & set(relevant_chunk_ids)) / len(relevant_chunk_ids)
                metrics[f'recall_at_{k}'].append(recall_k)
                
                # Precision@K
                precision_k = len(set(retrieved_ids) & set(relevant_chunk_ids)) / k
                metrics[f'precision_at_{k}'].append(precision_k)
            
            # MRR 계산
            mrr_score = self.calculate_mrr(search_results[10], relevant_chunk_ids)
            metrics['mrr'].append(mrr_score)
            
            # NDCG@10 계산
            ndcg_score = self.calculate_ndcg(search_results[10], relevant_chunk_ids)
            metrics['ndcg_at_10'].append(ndcg_score)
        
        # 평균 계산
        return {metric: np.mean(scores) for metric, scores in metrics.items()}
    
    def calculate_mrr(self, search_results, relevant_ids):
        """Mean Reciprocal Rank 계산"""
        for rank, (doc, _) in enumerate(search_results, 1):
            chunk_id = doc.metadata.get('chunk_id')
            if chunk_id in relevant_ids:
                return 1.0 / rank
        return 0.0
    
    def calculate_ndcg(self, search_results, relevant_ids, k=10):
        """Normalized Discounted Cumulative Gain 계산"""
        dcg = 0.0
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_ids), k)))
        
        for rank, (doc, _) in enumerate(search_results[:k], 1):
            chunk_id = doc.metadata.get('chunk_id')
            if chunk_id in relevant_ids:
                dcg += 1.0 / math.log2(rank + 1)
        
        return dcg / idcg if idcg > 0 else 0.0
```

### 🎯 **2. 법적 문서 특화 평가**

#### **도메인 특화 메트릭**
```python
class LegalRAGEvaluator:
    """법적 문서 특화 RAG 평가"""
    
    def evaluate_legal_retrieval(self, vectorstore, legal_queries):
        """법적 검색 특화 평가"""
        
        # 1) 법적 개념 검색 정확도
        legal_concept_accuracy = self.evaluate_legal_concept_retrieval(
            vectorstore, legal_queries['concept_queries']
        )
        
        # 2) 조문 참조 검색 정확도
        article_reference_accuracy = self.evaluate_article_reference_retrieval(
            vectorstore, legal_queries['article_queries']
        )
        
        # 3) 계약 조건 검색 정확도  
        contract_clause_accuracy = self.evaluate_contract_clause_retrieval(
            vectorstore, legal_queries['clause_queries']
        )
        
        # 4) 예외 조건 검색 정확도
        exception_handling_accuracy = self.evaluate_exception_handling_retrieval(
            vectorstore, legal_queries['exception_queries']
        )
        
        return {
            'legal_concept_accuracy': legal_concept_accuracy,
            'article_reference_accuracy': article_reference_accuracy,
            'contract_clause_accuracy': contract_clause_accuracy,
            'exception_handling_accuracy': exception_handling_accuracy
        }
    
    def evaluate_legal_concept_retrieval(self, vectorstore, concept_queries):
        """법적 개념 검색 평가 (예: 위약금, 손해배상)"""
        accuracies = []
        
        for query_item in concept_queries:
            query = query_item['question']  # "위약금은 언제 지불하나요?"
            expected_concepts = query_item['legal_concepts']  # ['위약금', '지급시기']
            
            # 검색 수행
            results = vectorstore.similarity_search(query, k=5)
            
            # 검색된 문서에서 법적 개념 추출
            retrieved_concepts = self.extract_legal_concepts_from_results(results)
            
            # 일치율 계산
            concept_match_rate = len(set(retrieved_concepts) & set(expected_concepts)) / len(expected_concepts)
            accuracies.append(concept_match_rate)
        
        return np.mean(accuracies)
```

### 📈 **3. 파라미터 민감도 분석**

#### **청킹 파라미터 영향 분석**
```python
def analyze_chunking_parameter_sensitivity():
    """청킹 파라미터가 검색 성능에 미치는 영향 분석"""
    
    # 결과 데이터 로드 (28,560개 조합 결과)
    results_df = pd.read_csv('experiments/outputs/experiment_results.csv')
    
    # 파라미터별 분석
    parameter_analysis = {}
    
    # 1) 청크 크기 영향 분석
    chunk_size_analysis = results_df.groupby('chunk_size').agg({
        'recall_at_5': ['mean', 'std'],
        'precision_at_5': ['mean', 'std'],
        'mrr': ['mean', 'std']
    })
    parameter_analysis['chunk_size'] = chunk_size_analysis
    
    # 2) 청크 겹침 비율 영향 분석
    overlap_analysis = results_df.groupby('chunk_overlap').agg({
        'recall_at_5': ['mean', 'std'],
        'precision_at_5': ['mean', 'std']
    })
    parameter_analysis['chunk_overlap'] = overlap_analysis
    
    # 3) 청커 유형별 성능 비교
    chunker_comparison = results_df.groupby('chunker_name').agg({
        'recall_at_5': ['mean', 'std', 'max'],
        'precision_at_5': ['mean', 'std', 'max'],
        'mrr': ['mean', 'std', 'max']
    })
    parameter_analysis['chunker_type'] = chunker_comparison
    
    # 4) 임베딩 모델별 성능 비교
    embedding_comparison = results_df.groupby('embedding_name').agg({
        'recall_at_5': ['mean', 'std', 'max'],
        'precision_at_5': ['mean', 'std', 'max']
    })
    parameter_analysis['embedding_model'] = embedding_comparison
    
    return parameter_analysis

def generate_sensitivity_report(parameter_analysis):
    """파라미터 민감도 분석 보고서 생성"""
    
    report = {
        'key_findings': {},
        'recommendations': {},
        'statistical_significance': {}
    }
    
    # 청크 크기 최적 범위 식별
    chunk_size_perf = parameter_analysis['chunk_size']
    best_chunk_size = chunk_size_perf['recall_at_5']['mean'].idxmax()
    report['key_findings']['optimal_chunk_size'] = best_chunk_size
    
    # 청커 유형 순위
    chunker_ranking = parameter_analysis['chunker_type']['recall_at_5']['mean'].sort_values(ascending=False)
    report['key_findings']['chunker_ranking'] = chunker_ranking.to_dict()
    
    # 임베딩 모델 순위
    embedding_ranking = parameter_analysis['embedding_model']['recall_at_5']['mean'].sort_values(ascending=False)
    report['key_findings']['embedding_ranking'] = embedding_ranking.to_dict()
    
    return report
```

---

## 🎨 **시각화 및 분석 도구**

### 📊 **1. 성능 매트릭스 히트맵**

```python
def create_performance_heatmap(results_df):
    """청커-임베딩 조합 성능 히트맵"""
    
    # 피벗 테이블 생성
    heatmap_data = results_df.pivot_table(
        values='recall_at_5',
        index='chunker_name', 
        columns='embedding_name',
        aggfunc='mean'
    )
    
    # 히트맵 생성
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.3f')
    plt.title('청커-임베딩 조합별 Recall@5 성능')
    plt.tight_layout()
    plt.savefig('performance_heatmap.png', dpi=300)
    
    return heatmap_data

def create_parameter_sensitivity_plots(parameter_analysis):
    """파라미터 민감도 시각화"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 청크 크기별 성능
    chunk_size_data = parameter_analysis['chunk_size']['recall_at_5']['mean']
    axes[0,0].plot(chunk_size_data.index, chunk_size_data.values, marker='o')
    axes[0,0].set_title('청크 크기별 Recall@5')
    axes[0,0].set_xlabel('청크 크기')
    axes[0,0].set_ylabel('Recall@5')
    
    # 청커 유형별 박스플롯
    chunker_box_data = results_df.boxplot(column='recall_at_5', by='chunker_name', ax=axes[0,1])
    axes[0,1].set_title('청커 유형별 성능 분포')
    
    # 임베딩 모델별 성능
    embedding_data = parameter_analysis['embedding_model']['recall_at_5']['mean']
    axes[1,0].bar(range(len(embedding_data)), embedding_data.values)
    axes[1,0].set_xticks(range(len(embedding_data)))
    axes[1,0].set_xticklabels(embedding_data.index, rotation=45)
    axes[1,0].set_title('임베딩 모델별 평균 성능')
    
    plt.tight_layout()
    plt.savefig('parameter_sensitivity.png', dpi=300)
```

### 🔍 **2. t-SNE 임베딩 공간 분석**

```python
def create_embedding_space_analysis(vector_databases, sample_queries):
    """임베딩 공간 시각화를 통한 정성적 분석"""
    
    embeddings_analysis = {}
    
    for db_name, vectorstore in vector_databases.items():
        # 샘플 쿼리에 대한 임베딩 생성
        query_embeddings = []
        doc_embeddings = []
        
        for query in sample_queries:
            # 쿼리 임베딩
            query_emb = vectorstore.embedding_function.embed_query(query)
            query_embeddings.append(query_emb)
            
            # 검색된 문서 임베딩
            results = vectorstore.similarity_search(query, k=5)
            for doc in results:
                doc_emb = vectorstore.embedding_function.embed_documents([doc.page_content])
                doc_embeddings.extend(doc_emb)
        
        # t-SNE 차원 축소
        all_embeddings = query_embeddings + doc_embeddings
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(all_embeddings)
        
        embeddings_analysis[db_name] = {
            'embeddings_2d': embeddings_2d,
            'query_count': len(query_embeddings),
            'doc_count': len(doc_embeddings)
        }
    
    return embeddings_analysis

def create_interactive_tsne_dashboard(embeddings_analysis, combination_metadata):
    """Gradio 기반 t-SNE 인터랙티브 대시보드"""
    
    def update_tsne_plot(chunker_type, embedding_model, threshold):
        # 선택된 조합에 해당하는 t-SNE 데이터 로드
        combination_key = f"{chunker_type}_{embedding_model}"
        
        if combination_key in embeddings_analysis:
            data = embeddings_analysis[combination_key]
            
            # 플롯 생성
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 쿼리 포인트 (빨간색)
            query_points = data['embeddings_2d'][:data['query_count']]
            ax.scatter(query_points[:, 0], query_points[:, 1], 
                      c='red', s=100, alpha=0.7, label='Queries')
            
            # 문서 포인트 (파란색)
            doc_points = data['embeddings_2d'][data['query_count']:]
            ax.scatter(doc_points[:, 0], doc_points[:, 1], 
                      c='blue', s=50, alpha=0.5, label='Documents')
            
            ax.set_title(f'{chunker_type} + {embedding_model}')
            ax.legend()
            
            return fig
        else:
            return plt.figure()
    
    # Gradio 인터페이스
    with gr.Blocks() as demo:
        with gr.Row():
            chunker_dropdown = gr.Dropdown(
                choices=['TokenChunker', 'SentenceChunker', 'SemanticChunker', 
                        'LateChunker', 'NeuralChunker', 'RecursiveChunker'],
                value='SentenceChunker',
                label="청커 유형"
            )
            embedding_dropdown = gr.Dropdown(
                choices=['qwen3_8b', 'qwen3_0_6b', 'jina_v3', 'bge_m3', 
                        'all_minilm_l6', 'multilingual_e5'],
                value='all_minilm_l6',
                label="임베딩 모델"
            )
        
        tsne_plot = gr.Plot(label="t-SNE 임베딩 공간")
        
        # 상호작용 설정
        for input_component in [chunker_dropdown, embedding_dropdown]:
            input_component.change(
                update_tsne_plot,
                inputs=[chunker_dropdown, embedding_dropdown],
                outputs=[tsne_plot]
            )
    
    return demo
```

---

## 📋 **실행 계획 및 구현 순서**

### 🚀 **Phase 3-1: 기본 평가 시스템 (Week 1-2)**

```python
# 1순위 구현 목록
priority_1_implementation = [
    "RetrievalEvaluator 클래스 구현",
    "기본 메트릭 (Recall@K, Precision@K, MRR) 계산",
    "28,560개 조합 자동 평가 파이프라인",
    "결과 저장 및 CSV 출력"
]

# 구현 코드 템플릿
evaluation_pipeline = """
python -c "
from src.evaluation.retrieval_evaluator import RetrievalEvaluator
evaluator = RetrievalEvaluator('qa_dataset.json', 'experiments/outputs/')
results = evaluator.evaluate_all_combinations()
results.to_csv('evaluation_results.csv')
"
"""
```

### 📊 **Phase 3-2: 분석 및 시각화 (Week 3-4)**

```python
# 2순위 구현 목록  
priority_2_implementation = [
    "파라미터 민감도 분석 시스템",
    "성능 히트맵 및 시각화 도구",
    "법적 문서 특화 메트릭 구현",
    "통계적 유의성 검증"
]
```

### 🎨 **Phase 3-3: 대시보드 및 보고서 (Week 5-6)**

```python
# 3순위 구현 목록
priority_3_implementation = [
    "Gradio 평가 대시보드 구축",
    "t-SNE 인터랙티브 시각화",
    "종합 성능 보고서 자동 생성",
    "최적 조합 추천 시스템"
]
```

---

## 🎯 **기대 성과 및 활용 방안**

### 📈 **정량적 성과**
- **28,560개 조합** 체계적 평가 완료
- **Top 10% 성능 조합** 명확히 식별 (약 2,856개)
- **베이스라인 대비 성능 개선률** 정량적 측정
- **파라미터별 기여도** 통계적 검증

### 🎓 **학술적 기여**
- **포괄적 비교 연구**: 기존 연구 대비 10배 이상 실험 규모
- **법적 문서 특화**: 도메인 특화 RAG 평가 방법론 제시
- **재현 가능한 프레임워크**: 오픈소스 평가 도구 제공

### 💼 **실용적 활용**
- **최적 설정 가이드**: 법적 문서용 Retriever 구성 가이드라인
- **성능 예측 모델**: 새로운 조합의 성능 예측 시스템
- **비용 효율성 분석**: 성능 대비 계산 비용 최적화

---

## 📚 **참고 자료 및 도구**

### 🛠️ **구현 도구**
```python
# 필요한 라이브러리
required_libraries = [
    "scikit-learn",      # 평가 메트릭
    "numpy", "pandas",   # 데이터 처리
    "matplotlib", "seaborn",  # 시각화
    "gradio",           # 대시보드
    "tqdm",             # 진행률 표시
    "scipy"             # 통계 검증
]

# 설치 명령
pip install scikit-learn numpy pandas matplotlib seaborn gradio tqdm scipy
```

### 📖 **평가 기준 참고**
- **MS MARCO**: 대규모 검색 평가 표준
- **BEIR**: 정보 검색 벤치마크  
- **RAGAS**: RAG 특화 평가 프레임워크
- **TruLens**: RAG 시스템 평가 및 모니터링

---

*이 문서는 현재 프로젝트 데이터와 상황에 특화된 실행 가능한 평가 방법들을 정리한 것으로, Phase 3 구현 시 직접 활용 가능합니다.*

**Last Updated**: 2025-09-03  
**Status**: Ready for Phase 3 Implementation

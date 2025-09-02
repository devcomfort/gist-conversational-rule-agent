from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.llms.litellm import LiteLLM
from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
import json
import textwrap

from dotenv import load_dotenv

load_dotenv()

text_splitter = TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=128)
title_extractor = TitleExtractor(
    nodes=5,
    llm=LiteLLM(
        model="fireworks_ai/accounts/fireworks/models/gpt-oss-20b",
    ),
)
qa_extractor = QuestionsAnsweredExtractor(
    questions=3,
    llm=LiteLLM(
        model="fireworks_ai/accounts/fireworks/models/gpt-oss-20b",
    ),
)

# assume documents are defined -> extract nodes

pipeline = IngestionPipeline(
    transformations=[
        text_splitter,
        title_extractor,
        qa_extractor,
    ]
)

documents = [
    Document(
        text=(
            "LlamaIndex는 LLM 애플리케이션을 위한 데이터 프레임워크입니다. "
            "텍스트 분할, 타이틀 추출, Q&A 추출과 같은 파이프라인 기반 전처리를 지원합니다. "
            "이 문서는 라이브러리 소개와 핵심 기능을 간단히 설명합니다."
        ),
        metadata={"source": "sample_1", "topic": "llamaindex", "lang": "ko"},
    ),
    Document(
        text=(
            "Svelte는 반응형 UI 프레임워크로, 컴파일 단계에서 최소한의 런타임을 생성합니다. "
            "상태 변화에 따른 업데이트가 효율적이며, 작은 단위 컴포넌트 구조를 권장합니다. "
            "본 문서는 Svelte의 장점과 사용 예시를 담고 있습니다."
        ),
        metadata={"source": "sample_2", "topic": "svelte", "lang": "ko"},
    ),
]

nodes = pipeline.run(
    documents=documents,
    in_place=True,
    show_progress=True,
    num_workers=2,
)


def _hr(char: str = "=", width: int = 80) -> None:
    print(char * width)


def _section(title: str, width: int = 80) -> None:
    print()
    _hr("=", width)
    print(title)
    _hr("=", width)


def _subsection(title: str, width: int = 80) -> None:
    print()
    print(title)
    _hr("-", width)


def _wrap(text: str, width: int = 80, indent: int = 2) -> str:
    cleaned = (text or "").replace("\n", " ").strip()
    return textwrap.fill(
        cleaned,
        width=width,
        initial_indent=" " * indent,
        subsequent_indent=" " * indent,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _kv(label: str, value) -> None:
    printable = "-" if value is None or value == "" else str(value)
    print(f"{label:>12}: {printable}")


_section("Ingestion Result")
print(f"Total nodes: {len(nodes)}")

for idx, node in enumerate(nodes, start=1):
    _section(f"Node {idx}")

    # Basic
    metadata = getattr(node, "metadata", {}) or {}
    node_id = getattr(node, "node_id", getattr(node, "id_", None))
    _subsection("Basic")
    _kv("Node ID", node_id)
    _kv("Source", metadata.get("source"))
    _kv("Topic", metadata.get("topic"))
    _kv("Lang", metadata.get("lang"))

    # Title
    title = metadata.get("document_title")
    if title:
        _subsection("Title")
        print(_wrap(title))

    # Questions (from QuestionsAnsweredExtractor)
    questions_md = metadata.get("questions_this_excerpt_can_answer")
    if questions_md:
        _subsection("Questions")
        print(_wrap(questions_md, width=80, indent=2))

    # Text snippet
    try:
        content = node.get_content()
    except Exception:
        content = getattr(node, "text", "")
    _subsection("Text Snippet")
    snippet = (content or "").strip()
    if len(snippet) > 800:
        snippet = snippet[:800] + "..."
    print(_wrap(snippet))

    # Other metadata
    excluded = {
        "source",
        "topic",
        "lang",
        "document_title",
        "questions_this_excerpt_can_answer",
    }
    other_keys = [k for k in metadata.keys() if k not in excluded]
    if other_keys:
        _subsection("Other Metadata")
        other = {k: metadata.get(k) for k in other_keys}
        print(json.dumps(other, ensure_ascii=False, indent=2))

from chonkie import AutoEmbeddings


def create_embedding_model(model_name: str):
    """임베딩 모델을 생성합니다.

    Args:
        model_name: Hugging Face 등에서 제공하는 임베딩 모델 이름 또는 경로.

    Returns:
        임베딩 모델 인스턴스.

    Raises:
        ValueError: 모델 이름이 잘못되었거나 다운로드/초기화에 실패한 경우.

    Example:
        >>> emb = create_embedding_model("Qwen/Qwen3-Embedding-8B")
    """
    try:
        return AutoEmbeddings.get_embeddings(model_name)
    except Exception as e:
        # 친절한 가이드 포함 에러 메시지
        hint = (
            "모델명이 올바른지 확인하세요. 예: 'BAAI/bge-m3', 'intfloat/multilingual-e5-small',\n"
            "네트워크 연결 및 권한(Private 모델)도 확인이 필요할 수 있습니다."
        )
        raise ValueError(
            f"임베딩 모델 초기화 실패: {model_name}\n{hint}\n원본 에러: {e}"
        ) from e

from __future__ import annotations

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core.config import get_settings


def get_chat_model(temperature: float = 0.2, streaming: bool = False) -> ChatOpenAI:
    settings = get_settings()
    model_name = settings.streaming_chat_model if streaming else settings.chat_model
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        streaming=streaming,
    )


def get_vision_model(temperature: float = 0.2) -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.vision_model,
        temperature=temperature,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


def get_embedding_model() -> OpenAIEmbeddings:
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        dimensions=settings.embedding_dimensions,
        chunk_size=settings.embedding_batch_size,
        check_embedding_ctx_length=False,
    )

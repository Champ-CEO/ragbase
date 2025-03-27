from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_groq import ChatGroq

from ragbase.config import Config


def create_llm(use_complex_model=False) -> BaseLanguageModel:
    model_name = Config.Model.COMPLEX_LLM if use_complex_model else Config.Model.GENERAL_LLM
    return ChatGroq(
        temperature=Config.Model.TEMPERATURE,
        model_name=model_name,
        max_tokens=Config.Model.MAX_TOKENS,
    )


def create_embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=Config.Model.EMBEDDINGS)


def create_reranker() -> FlashrankRerank:
    return FlashrankRerank(model=Config.Model.RERANKER)

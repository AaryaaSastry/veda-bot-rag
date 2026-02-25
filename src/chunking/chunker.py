import tiktoken
from typing import List
from structure.schema import KnowledgeChunk
from structure.metadata_extractor import (
    detect_dosha,
    detect_category,
    extract_topic,
    detect_disease_type,
    detect_srotas,
    detect_treatment_type,
    detect_level_of_care,
    detect_formulation_type,
)


class TokenChunker:
    def __init__(self, model_name="gpt-4", chunk_size=700, overlap=100):
        self.encoding = tiktoken.encoding_for_model(model_name)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        tokens = self.encoding.encode(text)
        chunks = []

        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)

            start += self.chunk_size - self.overlap

        return chunks


def chunk_text(text, chunk_size=800):
    """
    Splits text into chunks of roughly equal word counts.
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def create_structured_chunk(text, source, chapter):
    """
    Wraps a text chunk with metadata extracted from its content.
    """
    return KnowledgeChunk(
        text=text,
        source=source,
        chapter=chapter,
        topic=extract_topic(text),
        dosha=detect_dosha(text),
        category=detect_category(text),
        disease_type=detect_disease_type(text),
        srotas=detect_srotas(text),
        treatment_type=detect_treatment_type(text),
        level_of_care=detect_level_of_care(text),
        formulation_type=detect_formulation_type(text),
    )


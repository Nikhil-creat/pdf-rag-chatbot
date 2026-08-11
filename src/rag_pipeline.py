"""
Orchestrates the full RAG loop:

  ingest:  PDF -> chunks -> embeddings -> vector store
  answer:  question -> embed -> retrieve top-k chunks -> build prompt -> LLM

The prompt-engineering piece lives in `_build_prompt`. Two choices matter
most there and are worth being able to explain:
  1. The model is explicitly told to answer ONLY from the provided context
     and to say when it doesn't know — this is what keeps a RAG chatbot
     from hallucinating answers the source document never contained.
  2. Each retrieved chunk is tagged with its page number so the model can
     (and is instructed to) cite pages in its answer.
"""

from dataclasses import dataclass, field

import anthropic

from .config import settings
from .embeddings import get_embedding_model
from .pdf_processor import process_pdf
from .vector_store import VectorStore, SearchResult


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "excerpts provided below, which were retrieved from a user-uploaded PDF. "
    "Rules:\n"
    "1. Base your answer strictly on the provided excerpts. Do not use "
    "outside knowledge.\n"
    "2. If the excerpts don't contain enough information to answer, say so "
    "plainly instead of guessing.\n"
    "3. When you use a fact from an excerpt, cite its page number in "
    "parentheses, e.g. (p. 4).\n"
    "4. Be concise and direct."
)


@dataclass
class AnswerResult:
    answer: str
    sources: list[SearchResult] = field(default_factory=list)


class RAGPipeline:
    def __init__(self):
        self.embedder = get_embedding_model(settings.embedding_model)
        self.store = VectorStore(dimension=self.embedder.dimension)
        self._client: anthropic.Anthropic | None = None
        self.ingested_files: list[str] = []

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            if not settings.anthropic_api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Add it to your .env file."
                )
            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        return self._client

    def ingest_pdf(self, file_path: str, source_name: str) -> int:
        """Process a PDF, embed its chunks, and add them to the store.
        Returns the number of chunks added."""
        chunks = process_pdf(
            file_path,
            source_name,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(texts)
        records = [
            {"text": c.text, "page": c.page, "source": c.source} for c in chunks
        ]
        self.store.add(embeddings, records)
        self.ingested_files.append(source_name)
        return len(chunks)

    def _retrieve(self, question: str) -> list[SearchResult]:
        query_embedding = self.embedder.encode_one(question)
        return self.store.search(
            query_embedding,
            top_k=settings.top_k,
            min_score=settings.similarity_threshold,
        )

    def _build_prompt(self, question: str, sources: list[SearchResult]) -> str:
        if not sources:
            context = "(No relevant excerpts were found in the document.)"
        else:
            context = "\n\n".join(
                f"[Excerpt from page {s.page}]\n{s.text}" for s in sources
            )
        return (
            f"Context excerpts:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer the question using only the context above."
        )

    def answer_query(self, question: str) -> AnswerResult:
        if self.store.is_empty:
            return AnswerResult(
                answer="No PDF has been processed yet — upload one first.",
                sources=[],
            )

        sources = self._retrieve(question)
        prompt = self._build_prompt(question, sources)

        response = self.client.messages.create(
            model=settings.llm_model,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        answer_text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        return AnswerResult(answer=answer_text, sources=sources)

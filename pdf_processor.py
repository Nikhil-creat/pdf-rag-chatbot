"""
PDF ingestion: extract text page-by-page, then split it into overlapping
chunks suitable for embedding.

Two concepts worth knowing for an interview:
  - Overlap between chunks prevents a sentence that straddles a chunk
    boundary from losing its meaning in both halves.
  - We keep page numbers attached to every chunk so retrieved answers
    can cite "page 3" instead of just "somewhere in the PDF".
"""

from dataclasses import dataclass

import pdfplumber


@dataclass
class Chunk:
    text: str
    page: int
    chunk_index: int
    source: str


def extract_pages(file_path: str) -> list[tuple[int, str]]:
    """Return a list of (page_number, page_text) tuples, 1-indexed."""
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append((i, text))
    return pages


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Word-boundary-aware sliding window splitter.

    Splits on whitespace so we never cut a word in half, advances by
    (chunk_size - overlap) characters each step, and always makes
    forward progress even if a single "word" is longer than chunk_size.
    """
    if len(text) <= chunk_size:
        return [text]

    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        added_len = len(word) + (1 if current else 0)
        if current_len + added_len > chunk_size and current:
            chunks.append(" ".join(current))
            # keep the tail of the previous chunk for overlap
            overlap_words: list[str] = []
            overlap_len = 0
            for w in reversed(current):
                overlap_len += len(w) + 1
                if overlap_len > overlap:
                    break
                overlap_words.insert(0, w)
            current = overlap_words
            current_len = sum(len(w) + 1 for w in current)

        current.append(word)
        current_len += added_len

    if current:
        chunks.append(" ".join(current))

    return chunks


def process_pdf(file_path: str, source_name: str, chunk_size: int, overlap: int) -> list[Chunk]:
    """Extract text from a PDF and return a flat list of Chunk objects."""
    pages = extract_pages(file_path)
    chunks: list[Chunk] = []
    idx = 0
    for page_num, page_text in pages:
        for piece in _split_text(page_text, chunk_size, overlap):
            chunks.append(Chunk(text=piece, page=page_num, chunk_index=idx, source=source_name))
            idx += 1
    return chunks

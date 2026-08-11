# AI PDF Chatbot (RAG)

Upload a PDF and ask questions about it. Answers are grounded in the
document's actual content using retrieval-augmented generation (RAG) —
the model never sees the whole PDF, only the specific chunks retrieved
for each question.

## How it works

```
PDF → extract text (pdfplumber) → chunk with overlap
    → embed chunks (sentence-transformers, local) → store in FAISS
                                                            │
User question → embed question ──────────────────────► search FAISS
                                                            │
                                          top-k relevant chunks
                                                            │
                              build prompt (chunks + question)
                                                            │
                                    Claude API generates answer
                                     (cites page numbers, refuses
                                      to answer outside the context)
```

**Why this architecture:**
- **Embeddings run locally** (`sentence-transformers`, `all-MiniLM-L6-v2`) —
  free, no API key, and embedding is the highest-volume operation in a RAG
  system (every chunk on ingest, every query on search), so keeping it
  off a paid API keeps cost flat regardless of corpus size.
- **FAISS** stores vectors as a flat inner-product index. Vectors are
  L2-normalized on the way in, so inner product behaves as cosine
  similarity — simpler and faster than computing cosine distance by hand.
- **Only the generation step calls the Anthropic API**, and only with the
  handful of retrieved chunks, not the full document — this is what keeps
  the app cheap even on long PDFs.
- **Prompt engineering**: the system prompt instructs the model to answer
  only from the provided excerpts, admit when it doesn't know, and cite
  page numbers — this is what prevents hallucinated answers.

## Project structure

```
pdf-rag-chatbot/
├── app.py                 # Streamlit UI
├── src/
│   ├── config.py           # all tunables (chunk size, top-k, model names...)
│   ├── pdf_processor.py    # text extraction + chunking
│   ├── embeddings.py       # sentence-transformers wrapper
│   ├── vector_store.py     # FAISS wrapper
│   └── rag_pipeline.py     # ties retrieval + prompt + generation together
├── requirements.txt
└── .env.example
```

## Setup

```bash
git clone <your-repo-url>
cd pdf-rag-chatbot
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and add your ANTHROPIC_API_KEY (console.anthropic.com)

streamlit run app.py
```

First run will download the embedding model (~90 MB) — after that it's
fully offline for indexing.

## Usage

1. Upload a PDF in the sidebar and click **Process PDF**.
2. Ask questions in the chat box.
3. Expand **Sources** under any answer to see which page(s) it drew from.

## Known limitations

- No OCR — scanned/image-only PDFs won't extract text (pdfplumber reads
  embedded text layers only).
- In-memory FAISS index — re-upload PDFs after restarting the app (no
  persistence layer yet; `src/vector_store.py` is the place to add one).
- Single-user session state — not built for concurrent multi-user use.

## Possible extensions

- Persist the FAISS index to disk (`faiss.write_index`) so PDFs don't
  need re-indexing on restart.
- Add OCR (`pytesseract`) for scanned documents.
- Support multiple PDFs with per-document filtering.
- Swap the flat FAISS index for `IndexIVFFlat` if the corpus grows large
  enough that brute-force search becomes slow.

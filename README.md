# rag-doc-qa

A small Retrieval-Augmented Generation (RAG) system that answers questions
about a PDF document, with citations to the text chunks used.
Multilingual embeddings are used so English and Persian queries can be matched.

> Status: work in progress. Currently a command-line prototype.

## How it works

1. Extract text from a PDF and split it into overlapping chunks
2. Embed each chunk with `intfloat/multilingual-e5-base`
3. Embed the question and retrieve the top-3 most similar chunks (cosine similarity)
4. Send the chunks and the question to Gemini, which answers using only that context
   and cites chunk numbers

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

1. Create an API key in Google AI Studio
2. Copy `.env.example` to `.env` and put your key in it
3. Put a text-based PDF in `data/` and set its filename in `rag_answer.py`
   (I tested with the MobileNetV2 paper, arXiv 1801.04381)

```bash
python rag_answer.py
```

## Example

```
Question: What is SSDLite?
Answer: SSDLite is a mobile-friendly variant of SSD in which regular
convolutions in the prediction layers are replaced with separable
convolutions, reducing parameters and compute [chunk 39].
```

## Known limitations

- Chunks are split by character count, so they can start mid-word
- Embeddings are recomputed on every run (no vector database yet)
- PDF tables are extracted as noisy text
- Retrieval scores are close together, so the top result is not always the best one
- Persian PDFs are not tested yet

## Roadmap

- [ ] FastAPI endpoint (`POST /ask`)
- [ ] Store embeddings in PostgreSQL + pgvector (Docker)
- [ ] Retrieval evaluation on a small question set
- [ ] Better chunking (sentence/paragraph based)
- [ ] Persian document support
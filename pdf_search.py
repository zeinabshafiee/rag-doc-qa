import re
import numpy as np
from sentence_transformers import SentenceTransformer
from pdf_chunks import load_pdf, chunk_text


def clean_text(text):
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")   # fix ligatures
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # join words split by hyphen at line end
    text = re.sub(r"\s+", " ", text)                      # collapse whitespace/newlines
    return text


model = SentenceTransformer("intfloat/multilingual-e5-base")

text = clean_text(load_pdf("data/1801.04381v4.pdf"))
chunks = chunk_text(text)
print(f"{len(chunks)} chunks. Embedding them...")

chunk_vecs = model.encode(
    [f"passage: {c}" for c in chunks],
    normalize_embeddings=True,
    show_progress_bar=True,
)

while True:
    question = input("\nQuestion (or 'exit'): ").strip()
    if question.lower() == "exit":
        break
    q_vec = model.encode(f"query: {question}", normalize_embeddings=True)
    scores = chunk_vecs @ q_vec
    for i in np.argsort(-scores)[:3]:
        print(f"\n[{scores[i]:.3f}] chunk {i}\n{chunks[i]}")
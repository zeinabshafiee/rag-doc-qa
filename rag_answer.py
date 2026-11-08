import os
import re
import numpy as np
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer
from pdf_chunks import load_pdf, chunk_text

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
LLM_MODEL = "gemini-3.8-flash"   # check ai.google.dev for the current free Flash model


def clean_text(text):
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return re.sub(r"\s+", " ", text)


embedder = SentenceTransformer("intfloat/multilingual-e5-base")
chunks = chunk_text(clean_text(load_pdf("data/1801.04381v4.pdf")))
chunk_vecs = embedder.encode(
    [f"passage: {c}" for c in chunks], normalize_embeddings=True
)


def ask(question, k=3):
    q_vec = embedder.encode(f"query: {question}", normalize_embeddings=True)
    top = np.argsort(-(chunk_vecs @ q_vec))[:k]
    context = "\n\n".join(f"[chunk {i}]\n{chunks[i]}" for i in top)
    prompt = (
        "Answer the question using ONLY the context below. "
        "Cite the chunk numbers you used, like [chunk 12]. "
        "If the context does not contain the answer, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    response = client.models.generate_content(model=LLM_MODEL, contents=prompt)
    return response.text


while True:
    q = input("\nQuestion (or 'exit'): ").strip()
    if q.lower() == "exit":
        break
    print("\n" + ask(q))
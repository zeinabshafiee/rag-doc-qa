from pypdf import PdfReader


def load_pdf(path):
    """Read all pages and join their text."""
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text, size=800, overlap=150):
    """Split text into pieces of ~size characters.
    Each piece overlaps the previous one so a sentence
    cut at the boundary still appears whole in one of them."""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size].strip())
        start += size - overlap
    return [c for c in chunks if c]


if __name__ == "__main__":
    text = load_pdf("data/1801.04381v4.pdf")
    chunks = chunk_text(text)

    print(f"Characters: {len(text)}")
    print(f"Chunks: {len(chunks)}")
    print("\n--- chunk 0 ---")
    print(chunks[0])
    print("\n--- chunk 1 ---")
    print(chunks[1])
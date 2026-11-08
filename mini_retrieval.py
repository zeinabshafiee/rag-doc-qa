from sentence_transformers import SentenceTransformer
import numpy as np

# 1) Load a multilingual model that turns text into numbers (vectors)
model = SentenceTransformer("intfloat/multilingual-e5-base")

# 2) Our tiny "document", already split into 3 pieces
chunks = [
    "Rasa sells stationery through Digikala and Instagram.",
    "The refund window is 7 days after delivery.",
    "مدت زمان ارسال سفارش‌ها بین ۲ تا ۴ روز کاری است.",
]

# 3) The question
question = "Where can I buy your products?"

# 4) Convert pieces and question to vectors
chunk_vecs = model.encode([f"passage: {c}" for c in chunks], normalize_embeddings=True)
q_vec = model.encode(f"query: {question}", normalize_embeddings=True)

# 5) Measure how close each piece is to the question
scores = chunk_vecs @ q_vec

# 6) Print pieces from most to least relevant
for i in np.argsort(-scores):
    print(f"{scores[i]:.3f}  {chunks[i]}")
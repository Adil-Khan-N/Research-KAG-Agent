import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Vision Transformers apply self-attention to image patches.",
    "The attention mechanism computes weighted sums over input tokens.",
    "Convolutional neural networks use local filters to detect features.",
    "BERT is a transformer model pre-trained on masked language modeling.",
    "Random forests are ensembles of decision trees.",
]

# Embed all 5
embeddings = model.encode(sentences)
print(f"Embedding shape: {embeddings.shape}")  
# Should print: (5, 384)

# Compute cosine similarity between all pairs
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("\nPairwise similarities:")
closest_pair = None
highest_score = -1

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        score = cosine_similarity(embeddings[i], embeddings[j])
        print(f"  [{i}] vs [{j}]: {score:.4f}")
        if score > highest_score:
            highest_score = score
            closest_pair = (i, j)

print(f"\nClosest pair (score={highest_score:.4f}):")
print(f"  A: {sentences[closest_pair[0]]}")
print(f"  B: {sentences[closest_pair[1]]}")
print(f"\nModel output dimension: {embeddings.shape[1]}")

# NOTE: Update your schema to VECTOR(384) for this model


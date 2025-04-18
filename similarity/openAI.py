import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

openai.api_key = "your-api-key"

def get_openai_embedding(text: str) -> list:
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def compute_cosine_openai_embedding(file1: str, file2: str) -> float:
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        code1 = f1.read()
        code2 = f2.read()
    
    emb1 = np.array(get_openai_embedding(code1)).reshape(1, -1)
    emb2 = np.array(get_openai_embedding(code2)).reshape(1, -1)
    
    return float(cosine_similarity(emb1, emb2)[0][0])

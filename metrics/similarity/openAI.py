import openai  # type: ignore
import numpy as np  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from dotenv import load_dotenv  # type: ignore
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_KEY")

MAX_TOKENS = 8192

def count_tokens(text: str) -> int:
    # Simple token approximation (you can replace with tiktoken if needed)
    return len(text.split())

def get_openai_embedding(text: str) -> list:
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    return response.data[0].embedding

def compute_cosine_openai_embedding(code1: str, code2: str) -> float | str | None:
    total_tokens = count_tokens(code1) + count_tokens(code2)
    
    if total_tokens > MAX_TOKENS:
        print(f"⚠️ Skipping OpenAI cosine similarity: total tokens ({total_tokens}) exceed limit ({MAX_TOKENS})")
        return "skipped"

    try:
        emb1 = np.array(get_openai_embedding(code1)).reshape(1, -1)
        emb2 = np.array(get_openai_embedding(code2)).reshape(1, -1)
        return float(cosine_similarity(emb1, emb2)[0][0])
    
    except openai.BadRequestError as e:
        if "maximum context length" in str(e):
            print(f"⚠️ OpenAI context length exceeded: {e}")
            return "skipped"
        else:
            print(f"⚠️ OpenAI BadRequestError: {e}")
            return None
    except Exception as e:
        print(f"⚠️ Error computing OpenAI cosine similarity: {e}")
        return None

from google import genai
from google.genai import types

client = genai.Client(api_key="AIzaSyAtUfjEH-Mrvjq7COBItoAWBoDGzSx-gVo")

result = client.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents="What is the meaning of life?",
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
)
print(result.embeddings)
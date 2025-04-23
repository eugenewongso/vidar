import argparse
from langchain_community.vectorstores import FAISS # type: ignore 
from langchain_openai import OpenAIEmbeddings # type: ignore
from dotenv import load_dotenv # type: ignore

load_dotenv()

def read_file(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def semantic_search(index_path, query_code_path, top_k=3):
    print(f"🔍 Loading FAISS index from {index_path} ...")
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)


    print(f"📄 Reading query code from {query_code_path} ...")
    query_text = read_file(query_code_path)

    print(f"🧠 Performing semantic similarity search (top {top_k}) ...")
    results = db.similarity_search(query_text, k=top_k)

    print(f"\n🧾 Top {top_k} Matches:")
    for i, result in enumerate(results, start=1):
        print(f"\n🔹 Match #{i}")
        # print(f"📍 File: {result.metadata['source']}")
        print(f"📍 File: {result.metadata.get('source', '[unknown]')}")
        print(f"--- Snippet ---\n{result.page_content[:300]}...\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perfor# type: ignorem semantic code search on a FAISS index.")
    parser.add_argument("--index_path", required=True, help="Path to FAISS index directory (e.g. ./vector_indexes/faiss_index_<commit>)")
    parser.add_argument("--query_file", required=True, help="Path to the query code file (e.g. an upstream .c file)")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top matches to return")
    args = parser.parse_args()

    semantic_search(args.index_path, args.query_file, args.top_k)

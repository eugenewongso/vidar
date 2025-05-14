import os
import sys
import argparse
from dotenv import load_dotenv  # type: ignore
from vector_indexer import index_repo
from semantic_search import semantic_search
from datetime import datetime

load_dotenv()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code Search Tool: Index a repository and immediately search it.")

    parser.add_argument("--repo_path", required=True, help="Path to the code repository to index.")
    parser.add_argument("--output_dir", default="./code_search/vector_indexes", help="Directory to store FAISS index folders. Defaults to ./vector_indexes.")
    parser.add_argument("--query_file", required=True, help="Path to the query code file.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top matches to return.")

    args = parser.parse_args()

    index_repo(args.repo_path, args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    index_name = f"faiss_index_{timestamp}"
    index_path = os.path.join(args.output_dir, index_name)

    if not os.path.isdir(index_path):
         print(f"⚠️ Index directory not found at {index_path} after indexing attempt. Skipping search.")
         sys.exit(1) 

    semantic_search(index_path, args.query_file, args.top_k)

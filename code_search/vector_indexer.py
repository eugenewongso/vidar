import os
import argparse
from tqdm import tqdm  # type: ignore
from langchain_community.vectorstores import FAISS  # type: ignore
from langchain_community.embeddings import OpenAIEmbeddings  # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
from langchain_core.documents import Document  # type: ignore
from dotenv import load_dotenv  # type: ignore
from langchain_community.vectorstores import FAISS # type: ignore
from langchain_openai import OpenAIEmbeddings # type: ignore

load_dotenv()

def load_c_files_manually(repo_path):
    print(f"Searching for .c files in {repo_path} ...")
    docs = []
    all_c_files = []

    # Collect all .c file paths
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".c"):
                all_c_files.append(os.path.join(root, file))

    # Read each file with a progress bar
    for path in tqdm(all_c_files, desc="Loading C files"):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                docs.append(Document(page_content=content, metadata={"source": path}))
        except Exception as e:
            print(f"⚠️ Failed to read {path}: {e}")

    print(f"✅ Loaded {len(docs)} C files.")
    return docs

def index_repo(repo_path: str, commit_hash: str, output_dir: str):
    print(f"]Indexing repo snapshot for commit {commit_hash}")

    docs = load_c_files_manually(repo_path)

    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = []
    for doc in tqdm(docs, desc="Splitting"):
        split_docs = splitter.split_documents([doc])
        for chunk in split_docs:
            chunk.metadata = doc.metadata  
        chunks.extend(split_docs)

    print(f"Total chunks: {len(chunks)}")

    print("Generating embeddings and building FAISS index...")
    embeddings = OpenAIEmbeddings()

    if not chunks:
        print("⚠️ No chunks to index.")
        return

    # Initialize FAISS with the first chunk
    print("Initializing FAISS index...")
    db = FAISS.from_documents(chunks[:1], embeddings)
    print("Initial index created.")

    # Add remaining chunks in batches with progress bar
    batch_size = 800 # Adjust batch size as needed based on performance/memory and OPENAI API limit
    print(f"Adding remaining {len(chunks) - 1} chunks in batches of {batch_size}...")
    for i in tqdm(range(1, len(chunks), batch_size), desc="Generating embeddings and indexing"):
        batch = chunks[i:i + batch_size]
        if batch:
             db.add_documents(batch) 

    print("Finished adding all chunks.")
    index_path = os.path.join(output_dir, f"faiss_index_{commit_hash}")
    os.makedirs(index_path, exist_ok=True)
    db.save_local(index_path)

    print(f"\n✅ FAISS index saved to: {index_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index C source files from a Linux repo for semantic search fallback.")
    parser.add_argument("--repo_path", required=True, help="Path to the repo at a specific commit")
    parser.add_argument("--commit_hash", required=True, help="Commit hash used to tag the index")
    parser.add_argument("--output_dir", default="./vector_indexes", help="Where to store FAISS index folders")
    args = parser.parse_args()

    index_repo(args.repo_path, args.commit_hash, args.output_dir)

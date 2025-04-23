import os
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import argparse

def index_repo(repo_path: str, commit_hash: str, output_dir: str):
    # Step 1: Load .c source files from the given repo directory
    loader = DirectoryLoader(repo_path, glob="**/*.c", recursive=True)
    print(f"📂 Loading C source files from {repo_path}...")
    docs = loader.load()

    # Step 2: Split documents to handle large files
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    print(f"🧩 Splitting documents into chunks...")
    docs = text_splitter.split_documents(docs)

    # Step 3: Embed and build FAISS index
    print(f"📐 Generating embeddings and creating FAISS index...")
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embedding)

    # Step 4: Save index to disk
    index_path = os.path.join(output_dir, f"faiss_index_{commit_hash}")
    os.makedirs(index_path, exist_ok=True)
    vectorstore.save_local(index_path)
    print(f"✅ FAISS index saved to {index_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index a Linux fork repo snapshot using FAISS.")
    parser.add_argument("--repo_path", required=True, help="Path to the checked-out downstream repo at a specific commit.")
    parser.add_argument("--commit_hash", required=True, help="Commit hash used to label the FAISS index.")
    parser.add_argument("--output_dir", default="./vector_indexes", help="Directory to store generated FAISS indexes.")
    args = parser.parse_args()

    index_repo(args.repo_path, args.commit_hash, args.output_dir)

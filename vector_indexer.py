import os
import argparse
from langchain_community.vectorstores import FAISS # type: ignore
from langchain_community.document_loaders import DirectoryLoader # type: ignore
from langchain_community.embeddings import OpenAIEmbeddings # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain_core.documents import Document # type: ignore
from dotenv import load_dotenv # type: ignore

load_dotenv()


def load_c_files_manually(repo_path):
    docs = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".c"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    docs.append(Document(page_content=content, metadata={"source": path}))
    return docs

def index_repo(repo_path: str, commit_hash: str, output_dir: str):
    # Step 1: Load C source files
    print(f" Loading files from {repo_path} ...")
    # loader = DirectoryLoader(repo_path, glob="**/*.c", recursive=True)
    # docs = loader.load()

    docs = load_c_files_manually(repo_path)

    # Step 2: Split into manageable chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(docs)

    # Step 3: Generate embeddings and create FAISS index
    print(" Creating embeddings and building FAISS index...")
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    # Step 4: Save index
    index_path = os.path.join(output_dir, f"faiss_index_{commit_hash}")
    os.makedirs(index_path, exist_ok=True)
    db.save_local(index_path)

    print(f"✅ Index saved to: {index_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index C source files from a Linux repo for semantic search fallback.")
    parser.add_argument("--repo_path", required=True, help="Path to the repo at a specific commit")
    parser.add_argument("--commit_hash", required=True, help="Commit hash used to tag the index")
    parser.add_argument("--output_dir", default="./vector_indexes", help="Where to store FAISS index folders")
    args = parser.parse_args()

    index_repo(args.repo_path, args.commit_hash, args.output_dir)

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
    print(f"📁 Searching for .c files in {repo_path} ...")
    docs = []
    all_c_files = []

    # Collect all .c file paths
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".c"):
                all_c_files.append(os.path.join(root, file))

    # Read each file with a progress bar
    for path in tqdm(all_c_files, desc="📄 Loading C files"):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                docs.append(Document(page_content=content, metadata={"source": path}))
        except Exception as e:
            print(f"⚠️ Failed to read {path}: {e}")

    print(f"✅ Loaded {len(docs)} C files.")
    return docs


def index_repo(repo_path: str, commit_hash: str, output_dir: str):
    print(f"🔍 Indexing repo snapshot for commit {commit_hash}")

    # Step 1: Load C source files
    docs = load_c_files_manually(repo_path)

    # Step 2: Split into chunks
    print("🧩 Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = []
    for doc in tqdm(docs, desc="🔪 Splitting"):
        chunks.extend(splitter.split_documents([doc]))

    print(f"📦 Total chunks: {len(chunks)}")

    # Step 3: Generate embeddings and build FAISS index
    print("🧠 Generating embeddings and building FAISS index...")
    embeddings = OpenAIEmbeddings()
    # db = FAISS.from_documents(tqdm(chunks, desc="🔁 Embedding"), embeddings)

    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    embedded_vectors = []
    print("🔁 Generating embeddings (this may take a while)...")
    for text in tqdm(texts, desc="Embedding"):
        try:
            embedded_vectors.append(embeddings.embed_query(text))
        except Exception as e:
            print(f"⚠️ Embedding failed: {e}")
            embedded_vectors.append([0.0] * 1536)  # Fallback zero vector, or skip

    # db = FAISS.from_embeddings(embedded_vectors, texts, metadatas)
    text_embeddings = list(zip(texts, embedded_vectors))
    db = FAISS.from_embeddings(text_embeddings, metadatas)

    # Step 4: Save index
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

import os
import argparse
from tqdm import tqdm  # type: ignore
from langchain_community.vectorstores import FAISS  # type: ignore
from langchain_openai import OpenAIEmbeddings # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
from langchain_core.documents import Document  # type: ignore
from dotenv import load_dotenv  # type: ignore

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

def create_index_for_repo(repo_path: str, index_output_path: str):
    print(f"Starting indexing for repository: {repo_path}")
    print(f"Output FAISS index will be saved to: {index_output_path}")

    docs = load_c_files_manually(repo_path)

    if not docs:
        print("⚠️ No .c files found to index.")
        return

    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    chunks = []
    for doc in tqdm(docs, desc="Splitting documents"):
        split_docs = splitter.split_documents([doc])
        for chunk in split_docs:
            # Ensure metadata is preserved if it's important
            chunk.metadata = doc.metadata  
        chunks.extend(split_docs)

    print(f"Total chunks created: {len(chunks)}")

    if not chunks:
        print("⚠️ No chunks generated from documents. Cannot create index.")
        return

    print("Generating embeddings and building FAISS index...")
    embeddings = OpenAIEmbeddings()

    # Initialize FAISS with the first chunk
    print("Initializing FAISS index with the first chunk...")
    try:
        db = FAISS.from_documents(chunks[:1], embeddings)
        print("Initial FAISS index created.")
    except Exception as e:
        print(f"❌ Error initializing FAISS index: {e}")
        return

    # Add remaining chunks in batches
    batch_size = 500 
    print(f"Adding remaining {len(chunks) - 1} chunks in batches of {batch_size}...")
    for i in tqdm(range(1, len(chunks), batch_size), desc="Generating embeddings and adding to index"):
        batch = chunks[i:i + batch_size]
        if batch:
            try:
                db.add_documents(batch)
            except Exception as e:
                print(f"❌ Error adding batch to FAISS index: {e}")
                # Decide if you want to stop or continue
                # For now, let's stop if a batch fails
                return 

    print("Finished adding all chunks to the index.")
    
    # Save the FAISS index
    try:
        os.makedirs(index_output_path, exist_ok=True)
        db.save_local(index_output_path)
        print(f"\n✅ FAISS index successfully created and saved to: {index_output_path}")
    except Exception as e:
        print(f"❌ Error saving FAISS index to {index_output_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a FAISS index for C source files in a repository.")
    parser.add_argument("--repo_path", required=True, help="Path to the code repository to index.")
    parser.add_argument("--index_output_path", required=True, help="Path to the directory where the FAISS index will be saved (e.g., ./code_search/vector_indexes/my_repo_index). This directory will be created if it doesn't exist.")
    
    args = parser.parse_args()

    create_index_for_repo(args.repo_path, args.index_output_path)

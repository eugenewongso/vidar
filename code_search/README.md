# Semantic Code Search Tool (`code_search.py`)

This directory contains a Python script for indexing a code repository and immediately performing a semantic search on the generated index using FAISS and OpenAI embeddings.

## Prerequisites

1.  **Python 3:** Ensure you have Python 3 installed.
2.  **Dependencies:** Install the required Python packages:
    ```bash
    pip install langchain langchain-community langchain-openai faiss-cpu python-dotenv tqdm
    ```
    *(Note: Use `faiss-gpu` instead of `faiss-cpu` if you have a compatible GPU and CUDA installed.)*
3.  **OpenAI API Key:** You need an OpenAI API key for generating embeddings. Create a `.env` file in the root directory (`vidar-post-eval`) and add your key:
    ```
    OPENAI_API_KEY='your_api_key_here'
    ```

## Core Process Overview

The fundamental steps involved in this semantic code search process are:

1.  Load the files and parse the documents (e.g., using `DirectoryLoader` and potentially language-specific parsing like AST for code).
2.  Create vector embeddings for the document chunks using `OpenAIEmbeddings`.
3.  Store these embeddings in a vector database (this script uses FAISS).
4.  Query the created vector store using `similarity_search` with the query text.

## Usage

The `code_search.py` script performs indexing and searching in a single run. It first scans a specified repository path, extracts `.c` files, generates embeddings, and saves a FAISS vector index. Then, it immediately loads this index and performs a semantic similarity search using the content of a query file, returning the top K most similar code snippets.

**Arguments:**

*   `--repo_path`: (Required) Path to the code repository you want to index.
*   `--commit_hash`: (Required) The specific commit hash of the repository state you are indexing. This is used to tag the output index directory.
*   `--query_file`: (Required) Path to the file containing the code you want to search for similar snippets of.
*   `--output_dir`: (Optional) Directory where the FAISS index folders will be stored. Defaults to `./vector_indexes`.
*   `--top_k`: (Optional) The number of top matching snippets to return. Defaults to 3.

**Example Command:**

```bash
python3 code_search/code_search.py \
  --repo_path /Users/enricoprayogo/Desktop/CSE_390C/homework/hw2 \
  --commit_hash 388a0529214d579c6cc866800c803bf35eef056b \
  --query_file "./input/upstream_commit/CVE-2025-21655_60495b08_CVE-2025-21655_eventfd.c" \
  --top_k 1 \
  --output_dir ./vector_indexes
```

This command will:
1.  Create (or overwrite) an index directory named `faiss_index_388a0529214d579c6cc866800c803bf35eef056b` inside `./vector_indexes`.
2.  Search this newly created index for code similar to the content of the query file.
3.  Print the single most similar snippet found, including its source file path.

---
*Note: The original `vector_indexer.py` and `semantic_search.py` scripts still exist but are superseded by `code_search.py`.*

## References:
- https://github.com/facebookresearch/faiss
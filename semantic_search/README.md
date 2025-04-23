### Vector indexer example command:
```bash
python3 semantic_search/vector_indexer.py \
  --repo_path /Users/enricoprayogo/Desktop/CSE_390C/homework/hw2 \ 
  --commit_hash 388a0529214d579c6cc866800c803bf35eef056b \
  --output_dir ./vector_indexes
  ```

  ### Semantic Search example command:

  ```bash
   python3 semantic_search/semantic_search.py \
  --index_path "./vector_indexes/faiss_index_388a0529214d579c6cc866800c803bf35eef056b" \
  --query_file "./input/upstream_commit/CVE-2025-21655_60495b08_CVE-2025-21655_eventfd.c" \
  --top_k 1
  ```
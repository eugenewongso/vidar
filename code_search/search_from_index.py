import argparse
import sys
from semantic_search import semantic_search # Assuming semantic_search.py is in the same directory or accessible via PYTHONPATH
from dotenv import load_dotenv # type: ignore

load_dotenv()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform semantic code search using a pre-built FAISS index.")
    
    parser.add_argument("--index_path", 
                        required=True, 
                        help="Path to the FAISS index directory (e.g., ./code_search/vector_indexes/my_repo_index).")
    parser.add_argument("--query_file", 
                        required=True, 
                        help="Path to the query code file (e.g., a .c file containing the code snippet to search for).")
    parser.add_argument("--top_k", 
                        type=int, 
                        default=3, 
                        help="Number of top matches to return. Defaults to 3.")
    
    args = parser.parse_args()

    # Ensure the semantic_search function is available
    # If semantic_search.py is in a different relative path, adjust the import accordingly.
    # For example, if it's in the parent directory: from ..semantic_search import semantic_search
    # If it's in a sub-module: from .submodule.semantic_search import semantic_search
    
    try:
        semantic_search(args.index_path, args.query_file, args.top_k)
    except ImportError:
        print("Error: The 'semantic_search' function could not be imported.")
        print("Please ensure 'semantic_search.py' is in the correct location (e.g., same directory as this script or in PYTHONPATH).")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during the search: {e}")
        sys.exit(1)

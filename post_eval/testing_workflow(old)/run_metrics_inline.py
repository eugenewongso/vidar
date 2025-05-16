import os
import argparse
from metrics.line_metrics import relative_line_count_similarity 
from metrics.similarity.codeBERT import compute_codebertscore_c
from metrics.similarity.openAI import compute_cosine_openai_embedding
from metrics.distance.edit_distance import token_level_edit_similarity, normalized_edit_similarity

# supress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def read_file(file_path) -> str:
    with open(file_path, 'r') as f:
        return f.read()
    
def count_tokens(text):
    return len(text.split())

MAX_TOKENS = 8192

def main():
    parser = argparse.ArgumentParser(description="Compare two code snippets using various metrics.")
    parser.add_argument("--ground_code", required=True, help="Ground truth code snippet as a string.")
    parser.add_argument("--candidate_code", required=True, help="Candidate code snippet as a string.")
    args = parser.parse_args()

    ground_truth_code = args.ground_code
    candidate_code = args.candidate_code

    print("\n=== Evaluation Metrics ===")

    rel_line_diff = relative_line_count_similarity(candidate_code, ground_truth_code)
    print("Relative Line Count Difference:", round(rel_line_diff, 4))

    token_ed = token_level_edit_similarity(candidate_code, ground_truth_code)
    print("Token-Level Edit Similarity:", token_ed)

    norm_ed = normalized_edit_similarity(candidate_code, ground_truth_code)
    print("Normalized Edit Similarity:", round(norm_ed, 4))

    similarity_score_codebert_c = compute_codebertscore_c(candidate_code, ground_truth_code)
    print("CodeBERTScore for C file:")
    for metric, value in similarity_score_codebert_c.items():
        print(f"{metric}: {value:.4f}")

    total_tokens_ground_truth = count_tokens(ground_truth_code)
    total_tokens_candidate_code = count_tokens(candidate_code)
    total_tokens = total_tokens_ground_truth + total_tokens_candidate_code

    if total_tokens_ground_truth > MAX_TOKENS or total_tokens_candidate_code > MAX_TOKENS:
        print(f"Skipping OpenAI cosine similarity: total tokens ({total_tokens}) exceed limit ({MAX_TOKENS})")
    else:
        score = compute_cosine_openai_embedding(ground_truth_code, candidate_code)
        if isinstance(score, float):
            print(f"Cosine similarity (Open AI) = {score:.4f}")
        else:
            print(f"Cosine similarity (Open AI) = {score}")

if __name__ == "__main__":
    main()

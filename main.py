# from compile_check import compiles_with_gpp
from line_metrics import relative_line_count_diff 
from similarity.codeBERT import compute_codebertscore_c
from similarity.openAI import compute_cosine_openai_embedding
from edit_distance import token_level_edit_distance, normalized_edit_distance
# from similarity.sklearn import 


def read_file(file_path) -> str:
    with open(file_path, 'r') as f:
        return f.read()
    
def count_tokens(text):
    return len(text.split())

MAX_TOKENS = 8192

def main():
    ground_truth_path = "./testing_files/ground_truth/eventfd.c"
    candidate_patch_code_path = "./testing_files/candidate_patch/eventfd.c"

    ground_truth_code = read_file(ground_truth_path)
    candidate_code = read_file(candidate_patch_code_path)

    print("\n=== Evaluation Metrics ===")

    rel_line_diff = relative_line_count_diff(candidate_code, ground_truth_code)
    print("Relative Line Count Difference:", round(rel_line_diff, 4))

    token_ed = token_level_edit_distance(candidate_code, ground_truth_code)
    print("Token-Level Edit Distance:", token_ed)

    norm_ed = normalized_edit_distance(candidate_code, ground_truth_code)
    print("Normalized Edit Distance:", round(norm_ed, 4))

    similarity_score_codebert_c = compute_codebertscore_c(candidate_code, ground_truth_code)
    print("CodeBERTScore for C file:")
    for metric, value in similarity_score_codebert_c.items():
        print(f"{metric}: {value:.4f}")

    # Token check before OpenAI embedding
    total_tokens_ground_truth = count_tokens(ground_truth_code)
    total_tokens_candidate_code = count_tokens(candidate_code)
    total_tokens = total_tokens_ground_truth + total_tokens_candidate_code

    if total_tokens_ground_truth > MAX_TOKENS or total_tokens_candidate_code > MAX_TOKENS:
        print(f"Skipping OpenAI cosine similarity: total tokens ({total_tokens}) exceed limit ({MAX_TOKENS})")
    else:
        score = compute_cosine_openai_embedding(ground_truth_path, candidate_patch_code_path)
        print(f"Cosine similarity (Open AI) = {score:.4f}")

if __name__ == "__main__":
    main()
# from compile_check import compiles_with_gpp
# from line_metrics import relative_line_count_diff
from line_metrics import relative_line_count_diff # TODO 
from similarity.codeBERT import compute_cosine_similarity_from_files, compute_codebertscore_c
from similarity.openAI import compute_cosine_openai_embedding
# from similarity.sklearn import 

# from edit_distance import token_level_edit_distance, normalized_edit_distance
# from similarity import cosine_sim

def read_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()

def main():
    ground_truth_path = "./testing_files/ground_truth/eventfd.c"
    candidate_patch_code_path = "./testing_files/candidate_patch/eventfd.c"

    ground_truth_code = read_file(ground_truth_path)
    candidate_code = read_file(candidate_patch_code_path)

    # TODO: explore compilation checks (might do it at end after eval metrics due to time it takes)
    # print("=== Compilation Check ===")
    # compiles, compile_msg = compiles_with_gpp(candidate_code)
    # print("Compiles:", compiles)
    # if not compiles:
    #     print("Compilation Error:\n", compile_msg)
    #     return

    print("\n=== Evaluation Metrics ===")

    # rel_line_diff = relative_line_count_diff(candidate_code, ground_truth_code)
    # print("Relative Line Count Difference:", round(rel_line_diff, 4))

    # TODO: token edit distance
    # token_ed = token_level_edit_distance(candidate_code, ground_truth_code)
    # print("Token-Level Edit Distance:", token_ed)

    # TODO: normalized edit distance
    # norm_ed = normalized_edit_distance(candidate_code, ground_truth_code)
    # print("Normalized Edit Distance:", round(norm_ed, 4))

    # TODO: cosine similarity
    similarity_score_codebert = compute_cosine_similarity_from_files(ground_truth_path, candidate_patch_code_path)
    print(f"Cosine similarity (CodeBERT) = {similarity_score_codebert:.4f}")
    similarity_score_codebert_c = compute_codebertscore_c(ground_truth_path, candidate_patch_code_path)
    print("CodeBERTScore for C file:")
    for metric, value in similarity_score_codebert_c.items():
        print(f"{metric}: {value:.4f}")
    

    # Open AI text embedding 3 models can only be used for short files, max of 8192 tokens
    # score = compute_cosine_openai_embedding(ground_truth_path, candidate_patch_code_path)
    # print(f"Cosine similarity (Open AI) = {score:.4f}")


if __name__ == "__main__":
    main()
# from compile_check import compiles_with_gpp
# from line_metrics import relative_line_count_diff
from line_metrics import relative_line_count_diff # TODO 
from similarity.codeBERT import compute_cosine_similarity_from_files

# from edit_distance import token_level_edit_distance, normalized_edit_distance
# from similarity import cosine_sim

def read_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()

def main():
    ground_truth_path = "./testing_files/ground_truth/af_unix.c"
    candidate_patch_code_path = "./testing_files/candidate_patch/af_unix.c"

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
    # cosine = cosine_sim(candidate_code, ground_truth_code) 
    similarity_score = compute_cosine_similarity_from_files(ground_truth_path, candidate_patch_code_path)
    print(f"Cosine similarity (CodeBERT) = {similarity_score:.4f}")


if __name__ == "__main__":
    main()
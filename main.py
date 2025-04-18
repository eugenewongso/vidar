# from compile_check import compiles_with_gpp
# from line_metrics import relative_line_count_diff
from line_metrics import relative_line_count_diff

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

    # print("=== Compilation Check ===")
    # compiles, compile_msg = compiles_with_gpp(candidate_code)
    # print("Compiles:", compiles)
    # if not compiles:
    #     print("Compilation Error:\n", compile_msg)
    #     return

    # print("\n=== Evaluation Metrics ===")

    rel_line_diff = relative_line_count_diff(candidate_code, ground_truth_code)
    print("Relative Line Count Difference:", round(rel_line_diff, 4))

    # token_ed = token_level_edit_distance(candidate_code, ground_truth_code)
    # print("Token-Level Edit Distance:", token_ed)

    # norm_ed = normalized_edit_distance(candidate_code, ground_truth_code)
    # print("Normalized Edit Distance:", round(norm_ed, 4))

    # cosine = cosine_sim(candidate_code, ground_truth_code)
    # print("Cosine Similarity:", round(cosine, 4))

if __name__ == "__main__":
    main()
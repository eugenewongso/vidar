def relative_line_count_similarity(candidate_code, ground_truth_code):
    """
    Calculate the relative line count similarity between candidate code and ground truth code.

    Returns 1.0 if the line counts are the same, and decreases towards 0.0 as the difference increases.
    If both are empty, returns 1.0 (identical).
    If ground truth is empty but candidate is not, returns 0.0 (completely different).

    Args:
        candidate_code (str): The candidate code as a string
        ground_truth_code (str): The ground truth code as a string

    Returns:
        float: The relative line count similarity (1.0 = same, 0.0 = maximally different)
    """
    candidate_lines = [line for line in candidate_code.split('\n') if line.strip()]
    ground_truth_lines = [line for line in ground_truth_code.split('\n') if line.strip()]

    candidate_line_count = len(candidate_lines)
    ground_truth_line_count = len(ground_truth_lines)

    # Both empty: identical
    if ground_truth_line_count == 0 and candidate_line_count == 0:
        return 1.0
    # One empty, one not: maximally different
    if ground_truth_line_count == 0 or candidate_line_count == 0:
        return 0.0

    relative_diff = abs(candidate_line_count - ground_truth_line_count) / ground_truth_line_count
    similarity = 1.0 - min(1.0, relative_diff)
    
    return similarity
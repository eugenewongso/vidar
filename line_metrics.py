def relative_line_count_diff(candidate_code, ground_truth_code):
    """
    Calculate the relative line count difference between candidate code and ground truth code.
    
    Formula: (candidate_line_count - ground_truth_line_count) / ground_truth_line_count
    
    A positive value means the candidate code has more lines than the ground truth.
    A negative value means the candidate code has fewer lines than the ground truth.
    A value of 0 means they have the same number of lines.
    
    Args:
        candidate_code (str): The candidate code as a string
        ground_truth_code (str): The ground truth code as a string
        
    Returns:
        float: The relative line count difference
    """
    # Count non-empty lines in both code samples
    candidate_lines = [line for line in candidate_code.split('\n') if line.strip()]
    ground_truth_lines = [line for line in ground_truth_code.split('\n') if line.strip()]
    
    candidate_line_count = len(candidate_lines)
    ground_truth_line_count = len(ground_truth_lines)
    
    # Avoid division by zero
    if ground_truth_line_count == 0:
        if candidate_line_count == 0:
            return 0.0  # Both are empty, no difference
        else:
            # return float('inf')  # Ground truth is empty but candidate is not
            return 1.0  # Ground truth is empty but candidate is not → max difference
    
    # Calculate relative difference
    relative_diff = abs(candidate_line_count - ground_truth_line_count) / ground_truth_line_count
    
    return min(1.0, relative_diff)
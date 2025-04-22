import re

def tokenize_code(code_string):
    """
    Split code into tokens (identifiers, keywords, operators, etc.)
    
    Args:
        code_string (str): The code to tokenize
        
    Returns:
        list: List of tokens
    """
    # This regex pattern matches:
    # - Identifiers and keywords (words)
    # - Operators, punctuation, and other symbols
    # - Numbers
    # - String literals (both single and double quotes)
    # - Whitespace is ignored
    
    pattern = r'[a-zA-Z_]\w*|[-+*/=<>!&|^~%]+|[(){}\[\];,.]|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|\d+(?:\.\d+)?'
    tokens = re.findall(pattern, code_string)
    return tokens

def levenshtein_distance(tokens1, tokens2):
    """
    Calculate Levenshtein distance between two lists of tokens
    
    Args:
        tokens1 (list): First list of tokens
        tokens2 (list): Second list of tokens
        
    Returns:
        int: The edit distance between the token lists
    """
    # Initialize the matrix
    m, n = len(tokens1), len(tokens2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    
    # Initialize first row and column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens1[i-1] == tokens2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],   # deletion
                                 dp[i][j-1],     # insertion
                                 dp[i-1][j-1])   # substitution
    
    return dp[m][n]

def token_level_edit_distance(candidate_code, ground_truth_code):
    """
    Calculate the edit distance between two code samples at the token level
    
    Args:
        candidate_code (str): The candidate code
        ground_truth_code (str): The ground truth code
        
    Returns:
        int: The token-level edit distance
    """
    # Tokenize both code samples
    candidate_tokens = tokenize_code(candidate_code)
    ground_truth_tokens = tokenize_code(ground_truth_code)
    
    # Calculate Levenshtein distance between token lists
    distance = levenshtein_distance(candidate_tokens, ground_truth_tokens)
    
    return distance

def normalized_edit_distance(candidate_code, ground_truth_code):
    """
    Calculate the normalized edit distance between two code samples
    
    Normalized by the maximum possible edit operations (max length of the two token lists)
    
    Args:
        candidate_code (str): The candidate code
        ground_truth_code (str): The ground truth code
        
    Returns:
        float: The normalized edit distance (between 0 and 1)
    """
    # Tokenize both code samples
    candidate_tokens = tokenize_code(candidate_code)
    ground_truth_tokens = tokenize_code(ground_truth_code)
    
    # Calculate Levenshtein distance between token lists
    distance = levenshtein_distance(candidate_tokens, ground_truth_tokens)
    
    # Normalize by the maximum length of the two token lists
    max_length = max(len(candidate_tokens), len(ground_truth_tokens))
    
    # Avoid division by zero
    if max_length == 0:
        return 0.0
    
    normalized_distance = distance / max_length
    
    return min(1.0, normalized_distance)
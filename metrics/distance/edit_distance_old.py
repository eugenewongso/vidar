import re
from rapidfuzz.distance import Levenshtein

MAX_TOKENS = 8192  # You can adjust this limit as needed

def tokenize_code(code_string):
    pattern = r'[a-zA-Z_]\w*|[-+*/=<>!&|^~%]+|[(){}\[\];,.]|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|\d+(?:\.\d+)?'
    tokens = re.findall(pattern, code_string)
    return tokens

# TODO: optimize O(M*N) runtime
def levenshtein_distance(tokens1, tokens2):
    m, n = len(tokens1), len(tokens2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens1[i-1] == tokens2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]

def token_level_edit_distance(candidate_code, ground_truth_code):
    candidate_tokens = tokenize_code(candidate_code)
    ground_truth_tokens = tokenize_code(ground_truth_code)

    if len(candidate_tokens) > MAX_TOKENS or len(ground_truth_tokens) > MAX_TOKENS:
        return "skipped"

    # Convert tokens to space-separated string (preserves token boundaries)
    distance = Levenshtein.distance(" ".join(candidate_tokens), " ".join(ground_truth_tokens))
    # print("debugging distance", distance)
    max_len = max(len(candidate_tokens), len(ground_truth_tokens))

    return min(1.0, distance / max_len if max_len != 0 else 0)

def normalized_edit_distance(candidate_code, ground_truth_code):
    candidate_tokens = tokenize_code(candidate_code)
    ground_truth_tokens = tokenize_code(ground_truth_code)
    if len(candidate_tokens) > MAX_TOKENS or len(ground_truth_tokens) > MAX_TOKENS:
        return "skipped"
    distance = levenshtein_distance(candidate_tokens, ground_truth_tokens)
    max_length = max(len(candidate_tokens), len(ground_truth_tokens))
    if max_length == 0:
        return 0.0
    normalized_distance = distance / max_length
    return min(1.0, normalized_distance)

def token_level_edit_similarity(candidate_code, ground_truth_code):
    candidate_tokens = tokenize_code(candidate_code)
    ground_truth_tokens = tokenize_code(ground_truth_code)
    if len(candidate_tokens) > MAX_TOKENS or len(ground_truth_tokens) > MAX_TOKENS:
        return "skipped"
    distance = levenshtein_distance(candidate_tokens, ground_truth_tokens)
    max_len = max(len(candidate_tokens), len(ground_truth_tokens))
    if max_len == 0:
        return 1.0
    similarity = 1.0 - min(1.0, distance / max_len)
    return similarity

def normalized_edit_similarity(candidate_code, ground_truth_code):
    candidate_tokens = tokenize_code(candidate_code)
    ground_truth_tokens = tokenize_code(ground_truth_code)
    if len(candidate_tokens) > MAX_TOKENS or len(ground_truth_tokens) > MAX_TOKENS:
        return "skipped"
    distance = levenshtein_distance(candidate_tokens, ground_truth_tokens)
    max_length = max(len(candidate_tokens), len(ground_truth_tokens))
    if max_length == 0:
        return 1.0
    similarity = 1.0 - min(1.0, distance / max_length)
    return similarity
import re
from rapidfuzz.distance import Levenshtein #type: ignore
from tree_sitter import Language, Parser #type: ignore

MAX_TOKENS = 8192

# TODO: use tree-sitter for tokenization
# Build one-time (once per environment)
# Language.build_library(
#     'build/my-languages.so',
#     [
#         'vendor/tree-sitter-c',
#         'vendor/tree-sitter-cpp',
#         'vendor/tree-sitter-python',
#         'vendor/tree-sitter-java',
#         'vendor/tree-sitter-javascript',
#     ]
# )
"""
PY_LANGUAGE = Language('build/my-languages.so', 'python')
parser = Parser()
parser.set_language(PY_LANGUAGE)


def tokenize_with_tree_sitter(code: str) -> list:
    tree = parser.parse(bytes(code, "utf8"))
    root = tree.root_node

    def walk(node):
        if node.child_count == 0:
            return [code[node.start_byte:node.end_byte]]
        tokens = []
        for child in node.children:
            tokens.extend(walk(child))
        return tokens

    return walk(root)
"""

def tokenize_code(code_string):
    pattern = r'[a-zA-Z_]\w*|[-+*/=<>!&|^~%]+|[(){}\[\];,.]|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|\d+(?:\.\d+)?'
    return re.findall(pattern, code_string)

# TODO: determine if this is the best way to tokenize code and try to not skip
def token_level_edit_distance(candidate_code, ground_truth_code):
    """
    Normalized Levenshtein distance (0 = identical, 1 = max difference).
    """
    tokens1 = tokenize_code(candidate_code)
    tokens2 = tokenize_code(ground_truth_code)

    if len(tokens1) > MAX_TOKENS or len(tokens2) > MAX_TOKENS:
        return "skipped"

    return round(Levenshtein.normalized_distance(tokens1, tokens2), 4)

def token_level_edit_similarity(candidate_code, ground_truth_code):
    """
    Normalized Levenshtein similarity (1 = identical, 0 = completely different).
    """
    tokens1 = tokenize_code(candidate_code)
    tokens2 = tokenize_code(ground_truth_code)

    if len(tokens1) > MAX_TOKENS or len(tokens2) > MAX_TOKENS:
        return "skipped"

    return round(Levenshtein.normalized_similarity(tokens1, tokens2), 4)

def normalized_edit_distance(candidate_code, ground_truth_code):
    """
    Normalized Levenshtein distance (real definition): edit_distance / max_len(tokens).
    Not forcibly bounded to 1.0 — may exceed 1.0 for weighted edits.
    """
    tokens1 = tokenize_code(candidate_code)
    tokens2 = tokenize_code(ground_truth_code)

    if len(tokens1) > MAX_TOKENS or len(tokens2) > MAX_TOKENS:
        return "skipped"

    max_len = max(len(tokens1), len(tokens2))
    if max_len == 0:
        return 0.0

    return Levenshtein.distance(tokens1, tokens2) / max_len


def _test_edit_distance_metrics():
    code1 = "int a = b + 1;"
    code2 = "int a = b + 1;"
    code3 = "int x = c + 2;"
    code4 = "for (int i = 0; i < 10; ++i) { sum += i; }"

    print("Test 1: Identical code")
    print("  token_level_edit_distance:", token_level_edit_distance(code1, code2))  # Expect 0.0
    print("  token_level_edit_similarity:", token_level_edit_similarity(code1, code2))  # Expect 1.0
    print("  normalized_edit_distance:", normalized_edit_distance(code1, code2))  # Expect 0.0

    print("\nTest 2: One variable changed")
    print("  token_level_edit_distance:", token_level_edit_distance(code1, code3))  # Small > 0.0
    print("  token_level_edit_similarity:", token_level_edit_similarity(code1, code3))  # Close to 1.0
    print("  normalized_edit_distance:", normalized_edit_distance(code1, code3))

    print("\nTest 3: Completely different")
    print("  token_level_edit_distance:", token_level_edit_distance(code1, code4))  # Higher value
    print("  token_level_edit_similarity:", token_level_edit_similarity(code1, code4))
    print("  normalized_edit_distance:", normalized_edit_distance(code1, code4))

    print("\nTest 4: Empty code")
    print("  token_level_edit_distance:", token_level_edit_distance("", ""))  # Expect 0.0
    print("  token_level_edit_similarity:", token_level_edit_similarity("", ""))  # Expect 1.0
    print("  normalized_edit_distance:", normalized_edit_distance("", ""))  # Expect 0.0

# Run the test (uncomment to run tests)
if __name__ == "__main__":
    _test_edit_distance_metrics()
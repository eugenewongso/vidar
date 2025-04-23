import nltk
nltk.download('punkt')
from nltk.metrics.distance import jaccard_distance
from nltk.tokenize import word_tokenize

def compute_jaccard_distance(str1, str2):
    tokens1 = set(word_tokenize(str1))
    tokens2 = set(word_tokenize(str2))
    return jaccard_distance(tokens1, tokens2)

# Example usage
s1 = "The quick brown fox jumps over the lazy dog"
s2 = "A quick brown dog leaps over a sleepy cat"

print("Jaccard Distance:", compute_jaccard_distance(s1, s2))
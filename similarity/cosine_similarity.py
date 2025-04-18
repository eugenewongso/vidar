from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast


def vectorize_texts(text1, text2):
    vectorizer = CountVectorizer(tokenizer=basic_tokenize)
    vectors = vectorizer.fit_transform([text1, text2])
    return vectors

def basic_tokenize(text):
    return text.split()


def compute_cosine_similarity(text1: str, text2: str) -> float:
    vectorizer = CountVectorizer(tokenizer=basic_tokenize)
    vectors = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(vectors)[0, 1]

def tokenize_code(code_string):
  tree = ast.parse(code_string)
  tokens = []
  for node in ast.walk(tree):
    if isinstance(node, ast.Name):
      tokens.append(node.id)
    elif isinstance(node, ast.Str):
      tokens.append('STRING')
    elif isinstance(node, ast.Num):
      tokens.append('NUMBER')
    # Add more node types as needed
    return tokens
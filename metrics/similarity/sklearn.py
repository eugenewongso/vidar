from sklearn.feature_extraction.text import CountVectorizer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
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
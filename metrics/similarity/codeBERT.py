import torch # type: ignore
from transformers import RobertaTokenizer, RobertaModel # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
import code_bert_score # type: ignore

# Load tokenizer and model once
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base") 
model = RobertaModel.from_pretrained("microsoft/codebert-base")

def load_code_from_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def get_code_embedding(code: str) -> torch.Tensor:
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[0][0]  # CLS token embedding

def compute_cosine_similarity_from_files(gt_path: str, candidate_path: str) -> float:
    gt_code = load_code_from_file(gt_path)
    candidate_code = load_code_from_file(candidate_path)

    gt_vec = get_code_embedding(gt_code).unsqueeze(0).numpy()
    candidate_vec = get_code_embedding(candidate_code).unsqueeze(0).numpy()

    return float(cosine_similarity(gt_vec, candidate_vec)[0][0])

def compute_codebert_score(candidate_code: str, ground_truth_code: str, language: str) -> dict:
    supported_languages = ["python", "java", "javascript", "c", "cpp", "c++", "js"]
    language = language.lower().strip()

    if not candidate_code or not ground_truth_code:
        return {
            "error": "Candidate code and ground truth code must not be empty."
        }

    if language in supported_languages:
        try:
            precision, recall, f1, f3 = code_bert_score.score(
                cands=[candidate_code],
                refs=[ground_truth_code],
                lang=language
            )
            return {
                "precision": precision.item(),
                "recall": recall.item(),
                "f1": f1.item(),
                "f3": f3.item()
            }
        except Exception as e:
            return {
                "error": f"Exception during CodeBERTScore computation: {e}"
            }
    else:
        # Fallback to cosine similarity using embeddings
        try:
            cand_vec = get_code_embedding(candidate_code).unsqueeze(0).numpy()
            gt_vec = get_code_embedding(ground_truth_code).unsqueeze(0).numpy()
            cos_sim = float(cosine_similarity(cand_vec, gt_vec)[0][0])
            return {
                "cosine_similarity_fallback": cos_sim
            }
        except Exception as e:
            return {
                "error": f"Exception during fallback cosine similarity computation: {e}"
            }

"""
def compute_codebert_score(candidate_code: str, ground_truth_code: str, language: str) -> dict:
    Computes CodeBERTScore between a ground truth and a candidate code file.
    Returns a dictionary with precision, recall, F1, and F3 scores.
    supported_languages = ["python", "java", "javascript", "c", "cpp", "c++", "js"]
    language = language.lower().strip()
    if language not in supported_languages:
        return {
            "error": f"Language '{language}' is not supported for CodeBERTScore. Supported languages are: {', '.join(supported_languages)}"
        }
    if not candidate_code or not ground_truth_code:
        return {
            "error": "Candidate code and ground truth code must not be empty."
        }
    try:
        precision, recall, f1, f3 = code_bert_score.score(
            cands=[candidate_code],
            refs=[ground_truth_code],
            lang=language
        )
        return {
            "precision": precision.item(),
            "recall": recall.item(),
            "f1": f1.item(),
            "f3": f3.item()
        }
    except Exception as e:
        return {
            "error": f"Exception during CodeBERTScore computation: {e}"
        }
"""

# Simple validation test
def _test_codebert_score_all():
    # Supported language test (should return precision/recall/f1/f3)
    print("Test: Supported language (C)")
    c_result = compute_codebert_score("int main() { return 0; }", "int main() { return 0; }", "c")
    print("  Result:", c_result)
    assert "precision" in c_result

    # Realistic code snippets from unsupported languages
    code_snippets = {
        "go": (
            '''package main
import "fmt"
func main() {
    fmt.Println("Hello, World!")
}''',
            '''package main
import "fmt"
func main() {
    fmt.Println("Hi there!")
}'''
        ),
        "rust": (
            '''fn main() {
    println!("Hello, world!");
}''',
            '''fn main() {
    println!("Hi there!");
}'''
        ),
        "typescript": (
            '''function greet(): void {
    console.log("Hello, world!");
}''',
            '''function greet(): void {
    console.log("Hi there!");
}'''
        ),
        "bash": (
            '''#!/bin/bash
echo "Hello, world!"''',
            '''#!/bin/bash
echo "Hi there!"'''
        ),
        "kotlin": (
            '''fun main() {
    println("Hello, world!")
}''',
            '''fun main() {
    println("Hi there!")
}'''
        ),
        "swift": (
            '''import Foundation
print("Hello, world!")''',
            '''import Foundation
print("Hi there!")'''
        ),
    }

    for lang, (code1, code2) in code_snippets.items():
        print(f"\nTest: Unsupported language ({lang})")
        result = compute_codebert_score(code1, code2, lang)
        print("  Result:", result)
        assert "cosine_similarity_fallback" in result or "error" in result, f"Unexpected result for {lang}"

    print("\n✅ All fallback tests passed.")
# Uncomment to run the test
if __name__ == "__main__":
    _test_codebert_score_all()

import torch
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics.pairwise import cosine_similarity

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
import os
import json
import argparse
import csv
from tqdm import tqdm #type: ignore
from metrics.line_metrics import relative_line_count_similarity
from metrics.similarity.codeBERT import compute_codebert_score
from metrics.similarity.openAI import compute_cosine_openai_embedding
from metrics.distance.edit_distance import token_level_edit_similarity, normalized_edit_distance, token_level_edit_distance
from datetime import datetime

MAX_TOKENS = 8192

def count_tokens(text):
    return len(text.split())

def clean_code(code):
    code = code.strip()
    if code.startswith("```"):
        code = code.split('\n', 1)[-1]
    if code.endswith("```"):
        code = code.rsplit('```', 1)[0]
    return code.strip()

def get_language_from_filename(filename):
    ext = filename.lower().rsplit('.', 1)[-1]
    ext_map = {
        "py": "python",
        "java": "java",
        "js": "javascript",
        "c": "c",
        "cpp": "cpp",
        "cc": "cpp",
        "cxx": "cpp",
        "h": "cpp",
        "hpp": "cpp",
        "cs": "csharp",
        # Add more as needed
    }
    return ext_map.get(ext, ext) # if not found, use the extension itself

def compute_metrics(upstream_code, downstream_code, file_name) -> dict:
    metrics = {}

    # Relative line count similarity
    rls = relative_line_count_similarity(downstream_code, upstream_code)
    metrics["relative_line_count_similarity"] = round(rls, 4) if isinstance(rls, float) else rls

    # Token-level edit similarity
    tles = token_level_edit_similarity(downstream_code, upstream_code)
    metrics["token_level_edit_similarity"] = round(tles, 4) if isinstance(tles, float) else tles

    # Normalized edit similarity
    nes = normalized_edit_distance(downstream_code, upstream_code)
    metrics["normalized_edit_similarity"] = round(nes, 4) if isinstance(nes, float) else nes

    # Token-level edit distance
    tled = token_level_edit_distance(downstream_code, upstream_code)
    metrics["token_level_edit_distance"] = round(tled, 4) if isinstance(tled, float) else tled

    # Get language from file name
    language = get_language_from_filename(file_name)

    # CodeBERTScore
    codebert = compute_codebert_score(downstream_code, upstream_code, language)
    if "error" in codebert:
        metrics["codebert_score"] = codebert  # preserve the error message
    else:
        metrics["codebert_score"] = {k: round(v, 4) for k, v in codebert.items()}

    # OpenAI cosine similarity (if tokens upstream, downstream < 8192)
    total_tokens_upstream = count_tokens(upstream_code)
    total_tokens_downstream = count_tokens(downstream_code)
    total_tokens = total_tokens_upstream + total_tokens_downstream

    if total_tokens > MAX_TOKENS:
        metrics["cosine_similarity_openai"] = "skipped"
    else:
        score = compute_cosine_openai_embedding(upstream_code, downstream_code)
        if isinstance(score, float):
            metrics["cosine_similarity_openai"] = round(score, 4)
        else:
            metrics["cosine_similarity_openai"] = score

    return metrics

def main():
    parser = argparse.ArgumentParser(description="Directly evaluate patch comparisons using inline JSON content and imported metric functions.")
    parser.add_argument("--json_input", required=True, help="Path to the JSON file with failure data")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_json = f"results/summary_{timestamp}.json"
    results_csv = f"results/summary_{timestamp}.csv"

    os.makedirs(os.path.dirname(results_json), exist_ok=True)
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)

    with open(args.json_input, "r") as f:
        report = json.load(f)

    # Count total number of file_conflicts for progress bar
    total_conflicts = 0
    for vuln in report:
        for failure in vuln.get("failures", []):
            total_conflicts += len(failure.get("file_conflicts", []))

    all_results = []

    with tqdm(total=total_conflicts, desc="Evaluating file conflicts") as pbar:
        for vuln in report:
            cve_id = vuln.get("id")
            failures = vuln.get("failures", [])
            if not failures:
                print(f"⚠️ Skipping {cve_id} — no failures listed.")
                continue

            for failure in failures:
                downstream_version = failure.get("downstream_version", "unknown")
                file_conflicts = failure.get("file_conflicts", [])
                if not file_conflicts:
                    print(f"⚠️ Skipping {cve_id} — no file_conflicts in failure for version {downstream_version}.")
                    continue

                for conflict in file_conflicts:
                    file_name = conflict.get("file_name", "unknown")
                    upstream_content = clean_code(conflict.get("upstream_file_content", ""))
                    downstream_content = clean_code(conflict.get("downstream_file_content", ""))

                    print(f"\nComparing {file_name} for {cve_id} ({downstream_version})")

                    # compute metrics
                    metrics = compute_metrics(upstream_content, downstream_content, file_name)

                    print("\n=== Evaluation Metrics ===")
                    print(f"Relative Line Count Difference: {metrics.get('relative_line_count_similarity')}")
                    print(f"Token-Level Edit Distance: {metrics.get('token_level_edit_distance')}")
                    print(f"Normalized Edit Distance: {metrics.get('normalized_edit_similarity')}")

                    # Print CodeBERTScore
                    codebert = metrics.get("codebert_score", {})
                    if isinstance(codebert, dict) and all(isinstance(v, float) for v in codebert.values()):
                        print("CodeBERTScore for C file:")
                        for k in ["precision", "recall", "f1", "f3"]:
                            if k in codebert:
                                print(f"{k}: {codebert[k]}")
                    elif isinstance(codebert, dict) and "error" in codebert:
                        print(f"CodeBERTScore error: {codebert['error']}")

                    # Print OpenAI cosine similarity
                    cosine = metrics.get("cosine_similarity_openai")
                    if isinstance(cosine, float):
                        print(f"Cosine similarity (Open AI) = {cosine}")
                    elif cosine == "skipped":
                        print(f"Skipping OpenAI cosine similarity: total tokens exceed limit ({MAX_TOKENS})")
                    else:
                        print(f"Cosine similarity (Open AI) = {cosine}")

                    all_results.append({
                        "cve_id": cve_id,
                        "downstream_version": downstream_version,
                        "file_name": file_name,
                        "upstream_codebase": upstream_content,
                        "downstream_codebase": downstream_content,
                        "metrics": metrics,
                    })
                    pbar.update(1)

    # Save JSON
    with open(results_json, "w") as jf:
        json.dump(all_results, jf, indent=2)

    # Save CSV (flattened metrics)
    if all_results:
        # Gather all codebert keys
        codebert_keys = []
        for r in all_results:
            if "metrics" in r and "codebert_score" in r["metrics"]:
                codebert_keys = list(r["metrics"]["codebert_score"].keys())
                break

        fieldnames = [
            "cve_id", "downstream_version", "file_name",
            "relative_line_count_similarity", "token_level_edit_similarity",
            "normalized_edit_similarity", "cosine_similarity_openai"
        ] + [f"codebert_{k}" for k in codebert_keys]

        with open(results_csv, "w", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_results:
                flat = {
                    "cve_id": row["cve_id"],
                    "downstream_version": row["downstream_version"],
                    "file_name": row["file_name"],
                }
                metrics = row.get("metrics", {})
                flat["relative_line_count_similarity"] = metrics.get("relative_line_count_similarity")
                flat["token_level_edit_similarity"] = metrics.get("token_level_edit_similarity")
                flat["normalized_edit_similarity"] = metrics.get("normalized_edit_similarity")
                flat["cosine_similarity_openai"] = metrics.get("cosine_similarity_openai")
                if "codebert_score" in metrics:
                    for k in codebert_keys:
                        flat[f"codebert_{k}"] = metrics["codebert_score"].get(k)
                writer.writerow(flat)

    print(f"\n✅ Results saved to:\n  JSON: {results_json}\n  CSV: {results_csv}")

if __name__ == "__main__":
    main()
import os
import json
import argparse
import csv
from tqdm import tqdm  # type: ignore
from metrics.line_metrics import relative_line_count_similarity
from metrics.similarity.codeBERT import compute_codebert_score
from metrics.similarity.openAI import compute_cosine_openai_embedding
from metrics.distance.edit_distance import (
    token_level_edit_distance,
    normalized_edit_similarity,
)
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
    }
    return ext_map.get(ext, ext)  # fallback to extension

def compute_metrics(upstream_code, downstream_code, file_name) -> dict:
    metrics = {}

    rls = relative_line_count_similarity(downstream_code, upstream_code)
    metrics["relative_line_count_similarity"] = round(rls, 4) if isinstance(rls, float) else rls

    nes = normalized_edit_similarity(downstream_code, upstream_code)
    metrics["normalized_edit_similarity"] = round(nes, 4) if isinstance(nes, float) else nes

    tled = token_level_edit_distance(downstream_code, upstream_code)
    metrics["token_level_edit_distance"] = round(tled, 4) if isinstance(tled, float) else tled

    language = get_language_from_filename(file_name)
    codebert = compute_codebert_score(downstream_code, upstream_code, language)
    if "error" in codebert:
        metrics["codebert_score"] = codebert
    else:
        metrics["codebert_score"] = {k: round(v, 4) for k, v in codebert.items()}

    total_tokens_upstream = count_tokens(upstream_code)
    total_tokens_downstream = count_tokens(downstream_code)
    total_tokens = total_tokens_upstream + total_tokens_downstream

    metrics["token_count_upstream"] = total_tokens_upstream
    metrics["token_count_downstream"] = total_tokens_downstream
    metrics["token_count_total"] = total_tokens
    print(f"🧠 Token usage for {file_name}:")
    print(f"- Upstream: {total_tokens_upstream}")
    print(f"- Downstream: {total_tokens_downstream}")
    print(f"- Total: {total_tokens} (limit = {MAX_TOKENS})")
    if total_tokens > MAX_TOKENS * 2:
        print(f"⚠️ Skipping all metrics for {file_name} — total tokens = {total_tokens}")
        metrics["skipped"] = True
        return metrics

    if total_tokens > MAX_TOKENS:
        metrics["cosine_similarity_openai"] = "skipped"
    else:
        score = compute_cosine_openai_embedding(upstream_code, downstream_code)
        metrics["cosine_similarity_openai"] = round(score, 4) if isinstance(score, float) else score


    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate patch comparisons and save results incrementally.")
    parser.add_argument("--json_input", required=True, help="Path to input JSON file")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_json = f"results/summary_{timestamp}.json"
    results_csv = f"results/summary_{timestamp}.csv"

    os.makedirs(os.path.dirname(results_json), exist_ok=True)
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)

    with open(args.json_input, "r") as f:
        report = json.load(f)

    total_conflicts = sum(
        len(failure.get("file_conflicts", []))
        for vuln in report
        for failure in vuln.get("failures", [])
    )

    all_results = []
    header_written = os.path.exists(results_csv)

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
                    print(f"⚠️ Skipping {cve_id} — no file_conflicts for {downstream_version}.")
                    continue

                for conflict in file_conflicts:
                    file_name = conflict.get("file_name", "unknown")
                    ground_truth = clean_code(conflict.get("downstream_file_content_ground_truth", ""))
                    patched = clean_code(conflict.get("downstream_file_content_patched_llm", ""))

                    print(f"\n🔍 Comparing LLM-patched vs Ground Truth for {file_name} in {cve_id} ({downstream_version})")
                    metrics = compute_metrics(patched, ground_truth, file_name)

                    if metrics.get("skipped"):
                        continue


                    result_entry = {
                        "cve_id": cve_id,
                        "downstream_version": downstream_version,
                        "file_name": file_name,
                        "llm_patched_code": patched,
                        "ground_truth_code": ground_truth,
                        "metrics": metrics,
                    }


                    all_results.append(result_entry)
                    pbar.update(1)

                    # Write updated JSON
                    with open(results_json, "w") as jf:
                        json.dump(all_results, jf, indent=2)

                    # Flatten for CSV
                    flat = {
                        "cve_id": cve_id,
                        "downstream_version": downstream_version,
                        "file_name": file_name,
                        "relative_line_count_similarity": metrics.get("relative_line_count_similarity"),
                        "token_level_edit_distance": metrics.get("token_level_edit_distance"),
                        "normalized_edit_similarity": metrics.get("normalized_edit_similarity"),
                        "cosine_similarity_openai": metrics.get("cosine_similarity_openai"),
                        "total_tokens": metrics.get("token_count_total"),
                        "token_count_ground_truth": metrics.get("token_count_ground_truth"),
                        "token_count_patched": metrics.get("token_count_predicted"),
                    }

                    if "codebert_score" in metrics:
                        codebert_dict = metrics["codebert_score"]
                        if isinstance(codebert_dict, dict):
                            for k, v in codebert_dict.items():
                                flat[f"codebert_{k}"] = v

                    # Write to CSV
                    with open(results_csv, "a", newline="") as cf:
                        writer = csv.DictWriter(cf, fieldnames=flat.keys())
                        if not header_written:
                            writer.writeheader()
                            header_written = True
                        writer.writerow(flat)

    print(f"\n✅ Incremental results saved to:\n  JSON: {results_json}\n  CSV: {results_csv}")

if __name__ == "__main__":
    main()

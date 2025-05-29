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
    if total_tokens > MAX_TOKENS:
        metrics["cosine_similarity_openai"] = "skipped"
    else:
        score = compute_cosine_openai_embedding(upstream_code, downstream_code)
        metrics["cosine_similarity_openai"] = round(score, 4) if isinstance(score, float) else score

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Compare cleaned_ground_truth vs. cleaned_upstream_plus_llm for each JSON record."
    )
    parser.add_argument(
        "--json_input", "-i", required=True,
        help="Path to input JSON file (a list of records with cleaned_ground_truth and cleaned_upstream_plus_llm)."
    )
    args = parser.parse_args()

    # timestamped outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_json = f"results/gt_vs_llm_{timestamp}.json"
    results_csv  = f"results/gt_vs_llm_{timestamp}.csv"

    os.makedirs(os.path.dirname(results_json), exist_ok=True)
    os.makedirs(os.path.dirname(results_csv),  exist_ok=True)

    # load a flat list of records
    with open(args.json_input, "r") as f:
        records = json.load(f)

    all_results    = []
    header_written = os.path.exists(results_csv)

    with tqdm(total=len(records), desc="Comparing GT vs. LLM") as pbar:
        for rec in records:
            cve_id      = rec.get("cve_id",           "unknown")
            version     = rec.get("downstream_version","unknown")
            file_name   = rec.get("file_name",        "unknown")
            upstream    = clean_code(rec.get("cleaned_upstream_plus_llm", ""))
            downstream  = clean_code(rec.get("cleaned_ground_truth",      ""))

            print(f"\n⏳ Comparing {file_name} for {cve_id} (ver {version})")
            metrics = compute_metrics(upstream, downstream, file_name)

            # build our result object
            entry = {
                "cve_id":                 cve_id,
                "downstream_version":     version,
                "file_name":              file_name,
                "cleaned_upstream_plus_llm": upstream,
                "cleaned_ground_truth":      downstream,
                "metrics":                metrics,
            }
            all_results.append(entry)
            pbar.update(1)

            # write incremental JSON
            with open(results_json, "w") as jf:
                json.dump(all_results, jf, indent=2)

            # flatten for CSV
            flat = {
                "cve_id":                     cve_id,
                "downstream_version":         version,
                "file_name":                  file_name,
                "relative_line_count_similarity": metrics["relative_line_count_similarity"],
                "normalized_edit_similarity": metrics["normalized_edit_similarity"],
                "token_level_edit_distance":  metrics["token_level_edit_distance"],
                "cosine_similarity_openai":   metrics["cosine_similarity_openai"],
                "token_count_upstream":       metrics["token_count_upstream"],
                "token_count_downstream":     metrics["token_count_downstream"],
                "token_count_total":          metrics["token_count_total"],
            }
            # include any codebert sub-scores
            cb = metrics.get("codebert_score")
            if isinstance(cb, dict):
                for name, score in cb.items():
                    flat[f"codebert_{name}"] = score

            # append to CSV
            with open(results_csv, "a", newline="") as cf:
                writer = csv.DictWriter(cf, fieldnames=flat.keys())
                if not header_written:
                    writer.writeheader()
                    header_written = True
                writer.writerow(flat)

    print(f"\n✅ Done. Outputs:\n  • JSON: {results_json}\n  • CSV:  {results_csv}")


if __name__ == "__main__":
    main()

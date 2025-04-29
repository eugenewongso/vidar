import os
import json
import argparse
import subprocess
import csv
import re

def sanitize_filename(s):
    return s.replace("/", "_").replace("..", "").strip()

# TODO: fix such that this will directly call functions from run_metrics_inline.py instead of executing it on terminal (much safer approach)
def compare_with_metrics_inline(ground_code, candidate_code):
    # Run the metrics script and capture output
    result = subprocess.run(
        [
            "python3", "run_metrics_inline.py",
            "--ground_code", ground_code,
            "--candidate_code", candidate_code,
        ],
        capture_output=True,
        text=True,
    )
    return result.stdout


def parse_metrics(output):
    # Parse the output from run_metrics_inline.py into a dict
    metrics = {}
    # Relative Line Count Difference
    m = re.search(r"Relative Line Count Difference:\s*([-\d.]+)", output)
    if m:
        metrics["relative_line_count_diff"] = float(m.group(1))
    # Token-Level Edit Similarity
    m = re.search(r"Token-Level Edit Similarity:\s*(\d+)", output)
    if m:
        metrics["token_level_edit_similarity"] = int(m.group(1))
    # Normalized Edit Similarity
    m = re.search(r"Normalized Edit Similarity:\s*([-\d.]+)", output)
    if m:
        metrics["normalized_edit_similarity"] = float(m.group(1))
    # CodeBERTScore
    codebert = {}
    codebert_section = re.search(r"CodeBERTScore for C file:\s*((?:\w+: [\d.]+\s*)+)", output)
    if codebert_section:
        for line in codebert_section.group(1).strip().splitlines():
            k, v = line.split(":")
            codebert[k.strip()] = float(v.strip())
        metrics["codebert_score"] = codebert
    # Cosine similarity (Open AI)
    m = re.search(r"Cosine similarity \(Open AI\) = ([\d.]+)", output)
    if m:
        metrics["cosine_similarity_openai"] = float(m.group(1))
    elif "Skipping OpenAI cosine similarity" in output:
        metrics["cosine_similarity_openai"] = None
    return metrics

def clean_code(code):
    code = code.strip()
    if code.startswith("```"):
        code = code.split('\n', 1)[-1]
    if code.endswith("```"):
        code = code.rsplit('```', 1)[0]
    return code.strip()

def main():
    parser = argparse.ArgumentParser(description="Evaluate failed patch comparisons using inline JSON content.")
    parser.add_argument("--json_input", required=True, help="Path to the JSON file with failure data")
    parser.add_argument("--results_json", default="results/summary.json", help="Path to save result summary JSON")
    parser.add_argument("--results_csv", default="results/summary.csv", help="Path to save result summary CSV")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.results_json), exist_ok=True)

    with open(args.json_input, "r") as f:
        report = json.load(f)

    all_results = []

    for vuln in report:
        cve_id = vuln["id"]
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

            conflict = file_conflicts[0]
            file_name = conflict["file_name"]

            upstream_content = clean_code(conflict.get("upstream_file_content", ""))
            downstream_content = clean_code(conflict.get("downstream_file_content", ""))

            print(f"\n🔍 Comparing {file_name} for {cve_id} ({downstream_version})")
            metrics_output = compare_with_metrics_inline(upstream_content, downstream_content)
            metrics = parse_metrics(metrics_output)

            all_results.append({
                "cve_id": cve_id,
                "downstream_version": downstream_version,
                "file_name": file_name,
                "upstream_codebase": upstream_content,
                "downstream_codebase": downstream_content,
                "metrics": metrics,
            })

    # Save JSON
    with open(args.results_json, "w") as jf:
        json.dump(all_results, jf, indent=2)

    # Save CSV (flattened metrics)
    if all_results:
        fieldnames = [
            "cve_id", "downstream_version", "file_name",
            "relative_line_count_diff", "token_level_edit_distance",
            "normalized_edit_distance", "cosine_similarity_openai"
        ]
        # Add CodeBERTScore keys if present
        codebert_keys = []
        for r in all_results:
            if "metrics" in r and "codebert_score" in r["metrics"]:
                codebert_keys = list(r["metrics"]["codebert_score"].keys())
                break
        fieldnames += [f"codebert_{k}" for k in codebert_keys]
        with open(args.results_csv, "w", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_results:
                flat = {
                    "cve_id": row["cve_id"],
                    "downstream_version": row["downstream_version"],
                    "file_name": row["file_name"],
                }
                metrics = row.get("metrics", {})
                flat["relative_line_count_diff"] = metrics.get("relative_line_count_diff")
                flat["token_level_edit_distance"] = metrics.get("token_level_edit_distance")
                flat["normalized_edit_distance"] = metrics.get("normalized_edit_distance")
                flat["cosine_similarity_openai"] = metrics.get("cosine_similarity_openai")
                if "codebert_score" in metrics:
                    for k in codebert_keys:
                        flat[f"codebert_{k}"] = metrics["codebert_score"].get(k)
                writer.writerow(flat)

    print(f"\n✅ Results saved to:\n  JSON: {args.results_json}\n  CSV: {args.results_csv}")

if __name__ == "__main__":
    main()

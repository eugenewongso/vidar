import os
import json
import argparse
import subprocess
import csv

def extract_inline_file(content: str, output_path: str):
    try:
        with open(output_path, "w") as out_file:
            out_file.write(content)
        return True
    except Exception as e:
        print(f"Error writing file {output_path}: {e}")
        return False


def sanitize_filename(s):
    return s.replace("/", "_").replace("..", "").strip()


def compare_with_metrics(upstream_path, downstream_path):
    subprocess.run([
        "python3", "run_metrics.py",
        "--ground", upstream_path,
        "--candidate", downstream_path,
    ])


def main():
    parser = argparse.ArgumentParser(description="Evaluate failed patch comparisons using inline JSON content.")
    parser.add_argument("--json_input", required=True, help="Path to the JSON file with failure data")
    parser.add_argument("--upstream_dir", default="input/upstream_commit", help="Directory to save upstream files")
    parser.add_argument("--downstream_dir", default="input/downstream_commit", help="Directory to save downstream files")
    parser.add_argument("--index_path", required=True, help="Path to the FAISS index for similarity checks")
    parser.add_argument("--results_json", default="results/summary.json", help="Path to save result summary JSON")
    parser.add_argument("--results_csv", default="results/summary.csv", help="Path to save result summary CSV")
    args = parser.parse_args()

    os.makedirs(args.upstream_dir, exist_ok=True)
    os.makedirs(args.downstream_dir, exist_ok=True)
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

            upstream_content = conflict.get("upstream_file_content", "").strip("```java\n").strip("```")
            downstream_content = conflict.get("downstream_file_content", "").strip("```java\n").strip("```")

            suffix = sanitize_filename(f"{cve_id}_{downstream_version}_{file_name}")
            upstream_file_path = os.path.join(args.upstream_dir, f"{suffix}_upstream.java")
            downstream_file_path = os.path.join(args.downstream_dir, f"{suffix}_downstream.java")

            got_upstream = extract_inline_file(upstream_content, upstream_file_path)
            got_downstream = extract_inline_file(downstream_content, downstream_file_path)

            if got_upstream and got_downstream:
                print(f"\n🔍 Comparing {upstream_file_path} vs {downstream_file_path}")
                compare_with_metrics(upstream_file_path, downstream_file_path)

                all_results.append({
                    "cve_id": cve_id,
                    "downstream_version": downstream_version,
                    "file_name": file_name,
                    "upstream_file": upstream_file_path,
                    "downstream_file": downstream_file_path
                })

    # Save JSON
    with open(args.results_json, "w") as jf:
        json.dump(all_results, jf, indent=2)

    # Save CSV
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(args.results_csv, "w", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_results:
                writer.writerow(row)

    print(f"\n✅ Results saved to:\n  JSON: {args.results_json}\n  CSV: {args.results_csv}")


if __name__ == "__main__":
    main()

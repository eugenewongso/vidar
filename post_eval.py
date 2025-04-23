import os
import json
import argparse
import subprocess

def extract_file_at_commit(repo_path, commit_hash, file_path, output_path):
    try:
        with open(output_path, "w") as out_file:
            subprocess.run(
                ["git", "show", f"{commit_hash}:{file_path}"],
                cwd=repo_path,
                check=True,
                stdout=out_file,
                stderr=subprocess.DEVNULL,
            )
        return True
    except subprocess.CalledProcessError:
        print(f"⚠️ File {file_path} not found in commit {commit_hash}.")
        return False
    
def sanitize_filename(s):
    return s.replace("/", "_").replace("..", "").strip()

# if name of file metrics.py is changed, this must be changed.
def compare_with_metrics(upstream_path, downstream_path):
    subprocess.run([
        "python3", "run_metrics.py",
        "--ground", upstream_path,
        "--candidate", downstream_path
    ])

def main():
    parser = argparse.ArgumentParser(description="Run patch evaluation post-processing.")
    parser.add_argument("--repo", required=True, help="Path to the Linux kernel repository")
    parser.add_argument("--eval_report", required=True, help="Path to the JSON evaluation report")
    parser.add_argument("--upstream_dir", default="testing_files/upstream_commit", help="Directory to save upstream files")
    parser.add_argument("--downstream_dir", default="testing_files/downstream_commit", help="Directory to save downstream files")
    
    args = parser.parse_args()

    os.makedirs(args.upstream_dir, exist_ok=True)
    os.makedirs(args.downstream_dir, exist_ok=True)

    with open(args.eval_report, "r") as f:
        report = json.load(f)

    for cve in report["cves_with_all_failures"]:
        cve_id = os.path.basename(cve["cve_url"]).replace(".json", "")
        for attempt in cve["patch_attempts"]:
            upstream_commit = attempt["upstream_commit"]
            for file_path in attempt.get("file_paths", []):
                for result in attempt["patch_results"]:
                    downstream_commit = result["downstream_commit"]

                    filename_suffix = sanitize_filename(f"{cve_id}_{os.path.basename(file_path)}")
                    upstream_file_path = os.path.join(args.upstream_dir, f"{cve_id}_{upstream_commit[:8]}_{filename_suffix}")
                    downstream_file_path = os.path.join(args.downstream_dir, f"{cve_id}_{downstream_commit[:8]}_{filename_suffix}")

                    got_upstream = extract_file_at_commit(args.repo, upstream_commit, file_path, upstream_file_path)
                    got_downstream = extract_file_at_commit(args.repo, downstream_commit, file_path, downstream_file_path)

                    if got_upstream and got_downstream:
                        print(f"\n🔍 Comparing {upstream_file_path} vs {downstream_file_path}")
                        compare_with_metrics(upstream_file_path, downstream_file_path)

if __name__ == "__main__":
    main()

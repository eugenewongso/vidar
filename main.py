import os
import json
import argparse
import subprocess
from post_eval import extract_file_at_commit, sanitize_filename, compare_with_metrics

def run_code_search(repo_path, commit_hash, query_file, top_k=3):
    """Run semantic code search using the provided query file."""
    output_dir = os.path.join(os.getcwd(), "vector_indexes")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🔍 Running semantic code search for {query_file} in commit {commit_hash}")
    subprocess.run([
        "python3", "code_search/code_search.py",
        "--repo_path", repo_path,
        "--commit_hash", commit_hash,
        "--query_file", query_file,
        "--top_k", str(top_k),
        "--output_dir", output_dir
    ])

def main():
    parser = argparse.ArgumentParser(description="Run combined patch evaluation and code search.")
    parser.add_argument("--repo", required=True, help="Path to the Linux kernel repository")
    parser.add_argument("--eval_report", required=True, help="Path to the JSON evaluation report")
    parser.add_argument("--upstream_dir", default="input/upstream_commit", help="Directory to save upstream files")
    parser.add_argument("--downstream_dir", default="input/downstream_commit", help="Directory to save downstream files")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top matches to return in semantic search")
    
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
                        print(f"\n📊 Comparing metrics: {os.path.basename(upstream_file_path)} vs {os.path.basename(downstream_file_path)}")
                        compare_with_metrics(upstream_file_path, downstream_file_path)
                        
                        # Run semantic search using upstream file as query against downstream commit
                        print(f"\n🔎 Finding similar code in downstream repository...")
                        run_code_search(args.repo, downstream_commit, upstream_file_path, args.top_k)
                        
                        # Run semantic search using downstream file as query against upstream commit
                        print(f"\n🔍 Finding similar code in upstream repository...")
                        run_code_search(args.repo, upstream_commit, downstream_file_path, args.top_k)

if __name__ == "__main__":
    main()
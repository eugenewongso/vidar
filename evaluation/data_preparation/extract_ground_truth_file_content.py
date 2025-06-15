r"""Augments a failure report with ground truth source code content.

This script enriches a JSON failure report with the actual, "ground truth"
content of source files at the exact moment a patch was supposed to be applied.
This is crucial for later steps, such as providing context to an LLM for patch
correction.

The process for each failure in the report is as follows:
1.  Identifies the repository path, target filename, downstream commit hash, and
    Android release version associated with the failure.
2.  If the repository doesn't exist locally, it attempts to clone it from the
    AOSP Google Source server.
3.  Performs a `git clean` to ensure a pristine state.
4.  Checks out the correct downstream branch (e.g., `android14-release`).
5.  Checks out the specific commit hash where the patch was applied.
6.  Reads the full content of the target source file at that commit.
7.  Calculates token counts for the file content.
8.  Injects the ground truth content and token counts back into the JSON report
    under the corresponding failure entry.
9.  Saves the fully augmented report to a new JSON file.

Usage:
  python extract_ground_truth_file_content.py \
      --input <path_to_failures.json> \
      --output <path_to_augmented_report.json>
"""
import json
import os
import sys
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from android_patch_manager import AndroidPatchManager

# Input/output file paths
INPUT_FILE = "reports/20242025_combined_failures.json"
OUTPUT_FILE = "run_results/all_versions/2024_2025/20242025_combined_failures_with_ground_truth.json"
GCP_PROJECT = "neat-resolver-406722"

def extract_ground_truth_file_content(repo_path, file_name, commit_hash, downstream_version):
    """
    Ensures repo is cloned, checks out to android14-release then specific commit,
    and extracts ground truth file content + tokens.

    Args:
        repo_path (str): Local path where the repo should be located.
        file_name (str): File to extract content from.
        commit_hash (str): Downstream commit hash to get ground truth from.

    Returns:
        tuple[str, dict]: Content and token statistics of the ground truth file.
    """
    if not os.path.exists(repo_path):
        print(f"⚠️ Repo path not found: {repo_path}")
        repo_name = os.path.basename(repo_path)

        # Construct AOSP URL
        repo_url = f"https://android.googlesource.com/platform/{repo_name}"
        repo_base = os.path.dirname(repo_path)

        try:
            repo_path = AndroidPatchManager.clone_repo(repo_url, repo_base)
        except Exception as e:
            print(f"❌ Failed to clone {repo_url}: {e}")
            return "", {}

    try:
        AndroidPatchManager.clean_repo(repo_path)

        branch_name = f"android{downstream_version}-release"
        try:
            AndroidPatchManager.checkout_downstream_branch(repo_path, downstream_version)
        except Exception as e:
            print(f"⚠️ Could not checkout {branch_name}: {e}")


        # Checkout the actual downstream patch commit
        AndroidPatchManager.checkout_commit(repo_path, commit_hash)

        file_path = os.path.join(repo_path, file_name)
        if not os.path.exists(file_path):
            print(f"⚠️ File not found in repo: {file_path}")
            return "", {}

        with open(file_path, "r") as f:
            content = f.read()
            tokens = AndroidPatchManager.get_all_token_counts(content)
            return content.strip(), tokens

    except Exception as e:
        print(f"❌ Error extracting ground truth for {file_name} at {commit_hash}: {e}")
        return "", {}



def main():
    """
    Main function to augment a patch failure report with ground truth file content and token stats.

    For each failed patch case, this:
    - Looks up the repo and downstream commit
    - Extracts actual file content from that commit
    - Computes and stores ground truth file content + token counts
    - Saves updated JSON to a new file
    """
    # Load original failure report
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    # Iterate through all failure entries
    for vuln in data.get("failures", []):
        for result in vuln.get("failures", []):
            commit = result.get("downstream_patch")
            repo_path = result.get("repo_path")
            downstream_version = result.get("downstream_version")

            for fc in result.get("file_conflicts", []):
                file_name = fc.get("file_name")
                if not file_name or not commit or not repo_path or not downstream_version:
                    continue

                # Extract and attach ground truth data
                gt_content, gt_tokens = extract_ground_truth_file_content(repo_path, file_name, commit, downstream_version)
                fc["downstream_file_content_ground_truth"] = gt_content
                fc["downstream_file_ground_truth_tokens"] = gt_tokens

    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Save the updated report
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Updated file saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

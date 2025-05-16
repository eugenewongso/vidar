import os
import json
import subprocess
from pathlib import Path
import re
import tempfile
import difflib

# Context sizes to evaluate — determines how many surrounding lines were given to the LLM
CONTEXT_SIZES = [3, 5, 10, 20]

# Input and output file paths
INPUT_FILE = "filtered_failures_android_14_2025_with_context_3_5_10_20_with_llm.json"
OUTPUT_FILE = "llm_patch_apply_results.json"

def parse_patch_hunks(output):
    """
    Parse patch command output to extract how many hunks failed out of how many total.
    Returns (failed, total) or (0, 0) if not found.
    """
    match = re.search(r"(\d+) out of (\d+) hunks failed", output)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 0

def run(cmd, cwd):
    """
    Run a shell command in the specified directory, with timeout and output logging.
    
    Args:
        cmd (str): The shell command to run.
        cwd (str): The directory to execute the command in.
    
    Returns:
        tuple: (success: bool, output: str)
    """
    try:
        print(f"\n💻 Running command: {cmd} (in {cwd})")
        result = subprocess.run(
            cmd,
            cwd=cwd,
            shell=True,
            text=True,
            capture_output=True,
            executable="/bin/bash",  # ensure consistent shell
            timeout=60               # timeout in seconds to prevent hanging
        )
        output = result.stdout + result.stderr
        print(f"📤 Command output:\n{output}")
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout expired for command: {cmd}")
        return False, f"Timeout expired for: {cmd}"
    except Exception as e:
        print(f"❌ Exception while running command: {cmd}\n{e}")
        return False, str(e)


def apply_patch(patch_str, repo_path):
    """
    Apply a patch string to a repo using GNU patch.
    
    Args:
        patch_str (str): The unified diff patch content.
        repo_path (str): Path to the target Git repo.
    
    Returns:
        tuple: (success: bool, patch_output: str)
    """
    # Write patch to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix=".patch", delete=False) as tmp:
        tmp.write(patch_str)
        patch_path = tmp.name

    # Apply the patch
    cmd = f"patch -p1 --ignore-whitespace -i {patch_path}"
    success, output = run(cmd, cwd=repo_path)

    # Clean up temporary patch file
    os.remove(patch_path)
    return success, output


def evaluate_llm_patches(data):
    """
    Evaluate the application success of LLM-generated patches across all context sizes.
    
    Args:
        data (list): Loaded JSON data from input file.
    
    Returns:
        list: A list of result dictionaries per vulnerability.
    """
    results = []

    os.makedirs("test_patches", exist_ok=True)

    for entry in data:
        first_failure = entry.get("failures", [])[0] if entry.get("failures") else {}
        repo_path = first_failure.get("repo_path")
        downstream_patch = first_failure.get("downstream_patch")
        vuln_id = entry.get("id", "unknown")

        # Skip if required information is missing
        if not repo_path or not downstream_patch:
            print(f"⚠️ Skipping {vuln_id}: Missing repo_path or downstream_patch")
            continue

        repo_path = os.path.abspath(repo_path)
        if not Path(repo_path).exists():
            print(f"⚠️ Skipping {vuln_id}: repo_path does not exist")
            continue

        result_entry = {
            "id": vuln_id,
            "repo_path": repo_path,
            "downstream_patch": downstream_patch,
            "patch_attempts": {}
        }

        # Try each context window one by one
        for ctx in CONTEXT_SIZES:
            key = f"llm_patch_context_{ctx}"
            patches = []

            print(f"\n🚀 Checking out android14-release and resetting repo for {vuln_id} (context {ctx})")

            # Clean up before checking out
            run("git reset --hard", cwd=repo_path)
            run("git clean -fd", cwd=repo_path)

            # Checkout branch
            checkout_success, checkout_msg = run("git checkout android14-release", cwd=repo_path)
            if not checkout_success:
                result_entry["patch_attempts"][key] = {
                    "applied": False,
                    "message": f"Failed to checkout android14-release: {checkout_msg}"
                }
                continue

            # Reset to one commit before downstream
            reset_success, reset_msg = run(f"git reset --hard {downstream_patch}^", cwd=repo_path)
            clean_success, clean_msg = run("git clean -fd", cwd=repo_path)

            if not reset_success:
                result_entry["patch_attempts"][key] = {
                    "applied": False,
                    "message": f"Failed to reset: {reset_msg}"
                }
                continue


            if not reset_success:
                result_entry["patch_attempts"][key] = {
                    "applied": False,
                    "message": f"Failed to reset: {reset_msg}"
                }
                continue

            # Apply the upstream patch first
            # Collect upstream_patch_content from each failure in case it's split
            upstream_patch_content = first_failure.get("upstream_patch_content", "")
            for failure in entry.get("failures", []):
                if not upstream_patch_content and failure.get("upstream_patch_content"):
                    upstream_patch_content = failure["upstream_patch_content"]

            upstream_applied = False
            upstream_msg = ""

            if upstream_patch_content:
                upstream_applied, upstream_msg = apply_patch(upstream_patch_content.strip(), repo_path)
                upstream_failed, upstream_total = parse_patch_hunks(upstream_msg)
                print(f"📌 Upstream patch applied: {upstream_applied}")
                print(f"📝 Upstream patch output:\n{upstream_msg}")
            else:
                print("⚠️ No upstream_patch_content found.")

            # Even if upstream fails, we continue to try the LLM patch.

            # Collect and reconstruct the LLM patch for this context size
            for failure in entry.get("failures", []):
                for conflict in failure.get("file_conflicts", []):
                    if key in conflict:
                        patches.append(conflict[key])

            if not patches:
                result_entry["patch_attempts"][key] = {
                    "applied": False,
                    "message": "No patch found"
                }
                continue

            reconstructed_patches = []
            for failure in entry.get("failures", []):
                for conflict in failure.get("file_conflicts", []):
                    if key in conflict and "file_name" in conflict:
                        file_path = conflict["file_name"]
                        modified_str = conflict[key]

                        # Create raw diff using LLM patch body and standard headers
                        diff_patch = (
                            f"diff --git a/{file_path} b/{file_path}\n"
                            f"--- a/{file_path}\n"
                            f"+++ b/{file_path}\n"
                            f"{modified_str.strip()}"
                        )
                        reconstructed_patches.append(diff_patch)



            combined_patch = "\n".join(reconstructed_patches).strip()

            # Save the combined patch to a test file for inspection
            test_patch_path = f"test_patches/test_patch_{vuln_id}_ctx{ctx}.patch"

            with open(test_patch_path, "w", encoding="utf-8") as f:
                f.write(combined_patch)
            print(f"📄 Saved test patch to: {test_patch_path}")

            # Try applying the combined LLM patch
            success, output = apply_patch(combined_patch, repo_path)
            llm_failed, llm_total = parse_patch_hunks(output)
            result_entry["patch_attempts"][key] = {
                "applied": success,
                "message": output.strip(),
                "llm_failed_hunks": llm_failed,
                "llm_total_hunks": llm_total,
                "upstream_applied": upstream_applied,
                "upstream_message": upstream_msg.strip(),
                "upstream_failed_hunks": upstream_failed,
                "upstream_total_hunks": upstream_total
            }


        results.append(result_entry)

    return results

def main():
    """Main function to load data, evaluate patches, and save results."""
    with open(INPUT_FILE) as f:
        data = json.load(f)

    results = evaluate_llm_patches(data)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

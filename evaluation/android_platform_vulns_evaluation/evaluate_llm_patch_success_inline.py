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
INPUT_FILE = "reports/android_14_llm_processed_output_per_reject.json"
OUTPUT_FILE = "android_14_per_reject_llm_patch_apply_results.json"

def parse_patch_hunks(output):
    # Match both "X out of Y hunks failed" and "X out of Y hunks ignored"
    match = re.search(r"(\d+)\s+out of\s+(\d+)\s+hunks\s+(failed|ignored)", output, re.IGNORECASE)
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


def apply_patch(patch_str, repo_path, use_merge=False, strip_level=1):
    """
    Apply a patch string to a repo using GNU patch (gpatch), with optional --merge.
    
    Args:
        patch_str (str): The unified diff patch content.
        repo_path (str): Path to the target Git repo.
        use_merge (bool): Whether to use --merge option.
    
    Returns:
        tuple: (success: bool, patch_output: str)
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix=".patch", delete=False) as tmp:
        tmp.write(patch_str)
        patch_path = tmp.name

    # Build the patch command
    cmd_parts = ["gpatch", f"-p{strip_level}", "-i", patch_path, "--ignore-whitespace"]
    if use_merge:
        cmd_parts.insert(1, "--merge")


    success, output = run(" ".join(cmd_parts), cwd=repo_path)

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

            run("git fetch origin android14-release", cwd=repo_path)

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

            # Apply the upstream patch first
            # Collect upstream_patch_content from each failure in case it's split
            upstream_patch_content = first_failure.get("upstream_patch_content", "")
            for failure in entry.get("failures", []):
                if not upstream_patch_content and failure.get("upstream_patch_content"):
                    upstream_patch_content = failure["upstream_patch_content"]

            upstream_applied = False
            upstream_msg = ""
            upstream_failed = 0
            upstream_total = 0 

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

            # Apply .rej patch content first with --merge
            rej_patches = []
            for failure in entry.get("failures", []):
                for conflict in failure.get("file_conflicts", []):
                    if "rej_file_content" in conflict and "file_name" in conflict:
                        file_path = conflict["file_name"]
                        rej_body = conflict["rej_file_content"]
                        if rej_body.strip():
                            rej_patch = (
                                f"diff --git a/{file_path} b/{file_path}\n"
                                f"--- a/{file_path}\n"
                                f"+++ b/{file_path}\n"
                                f"{rej_body.strip()}"
                            )
                            rej_patches.append(rej_patch)

            rej_combined_patch = "\n".join(rej_patches).strip()
            rej_success, rej_output = False, ""
            if rej_combined_patch:
                print("🧪 Applying .rej patch with --merge")
                rej_success, rej_output = apply_patch(rej_combined_patch, repo_path, use_merge=True, strip_level=0)

            # Try applying the combined LLM patch
            success, output = apply_patch(combined_patch, repo_path)
            llm_failed, llm_total = parse_patch_hunks(output)

            # Store patch content in result for reference
            result_entry["patch_attempts"][key] = {
                "applied": success,
                "message": output.strip(),
                "llm_failed_hunks": llm_failed,
                "llm_total_hunks": llm_total,
                "upstream_applied": upstream_applied,
                "upstream_message": upstream_msg.strip(),
                "upstream_failed_hunks": upstream_failed,
                "upstream_total_hunks": upstream_total,
                "llm_patch_content": combined_patch,
                "rej_applied": rej_success,
                "rej_output": rej_output.strip(),
                "rej_patch_content": rej_combined_patch
            }

        results.append(result_entry)

    return results

def main():
    """Main function to load data, evaluate patches, and save results."""
    with open(INPUT_FILE) as f:
        raw_data = json.load(f)

    # Fix: check if it's a dict with 'failures' or already a list
    if isinstance(raw_data, dict) and "failures" in raw_data:
        data = raw_data["failures"]
    else:
        data = raw_data

    results = evaluate_llm_patches(data)



    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

"""
This script processes a JSON dataset of Android CVE patching attempts,
navigates to each local Git repository, checks out the relevant downstream branch,
applies the upstream patch followed by the LLM-generated downstream patch,
and records whether the patches succeeded or failed along with any output messages.

It uses the AndroidPatchManager class to handle repo operations.

Input:
- JSON file with fields: id, upstream_patch_content, and failures (containing repo_path, branch_used, downstream_llm_diff_output)

Output:
- A new JSON file "patch_application_results.json" containing:
  - Patch application results (success/failure)
  - Patch outputs for debugging
  - Any errors encountered

Requirements:
- Repositories must exist in the "android_repos/" folder
- gpatch or patch must be installed and accessible from the command line
- android_patch_manager.py must be importable
"""

import json
import os
import tempfile
import difflib
from pathlib import Path
from android_patch_manager import AndroidPatchManager

# Configurable input and base repo location
INPUT_FILE = "approach_2_android_14_2024_with_llm_output_smart_retry.json"
REPO_BASE = Path("android_repos")

# Load input JSON data
with open(INPUT_FILE, "r") as f:
    data = json.load(f)

results = []

# Iterate over each CVE entry
for entry in data:
    entry_id = entry.get("id")
    upstream_patch = entry.get("upstream_patch_content", "")
    failures = entry.get("failures", [])

    # For each failed downstream application attempt
    for failure in failures:
        file_conflicts = failure.get("file_conflicts", [])
        repo_path = Path(failure["repo_path"])
        downstream_version = failure["downstream_version"]

        for conflict in file_conflicts:
            downstream_patch = conflict.get("downstream_llm_diff_output", "")

            # Initialize result object
            result = {
                "id": entry_id,
                "repo_path": str(repo_path),
                "downstream_version": downstream_version,
                "file_name": conflict.get("file_name", ""),
                "upstream_patch": {"success": False, "output": ""},
                "downstream_patch": {"success": False, "output": ""},
                "error": ""
            }

            try:
                if not repo_path.exists():
                    result["error"] = f"❌ Repo not found: {repo_path}"
                    results.append(result)
                    continue

                AndroidPatchManager.clean_repo(str(repo_path))
                AndroidPatchManager.checkout_downstream_branch(str(repo_path), downstream_version)

                # Reset to parent of the downstream patch commit
                downstream_patch_commit = failure.get("downstream_patch")
                if downstream_patch_commit:
                    AndroidPatchManager.checkout_commit(str(repo_path), f"{downstream_patch_commit}^")

                # Apply upstream patch
                with tempfile.NamedTemporaryFile(delete=False, suffix=".diff", mode="w") as f:
                    f.write(upstream_patch)
                    upstream_patch_path = f.name

                success_up, out_up, *_ = AndroidPatchManager.apply_patch(str(repo_path), upstream_patch_path)
                result["upstream_patch"] = {"success": success_up, "output": out_up.strip()}

                # Save file content after upstream patch but before downstream patch
                file_rel_path = conflict.get("file_name")
                intermediate_file_path = repo_path / file_rel_path

                if intermediate_file_path.exists():
                    try:
                        with open(intermediate_file_path, "r", encoding="utf-8", errors="replace") as f:
                            upstream_only_content = f.read()
                        conflict["downstream_file_content_patched_upstream_only"] = f"```{file_rel_path.split('.')[-1]}\n{upstream_only_content.strip()}\n```"
                    except Exception as e:
                        conflict["downstream_file_content_patched_upstream_only"] = f"❌ Failed to read upstream-patched file: {e}"
                else:
                    conflict["downstream_file_content_patched_upstream_only"] = "❌ File not found after upstream patch"


                # Apply downstream (LLM) patch
                with tempfile.NamedTemporaryFile(delete=False, suffix=".diff", mode="w") as f:
                    f.write(downstream_patch)
                    downstream_patch_path = f.name

                success_down, out_down, *_ = AndroidPatchManager.apply_patch(str(repo_path), downstream_patch_path)
                result["downstream_patch"] = {"success": success_down, "output": out_down.strip()}

                # Save final file content after patching
                file_rel_path = conflict.get("file_name")
                final_file_path = repo_path / file_rel_path

                if final_file_path.exists():
                    try:
                        with open(final_file_path, "r", encoding="utf-8", errors="replace") as f:
                            patched_content = f.read()
                        conflict["downstream_file_content_patched_llm"] = f"```{file_rel_path.split('.')[-1]}\n{patched_content.strip()}\n```"

                        # ✅ Add diff here
                        original_downstream_content = conflict.get("downstream_file_content", "")
                        if isinstance(original_downstream_content, str):
                            try:
                                original_lines = original_downstream_content.strip().splitlines(keepends=True)
                                final_lines = patched_content.strip().splitlines(keepends=True)
                                diff_lines = difflib.unified_diff(
                                    original_lines,
                                    final_lines,
                                    fromfile="downstream_file_content",
                                    tofile="downstream_file_content_patched_llm"
                                )
                                conflict["diff_patched_downstream_file_content"] = "".join(diff_lines)
                            except Exception as e:
                                conflict["diff_patched_downstream_file_content"] = f"❌ Failed to generate diff: {e}"
                        else:
                            conflict["diff_patched_downstream_file_content"] = "❌ Original downstream content not available"

                    except Exception as e:
                        conflict["downstream_file_content_patched_llm"] = f"❌ Failed to read patched file: {e}"



            except Exception as e:
                result["error"] = str(e)

            results.append(result)


# Save updated original JSON structure with new field
with open("approach_2_android_14_2024_with_llm_output_smart_retry_patched.json", "w") as f:
    json.dump(data, f, indent=2)


# Save patch application summary results separately
with open("patch_summary_approach_2_android_14_2024_with_llm_output_smart_retry.json", "w") as f:
    json.dump(results, f, indent=2)


print("✅ Patch evaluation complete. Results saved to patch_summary_approach_2_android_14_2024_with_llm_output_smart_retry.json")

import json
import os
import tempfile
import difflib
from pathlib import Path
from android_patch_manager import AndroidPatchManager

INPUT_FILE = "run_results/all_versions/2024_2025/20242025_combined_failures_with_ground_truth.json"
REPO_BASE = Path("android_repos")

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

results = []

for entry in data:
    entry_id = entry.get("id")
    upstream_patch = entry.get("upstream_patch_content", "")
    failures = entry.get("failures", [])

    for failure in failures:
        file_conflicts = failure.get("file_conflicts", [])
        repo_path = Path(failure["repo_path"])
        downstream_version = failure["downstream_version"]

        for conflict in file_conflicts:
            result = {
                "id": entry_id,
                "repo_path": str(repo_path),
                "downstream_version": downstream_version,
                "file_name": conflict.get("file_name", ""),
                "upstream_patch": {"success": False, "output": ""},
                "error": ""
            }

            try:
                if not repo_path.exists():
                    result["error"] = f"❌ Repo not found: {repo_path}"
                    results.append(result)
                    continue

                AndroidPatchManager.clean_repo(str(repo_path))
                AndroidPatchManager.checkout_downstream_branch(str(repo_path), downstream_version)

                downstream_patch_commit = failure.get("downstream_patch")
                if downstream_patch_commit:
                    AndroidPatchManager.checkout_commit(str(repo_path), f"{downstream_patch_commit}^")

                # Apply upstream patch
                with tempfile.NamedTemporaryFile(delete=False, suffix=".diff", mode="w") as f:
                    f.write(upstream_patch)
                    upstream_patch_path = f.name

                success_up, out_up, *_ = AndroidPatchManager.apply_patch(str(repo_path), upstream_patch_path)
                result["upstream_patch"] = {"success": success_up, "output": out_up.strip()}

                # Save file after upstream patch
                file_rel_path = conflict.get("file_name")
                intermediate_path = repo_path / file_rel_path

                if intermediate_path.exists():
                    try:
                        with open(intermediate_path, "r", encoding="utf-8", errors="replace") as f:
                            upstream_only_content = f.read()
                        ext = file_rel_path.split(".")[-1]
                        conflict["downstream_file_content_patched_upstream_only"] = f"```{ext}\n{upstream_only_content.strip()}\n```"

                        # Optional diff from original downstream content
                        original = conflict.get("downstream_file_content", "")
                        if isinstance(original, str):
                            diff = difflib.unified_diff(
                                original.strip().splitlines(keepends=True),
                                upstream_only_content.strip().splitlines(keepends=True),
                                fromfile="original_downstream_file",
                                tofile="patched_upstream_only"
                            )
                            conflict["diff_patched_upstream_only"] = "".join(diff)
                        else:
                            conflict["diff_patched_upstream_only"] = "❌ Original downstream content missing"
                    except Exception as e:
                        conflict["downstream_file_content_patched_upstream_only"] = f"❌ Failed to read file after upstream patch: {e}"
                else:
                    conflict["downstream_file_content_patched_upstream_only"] = "❌ File not found after upstream patch"

            except Exception as e:
                result["error"] = str(e)

            results.append(result)

# Save updated data
with open("run_results/all_versions/2024_2025/20242025_combined_failures_with_ground_truth_and_patched_upstream.json", "w") as f:
    json.dump(data, f, indent=2)

# Save summary
with open("run_results/all_versions/2024_2025/summary_20242025_combined_failures_with_ground_truth_and_patched_upstream.json", "w") as f:
    json.dump(results, f, indent=2)

print("✅ Upstream-only patch evaluation complete. Results saved.")

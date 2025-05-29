import os
import json

INPUT_FILE = "reports/android_14_llm_processed_output_per_reject.json"
OUTPUT_DIR = "patches_to_test"

# Make sure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# If it's wrapped in {"failures": [...]}, unwrap
if isinstance(data, dict) and "failures" in data:
    data = data["failures"]

for entry in data:
    vuln_id = entry.get("id", "unknown")
    failures = entry.get("failures", [])
    if not failures:
        continue

    # Get upstream patch content from any failure entry
    upstream_patch = ""
    for failure in failures:
        if failure.get("upstream_patch_content"):
            upstream_patch = failure["upstream_patch_content"]
            break

    if upstream_patch.strip():
        upstream_path = os.path.join(OUTPUT_DIR, f"{vuln_id}_upstream.patch")
        with open(upstream_path, "w", encoding="utf-8") as f:
            f.write(upstream_patch.strip())

    # Save all LLM patch contexts
    for failure in failures:
        file_conflicts = failure.get("file_conflicts", [])
        for conflict in file_conflicts:
            for key, content in conflict.items():
                if key.startswith("llm_patch_context_"):
                    context_num = key.split("_")[-1]
                    filename = conflict.get("file_name", "unknown_file").replace("/", "_")
                    patch_text = content.strip()
                    if patch_text:
                        out_path = os.path.join(OUTPUT_DIR, f"{vuln_id}_{filename}_ctx{context_num}.patch")
                        with open(out_path, "w", encoding="utf-8") as f:
                            f.write(
                                f"diff --git a/{conflict['file_name']} b/{conflict['file_name']}\n"
                                f"--- a/{conflict['file_name']}\n"
                                f"+++ b/{conflict['file_name']}\n"
                                f"{patch_text}"
                            )

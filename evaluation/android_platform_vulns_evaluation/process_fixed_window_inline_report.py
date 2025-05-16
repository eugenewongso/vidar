import json
from collections import defaultdict
import os

# Load the provided JSON file
input_path = "filtered_failures_android_14_2025_with_context_3_5_10_20_with_llm.json"

with open(input_path, "r") as f:
    data = json.load(f)

summary = {
    "vulnerabilities_processed": 0,
    "contexts_processed": set(),
    "llm_patch_outputs": defaultdict(int),
    "per_vulnerability_summary": []
}

for entry in data:
    summary["vulnerabilities_processed"] += 1
    vuln_id = entry.get("id", "unknown_vuln")
    context_stats = defaultdict(int)

    for failure in entry.get("failures", []):
        for conflict in failure.get("file_conflicts", []):
            for key in conflict:
                if key.startswith("llm_patch_context_") and not key.endswith("_duration_seconds"):
                    ctx_size = key.replace("llm_patch_context_", "")
                    try:
                        ctx_size = int(ctx_size)
                        summary["contexts_processed"].add(ctx_size)
                        summary["llm_patch_outputs"][key] += 1
                        context_stats[ctx_size] += 1
                    except ValueError:
                        continue

    summary["per_vulnerability_summary"].append({
        "vulnerability_id": vuln_id,
        "llm_patch_counts_by_context": dict(sorted(context_stats.items()))
    })

# Final formatting
summary["contexts_processed"] = sorted(summary["contexts_processed"])
summary["llm_patch_outputs"] = dict(sorted(summary["llm_patch_outputs"].items()))

# Output the result
output_path = "llm_patch_summary_report.json"
with open(output_path, "w") as f:
    json.dump(summary, f, indent=2)

output_path

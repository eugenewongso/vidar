import json
import argparse
import os
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Separate LLM outputs by downstream version")
    parser.add_argument("--file", required=True, help="Path to the LLM output JSON file")
    parser.add_argument("--output_dir", default="version_separated", help="Directory to save version-specific files")
    args = parser.parse_args()

    base_filename = os.path.splitext(os.path.basename(args.file))[0]
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.file, "r") as f:
        data = json.load(f)

    version_dict = {}
    report_data = {
        "summary": {
            "total_file_conflicts_matching_version": 0,
            "files_skipped_pre_llm_call": 0,
            "files_attempted_for_llm_diff_generation": 0,
            "successful_llm_outputs": 0
        },
        "skipped_or_errored_diff_generation_log": []
    }

    found_versions = set()
    per_version_stats = defaultdict(lambda: {
        "successful": 0,
        "total": 0,
        "attempt_1": 0,
        "attempt_2": 0,
        "attempt_3": 0,
        "runtime_all": 0.0,
        "runtime_success": 0.0
    })

    for entry in data:
        if "failures" not in entry:
            continue

        for failure in entry["failures"]:
            version = failure.get("downstream_version", "unknown")
            found_versions.add(version)
            skip_entry = False
            entry_has_valid_output = False

            for file_conflict in failure.get("file_conflicts", []):
                report_data["summary"]["total_file_conflicts_matching_version"] += 1

                if not all([
                    file_conflict.get("file_name"),
                    file_conflict.get("rej_file_content"),
                    file_conflict.get("downstream_file_content_patched_upstream_only")
                ]):
                    report_data["summary"]["files_skipped_pre_llm_call"] += 1
                    report_data["skipped_or_errored_diff_generation_log"].append({
                        "vulnerability_id": entry['id'],
                        "file_name": file_conflict.get("file_name", "Unknown"),
                        "patch_sha": failure.get('downstream_patch', 'N/A'),
                        "reason": "Missing required fields"
                    })
                    skip_entry = True
                    break

                per_version_stats[version]["total"] += 1

                # Get runtime for this file conflict
                runtime_seconds = file_conflict.get("runtime_seconds", 0.0)
                per_version_stats[version]["runtime_all"] += runtime_seconds

                if file_conflict.get("llm_output_valid", False):
                    entry_has_valid_output = True
                    report_data["summary"]["successful_llm_outputs"] += 1
                    per_version_stats[version]["successful"] += 1
                    per_version_stats[version]["runtime_success"] += runtime_seconds
                    
                    # Track attempts made for successful outputs
                    attempts_made = file_conflict.get("attempts_made", 1)
                    if attempts_made == 1:
                        per_version_stats[version]["attempt_1"] += 1
                    elif attempts_made == 2:
                        per_version_stats[version]["attempt_2"] += 1
                    elif attempts_made == 3:
                        per_version_stats[version]["attempt_3"] += 1

            if not skip_entry:
                report_data["summary"]["files_attempted_for_llm_diff_generation"] += 1
                if entry_has_valid_output:
                    version_dict.setdefault(version, []).append(entry)

    for version, entries in version_dict.items():
        output_file = os.path.join(args.output_dir, f"{base_filename}_version_{version}.json")
        with open(output_file, "w") as f:
            json.dump(entries, f, indent=2)
        print(f"💾 Saved version {version} data to {output_file} ({len(entries)} entries)")

    print("\n📊 Summary:")
    print(f"Total file conflicts matching version: {report_data['summary']['total_file_conflicts_matching_version']}")
    print(f"Files skipped pre-LLM call: {report_data['summary']['files_skipped_pre_llm_call']}")
    print(f"Files attempted for LLM diff generation: {report_data['summary']['files_attempted_for_llm_diff_generation']}")
    print(f"Successful LLM outputs: {report_data['summary']['successful_llm_outputs']}")
    print(f"Total successful entries: {report_data['summary']['successful_llm_outputs']}/106")

    print("\n🧮 Per-Version Attempt & Runtime Breakdown:")
    print("Version\tSuccess\t\tCoverage\tRun 1\tRun 2\tRun 3\tRuntime (All)\tRuntime (Success)")
    for version in sorted(per_version_stats.keys()):
        stats = per_version_stats[version]
        success_str = f"{stats['successful']}/{stats['total']}"
        coverage = f"{(stats['successful'] / stats['total'] * 100):.2f}%" if stats['total'] > 0 else "0.00%"
        print(f"{version}\t{success_str}\t{coverage}\t{stats['attempt_1']}\t{stats['attempt_2']}\t{stats['attempt_3']}\t"
              f"{stats['runtime_all']:.2f}s\t{stats['runtime_success']:.2f}s")

    skip_log_file = os.path.join(args.output_dir, "skipped_entries_log.json")
    with open(skip_log_file, "w") as f:
        json.dump(report_data["skipped_or_errored_diff_generation_log"], f, indent=2)
    print(f"\n💾 Saved skip log to {skip_log_file}")

if __name__ == "__main__":
    main()

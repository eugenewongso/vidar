import json
import os
import argparse
from datetime import datetime

def extract_merge_conflicts(input_file):
    """ Extract failed patches, track success rates, and reorganize results by category """
    with open(input_file, "r") as file:
        full_report = json.load(file)

    report_data = {
        "summary": {},
        "cves_with_all_failures": [],
        "cves_with_partial_failures": [],
        "cves_with_all_successful_patches": []
    }

    total_cves_tested = len(full_report["CVEs"])
    total_versions_tested = 0
    total_failed_patches = 0
    total_cves_with_all_failures = 0
    total_cves_with_partial_failures = 0
    total_cves_with_all_successful_patches = 0
    unique_failed_versions = set()
    unique_tested_versions = set()

    for cve_entry in full_report["CVEs"]:
        cve_versions_tested = 0
        cve_failed_patches = 0
        all_failed = True
        any_failed = False
        patch_attempts_dict = {}

        for attempt in cve_entry["patch_attempts"]:
            total_versions_tested += attempt["total_versions_tested"]
            cve_versions_tested += attempt["total_versions_tested"]
            upstream_patch = attempt["upstream_patch"]

            if upstream_patch not in patch_attempts_dict:
                patch_attempts_dict[upstream_patch] = {
                    "upstream_patch": upstream_patch,
                    "total_versions_tested": 0,
                    "successful_patches": 0,
                    "failed_patches": 0,
                    "patch_results": []
                }

            failed_patches = [
                result for result in attempt["patch_results"] if result["result"] == "failure"
            ]
            successful_patches = [
                result for result in attempt["patch_results"] if result["result"] == "success"
            ]

            # ✅ Track unique versions tested
            for result in attempt["patch_results"]:
                unique_tested_versions.add(result["version"])

            # ✅ Track unique failed versions
            for result in failed_patches:
                unique_failed_versions.add(result["version"])

            patch_attempts_dict[upstream_patch]["total_versions_tested"] += attempt["total_versions_tested"]
            patch_attempts_dict[upstream_patch]["successful_patches"] += len(successful_patches)
            patch_attempts_dict[upstream_patch]["failed_patches"] += len(failed_patches)
            patch_attempts_dict[upstream_patch]["patch_results"].extend(failed_patches + successful_patches)

            cve_failed_patches += len(failed_patches)

            if failed_patches:
                any_failed = True

            if len(failed_patches) != attempt["total_versions_tested"]:
                all_failed = False

        patch_attempts_list = list(patch_attempts_dict.values())

        if all_failed and cve_versions_tested > 0:
            total_cves_with_all_failures += 1
            report_data["cves_with_all_failures"].append({
                "cve_url": cve_entry["cve_url"],
                "patch_attempts": patch_attempts_list
            })
        elif any_failed:
            total_cves_with_partial_failures += 1
            report_data["cves_with_partial_failures"].append({
                "cve_url": cve_entry["cve_url"],
                "patch_attempts": patch_attempts_list
            })
        else:
            total_cves_with_all_successful_patches += 1
            report_data["cves_with_all_successful_patches"].append({
                "cve_url": cve_entry["cve_url"],
                "patch_attempts": patch_attempts_list
            })

        total_failed_patches += cve_failed_patches

    # ✅ Updated summary with unique counts
    report_data["summary"] = {
        "total_cves_tested": total_cves_tested,
        "total_versions_tested": total_versions_tested,
        "total_failed_patches": total_failed_patches,
        "total_unique_versions_tested": len(unique_tested_versions),
        "total_unique_failed_patches": len(unique_failed_versions),
        "cves_with_all_failures": total_cves_with_all_failures,
        "cves_with_partial_failures": total_cves_with_partial_failures,
        "cves_with_all_successful_patches": total_cves_with_all_successful_patches
    }

    if not report_data["cves_with_all_failures"] and not report_data["cves_with_partial_failures"]:
        print("✅ No failed patches found. No report generated.")
        return

    # ✅ Save the reorganized report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = os.path.join("reports", f"merge_conflicts_report_{timestamp}.json")

    with open(report_filename, "w") as file:
        json.dump(report_data, file, indent=4)

    print(f"\n📄 Merge conflict report saved to {report_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract failed patches and generate a categorized summary.")
    parser.add_argument("input_file", help="Path to the full CVE report JSON file")

    args = parser.parse_args()

    extract_merge_conflicts(args.input_file)

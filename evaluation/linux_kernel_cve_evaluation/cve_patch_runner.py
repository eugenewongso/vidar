import json
import os
import argparse
from datetime import datetime
from urllib.parse import urlparse
from kernel_cve_patch_manager import KernelCVEPatchManager
import requests
from bs4 import BeautifulSoup
from collections import OrderedDict
from pathlib import Path


class CVEPatchRunner:
    def __init__(self, cve_input, report_path=None):
        self.cve_input = cve_input
        self.report_path = report_path or os.path.join(os.getcwd(), "reports", "full_cve_report.json")
        self.reports_folder = os.path.dirname(self.report_path)
        os.makedirs(self.reports_folder, exist_ok=True)

    def is_directory_url(self):
        return self.cve_input.endswith("/") or (
            "/tree/" in self.cve_input and not self.cve_input.endswith(".json")
        )


    def get_base_url(self):
        """ Extracts the base URL dynamically from the provided input """
        parsed_url = urlparse(self.cve_input)
        return f"{parsed_url.scheme}://{parsed_url.netloc}"

    def fetch_cve_list(self):
        """ Fetch only valid CVE JSON files from the directory listing, remove duplicates while keeping order, and ensure consistency """
        print(f"Fetching list of CVE JSON files from: {self.cve_input}")

        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(self.cve_input, headers=headers)

        if response.status_code != 200:
            raise Exception(f"Failed to retrieve CVE list: {response.status_code}")


        soup = BeautifulSoup(response.text, "html.parser")

        # Get dynamic base URL
        base_url = self.get_base_url()

        cve_files = []

        for a in soup.find_all("a", href=True):
            filename = a["href"].strip()

            if filename.endswith(".json") and "CVE-" in filename:
                full_url = f"{base_url}/{filename}".replace("/tree/", "/plain/").replace("//", "/").replace(":/", "://")
                if "/plain/cve/" in full_url:
                    cve_files.append(full_url)

        cve_files = list(OrderedDict.fromkeys(cve_files))
        cve_files.sort()

        return cve_files

    def load_existing_report(self):
        """ Load existing report if it exists, otherwise return a new structure """
        if os.path.exists(self.report_path):
            print(f"📂 Loading existing report from {self.report_path}")
            with open(self.report_path, "r") as file:
                return json.load(file)

        print(f"📄 No existing report found. Creating a new report at {self.report_path}")
        return {
            "summary": {
                "total_cves_tested": 0,
                "total_versions_tested": 0,
                "total_failed_patches": 0,
                "total_unique_downstream_versions_tested": 0,
                "total_unique_downstream_failed_patches": 0,
                "cves_with_all_failures": 0,
                "cves_with_partial_failures": 0,
                "cves_with_all_successful_patches": 0,
                "cves_skipped": 0,
            },
            "cves_with_all_failures": [],
            "cves_with_partial_failures": [],
            "cves_with_all_successful_patches": [],
            "cves_skipped": []
        }

    def save_android_format_result(self, report_entry):
        cve_id = os.path.basename(report_entry["cve_url"]).replace(".json", "")
        export_path = Path("reports_android_style") / f"{cve_id}_android_format.json"
        os.makedirs(export_path.parent, exist_ok=True)

        # Initialize required fields
        successes = []
        failures = []
        upstream_commit = ""
        upstream_patch_content = ""

        for attempt in report_entry.get("patch_attempts", []):
            upstream_commit = attempt.get("upstream_commit", upstream_commit)
            upstream_patch_content = attempt.get("upstream_patch_content", upstream_patch_content)

            for result in attempt["patch_results"]:
                entry = {
                    "downstream_version": result["downstream_commit"],
                    "downstream_patch": result["downstream_patch"],
                    "commit_date": result["commit_date"],
                    "result": result["result"],
                    "downstream_patch_content": result.get("downstream_patch_content", "")
                }

                if result["result"] == "failure":
                    entry.update({
                        "error": result.get("error", ""),
                        "total_hunks": result.get("total_hunks", 0),
                        "total_failed_hunks": result.get("total_failed_hunks", 0),
                        "failed_hunks": result.get("failed_hunks", []),
                        "file_conflicts": result.get("file_conflicts", [])
                    })
                    failures.append(entry)
                else:
                    successes.append(entry)

        # Final formatted output
        formatted = {
            "cve_id": cve_id,
            "upstream_commit": upstream_commit,
            "upstream_patch_content": upstream_patch_content,
            "successes": successes,
            "failures": failures
        }

        # Save full android-style report
        with open(export_path, "w") as f:
            json.dump(formatted, f, indent=2)
        print(f"✅ Android-style report saved to {export_path}")

        # Save failures-only report
        if failures:
            fail_export_path = Path("reports_android_style/failures") / f"{cve_id}_failures_only.json"
            os.makedirs(fail_export_path.parent, exist_ok=True)
            with open(fail_export_path, "w") as f:
                json.dump({"cve_id": cve_id, "failures": failures}, f, indent=2)
            print(f"❌ Failure-only report saved to {fail_export_path}")




    def run_all_cves(self, cve_list, start_index=1):
        """ Run all CVEs and generate a consolidated report with summary statistics """

        report_data = self.load_existing_report()  # Load existing report data
        processed_cves = {entry["cve_url"] for entry in (report_data.get("cves_with_all_failures", []) +
                                                          report_data.get("cves_with_partial_failures", []) +
                                                          report_data.get("cves_with_all_successful_patches", []) +
                                                          report_data.get("cves_skipped", []))}

        total_cves = len(cve_list)

        if start_index > total_cves:
            print(f"⚠️ The --start value ({start_index}) is greater than the total CVEs available ({total_cves}).")
            print("🔄 Proceeding with an empty set of CVEs.")
            return  # Instead of exiting, just return gracefully

        # Apply the start index (inclusive)
        cve_list = cve_list[start_index - 1:]

        print(f"🔄 Starting from CVE {start_index} (inclusive), processing {len(cve_list)} CVEs.")

        failures_summary_path = Path("reports_android_style") / "all_failures_summary.json"
        os.makedirs(failures_summary_path.parent, exist_ok=True)

        if failures_summary_path.exists():
            with open(failures_summary_path, "r") as f:
                all_failures_summary = json.load(f)
        else:
            all_failures_summary = []


        combined_android_results = []
        combined_android_failures = []

        
        for index, cve_url in enumerate(cve_list, start=start_index):
            if cve_url in processed_cves:
                print(f"⚠️ Skipping already processed CVE {cve_url}")
                continue

            print(f"\n🔄 Processing CVE {index}/{total_cves}: {cve_url}")

            manager = KernelCVEPatchManager(cve_url)
            report_entry = manager.process_cve()
            self.save_android_format_result(report_entry)
            cve_id = os.path.basename(report_entry["cve_url"]).replace(".json", "")
            successes = []
            failures = []
            upstream_patch_content = ""

            for attempt in report_entry.get("patch_attempts", []):
                if "upstream_patch_content" in attempt:
                    upstream_patch_content = attempt["upstream_patch_content"]

                for result in attempt["patch_results"]:
                    entry = {
                        "downstream_version": result["downstream_commit"],
                        "downstream_patch": result["downstream_patch"],
                        "commit_date": result["commit_date"],
                        "result": result["result"],
                        "downstream_patch_content": result.get("downstream_patch_content", ""),
                    }

                    if result["result"] == "failure":
                        entry.update({
                            "error": result.get("error", ""),
                            "total_hunks": result.get("total_hunks", 0),
                            "total_failed_hunks": result.get("total_failed_hunks", 0),
                            "failed_hunks": result.get("failed_hunks", []),
                            "file_conflicts": result.get("file_conflicts", [])
                        })
                        failures.append(entry)
                    else:
                        successes.append(entry)


            combined_android_results.append({
                "cve_id": cve_id,
                "upstream_patch_content": upstream_patch_content,
                "successes": successes,
                "failures": failures
            })

            if failures:
                combined_android_failures.append({
                    "cve_id": cve_id,
                    "failures": failures
                })

                for fail in failures:
                    all_failures_summary.append({
                        "cve_id": cve_id,
                        "downstream_commit": fail.get("downstream_commit"),
                        "downstream_patch": fail.get("downstream_patch"),
                        "commit_date": fail.get("commit_date"),
                        "error": fail.get("error"),
                        "total_hunks": fail.get("total_hunks"),
                        "total_failed_hunks": fail.get("total_failed_hunks"),
                        "failed_hunks": fail.get("failed_hunks"),
                        "file_conflicts": fail.get("file_conflicts", [])
                    })

                    with open(failures_summary_path, "w") as f:
                        json.dump(all_failures_summary, f, indent=2)
                    print(f"💾 Saved incremental failures to {failures_summary_path}")



            if "skipped" in report_entry:
                report_data["cves_skipped"].append(report_entry)
                report_data["summary"]["cves_skipped"] += 1
                print(f"⚠️ Skipped {cve_url}: {report_entry['error']}")

            else:
                report_data["summary"]["total_cves_tested"] += 1
                total_versions = sum([p["total_versions_tested"] for p in report_entry["patch_attempts"]])
                total_success = sum([p["successful_patches"] for p in report_entry["patch_attempts"]])
                total_failures = sum([p["failed_patches"] for p in report_entry["patch_attempts"]])

                report_data["summary"]["total_versions_tested"] += total_versions
                report_data["summary"]["total_failed_patches"] += total_failures

                unique_versions = set()
                unique_failed_versions = set()

                for attempt in report_entry["patch_attempts"]:
                    for result in attempt["patch_results"]:
                        unique_versions.add(result["downstream_commit"])
                        if result["result"] == "failure":
                            unique_failed_versions.add(result["downstream_commit"])

                report_data["summary"]["total_unique_downstream_versions_tested"] += len(unique_versions)
                report_data["summary"]["total_unique_downstream_failed_patches"] += len(unique_failed_versions)

                # Categorize CVE based on patching outcome
                if total_failures == total_versions:
                    report_data["cves_with_all_failures"].append(report_entry)
                    report_data["summary"]["cves_with_all_failures"] += 1
                elif total_success > 0 and total_failures > 0:
                    report_data["cves_with_partial_failures"].append(report_entry)
                    report_data["summary"]["cves_with_partial_failures"] += 1
                else:
                    report_data["cves_with_all_successful_patches"].append(report_entry)
                    report_data["summary"]["cves_with_all_successful_patches"] += 1

                # Print summary for each CVE
                print(f"\n📊 Summary for {cve_url}:")
                print(f"   ✅ {total_success} successful patches")
                print(f"   ❌ {total_failures} failed patches\n")

            # Save progress after each CVE is processed s
            with open(self.report_path, "w") as file:
                json.dump(report_data, file, indent=4)

        
        combined_path = Path("reports_android_style") / "combined_android_format.json"
        os.makedirs(combined_path.parent, exist_ok=True)

        with open(combined_path, "w") as f:
            json.dump(combined_android_results, f, indent=2)



        print(f"📦 Combined Android-style report saved to {combined_path}")

        print(f"\n📄 Final report saved to {self.report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate Linux Kernel CVE patching.")
    parser.add_argument("cve_input", help="URL to the CVE directory, a single CVE JSON URL, or a local folder path")
    parser.add_argument("--report", help="Path to the existing report JSON file to append to", default=None)
    parser.add_argument("--start", type=int, default=1, help="Start processing from the nth CVE (inclusive)")
    parser.add_argument("--local", action="store_true", help="Interpret input as a local directory of CVE JSONs")


    args = parser.parse_args()

    runner = CVEPatchRunner(args.cve_input, report_path=args.report)



    if args.local:
        # Process files from local directory
        cve_list = []
        for root, _, files in os.walk(args.cve_input):
            for file in sorted(files):
                if file.endswith(".json") and "CVE-" in file:
                    cve_list.append(os.path.join(root, file))
    else:
        if runner.is_directory_url():
            cve_list = runner.fetch_cve_list()
        else:
            cve_list = [runner.cve_input]

    runner.run_all_cves(cve_list, start_index=args.start)

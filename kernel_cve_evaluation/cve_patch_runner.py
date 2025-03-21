import json
import os
import argparse
from datetime import datetime
from urllib.parse import urlparse
from kernel_cve_patch_manager import KernelCVEPatchManager
import requests
from bs4 import BeautifulSoup
from collections import OrderedDict

class CVEPatchRunner:
    def __init__(self, cve_input, report_path=None):
        self.cve_input = cve_input
        self.report_path = report_path or os.path.join(os.getcwd(), "reports", "full_cve_report.json")
        self.reports_folder = os.path.dirname(self.report_path)
        os.makedirs(self.reports_folder, exist_ok=True)

    def is_directory_url(self):
        """ Check if the input is a directory or a direct CVE JSON URL """
        return self.cve_input.endswith("/") or "/tree/" in self.cve_input

    def get_base_url(self):
        """ Extracts the base URL dynamically from the provided input """
        parsed_url = urlparse(self.cve_input)
        return f"{parsed_url.scheme}://{parsed_url.netloc}"

    def fetch_cve_list(self):
        """ Fetch only valid CVE JSON files from the directory listing, remove duplicates while keeping order, and ensure consistency """
        print(f"Fetching list of CVE JSON files from: {self.cve_input}")

        response = requests.get(self.cve_input)
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

        for index, cve_url in enumerate(cve_list, start=start_index):
            if cve_url in processed_cves:
                print(f"⚠️ Skipping already processed CVE {cve_url}")
                continue

            print(f"\n🔄 Processing CVE {index}/{total_cves}: {cve_url}")

            manager = KernelCVEPatchManager(cve_url)
            report_entry = manager.process_cve()

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

            # Save progress after each CVE is processed
            with open(self.report_path, "w") as file:
                json.dump(report_data, file, indent=4)

        print(f"\n📄 Final report saved to {self.report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate Linux Kernel CVE patching.")
    parser.add_argument("cve_input", help="URL to the CVE directory or a single CVE JSON URL")
    parser.add_argument("--report", help="Path to the existing report JSON file to append to", default=None)
    parser.add_argument("--start", type=int, default=1, help="Start processing from the nth CVE (inclusive)")

    args = parser.parse_args()

    runner = CVEPatchRunner(args.cve_input, report_path=args.report)

    cve_list = runner.fetch_cve_list()
    runner.run_all_cves(cve_list, start_index=args.start)

import subprocess
import os
import re
import json
import time
import argparse

class PatchAdopter:
    def __init__(self, kernel_path, patch_dir, report_output_path):
        self.kernel_path = kernel_path
        self.patch_dir = patch_dir
        self.report_output_path = report_output_path
        self.strip_level = 1
        self.patch_command = "patch" if os.name != "darwin" else "gpatch"
        self.patch_results = {"patches": []}

    def apply_patch(self, patch_file: str, patch_url: str, source: str = "Vanir"):
        if not os.path.exists(patch_file):
            print(f"❌ Patch file not found: {patch_file}")
            return {
                "patch_file": os.path.basename(patch_file),
                "patch_url": patch_url,
                "source": source,
                "status": "Rejected: Missing Patch File",
                "rejected_files": [],
                "message_output": "Patch file not found."
            }

        try:
            result = subprocess.run(
                [self.patch_command, "-p", str(self.strip_level), "-i", patch_file, "-f"],
                text=True,
                capture_output=True,
                input=""
            )

            console_output = result.stdout + result.stderr
            print(console_output)

            detailed_status = self.determine_detailed_status(console_output)
            rejected_files = self.extract_failed_files(console_output)
            reject_file_paths = self.get_rej_files()
            formatted_rejected_files = self.map_rejected_files(rejected_files, reject_file_paths)

            overall_status = "Applied Successfully" if not formatted_rejected_files else "Rejected"
            if detailed_status == "Skipped: Files Not Found" and overall_status == "Applied Successfully":
                overall_status = "Applied Successfully"

            return {
                "patch_file": os.path.basename(patch_file),
                "patch_url": patch_url,
                "source": source,
                "status": overall_status,
                "detailed_status": detailed_status,
                "rejected_files": formatted_rejected_files,
                "message_output": console_output
            }

        except subprocess.CalledProcessError as e:
            console_output = (e.stdout or "") + (e.stderr or "")
            print(console_output)
            return {
                "patch_file": os.path.basename(patch_file),
                "patch_url": patch_url,
                "source": source,
                "status": "Rejected",
                "detailed_status": "Rejected: Error Running Patch Command",
                "rejected_files": [],
                "message_output": console_output
            }

    def determine_detailed_status(self, console_output):
        if "Reversed (or previously applied) patch detected" in console_output:
            return "Applied Successfully: Already Applied"
        if "can't find file to patch" in console_output:
            return "Skipped: Files Not Found"
        if "FAILED" in console_output and "hunk" in console_output:
            return "Rejected: Failed Hunks"
        if "offset" in console_output and "FAILED" not in console_output:
            return "Applied Successfully: With Offsets"
        return "Applied Successfully: Clean"

    def extract_failed_files(self, console_output):
        pattern = re.compile(r"patching file (\S+)\nHunk #\d+ FAILED")
        return [match.strip() for match in pattern.findall(console_output)]

    def get_rej_files(self):
        time.sleep(1)
        reject_files = []
        for root, _, files in os.walk(self.kernel_path):
            for file in files:
                if file.endswith(".rej"):
                    reject_files.append(os.path.join(root, file))
        return reject_files

    def map_rejected_files(self, failed_files, reject_files):
        rejected_mappings = []
        for failed_file in failed_files:
            reject_file = os.path.join(self.kernel_path, failed_file + ".rej")
            if reject_file in reject_files:
                rejected_mappings.append({"failed_file": failed_file, "reject_file": reject_file})
            else:
                rejected_mappings.append({"failed_file": failed_file, "reject_file": None})
        return rejected_mappings

    def save_report(self):
        with open(self.report_output_path, "w") as report:
            json.dump(self.patch_results, report, indent=4)
        print(f"📄 Patch report saved to: {self.report_output_path}")

    def generate_summary(self):
        status_counts = {}
        for patch in self.patch_results["patches"]:
            detailed_status = patch.get("detailed_status", "Unknown")
            status_counts[detailed_status] = status_counts.get(detailed_status, 0) + 1
        summary = {
            "total_patches": len(self.patch_results["patches"]),
            "status_counts": status_counts
        }
        print("\n===== Patch Application Summary =====")
        print(f"Total patches processed: {summary['total_patches']}")
        print("Status breakdown:")
        for status, count in status_counts.items():
            print(f"  - {status}: {count}")
        return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["Vanir", "LLM"], default="Vanir", help="Source of the patches")
    args = parser.parse_args()

    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    kernel_path = os.environ.get("KERNEL_PATH", "/data/androidOS14")
    patch_dir = os.path.join(PROJECT_ROOT, "..", "fetch_patch_output", "diff_output") if args.source == "Vanir" else os.path.join(PROJECT_ROOT, "..", "patch_adoption", "generated_patches")
    parsed_report_path = os.path.join(PROJECT_ROOT, "reports", "parsed_report.json") if args.source == "Vanir" else os.path.join(PROJECT_ROOT, "reports", "1_llm_output.json")
    report_output_path = os.path.join(PROJECT_ROOT, "reports", "patch_application_report.json")

    os.makedirs(os.path.dirname(parsed_report_path), exist_ok=True)
    if not os.path.isdir(kernel_path):
        print(f"❌ Error: Kernel directory not found at {kernel_path}")
        exit(1)

    os.chdir(kernel_path)

    with open(parsed_report_path, "r") as f:
        parsed_report = json.load(f)

    patcher = PatchAdopter(kernel_path, patch_dir, report_output_path)
    for patch in parsed_report["patches"]:
        patch_file = os.path.join(patch_dir, patch["patch_file"] if args.source == "Vanir" else os.path.basename(patch["output_path"]))
        print(f"\n🔍 Attempting to apply patch: {patch_file}")
        patch_result = patcher.apply_patch(patch_file, patch["patch_url"], source=args.source)
        patcher.patch_results["patches"].append(patch_result)

    patcher.save_report()
    patcher.generate_summary()

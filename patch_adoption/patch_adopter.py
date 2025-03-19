import subprocess
import os
import re
import json
import time

class PatchAdopter:
    """Handles applying patches and generates a JSON report with rejected files and console output."""

    def __init__(self, kernel_path, patch_dir, report_output_path):
        """
        Initializes the PatchAdopter.

        :param kernel_path: Path to the kernel source where patches will be applied.
        :param patch_dir: Directory where patch files are stored.
        :param report_output_path: Path to save the patch application report.
        """
        self.kernel_path = kernel_path
        self.patch_dir = patch_dir
        self.report_output_path = report_output_path
        self.strip_level = 1
        self.patch_command = "gpatch"  # Use "patch" if on Linux (TODO: hardcode)
        self.patch_results = {"patches": []}

    def apply_patch(self, patch_file: str, patch_url: str):
        """
        Applies a single patch file using GNU patch.

        :param patch_file: Path to the patch file.
        :param patch_url: URL of the patch.
        :return: Patch application details including rejected files and output message.
        """
        if not os.path.exists(patch_file):
            print(f"❌ Patch file not found: {patch_file}")
            return {
                "patch_file": os.path.basename(patch_file),
                "patch_url": patch_url,
                "status": "Rejected",
                "rejected_files": [],
                "message_output": "Patch file not found."
            }

        try:
            # Run the patch command
            result = subprocess.run(
                [self.patch_command, "-p", str(self.strip_level), "-i", patch_file],
                text=True,
                capture_output=True
            )

            console_output = result.stdout + result.stderr
            print(console_output)

            # Extract failed file paths from the output
            rejected_files = self.extract_failed_files(console_output)

            # Find actual .rej files
            reject_file_paths = self.get_rej_files()

            # Map failed files to their corresponding .rej files
            formatted_rejected_files = self.map_rejected_files(rejected_files, reject_file_paths)

            # Determine patch status
            status = "Applied Successfully" if not formatted_rejected_files else "Rejected"

            return {
                "patch_file": os.path.basename(patch_file),
                "patch_url": patch_url,
                "status": status,
                "rejected_files": formatted_rejected_files,
                "message_output": console_output
            }

        except subprocess.CalledProcessError as e:
            console_output = (e.stdout or "") + (e.stderr or "")
            print(console_output)
            return {
                "patch_file": os.path.basename(patch_file),
                "patch_url": patch_url,
                "status": "Rejected",
                "rejected_files": [],
                "message_output": console_output
            }

    def extract_failed_files(self, console_output):
        """
        Extracts failed file paths from the patch output.

        :param console_output: The output of the patch command.
        :return: List of failed file paths.
        """
        failed_files = []
        pattern = re.compile(r"patching file (\S+)\nHunk #\d+ FAILED")

        for match in pattern.findall(console_output):
            failed_files.append(match.strip())

        return failed_files

    def get_rej_files(self):
        """
        Finds all .rej files in the kernel source directory.

        :return: List of .rej file paths.
        """
        time.sleep(1)  # Ensure file system updates
        reject_files = []

        for root, _, files in os.walk(self.kernel_path):
            for file in files:
                if file.endswith(".rej"):
                    reject_files.append(os.path.join(root, file))

        return reject_files

    def map_rejected_files(self, failed_files, reject_files):
        """
        Maps failed files to their corresponding .rej files.

        :param failed_files: List of failed source files.
        :param reject_files: List of reject (.rej) files.
        :return: List of dictionaries containing failed files and corresponding reject files.
        """
        rejected_mappings = []

        for failed_file in failed_files:
            # Convert source file name to .rej format
            reject_file = os.path.join(self.kernel_path, failed_file + ".rej")
            if reject_file in reject_files:
                rejected_mappings.append({
                    "failed_file": failed_file,
                    "reject_file": reject_file
                })
            else:
                rejected_mappings.append({
                    "failed_file": failed_file,
                    "reject_file": None
                })

        return rejected_mappings

    def save_report(self):
        """
        Saves the patch application report as a JSON file.
        """
        with open(self.report_output_path, "w") as report:
            json.dump(self.patch_results, report, indent=4)

        print(f"📄 Patch report saved to: {self.report_output_path}")


# === Main Patch Application Logic ===

if __name__ == "__main__":
    # Paths
    kernel_path = input("Input Repository path to apply diff files here: ")
    patch_dir = "reports/parsed_report.json" 
    parsed_report_path = "reports/parsed_report.json"
    report_output_path = "reports/patch_application_report.json"

    # Ensure the Xiaomi Kernel directory exists
    if not os.path.isdir(kernel_path):
        print(f"❌ Error: Xiaomi Kernel directory not found at {kernel_path}")
        exit(1)

    # Change to kernel directory before applying patches
    os.chdir(kernel_path)

    # Load parsed report JSON
    with open(parsed_report_path, "r") as f:
        parsed_report = json.load(f)

    # Initialize patch handler
    patcher = PatchAdopter(kernel_path, patch_dir, report_output_path)

    # Iterate through patches
    for patch in parsed_report["patches"]:
        patch_file = os.path.join(patch_dir, patch["patch_file"])

        print(f"\n🔍 Attempting to apply patch: {patch_file}")
        patch_result = patcher.apply_patch(patch_file, patch["patch_url"])

        patcher.patch_results["patches"].append(patch_result)

    # Save the final report
    patcher.save_report()

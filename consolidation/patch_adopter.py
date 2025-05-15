import subprocess
import os
import re
import json
import time

class PatchAdopter:
    """
    Handles the application of security patches to kernel source code and generates detailed reports.
    
    This class is responsible for:
    1. Applying patches using the GNU patch utility
    2. Analyzing patch application results
    3. Identifying failed patches and rejected hunks
    4. Generating comprehensive reports on patch application status
    5. Creating summaries of patch application results
    
    The class handles various patch application scenarios including:
    - Successfully applied patches (clean or with offsets)
    - Already applied patches
    - Rejected patches (with failed hunks)
    - Missing files
    - Error conditions
    
    Usage:
        patcher = PatchAdopter(kernel_path, patch_dir, report_path)
        result = patcher.apply_patch(patch_file, patch_url)
        patcher.patch_results["patches"].append(result)
        patcher.save_report()
        patcher.generate_summary()
    """

    def __init__(self, kernel_path, patch_dir, report_output_path):
        """
        Initializes the PatchAdopter with paths and default settings.

        :param kernel_path: Path to the kernel source where patches will be applied.
        :param patch_dir: Directory where patch files are stored.
        :param report_output_path: Path to save the patch application report.
        """
        self.kernel_path = kernel_path
        self.patch_dir = patch_dir
        self.report_output_path = report_output_path
        self.strip_level = 1  # Default strip level for patch command (-p1)
        
        # Use 'patch' on Linux and 'gpatch' on macOS
        self.patch_command = "patch" if os.name != "darwin" else "gpatch"
        self.patch_results = {"patches": []}  # Initialize empty results container

    def apply_patch(self, patch_file: str, patch_url: str):
        """
        Applies a single patch file using GNU patch and analyzes the results.
        
        This function:
        1. Verifies the patch file exists
        2. Runs the patch command with appropriate flags
        3. Captures and analyzes the command output
        4. Identifies failed files and rejection files
        5. Determines detailed status based on output analysis
        
        :param patch_file: Path to the patch file.
        :param patch_url: URL of the patch (for reporting purposes).
        :return: Dictionary with comprehensive patch application details.
        """
        # Check if patch file exists
        if not os.path.exists(patch_file):
            print(f"❌ Patch file not found: {patch_file}")
            return {
                "patch_file": os.path.basename(patch_file),
                "patch_url": patch_url,
                "status": "Rejected: Missing Patch File",
                "rejected_files": [],
                "message_output": "Patch file not found."
            }

        try:
            # Run the patch command with -f flag to avoid interactive prompts
            # capture_output=True captures stdout and stderr
            # text=True ensures the output is returned as strings, not bytes
            result = subprocess.run(
                [self.patch_command, "-p", str(self.strip_level), "-i", patch_file, "-f"],
                text=True,
                capture_output=True,
                input=""  # Provide empty input to avoid interactive prompts
            )

            # Combine stdout and stderr for analysis
            console_output = result.stdout + result.stderr
            print(console_output)

            # Analyze output to determine detailed status
            detailed_status = self.determine_detailed_status(console_output)

            # Extract paths of files that failed to patch
            rejected_files = self.extract_failed_files(console_output)

            # Find all .rej files in the repository
            reject_file_paths = self.get_rej_files()

            # Map failed files to their corresponding .rej files
            formatted_rejected_files = self.map_rejected_files(rejected_files, reject_file_paths)

            # Overall status (for backward compatibility)
            overall_status = "Applied Successfully" if not formatted_rejected_files else "Rejected"
            
            # Edge case: Files not found but no rejections should still be successful
            if detailed_status == "Skipped: Files Not Found" and overall_status == "Applied Successfully":
                overall_status = "Applied Successfully"

            # Return comprehensive result object
            return {
                "patch_file": os.path.basename(patch_file),
                "patch_url": patch_url,
                "status": overall_status,
                "detailed_status": detailed_status,
                "rejected_files": formatted_rejected_files,
                "message_output": console_output
            }

        except subprocess.CalledProcessError as e:
            # Handle cases where subprocess.run raises an exception
            console_output = (e.stdout or "") + (e.stderr or "")
            print(console_output)
            return {
                "patch_file": os.path.basename(patch_file),
                "patch_url": patch_url,
                "status": "Rejected",
                "detailed_status": "Rejected: Error Running Patch Command",
                "rejected_files": [],
                "message_output": console_output
            }

    def determine_detailed_status(self, console_output):
        """
        Determines the detailed status of the patch application by analyzing console output.
        
        This function categorizes patch application results into one of several statuses:
        - Already Applied: Patch was previously applied to the source
        - Files Not Found: Target files don't exist in the repository
        - Failed Hunks: Some parts of the patch couldn't be applied
        - Applied With Offsets: Patch applied but with line number adjustments
        - Clean Application: Perfect patch application without issues
        
        :param console_output: The output of the patch command.
        :return: Detailed status string.
        """
        # Check if patch was already applied
        if "Reversed (or previously applied) patch detected" in console_output:
            return "Applied Successfully: Already Applied"
        
        # Check if files weren't found
        if "can't find file to patch" in console_output:
            return "Skipped: Files Not Found"
        
        # Check if any hunks failed
        if "FAILED" in console_output and "hunk" in console_output:
            return "Rejected: Failed Hunks"
        
        # Check if there were offsets but all hunks were applied
        if "offset" in console_output and "FAILED" not in console_output:
            return "Applied Successfully: With Offsets"
        
        # Default to clean application if none of the above
        return "Applied Successfully: Clean"

    def extract_failed_files(self, console_output):
        """
        Extracts failed file paths from the patch output using regex pattern matching.
        
        Analyzes the patch command output to identify files where one or more hunks failed.
        The pattern looks for lines like "patching file path/to/file" followed by 
        "Hunk #X FAILED".
        
        :param console_output: The output of the patch command.
        :return: List of failed file paths.
        """
        failed_files = []
        # Regex pattern to find failed files
        # Looks for "patching file X" followed by "Hunk #Y FAILED"
        pattern = re.compile(r"patching file (\S+)\nHunk #\d+ FAILED")

        for match in pattern.findall(console_output):
            failed_files.append(match.strip())

        return failed_files

    def get_rej_files(self):
        """
        Finds all .rej files in the kernel source directory.
        
        The patch utility creates .rej files when it cannot apply a hunk.
        These files contain the rejected portions of the patch and are useful
        for manual resolution of patch conflicts.
        
        The method includes a short delay to ensure the file system has completed
        any pending operations before scanning.
        
        :return: List of .rej file paths.
        """
        time.sleep(1)  # Ensure file system updates are complete
        reject_files = []

        # Walk the entire kernel directory to find .rej files
        for root, _, files in os.walk(self.kernel_path):
            for file in files:
                if file.endswith(".rej"):
                    reject_files.append(os.path.join(root, file))

        return reject_files

    def map_rejected_files(self, failed_files, reject_files):
        """
        Maps failed files to their corresponding .rej files.
        
        When the patch utility cannot apply a hunk, it creates a .rej file with the
        same name as the original file plus .rej extension. This method creates a mapping
        between the file that failed to patch and its corresponding rejection file.
        
        This mapping is crucial for developers who need to manually resolve patch conflicts.
        
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
                # Failed file but no .rej file found (unusual case)
                rejected_mappings.append({
                    "failed_file": failed_file,
                    "reject_file": None
                })

        return rejected_mappings

    def save_report(self):
        """
        Saves the patch application report as a JSON file.
        
        The report contains detailed information about each patch application attempt,
        including status, rejected files, and console output. This information is valuable
        for both automated processing and manual review.
        """
        with open(self.report_output_path, "w") as report:
            json.dump(self.patch_results, report, indent=4)

        print(f"📄 Patch report saved to: {self.report_output_path}")

    def generate_summary(self):
        """
        Generates a summary of patch application results by detailed status.
        
        Creates a high-level overview of the patch application process, counting
        the number of patches in each status category. This provides a quick
        assessment of the overall patch application success rate.
        
        :return: Dictionary with counts of each status type.
        """
        status_counts = {}
        
        # Count patches by detailed status
        for patch in self.patch_results["patches"]:
            detailed_status = patch.get("detailed_status", "Unknown")
            status_counts[detailed_status] = status_counts.get(detailed_status, 0) + 1
        
        summary = {
            "total_patches": len(self.patch_results["patches"]),
            "status_counts": status_counts
        }
        
        # Print summary to console
        print("\n===== Patch Application Summary =====")
        print(f"Total patches processed: {summary['total_patches']}")
        print("Status breakdown:")
        for status, count in status_counts.items():
            print(f"  - {status}: {count}")
        
        return summary


# === Main Patch Application Logic ===

if __name__ == "__main__":
    """
    Main entry point for patch application.
    
    When executed directly, this script:
    1. Sets up paths based on the project structure
    2. Loads the parsed report containing patch information
    3. Initializes the PatchAdopter
    4. Processes each patch in the report
    5. Saves the detailed report and summary
    
    Environment Variables:
    - KERNEL_PATH: Override the default kernel source path
    """
    # Get the project root directory
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths relative to project root
    kernel_path = os.environ.get("KERNEL_PATH", "/data/androidOS14")  # Default but overridable
    patch_dir = os.path.join(PROJECT_ROOT, "..", "fetch_patch_output", "diff_output")
    parsed_report_path = os.path.join(PROJECT_ROOT, "reports", "parsed_report.json")
    report_output_path = os.path.join(PROJECT_ROOT, "reports", "patch_application_report.json")
    
    # Ensure the reports directory exists
    os.makedirs(os.path.dirname(parsed_report_path), exist_ok=True)
    
    # Ensure the Kernel directory exists
    if not os.path.isdir(kernel_path):
        print(f"❌ Error: Kernel directory not found at {kernel_path}")
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
    
    # Generate and display summary
    patcher.generate_summary()
import subprocess
import os
import re
import json
import time

class PatchAdopter:
    """Handles applying patches from a single Vanir report and generates a JSON report with rejected files."""

    def __init__(self, kernel_path, report_output_path):
        """
        Initializes the PatchAdopter.

        :param kernel_path: Path to the kernel source where patches will be applied.
        :param report_output_path: Path to save the patch application report.
        """

        if os.path.isdir(report_output_path):
            timestamp = time.strftime("%Y%m%d_%H%M%S")  
            report_output_path = os.path.join(report_output_path, f"patch_application_report_{timestamp}.json")

        self.kernel_path = kernel_path
        self.report_output_path = report_output_path
        self.strip_level = 1
        self.patch_command = "gpatch"  # Use "patch" if on Linux (TODO: implement OS-agnostic approach)
        self.patch_results = {"patches": []}
        self.console_output = ""
        self.infile_merge_conflict = ""

    def get_console_output(self):
        return self.console_output

    def apply_patch(self, patch_file: str, patch_url: str):
        """
        Applies a single patch file using GNU patch.

        :param patch_file: Path to the patch file.
        :param patch_url: URL of the patch.
        :return: Patch application details including rejected files and output message.
        """
        if not os.path.exists(patch_file):
            print(f"❌ Patch file not found: {patch_file}")
            self.console_output = f"Patch file not found: {patch_file}"
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
                [self.patch_command, "-p", str(self.strip_level), "-i", patch_file, "--ignore-whitespace"],
                text=True,
                check=True,
                capture_output=True
            )

            # Store the output
            self.console_output = result.stdout + result.stderr

            if "Reversed (or previously applied) patch detected!" in self.console_output:
                return {
                    "patch_file": os.path.basename(patch_file),
                    "patch_url": patch_url,
                    "status": "Already Applied",  # New status
                    "rejected_files": [],  # No rejected files since it's already applied
                    "message_output": self.console_output
    }
            
            # Extract failed file paths from the output
            rejected_files = self.extract_failed_files(self.console_output)

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
                "message_output": self.console_output
            }

        except subprocess.CalledProcessError as e:
            # Handle errors from the patch command
            self.console_output = (e.stdout or "") + (e.stderr or "")
            print(self.console_output)
            
            # Even for errors, we should try to extract rejected files
            rejected_files = self.extract_failed_files(self.console_output)
            reject_file_paths = self.get_rej_files()
            formatted_rejected_files = self.map_rejected_files(rejected_files, reject_file_paths)

            return {
                "patch_file": os.path.basename(patch_file),
                "patch_url": patch_url,
                "status": "Rejected",
                "rejected_files": formatted_rejected_files,
                "message_output": self.console_output
            }

    def extract_failed_files(self, console_output):
        """
        Extracts failed file paths from the patch output.

        :param console_output: The output of the patch command.
        :return: List of failed file paths.
        """
        failed_files = []
        
        # Pattern for "patching file X" followed by any failure indicator
        patching_file_pattern = re.compile(r"patching file (\S+)")
        
        # Extract all files being patched
        for line in console_output.split('\n'):
            if "patching file" in line:
                match = patching_file_pattern.search(line)
                if match:
                    file_path = match.group(1).strip()
                    
                    # Only add files that have rejection indicators
                    if (f"saving rejects to file {file_path}.rej" in console_output or
                        f"hunks FAILED -- saving rejects to file {file_path}.rej" in console_output or
                        f"hunks ignored -- saving rejects to file {file_path}.rej" in console_output):
                        failed_files.append(file_path)
        
        return failed_files

    def get_rej_files(self, message_output=None):
        """
        Extracts all .rej files from message output and finds them in the file system.
        
        :param message_output: Optional message output to parse. If None, uses self.console_output
        :return: List of .rej file paths.
        """
        # Use provided message_output or fall back to self.console_output
        output_to_parse = message_output or self.console_output
        
        # Pattern to match both "saving rejects to file X.rej" and "hunks ignored -- saving rejects to file X.rej"
        rej_pattern = re.compile(r"(?:saving rejects to file|ignored -- saving rejects to file) (.+\.rej)") 
        matches = rej_pattern.findall(output_to_parse)
        
        rej_files = []
        for rej_file in matches:
            # Ensure the path is relative to the kernel path
            full_path = os.path.join(self.kernel_path, rej_file.strip())
            if os.path.exists(full_path):
                rej_files.append(full_path)
            else:
                print(f"Warning: Rejection file not found: {full_path}")
        
        return rej_files

    def combine_rejected_hunks(self, output_file="combined.rej", patch_results=None):
        """
        Combines all .rej files related to applied patches into one file,
        maintaining the original diff format. Can process either from the 
        last applied patch or from a list of patch results.
        
        :param output_file: Name of the combined output file.
        :param patch_results: Optional patch results to extract rejection files from.
        :return: The full path to the combined .rej file, or None if no files were found.
        """
        all_rej_files = []
        
        # If patch_results is provided, extract from message_output of all patches
        if patch_results:
            for patch in patch_results:
                if "message_output" in patch and patch["message_output"]:
                    patch_rej_files = self.get_rej_files(patch["message_output"])
                    all_rej_files.extend(patch_rej_files)
        else:
            # Otherwise use the standard approach for the last applied patch
            all_rej_files = self.get_rej_files()
        
        # Remove duplicates while preserving order
        unique_rej_files = []
        for file in all_rej_files:
            if file not in unique_rej_files:
                unique_rej_files.append(file)
        
        if not unique_rej_files:
            print("No .rej files found.")
            return None
        
        output_path = os.path.abspath(output_file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w") as combined_file:
            for rej_file in sorted(unique_rej_files):
                try:
                    with open(rej_file, "r") as file:
                        combined_file.write(f"### Rejected hunks from {os.path.basename(rej_file)} ###\n")
                        combined_file.write(file.read().strip() + "\n\n")
                except Exception as e:
                    print(f"Error reading {rej_file}: {str(e)}")
        
        print(f"Combined {len(unique_rej_files)} .rej files into {output_path}:")
        for rej_file in unique_rej_files:
            print(f" - {rej_file}")
        
        return output_path
    
    def map_rejected_files(self, failed_files, reject_files):
        """
        Maps failed files to their corresponding .rej files.

        :param failed_files: List of failed source files.
        :param reject_files: List of reject (.rej) files.
        :return: List of dictionaries containing failed files and corresponding reject files.
        """
        rejected_mappings = []

        for failed_file in failed_files:
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
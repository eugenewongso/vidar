import subprocess
import os
import re
import json
import time
import shutil

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
        self.patch_command = "gpatch"  # Use "patch" if on Linux (TODO: inplement OS-agnostic approach)
        self.patch_results = {"patches": []}
        self.console_output = """""" # TODO: determine whether we need console output or not since we're fetching from patch application report already
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

        console_output = ""

        try:
            # Run the patch command
            result = subprocess.run(
                [self.patch_command, "-p", str(self.strip_level), "-i", patch_file, "--ignore-whitespace"],
                text=True,
                check=True,
                capture_output=True,
                cwd=self.kernel_path #tester caro
            )

            self.console_output = result.stdout + result.stderr
            # print(console_output) # TODO: remove in prod.

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

    def get_rej_files(self, message_output=None):
        """
        Extracts all .rej files from message output and finds them in the file system.
        
        :param message_output: Optional message output to parse. If None, uses self.console_output
        :return: List of .rej file paths.
        """
        # Use provided message_output or fall back to self.console_output
        output_to_parse = message_output or self.console_output
        
        # added caro
        output_rej_dir = os.path.join(self.kernel_path, "outputs/rejected_patches")
        os.makedirs(output_rej_dir, exist_ok=True)  # Ensure the output directory exists

        # Pattern to match both "saving rejects to file X.rej" and "hunks ignored -- saving rejects to file X.rej"
        rej_pattern = re.compile(r"(?:saving rejects to file|ignored -- saving rejects to file) (.+\.rej)") # TODO: testing
        # TODO: incorporate logic so that rej files do not overwrite when same rej file name
        matches = rej_pattern.findall(output_to_parse)
        
        rej_files = []

        # Added caro
        if not matches:
            print("⚠️ No .rej files found in output.")
            return rej_files  # ✅ Returns an empty list if no `.rej` files are found


        for rej_file in matches:
            full_path = os.path.join(self.kernel_path, rej_file.strip())

            if os.path.exists(full_path):
                # Prevent overwriting by using a numbered suffix (_1, _2, etc.)
                # base, ext = os.path.splitext(full_path)
                # counter = 1
                # new_path = f"{base}_{counter}{ext}"

                # while os.path.exists(new_path):  # Find next available number
                #     counter += 1
                #     new_path = f"{base}_{counter}{ext}"

                # shutil.move(full_path, new_path)  # ✅ Rename instead of copying
                # rej_files.append(new_path)

                base_name = os.path.basename(full_path)
                dest_path = os.path.join(output_rej_dir, base_name)

                counter = 1
                while os.path.exists(dest_path):
                    dest_path = os.path.join(output_rej_dir, f"{os.path.splitext(base_name)[0]}_{counter}.rej")
                    counter += 1

                shutil.copy(full_path, dest_path)  # ✅ Copy instead of renaming
                rej_files.append(dest_path)

                
            else:
                print(f"⚠️ Warning: Expected .rej file not found: {full_path}")


        # try 3
        # for rej_file in matches:
        #     # Ensure the path is relative to the kernel path
        #     full_path = os.path.join(self.kernel_path, rej_file.strip())
            
        #     if os.path.exists(full_path):
        #         # Prevent overwriting by copying the file immediately
        #         timestamp = time.strftime("%Y%m%d_%H%M%S")  # Adds timestamp to ensure uniqueness
        #         new_path = f"{full_path}_{timestamp}.rej"

        #         shutil.copy(full_path, new_path)  # ✅ Copy instead of renaming
        #         rej_files.append(new_path)
        #     else:
        #         print(f"⚠️ Warning: Expected .rej file not found: {full_path}")


            # try 2
            # if os.path.exists(full_path):
            #     # If a duplicate exists, rename the old file
            #     base, ext = os.path.splitext(full_path)
            #     counter = 1
            #     new_path = full_path

            #     while os.path.exists(new_path):
            #         new_path = f"{base}_{counter}{ext}"
            #         counter += 1

            #     # Rename the original `.rej` file
            #     shutil.move(full_path, new_path)
            #     rej_files.append(new_path)
            # else:
            #     print(f"⚠️ Warning: Expected .rej file not found: {full_path}")

            # try 1
            # base, ext = os.path.splitext(full_path)
            # counter = 1
            # while os.path.exists(full_path):
            #     full_path = f"{base}_{counter}{ext}"
            #     counter += 1

            # rej_files.append(full_path)
            
            # original
            # if os.path.exists(full_path):
            #     rej_files.append(full_path)
            # else:
            #     print(f"Warning: Rejection file not found: {full_path}")
        
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
        
    # TODO: implement in main, make handler in main, iterate through each, then find inline merge conflicts of each 
    def generate_infile_merge_conflict(self, combined_rej_file):
        """
        Generates an in-file merge conflict message by capturing it directly from the patch output,
        ensuring surrounding context is included.

        :param combined_rej_file: Path to the combined .rej file.
        """
        if not os.path.exists(combined_rej_file):
            print(f"No combined .rej file found at {combined_rej_file}. Skipping merge attempt.")
            return

        try:
            result = subprocess.run(
                [self.patch_command, "--merge", "-p0", "--output=-", "-i", combined_rej_file],
                text=True,
                capture_output=True
            )

            full_output = result.stdout.splitlines()  # Full file content with conflict markers

            conflict_pattern = re.compile(r"(<<<<<<<.*?=======.*?>>>>>>>)", re.DOTALL)
            conflicts = conflict_pattern.finditer(result.stdout)

            if not conflicts:
                print("No merge conflicts detected.")
                return

            formatted_conflicts = []
            lines = result.stdout.split("\n")  # Convert to list of lines for easier indexing

            for match in conflicts:
                conflict_start = result.stdout[:match.start()].count("\n")  # Get line number estimate
                conflict_end = result.stdout[:match.end()].count("\n")

                # Extract 10 lines before and after
                start_context = max(0, conflict_start - 10)
                end_context = min(len(lines), conflict_end + 10)

                context_before = "\n".join(lines[start_context:conflict_start])
                conflict_text = "\n".join(lines[conflict_start:conflict_end])
                context_after = "\n".join(lines[conflict_end:end_context])

                # Modify conflict markers for clarity
                conflict_text = conflict_text.replace(
                    "<<<<<<<", f"<<<<<<< source code (line {conflict_start})"
                )
                conflict_text = conflict_text.replace(
                    ">>>>>>>", ">>>>>>> rejected patch from combined.rej"
                )

                # Format the final output with context
                formatted_conflict = (
                    f"{context_before}\n"
                    f"{conflict_text}\n"
                    f"{context_after}"
                )
                formatted_conflicts.append(formatted_conflict)

            self.infile_merge_conflict = "\n\n".join(formatted_conflicts).strip()

            print("\nExtracted In-File Merge Conflict Content with Context:")
            print(self.infile_merge_conflict)

        except subprocess.CalledProcessError as e:
            self.infile_merge_conflict = (e.stdout or "") + (e.stderr or "")
            print("\nFailed to process in-file merge conflict. Remaining conflicts:")
            print(self.infile_merge_conflict)

            """
            Generates an in-file merge conflict message by attempting to apply the combined .rej file using 
            `patch --merge --output=-`. Extracts full conflict blocks including markers.

            :param combined_rej_file: Path to the combined .rej file.
            """
        
            if not os.path.exists(combined_rej_file):
                print(f"No combined .rej file found at {combined_rej_file}. Skipping merge attempt.")
                return

            try:
                # Apply combined .rej using --merge but capture conflict output
                result = subprocess.run(
                    [self.patch_command, "--merge", "-p0", "--output=-", "-i", combined_rej_file],
                    # [self.patch_command, "--merge", "-p0", "-i", combined_rej_file],

                    text=True,
                    capture_output=True
                )

                # Store full output first
                self.infile_merge_conflict = result.stdout

                # Extract full conflicts, including markers <<<<<<< and >>>>>>>
                conflict_pattern = re.compile(r"(<<<<<<<.*?=======.*?>>>>>>>)", re.DOTALL)
                conflicts = conflict_pattern.findall(self.infile_merge_conflict)

                # If there are conflicts, format them explicitly
                if conflicts:
                    formatted_conflicts = []
                    for conflict in conflicts:
                        # Label the original code
                        conflict_text = conflict_text.replace(
                            "<<<<<<<", f"<<<<<<< source code (line {conflict_start})"
                        )
                        conflict_text = conflict_text.replace(
                            ">>>>>>>", ">>>>>>> rejected patch from combined.rej"
                        )
                        formatted_conflicts.append(conflict)

                    self.infile_merge_conflict = "\n\n".join(formatted_conflicts).strip()

                print("\nExtracted In-File Merge Conflict Content:")
                print(self.infile_merge_conflict)

            except subprocess.CalledProcessError as e:
                self.infile_merge_conflict = (e.stdout or "") + (e.stderr or "")
                print("\nFailed to process in-file merge conflict. Remaining conflicts:")
                print(self.infile_merge_conflict)

    def get_infile_merge_conflict(self) -> str:
        """
        Retrieves the in-file merge conflict content as a string.

        :return: The merge conflict output as a string.
        """
        return self.infile_merge_conflict


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
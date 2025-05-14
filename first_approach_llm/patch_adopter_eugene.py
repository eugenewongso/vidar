import time  # Import time module for timestamp
import subprocess
import os
import re
import json

class PatchAdopter:
    """Handles applying a single patch file using GNU patch and stores console output."""

    def __init__(self, strip_level=1):
        """
        Initializes the PatchAdopter.

        :param strip_level: Number of leading path components to strip.
        """
        self.strip_level = strip_level
        self.patch_command = "patch"
        self.console_output = ""
        self.infile_merge_conflict = ""
        self.patch_results = {"patches": []}

    def apply_patch(self, patch_file: str):
        """
        Applies a single patch file using GNU patch (without --merge).
        This applies hunks that work and generates .rej files for failed hunks.

        :param patch_file: Path to the patch file.
        :return: Console output as a string.
        """
        if not os.path.exists(patch_file):
            self.console_output = f"Patch file not found: {patch_file}"
            print(self.console_output)
            return self.console_output

        try:
            # Apply the patch normally (without --merge)
            result = subprocess.run(
                [self.patch_command, "-p", str(self.strip_level), "-i", patch_file, "--ignore-whitespace"],
                check=True,
                text=True,
                capture_output=True
            )

            self.console_output = result.stdout + result.stderr
            print(self.console_output)
            return self.console_output

        except subprocess.CalledProcessError as e:
            self.console_output = (e.stdout or "") + (e.stderr or "")
            print(self.console_output)
            return self.console_output

    def get_rej_files(self):
        """
        Extracts all .rej files from console output and finds them in the file system.

        :return: List of .rej file paths.
        """
        rej_pattern = re.compile(r"saving rejects to file (.+\.rej)")
        matches = rej_pattern.findall(self.console_output)

        rej_files = []
        for rej_file in matches:
            if os.path.exists(rej_file):
                rej_files.append(rej_file)

        return rej_files

    def combine_rejected_hunks(self, output_file="combined.rej"):
        """
        Combines all .rej files related to the last applied patch into one file,
        maintaining the original diff format.

        :param output_file: Name of the combined output file.
        :return: The full path to the combined .rej file, or None if no files were found.
        """
        rej_files = self.get_rej_files()

        if not rej_files:
            print("No .rej files found.")
            return None

        output_path = os.path.abspath(output_file)

        with open(output_path, "w") as combined_file:
            for rej_file in sorted(rej_files):
                with open(rej_file, "r") as file:
                    combined_file.write(file.read().strip() + "\n\n")

        print(f"Combined {len(rej_files)} .rej files into {output_path}:")
        for rej_file in rej_files:
            print(f" - {rej_file}")

        return output_path

    def generate_infile_merge_conflict(self, combined_rej_file):
        """
        Extracts and processes all in-file merge conflicts from the given .rej file.
        Each conflict is stored as a structured dictionary in a list and saved in JSON format.

        :param combined_rej_file: Path to the combined .rej file.
        :return: Path to the saved JSON file.
        """
        if not os.path.exists(combined_rej_file):
            print(f"⚠️ No combined .rej file found at {combined_rej_file}. Skipping merge attempt.")
            return None

        try:
            result = subprocess.run(
                [self.patch_command, "--merge", "-p0", "--output=-", "-i", combined_rej_file],
                text=True,
                capture_output=True
            )

            lines = result.stdout.split("\n")  # Convert output to list of lines

            # Improved regex to detect conflict markers
            conflict_pattern = re.compile(r"<<<<<<<.*?=======.*?>>>>>>>", re.DOTALL)
            conflicts = list(conflict_pattern.finditer(result.stdout))

            if not conflicts:
                print("✅ No merge conflicts detected.")
                return None

            formatted_conflicts = []  # List to store structured conflicts

            for i, match in enumerate(conflicts):
                conflict_start = result.stdout[:match.start()].count("\n")  # Approximate line number
                conflict_end = result.stdout[:match.end()].count("\n")

                # Extract 10 lines before and after conflict
                start_context = max(0, conflict_start - 10)
                end_context = min(len(lines), conflict_end + 10)

                before_context = "\n".join(lines[start_context:conflict_start])
                conflict_text = "\n".join(lines[conflict_start:conflict_end])
                after_context = "\n".join(lines[conflict_end:end_context])

                # Remove unwanted formatting (convert tabs to spaces)
                before_context = before_context.replace("\t", "  ")
                conflict_text = conflict_text.replace("\t", "  ")
                after_context = after_context.replace("\t", "  ")

                # Modify conflict markers for clarity
                conflict_text = conflict_text.replace(
                    "<<<<<<<", f"<<<<<<< source code (line {conflict_start})"
                ).replace(
                    ">>>>>>>", ">>>>>>> rejected patch from combined.rej"
                )

                # Store conflict as a structured dictionary
                conflict_entry = {
                    "conflict_id": i + 1,
                    "line_start": conflict_start,
                    "before": before_context.strip(),
                    "conflict": conflict_text.strip(),
                    "after": after_context.strip(),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")  # Add timestamp
                }

                formatted_conflicts.append(conflict_entry)  # Store in list

            return formatted_conflicts # Return conflicts list

        except subprocess.CalledProcessError as e:
            print("\n❌ Failed to process in-file merge conflict. Error:")
            print(e.stderr or e.stdout)
            return None
        
    def get_infile_merge_conflict(self) -> str:
        """
        Retrieves the in-file merge conflict content as a string.

        :return: The merge conflict output as a string.
        """
        return self.infile_merge_conflict
    
    def save_report(self):
        """
        Saves the patch application report as a JSON file.
        """
        with open(self.report_output_path, "w") as report:
            json.dump(self.patch_results, report, indent=4)

        print(f"📄 Patch report saved to: {self.report_output_path}")

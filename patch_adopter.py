import time  # Import time module for timestamp
import subprocess
import os
import re
import json
import shutil

class PatchAdopter:
    """Handles applying a single patch file using GNU patch and stores console output."""

    def __init__(self, kernel_path, strip_level=1):
        """
        Initializes the PatchAdopter.

        :param strip_level: Number of leading path components to strip.
        """
        self.kernel_path = kernel_path
        self.strip_level = strip_level
        self.patch_command = "gpatch"
        self.console_output = ""
        self.infile_merge_conflict = ""

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

        # ADDED CARO
        output_rej_dir = os.path.join(self.kernel_path, "outputs/failed_patches")
        os.makedirs(output_rej_dir, exist_ok=True)  # Ensure the output directory exists


        rej_pattern = re.compile(r"saving rejects to file (.+\.rej)")
        matches = rej_pattern.findall(self.console_output)

        rej_files = []

        # ADDED CARO
        if not matches:
            print("⚠️ No .rej files found in output.")
            return rej_files 


        for rej_file in matches:
            # ADDED & CHANGED CARO
            full_path = os.path.join(self.kernel_path, rej_file.strip())

            if os.path.exists(full_path):
                base_name = os.path.basename(full_path)
                dest_path = os.path.join(output_rej_dir, base_name)

                # Prevent overwriting by adding numbered suffixes (_1, _2, etc.)
                counter = 1
                while os.path.exists(dest_path):
                    dest_path = os.path.join(output_rej_dir, f"{os.path.splitext(base_name)[0]}_{counter}.rej")
                    counter += 1

                shutil.copy(full_path, dest_path)  # ✅ Copy instead of renaming
                rej_files.append(dest_path)
            else:
                print(f"⚠️ Warning: Expected .rej file not found: {full_path}")

        return rej_files

    def combine_rejected_hunks(self, output_file="combined.rej"):
        """
        Combines all .rej files related to the last applied patch into one file,
        maintaining the original diff format.

        :param output_file: Name of the combined output file.
        :return: The full path to the combined .rej file, or None if no files were found.
        """
        
        # CHANGED CARO
        output_rej_dir = os.path.join(self.kernel_path, "outputs/failed_patches")
        os.makedirs(output_rej_dir, exist_ok=True)
        all_rej_files = [os.path.join(output_rej_dir, f) for f in os.listdir(output_rej_dir) if f.endswith(".rej")]


        if not all_rej_files:
            print("No .rej files found.")
            return None

        # CHANGED CARO
        output_dir = os.path.join(self.kernel_path, "outputs/combined_rejected_hunks")
        os.makedirs(output_dir, exist_ok=True)

        if not output_file or os.path.dirname(output_file) == "":
            output_file = os.path.join(output_dir, "combined.rej") 

        with open(output_file, "w") as combined_file:
            for rej_file in sorted(all_rej_files):
                try:
                    with open(rej_file, "r") as file:
                        combined_file.write(f"### Rejected hunks from {os.path.basename(rej_file)} ###\n")
                        combined_file.write(file.read().strip() + "\n\n")
                except Exception as e:
                    print(f"⚠️ Error reading {rej_file}: {str(e)}")

        print(f"📁 Combined rejected hunks saved to: {output_file}")
        return output_file

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

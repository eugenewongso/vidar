import subprocess
import os
import re

class PatchAdopter:
    """Handles applying a single patch file using GNU patch and stores console output."""

    def __init__(self, strip_level=1):
        """
        Initializes the PatchAdopter.

        :param strip_level: Number of leading path components to strip.
        """
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

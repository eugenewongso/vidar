import subprocess
import os
import re

class PatchAdopter:
    """Handles applying a single patch file using GNU patch and stores console output."""

    def __init__(self, strip_level=1):
        """
        Initializes the PatchAdopter.

        :param strip_level: Number of leading path components to strip (equivalent to `-p` option in patch command).
        """
        self.strip_level = strip_level
        self.patch_command = "gpatch"
        self.console_output = ""

    def apply_patch(self, patch_file: str):
        """
        Applies a single patch file using GNU patch.

        :param patch_file: Path to the patch file.
        :return: Console output as a string.
        """
        if not os.path.exists(patch_file):
            self.console_output = f"Patch file not found: {patch_file}"
            print(self.console_output)
            return self.console_output

        try:
            result = subprocess.run(
                [self.patch_command, "-p", str(self.strip_level), "-i", patch_file],
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
        
    def get_console_output(self) -> str:
        """
        Retrieves the stored console output.

        :return: Console output as a string.
        """
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
            combined_file.write("# Combined .rej files:\n")
            for rej_file in sorted(rej_files):
                combined_file.write(f"# {rej_file}\n")
            
            combined_file.write("\n")

            for rej_file in sorted(rej_files):
                with open(rej_file, "r") as file:
                    combined_file.write(file.read().strip() + "\n\n")

        print(f"Combined {len(rej_files)} .rej files into {output_path}:")
        for rej_file in rej_files:
            print(f" - {rej_file}")

        return output_path


# if __name__ == "__main__":
#     adopter = PatchAdopter()
#     adopter.apply_patch("c1a7b4b4a736fa175488122cca9743cff2ae72e8_6.diff")

#     # Combine .rej files related to the patch
#     adopter.combine_rejected_hunks()

import os
import subprocess
import re

class PatchLLMAdopter:
    """Handles applying an LLM-generated patch file using GNU patch and stores console output."""

    def __init__(self, kernel_path, patch_file, strip_level=1):
        """
        Initializes the PatchLLMAdopter.

        :param kernel_path: Path to the kernel source where patches will be applied.
        :param patch_file: Path to the LLM-generated patch file.
        :param strip_level: Number of leading path components to strip (equivalent to `-p` option in patch command).
        """
        self.kernel_path = kernel_path
        self.patch_file = patch_file
        self.strip_level = strip_level
        self.patch_command = "gpatch"  # Use "patch" if on Linux
        self.console_output = ""

    def apply_patch(self, dry_run=False):
        """
        Applies the LLM-generated patch file using GNU patch.

        :param dry_run: If True, performs a dry run without modifying files.
        :return: Console output as a string.
        """
        if not os.path.exists(self.patch_file):
            self.console_output = f"❌ Patch file not found: {self.patch_file}"
            print(self.console_output)
            return self.console_output

        # Change directory to kernel source
        os.chdir(self.kernel_path)

        command = [self.patch_command, "-p", str(self.strip_level), "-i", self.patch_file]
        if dry_run:
            command.append("--dry-run")

        try:
            result = subprocess.run(
                command,
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
        Extracts all .rej files from console output and locates them in the file system.

        :return: List of .rej file paths.
        """
        rej_pattern = re.compile(r"saving rejects to file (.+\.rej)")
        matches = rej_pattern.findall(self.console_output)

        rej_files = []
        for rej_file in matches:
            reject_path = os.path.join(self.kernel_path, rej_file)
            if os.path.exists(reject_path):
                rej_files.append(reject_path)

        return rej_files

    def combine_rejected_hunks(self, output_file="combined.rej"):
        """
        Combines all .rej files related to the last applied patch into one file.

        :param output_file: Name of the combined output file.
        :return: The full path to the combined .rej file, or None if no files were found.
        """
        rej_files = self.get_rej_files()

        if not rej_files:
            print("✅ No .rej files found. Patch applied cleanly.")
            return None

        output_path = os.path.abspath(output_file)

        with open(output_path, "w") as combined_file:
            combined_file.write("# Combined .rej files:\n")
            for rej_file in sorted(rej_files):
                combined_file.write(f"# {rej_file}\n\n")

            for rej_file in sorted(rej_files):
                with open(rej_file, "r") as file:
                    combined_file.write(file.read().strip() + "\n\n")

        print(f"⚠️ Combined {len(rej_files)} .rej files into {output_path}:")
        return output_path


# === Run the Patch Process ===

if __name__ == "__main__":
    # Paths
    kernel_path = input("Input Repository path to apply diff files here: ")
    patch_file = "/Users/theophilasetiawan/Desktop/files/capstone/vidar/f913f0123e6cff4dbc7c1e17d13b7a59a54475d2.diff_fixed.diff"

    # Ensure the kernel directory exists
    if not os.path.isdir(kernel_path):
        print(f"❌ Error: Kernel directory not found at {kernel_path}")
        exit(1)

    # Ensure the patch file exists
    if not os.path.exists(patch_file):
        print(f"❌ Error: Patch file not found at {patch_file}")
        exit(1)

    # Initialize PatchLLMAdopter
    patcher = PatchLLMAdopter(kernel_path, patch_file, strip_level=1)

    print(f"\n🔍 Attempting to apply LLM-generated patch: {patch_file}")

    # Dry run check
    dry_run_output = patcher.apply_patch(dry_run=True)

    if "FAILED" in dry_run_output or "reject" in dry_run_output:
        print(f"⚠️ Patch {patch_file} may not apply cleanly. Skipping...")
        exit(1)

    print(f"✅ Patch {patch_file} can be applied successfully. Applying now...")
    patcher.apply_patch()

    # Look for rejected hunks
    rejected_files = patcher.get_rej_files()

    if rejected_files:
        print(f"⚠️ Some hunks were rejected. See files:")
        for rfile in rejected_files:
            print(f" - {rfile}")

        patcher.combine_rejected_hunks()
    else:
        print(f"🎉 Patch {patch_file} applied successfully!")

    print("\n🎉 Patch application process complete!")

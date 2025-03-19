import os
import subprocess
import re

class PatchLLMAdopter:
    """Handles applying an LLM-generated patch file using GNU patch and stores console output."""

    def __init__(self, kernel_path, strip_level=1):
        """
        Initializes the PatchLLMAdopter.

        :param kernel_path: Path to the kernel source where patches will be applied.
        :param strip_level: Number of leading path components to strip (equivalent to `-p` option in patch command).
        """
        self.kernel_path = kernel_path
        self.strip_level = strip_level
        self.patch_command = "gpatch"  # Use "patch" if on Linux
        self.console_output = ""

    def apply_patch(self, patch_file: str, dry_run=False):
        """
        Applies a single patch file using GNU patch.

        :param patch_file: Path to the patch file.
        :param dry_run: If True, performs a dry run without modifying files.
        :return: Console output as a string.
        """
        if not os.path.exists(patch_file):
            self.console_output = f"❌ Patch file not found: {patch_file}"
            print(self.console_output)
            return self.console_output

        # Change directory to kernel source
        os.chdir(self.kernel_path)

        command = [self.patch_command, "-p", str(self.strip_level), "-i", patch_file]
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
    llm_patch_dir = "/Users/theophilasetiawan/Desktop/files/capstone/vidar/patch_adoption/generated_patches"

    # Ensure the kernel directory exists
    if not os.path.isdir(kernel_path):
        print(f"❌ Error: Kernel directory not found at {kernel_path}")
        exit(1)

    # Ensure generated patch directory exists
    if not os.path.isdir(llm_patch_dir):
        print(f"❌ Error: Patch directory not found at {llm_patch_dir}")
        exit(1)

    # Initialize PatchLLMAdopter
    patcher = PatchLLMAdopter(kernel_path, strip_level=1)

    # Get all LLM-generated patch files
    patch_files = [f for f in os.listdir(llm_patch_dir) if f.endswith(".diff")]

    if not patch_files:
        print("⚠️ No LLM-generated patches found.")
        exit(1)

    for patch_file in patch_files:
        full_patch_path = os.path.join(llm_patch_dir, patch_file)

        print(f"\n🔍 Attempting to apply LLM-generated patch: {patch_file}")

        # Dry run check
        dry_run_output = patcher.apply_patch(full_patch_path, dry_run=True)

        if "FAILED" in dry_run_output or "reject" in dry_run_output:
            print(f"⚠️ Patch {patch_file} may not apply cleanly. Skipping...")
            continue

        print(f"✅ Patch {patch_file} can be applied successfully. Applying now...")
        patcher.apply_patch(full_patch_path)

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

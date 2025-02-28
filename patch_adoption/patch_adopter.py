import subprocess
import os

class PatchAdopter:
    """Handles applying a single patch file using GNU patch."""

    def __init__(self, strip_level=1):
        """
        Initializes the PatchAdopter.

        :param strip_level: Number of leading path components to strip (equivalent to `-p` option in patch command).
        """
        self.strip_level = strip_level
        # Assume GNU patch is installed
        self.patch_command = "gpatch"  

    def apply_patch(self, patch_file: str):
        """
        Applies a single patch file using GNU patch.

        :param patch_file: Path to the patch file.
        :return: True if the patch was applied successfully, False otherwise.
        """
        if not os.path.exists(patch_file):
            print(f"Patch file not found: {patch_file}")
            return False

        try:
            result = subprocess.run([
                self.patch_command, "-p", str(self.strip_level), "-i", patch_file
            ], check=True, text=True)
            
            print(f"Successfully applied patch: {patch_file}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to apply patch: {patch_file}")
            return False

# Example:
if __name__ == "__main__":
    adopter = PatchAdopter(strip_level=0)
    adopter.apply_patch("generated_patch.diff")

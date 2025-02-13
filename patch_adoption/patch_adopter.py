import patch

class PatchAdopter:
    """Handles applying multiple patch files to a codebase."""

    def __init__(self, patch_files=None):
        """
        Initializes the PatchAdopter.
        
        :param patch_files: Optional list of patch file paths to apply. Defaults to None.
        """
        self.patch_files = patch_files if patch_files else []
        # Stores success/failure per patch
        self.results = {}

    def apply_all(self):
        """Applies all patches in the list and stores the results."""
        for patch_file in self.patch_files:
            success = self.apply_patch(patch_file)
            self.results[patch_file] = success

        self.print_summary()

    def apply_patch(self, patch_file: str):
        """
        Applies a single patch file.

        :param patch_file: Path to the patch file.
        :return: True if the patch was applied successfully, False otherwise.
        """
        patch_set = patch.fromfile(patch_file)
        
        if patch_set is None:
            print(f"Failed to load patch file: {patch_file}")
            return False

        success = patch_set.apply()

        if success:
            print(f"Successfully applied patch: {patch_file}")
        else:
            print(f"Failed to apply patch: {patch_file}")

        return success

    def print_summary(self):
        """Prints a summary of applied patches."""
        print("\nPatch Adoption Summary:")
        for patch_file, success in self.results.items():
            status = "Success" if success else "Failed"
            print(f"{patch_file}: {status}")

# # Example:
# if __name__ == "__main__":
#     patch_files = ["patch1.diff", "patch2.diff", "patch3.diff"] 
#     adopter = PatchAdopter(patch_files)
#     adopter.apply_all()

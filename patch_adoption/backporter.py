import subprocess
import logging


class Backporter:
    """Handles backporting patches to previous versions."""

    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    def backport_patch(self, patch_file: str):
        """
        Attempts to backport the patch to older Android versions.

        Args:
            patch_file: Path to the patch file to be backported.
        """
        versions = ["android-12", "android-11", "android-10"]

        for version in versions:
            logging.info(f"Attempting to backport patch to {version}...")
            try:
                # Checkout the target version
                subprocess.run(["git", "checkout", version], cwd=self.repo_path, check=True)

                # Apply the patch using cherry-pick
                subprocess.run(["git", "cherry-pick", patch_file], cwd=self.repo_path, check=True)

                logging.info(f"Successfully backported patch to {version}.")
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to backport patch to {version}: {e}")

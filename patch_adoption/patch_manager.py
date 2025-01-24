import subprocess
import logging
from patch_adoption.backporter import Backporter
import requests

class PatchManager:
    """Manages the patch adoption process."""

    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.backporter = Backporter(repo_path)

    def apply_patch(self, patch_url: str) -> bool:
        """Downloads and applies a patch to the repository."""

        logging.info("Starting patch application from URL: %s", patch_url)
        try:
            # Download the patch
            patch_file = self._download_patch(patch_url)

            # Apply the patch
            subprocess.run(["git", "apply", patch_file], cwd=self.repo_path, check=True)
            logging.info("Patch successfully applied to the current version.")

            # Trigger backporting for older versions
            self.backporter.backport_patch(patch_file)

            return True
        except requests.RequestException as e:
            logging.error("Failed to download patch: %s", e)
            return False
        except subprocess.CalledProcessError as e:
            logging.error("Failed to apply patch: %s", e)
            return False


    def _download_patch(self, patch_url: str) -> str:
        """Downloads a patch file from the given URL and returns the path to the downloaded file."""

        response = requests.get(patch_url)

        if response.status_code == 200:
            patch_file = "/tmp/patch.diff"
            with open(patch_file, "wb") as f:
                f.write(response.content)

                logging.info("Downloaded patch to: $s", patch_file)
                return patch_file
            
        else:
            raise requests.RequestException(
            f"Failed to download patch from {patch_url} with status code {response.status_code}"
        )
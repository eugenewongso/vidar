r"""Parses raw Vanir security reports into a structured format.

This module is the first step in the patch management pipeline. It reads the
raw JSON output from a Vanir scan, extracts key information about security
patches, and reorganizes it into a clean, structured format. The primary goal
is to create a canonical list of unique patches and map them to the specific
files, functions, and **projects** they affect.

This structured output is essential for the subsequent steps in the pipeline,
such as fetching and applying the patches in their correct respective source
directories.

Usage:
  python vanir_report_parser.py
"""

import json
import os
import re
import sys
from urllib.parse import urlparse
import logging
import yaml
from pathlib import Path

# Get a logger for this module
logger = logging.getLogger(__name__)


# --- Load Configuration ---
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

PATHS_CONFIG = config.get("paths", {})
# --- End Load Configuration ---


class VanirParser:
    """
    Processes a raw Vanir security vulnerability report into a structured format.

    This class handles the loading, parsing, and saving of vulnerability data.
    It extracts essential details like patch URLs, the project path, and
    affected code locations, producing a clean JSON file that serves as the
    input for the next stage of the pipeline.
    """

    def __init__(self, file_path, output_path=None):
        """Initializes the parser and runs the full parsing process.

        Args:
            file_path: The path to the input Vanir report JSON file.
            output_path: The path to save the parsed output.
        """
        self.file_path = file_path
        self.output_path = output_path
        
        # Ensure the output directory exists.
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # Load, parse, and write the report.
        self.vanir_report = self._load_vanir_report()
        self.reorganized_report = self._parse_vanir_report()
        self._write_output_to_json()

    def _load_vanir_report(self):
        """Loads the Vanir security report from the specified JSON file."""
        with open(self.file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _get_patch_info(self, patch_url: str) -> tuple[str, str] | tuple[None, None]:
        """
        Extracts the commit hash and project path from a patch URL.

        Args:
            patch_url: The URL of the patch from Googlesource or CodeLinaro.

        Returns:
            A tuple of (commit_hash, project_path), or (None, None) on failure.
        """
        try:
            parsed_url = urlparse(patch_url)
            commit_hash = re.search(r'([a-fA-F0-9]{40})', patch_url).group(1)

            path_without_commit = ""
            if "android.googlesource.com" in parsed_url.netloc:
                match = re.search(r'(.+?)/\+/', parsed_url.path)
                if match:
                    # Strip leading /platform/ as it's not part of the local checkout path
                    path_without_commit = match.group(1).replace('/platform/', '', 1)
            elif "git.codelinaro.org" in parsed_url.netloc:
                match = re.search(r'(.+?)/-/commit/', parsed_url.path)
                if match:
                    # Strip leading /clo/la/ for CodeLinaro
                    path_without_commit = re.sub(r'^/clo/la/', '', match.group(1))

            if commit_hash and path_without_commit:
                return commit_hash, path_without_commit.strip('/')
                
        except (AttributeError, IndexError):
            pass # Fall through to return None on parsing failure
        
        logger.warning(f"Could not parse commit hash or project from URL: {patch_url}")
        return None, None


    def _parse_vanir_report(self):
        """Parses the loaded Vanir report into a structured format.

        This method iterates through the 'missing_patches' in the report,
        groups them by a unique patch URL, and records which files and
        functions are affected by each patch.

        Returns:
            A dictionary containing a list of structured patch information.
        """
        patch_list = []
        patches_by_url = {}

        for patch_info in self.vanir_report.get("missing_patches", []):
            for detail in patch_info["details"]:
                patch_url = detail["patch"]
                file_function = detail["unpatched_code"]

                # Extract file path and function name.
                parts = file_function.split("::", 1)
                file_path = parts[0]
                function_name = parts[1] if len(parts) > 1 else None

                # If we haven't seen this patch URL yet, create a new entry.
                if patch_url not in patches_by_url:
                    patch_hash, project_path = self._get_patch_info(patch_url)
                    if not patch_hash or not project_path:
                        continue

                    patch_entry = {
                        "patch_url": patch_url,
                        "patch_file": f"{patch_hash}.diff",
                        "project": project_path,
                        "files": {}
                    }
                    patches_by_url[patch_url] = patch_entry
                    patch_list.append(patch_entry)
                
                patch_entry = patches_by_url[patch_url]

                # Add the file to this patch's entry if it's not there.
                if file_path not in patch_entry["files"]:
                    patch_entry["files"][file_path] = {"functions": []}

                # Add the function name if it's new for this file.
                if function_name and function_name not in patch_entry["files"][file_path]["functions"]:
                    patch_entry["files"][file_path]["functions"].append(function_name)

        return {"patches": patch_list}

    def _write_output_to_json(self):
        """Writes the structured data to the output JSON file."""
        with open(self.output_path, "w", encoding="utf-8") as file:
            json.dump(self.reorganized_report, file, indent=4)
        logger.info(f"Parsed report saved to {self.output_path}")


def main():
    """Main entry point for the Vanir report parser script."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    vanir_report_path = os.path.join(project_root, PATHS_CONFIG.get("vanir_source_report"))
    parsed_report_path = os.path.join(project_root, PATHS_CONFIG.get("parsed_vanir_report"))
    
    if not os.path.exists(vanir_report_path):
        logger.error(f"Input file not found at '{vanir_report_path}'.")
        logger.error("   Please place the raw Vanir report at that location.")
        sys.exit(1)
        
    VanirParser(vanir_report_path, parsed_report_path)


if __name__ == "__main__":
    main()
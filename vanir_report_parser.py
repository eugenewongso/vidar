r"""Parses raw Vanir security reports into a structured format.

This module is the first step in the patch management pipeline. It reads the
raw JSON output from a Vanir scan, extracts key information about security
patches, and reorganizes it into a clean, structured format. The primary goal
is to create a canonical list of unique patches and map them to the specific
files and functions they affect.

This structured output is essential for the subsequent steps in the pipeline,
such as fetching and applying the patches.

Usage:
  python vanir_report_parser.py
"""

import json
import os
import re


class VanirParser:
    """
    Processes a raw Vanir security vulnerability report into a structured format.

    This class handles the loading, parsing, and saving of vulnerability data.
    It extracts essential details like patch URLs and affected code locations,
    producing a clean JSON file that serves as the input for the next stage of
    the pipeline.
    """

    def __init__(self, file_path, output_path=None):
        """Initializes the parser and runs the full parsing process.

        Args:
            file_path: The path to the input Vanir report JSON file.
            output_path: The path to save the parsed output. Defaults to
              'reports/parsed_report.json' within the same directory.
        """
        project_root = os.path.dirname(os.path.abspath(__file__))
        
        self.file_path = file_path
        self.output_path = output_path or os.path.join(
            project_root, "reports", "parsed_report.json")
        
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

    def _extract_patch_hash(self, patch_url):
        """Extracts a Git commit hash from a patch URL using a regex.

        Args:
            patch_url: The URL of the patch, from Googlesource or CodeLinaro.

        Returns:
            The 40-character commit hash if found, otherwise "N/A".
        """
        match = re.search(r"([a-f0-9]{40})$", patch_url)
        return match.group(1) if match else "N/A"

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
                    patch_hash = self._extract_patch_hash(patch_url)
                    patch_entry = {
                        "patch_url": patch_url,
                        "patch_file": f"{patch_hash}.diff",
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
        print(f"✅ Parsed report saved to {self.output_path}")


def main():
    """Main entry point for the Vanir report parser script."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Define the default input path for the raw Vanir report.
    vanir_report_path = os.path.join(
        project_root, "reports", "vanir_output.json")
    
    if not os.path.exists(vanir_report_path):
        print(f"❌ Error: Input file not found at '{vanir_report_path}'.")
        print("   Please place the raw Vanir report at that location.")
        sys.exit(1)
        
    VanirParser(vanir_report_path)


if __name__ == "__main__":
    main()
import os
import json
import re

class VanirParser:
    """A class to handle parsing, structuring, and extracting metadata from a Vanir report."""

    def __init__(self, file_path, output_path=None):
        self.file_path = file_path
        self.output_path = output_path or os.path.abspath("reports/parsed_report.json")

        # Load, parse, and write the report automatically
        self.vanir_report = self.load_vanir_report()
        self.reorganized_report = self.parse_vanir_report()
        self.write_output_to_json()

    def load_vanir_report(self):
        """Loads the Vanir report from a file."""
        with open(self.file_path, "r") as file:
            return json.load(file)

    def extract_patch_hash(self, patch_url):
        """
        Extracts the commit hash from a patch URL using regex.
        Constraint:  7 <= len(commit_hash) <= 40
        """
        match = re.search(r"([a-f0-9]{7,40})$", patch_url)  
        return match.group(1) if match else "N/A"


    def parse_vanir_report(self):
        """Parses the Vanir report into a structured format with a list of patch entries."""
        patch_list = []

        for patch_info in self.vanir_report.get("missing_patches", []):
            for detail in patch_info["details"]:
                patch_url = detail["patch"]
                file_function = detail["unpatched_code"]

                # Extract file path and function (if present)
                parts = file_function.split("::", 1)
                file_path = parts[0]
                function_name = parts[1] if len(parts) > 1 else None

                # Extract commit hash
                patch_hash = self.extract_patch_hash(patch_url)

                # Check if this patch URL already exists in the list
                patch_entry = next((p for p in patch_list if p["patch_url"] == patch_url), None)

                if not patch_entry:
                    patch_entry = {
                        "patch_url": patch_url,
                        "patch_file": f"{patch_hash}.diff" if patch_hash != "N/A" else "N/A",
                        "files": {}
                    }
                    patch_list.append(patch_entry)

                # Check if the file path exists under this patch
                if file_path not in patch_entry["files"]:
                    patch_entry["files"][file_path] = {"functions": []}

                # Add function name if present and not already listed
                if function_name and function_name not in patch_entry["files"][file_path]["functions"]:
                    patch_entry["files"][file_path]["functions"].append(function_name)

        return {"patches": patch_list}

    def write_output_to_json(self):
        """Writes the parsed structured data to a JSON file."""
        with open(self.output_path, "w") as file:
            json.dump(self.reorganized_report, file, indent=4)
        # print(f"✅ Parsed report saved to {self.output_path}")

# Automatically execute when the script is run directly
# if __name__ == "__main__":
#     VANIR_REPORT_PATH = os.path.abspath("reports/xiaomi_output.json")
#     VanirParser(VANIR_REPORT_PATH)

import json
import pprint
import re

class VanirParser:
    """A class to handle parsing, structuring, and extracting metadata from a Vanir report."""

    def __init__(self, file_path):
        self.file_path = file_path
        self.vanir_report = self.load_vanir_report()
        self.reorganized_report = self.parse_vanir_report()

    def load_vanir_report(self):
        """Loads the Vanir report from a file."""
        with open(self.file_path, "r") as file:
            return json.load(file)

    def extract_patch_hash(self, patch_url):
        """Extracts the commit hash from a patch URL using regex."""
        # Match any valid commit hash (variable length) before a non-alphanumeric character
        match = re.search(r"/([a-f0-9]+)\b", patch_url)
        return match.group(1) if match else "N/A"

    def parse_vanir_report(self):
        """Parses the Vanir report into a structured format with a list of patch entries."""
        patch_dict = {} 

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

                # Ensure patch URL is unique
                if patch_url not in patch_dict:
                    patch_dict[patch_url] = {
                        "patch_url": patch_url,
                        "patch_file": f"{patch_hash}.diff" if patch_hash != "N/A" else "N/A",
                        "files": {}
                    }

                # Ensure file path exists under this patch
                if file_path not in patch_dict[patch_url]["files"]:
                    patch_dict[patch_url]["files"][file_path] = {"functions": []}

                # Add function name if present and not already listed
                if function_name and function_name not in patch_dict[patch_url]["files"][file_path]["functions"]:
                    patch_dict[patch_url]["files"][file_path]["functions"].append(function_name)

        return {"patches": list(patch_dict.values())}  # Convert dict back to list for output

    def write_output_to_json(self, output_file="parsed_report.json"):
        """Writes the parsed structured data to a JSON file."""
        with open(output_file, "w") as file:
            json.dump(self.reorganized_report, file, indent=4)
        print(f"Parsed report successfully written to {output_file}")

# Example Usage:
if __name__ == "__main__":
    parser = VanirParser("2018-report-20250312215759.json")

    # Print structured data
    pprint.pprint(parser.reorganized_report)

    # Write parsed output to a JSON file
    parser.write_output_to_json("2018-parsed-20250312215759.json")

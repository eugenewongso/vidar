import os
import json
import re

class VanirParser:
    """
    Parser for Vanir security vulnerability reports.
    
    This class is responsible for:
    1. Loading raw Vanir vulnerability reports in JSON format
    2. Extracting structured information about security patches
    3. Organizing patch data by affected files and functions
    4. Saving the processed data in a structured format for further processing
    
    The parser extracts key information such as:
    - Patch URLs (from Googlesource or CodeLinaro)
    - Commit hashes from patch URLs
    - Affected files and functions
    
    The output is a structured JSON file that maps patches to their affected files,
    which is used by subsequent modules for downloading and applying patches.
    
    Usage:
        parser = VanirParser('path/to/vanir_report.json')
        # Parsing is done automatically in __init__
    """

    def __init__(self, file_path, output_path=None):
        """
        Initialize the parser with input and output file paths.
        
        The parser loads the Vanir report, processes it, and automatically 
        writes the parsed data to the output file.
        
        :param file_path: Path to the input Vanir report JSON file
        :param output_path: Path to save the parsed output (default: reports/parsed_report.json)
        """
        # Get the project root directory
        project_root = os.path.dirname(os.path.abspath(__file__))
        
        self.file_path = file_path
        self.output_path = output_path or os.path.join(project_root, "reports", "parsed_report.json")
        
        # Ensure the reports directory exists
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # Load, parse, and write the report automatically
        self.vanir_report = self.load_vanir_report()
        self.reorganized_report = self.parse_vanir_report()
        self.write_output_to_json()

    def load_vanir_report(self):
        """
        Loads the Vanir security report from a JSON file.
        
        This method simply reads the Vanir report file and parses it as JSON.
        
        :return: The loaded JSON data as a Python dictionary.
        """
        with open(self.file_path, "r") as file:
            return json.load(file)

    def extract_patch_hash(self, patch_url):
        """
        Extracts the commit hash from a patch URL using regex.
        
        The function looks for a 40-character hexadecimal string at the end of the URL,
        which is the standard format for Git commit hashes.
        
        :param patch_url: URL of the patch, typically from Googlesource or CodeLinaro.
        :return: The commit hash if found, otherwise "N/A".
        """
        match = re.search(r"([a-f0-9]{40})$", patch_url)  # Match 40-char commit hashes
        return match.group(1) if match else "N/A"

    def parse_vanir_report(self):
        """
        Parses the Vanir report into a structured format.
        
        This method:
        1. Extracts patch URLs and affected files/functions from the report
        2. Organizes patches by URL, avoiding duplicates
        3. Maps each patch to its affected files and functions
        4. Creates a structured representation suitable for downstream processing
        
        The result is a dictionary with a "patches" key containing a list of patch entries,
        each with patch URL, filename, and affected files with their functions.
        
        :return: Dictionary with structured patch information.
        """
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
        """
        Writes the parsed structured data to a JSON file.
        
        This method saves the processed patch information to a JSON file,
        which will be used by subsequent processing steps (downloading patches,
        applying patches, etc.).
        """
        with open(self.output_path, "w") as file:
            json.dump(self.reorganized_report, file, indent=4)
        print(f"✅ Parsed report saved to {self.output_path}")


# Automatically execute when the script is run directly
if __name__ == "__main__":
    """
    Main entry point for Vanir report parsing.
    
    When executed directly, this script:
    1. Defines the path to the Vanir output file
    2. Creates a VanirParser instance to process the report
    3. The parser automatically loads, processes, and saves the structured data
    """
    # Get the project root directory
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    
    # Use vanir_output.json for the input file
    VANIR_REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "vanir_output.json")
    
    # Ensure the reports directory exists
    os.makedirs(os.path.dirname(VANIR_REPORT_PATH), exist_ok=True)
    
    VanirParser(VANIR_REPORT_PATH)
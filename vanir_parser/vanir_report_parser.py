import json
import pprint

class VanirParser:
    """A class to handle parsing, structuring, and extracting metadata from Vanir report."""

    def __init__(self, file_path):
        self.file_path = file_path
        self.vanir_report = self.load_vanir_report()
        self.structured_data = self.parse_vanir_report()

    def load_vanir_report(self):
        """Loads the Vanir report from a file."""
        with open(self.file_path, "r") as file:
            return json.load(file)

    def parse_vanir_report(self):
        """Parses the Vanir report into the desired structured format."""
        patch_map = {}

        for patch_info in self.vanir_report.get("missing_patches", []):
            for detail in patch_info["details"]:
                patch_url = detail["patch"]
                file_function = detail["unpatched_code"]
                
                # Extract file path and function (if present)
                parts = file_function.split("::", 1)
                file_path = parts[0]
                function_name = parts[1] if len(parts) > 1 else None
                
                _, _, patch_hash = patch_url.rpartition("/+/")
                
                if patch_hash:
                    # Add to patch_map
                    if patch_url not in patch_map:
                        patch_map[patch_url] = {"patch_file": f"{patch_hash}.diff", "files": {}}

                    if file_path not in patch_map[patch_url]["files"]:
                        patch_map[patch_url]["files"][file_path] = []
                    
                    if function_name:
                        patch_map[patch_url]["files"][file_path].append(function_name)

        return patch_map


    def get_affected_files(self):
        """Returns a unique list of affected file paths."""
        affected_files = set()

        for patch_data in self.structured_data.values():
            affected_files.update(patch_data["files"].keys())

        return affected_files
    
    def get_commit_hashes(self):
        """Returns a unique list of commit hashes without the .diff extension."""
        commit_hashes = {data["patch_file"].replace(".diff", "") for data in self.structured_data.values()}

        return commit_hashes


    def write_output_to_json(self, output_file="parsed_report.json"):
        """Writes the parsed structured data to a JSON file."""
        with open(output_file, "w") as file:
            json.dump(self.structured_data, file, indent=4)
        print(f"Parsed report successfully written to {output_file}")

    def print_commit_hashes(self):
        """Pretty prints the list of commit hashes."""
        pprint.pprint(self.get_commit_hashes())


# # Example:
# if __name__ == "__main__":
#     # Initialize the parser
#     parser = VanirParser("LLM_map_input_test.json")

#     # Print structured data
#     pprint.pprint(parser.structured_data)

#     # Print affected files
#     print("\nAffected Files:")
#     pprint.pprint(parser.get_affected_files())

#     # Print commit hashes
#     print("\nList of Commit Hashes:")
#     pprint.pprint(parser.get_commit_hashes())

#     # Write parsed output to a JSON file
#     parser.write_output_to_json("parsed_report.json")

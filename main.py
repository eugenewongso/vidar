import json
import os
import time
from vanir_parser import VanirParser  # Importing the external VanirParser class
from patch_adoption import PatchAdopter  # Importing the external PatchAdopter class
from download_diff import DiffFetcher  # Importing the external diff downloader
from llm_integration import LLMPatchGenerator  # Importing the LLM patch generator

def main(file_path):
    parser = VanirParser(file_path)
    output_map = parser.structured_data
    if not output_map:
        print("Vanir execution failed. Exiting.")
        return

    generator = LLMPatchGenerator()
    patch_adopter = PatchAdopter()
    
    for patch_url, patch_data in output_map.items():
        commit_hash = patch_data["patch_file"].replace(".diff", "")
        patch_file = DiffFetcher(commit_hash)
        if not patch_file:
            continue

        llm_patch = generator.generate_patch([patch_file])
        success, error_log = patch_adopter.apply_patch(llm_patch)

        retries = 3
        while not success and retries > 0:
            print("Patch did not apply cleanly. Refining with LLM...")
            llm_patch = generator.refine_patch(llm_patch, error_log)
            success, error_log = patch_adopter.apply_patch(llm_patch)
            retries -= 1

        if success:
            print("Patch successfully applied and saved.")
        else:
            print("Patch could not be applied after multiple refinements.")

    print("Process completed.")


if __name__ == "__main__":
    FILE_PATH = "LLM_map_input_test.json"  # Replace with actual file path
    main(FILE_PATH)

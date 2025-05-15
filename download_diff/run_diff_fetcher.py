import json
import sys
import os
from pathlib import Path

# Ensure fetch_diff.py is properly imported
sys.path.append(str(Path(__file__).resolve().parent))
from fetch_diff import fetch_patch

"""
Patch Fetcher Runner

This script orchestrates the batch downloading of patches identified in the
Vanir report. It reads the structured report data, processes each patch entry,
and downloads the corresponding patch files.

This is the main entry point for the patch downloading process, which:
1. Reads the parsed Vanir report
2. For each patch, extracts the URL and affected files
3. Calls the fetch_patch function to download and filter the patch
4. Reports success or failure for each patch

The script handles errors for individual patches and continues processing
the remaining patches even if some fail.
"""

if __name__ == "__main__":
    """
    Main entry point for batch patch downloading.
    
    This script:
    1. Loads the parsed Vanir report
    2. For each patch, downloads it from the source repository
    3. Filters the patch to include only relevant files
    4. Saves the patch to the output directory
    5. Reports success or failure for each patch
    """
    # Get the project root directory
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define path to parsed report
    parsed_report_path = os.path.join(PROJECT_ROOT, "reports", "parsed_report.json")
    
    with open(parsed_report_path, "r") as f:
        parsed_report = json.load(f)

    for patch in parsed_report["patches"]:
        patch_url = patch["patch_url"]
        files_to_include = list(patch["files"].keys())
        print(f"🔍 Processing patch: {patch_url} | Filtering files: {files_to_include}")

        try:
            # Call fetch_patch to download and filter the patch
            diff_file = fetch_patch(patch_url, files_to_include)

            if diff_file:
                print(f"✅ Patch saved: {diff_file}")
            else:
                print(f"❌ Failed to fetch patch: {patch_url}")

        except Exception as e:
            # Catch and report any exceptions, but continue processing other patches
            print(f"⚠️ Error processing {patch_url}: {e}")
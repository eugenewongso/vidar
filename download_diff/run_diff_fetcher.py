import json
from fetch_diff import fetch_patch
import sys
from pathlib import Path

# Add the project root (vidar/) to Python's module search path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from paths import PARSED_REPORT_PATH  # Now this works!


# Load the parsed report JSON
with open(PARSED_REPORT_PATH, "r") as f:
    parsed_report = json.load(f)

# Process each patch in the report
for patch in parsed_report["patches"]:
    patch_url = patch["patch_url"]
    files_to_include = list(patch["files"].keys())
    print(f"🔍 Processing patch: {patch_url} | Filtering files: {files_to_include}")
    
    try:
        # Fetch and save the patch, passing the relevant file paths
        diff_file = fetch_patch(patch_url, files_to_include)

        if diff_file:
            print(f"✅ Patch saved: {diff_file}")
        else:
            print(f"❌ Failed to fetch patch: {patch_url}")

    except Exception as e:
        print(f"⚠️ Error processing {patch_url}: {e}")

import json
import sys
from pathlib import Path

# Ensure fetch_diff.py is properly imported
sys.path.append(str(Path(__file__).resolve().parent))
from fetch_diff import fetch_patch

if __name__ == "__main__":
    with open("reports/parsed_report.json", "r") as f:
        parsed_report = json.load(f)

    for patch in parsed_report["patches"]:
        patch_url = patch["patch_url"]
        files_to_include = list(patch["files"].keys())
        print(f"🔍 Processing patch: {patch_url} | Filtering files: {files_to_include}")

        try:
            diff_file = fetch_patch(patch_url, files_to_include)

            if diff_file:
                print(f"✅ Patch saved: {diff_file}")
            else:
                print(f"❌ Failed to fetch patch: {patch_url}")

        except Exception as e:
            print(f"⚠️ Error processing {patch_url}: {e}")

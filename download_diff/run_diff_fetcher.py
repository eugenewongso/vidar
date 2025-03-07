import json
from fetch_diff import fetch_patch  # ✅ Correct import

# Load the parsed report JSON
parsed_report_path = "llm_integration/parsed_report.json"

with open(parsed_report_path, "r") as f:
    parsed_report = json.load(f)

# Process each patch in the report
for patch in parsed_report["patches"]:
    patch_url = patch["patch_url"]
    print(f"🔍 Processing patch: {patch_url}")
    
    try:
        # Fetch and save the patch
        diff_file = fetch_patch(patch_url)

        if diff_file:
            print(f"✅ Patch saved: {diff_file}")
        else:
            print(f"❌ Failed to fetch patch: {patch_url}")

    except Exception as e:
        print(f"⚠️ Error processing {patch_url}: {e}")

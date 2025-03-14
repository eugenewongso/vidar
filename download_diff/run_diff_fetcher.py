import json
from download_diff.fetch_diff import fetch_patch

def run_diff_fetcher():
    parsed_report_path = "reports/parsed_report.json" # Load the parsed report JSON hard coded
    with open(parsed_report_path, "r") as f:
        parsed_report = json.load(f)

    # Process each patch in the report
    print("Starting the diff fetching process...")
    for patch in parsed_report["patches"]:
        patch_url = patch["patch_url"]
        files_to_include = list(patch["files"].keys()) 
        # print(f"Processing patch: {patch_url} | Filtering files: {files_to_include}")
        
        
        try:
            diff_file = fetch_patch(patch_url, files_to_include)

            if not diff_file:
                print(f"Failed to fetch patch: {patch_url}")

        except Exception as e:
            print(f"Error processing {patch_url}: {e}")
            
    print("Diff fetching process completed.")

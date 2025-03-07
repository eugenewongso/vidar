import sys
import os
import json
import re
from patch_adoption.patch_adopter import PatchAdopter
from vanir_parser.vanir_report_parser import VanirParser
from download_diff.fetch_diff import fetch_patch

# Ensure Python can find all project modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Paths (Relative to `vidar/`)
VANIR_REPORT_PATH = os.path.abspath("reports/xiaomi_output.json")  # Vanir JSON report
PATCH_SAVE_DIR = os.path.abspath("llm_integration/")  # Directory to save downloaded patches
XIAOMI_KERNEL_PATH = os.path.abspath("Xiaomi_Kernel_OpenSource")  # Xiaomi Kernel repository path
PATCH_REPORT_PATH = os.path.abspath("reports/patch_report.json")  # Final JSON report

def extract_commit_id(patch_url):
    """Extracts the commit hash from a patch URL using regex."""
    match = re.search(r"([a-f0-9]{40})$", patch_url)  # AOSP 40-char hashes
    if match:
        return match.group(1)
    
    match = re.search(r"commit/([a-f0-9]+)", patch_url)  # CodeLinaro commits
    if match:
        return match.group(1)

    return "N/A"

def extract_code_from_diff(diff_file):
    """
    Extracts modified files and functions from a diff file.
    Returns a dictionary mapping files to functions.
    """
    file_changes = {}

    if not os.path.exists(diff_file):
        return {}

    with open(diff_file, "r", encoding="utf-8") as f:
        diff_content = f.readlines()

    current_file = None
    function_list = []
    in_diff_section = False

    for line in diff_content:
        if line.startswith("--- "):  # Original file (before)
            current_file = line.split()[-1].strip().replace("a/", "")
            function_list = []  # Reset function list
            in_diff_section = False

        elif line.startswith("+++ "):  # Modified file (after)
            in_diff_section = True  # Start capturing changes

        elif in_diff_section and line.startswith("@@"):
            # Attempt to extract function name from context lines
            function_match = re.search(r"@@ .* @@ (\w+)", line)
            if function_match:
                function_name = function_match.group(1)
                if function_name not in function_list:
                    function_list.append(function_name)

        if current_file:
            file_changes[current_file] = {"functions": function_list}

    return file_changes

def main():
    """Automates the entire patching process from Vanir analysis to patch application."""
    
    # Step 1: Parse Vanir report
    print("📄 Parsing Vanir report...")
    parser = VanirParser(VANIR_REPORT_PATH)
    patches = parser.reorganized_report["patches"]

    if not patches:
        print("⚠️ No patches found in Vanir report. Exiting.")
        return

    # Save parsed output
    parser.write_output_to_json(os.path.join(PATCH_SAVE_DIR, "parsed_report.json"))

    # Step 2: Download patches
    patch_results = []
    downloaded_files = []  # Store downloaded patch paths

    for patch in patches:
        patch_url = patch["patch_url"]
        commit_id = extract_commit_id(patch_url)
        patch_filename = f"{commit_id}.diff" if commit_id != "N/A" else "unknown.diff"
        patch_file_path = os.path.join(PATCH_SAVE_DIR, patch_filename)

        print(f"🌐 Downloading patch: {patch_url}")
        downloaded_file = fetch_patch(patch_url)

        if not downloaded_file:
            print(f"❌ Failed to download patch: {patch_url}")
            continue
        
        print(f"✅ Patch downloaded: {downloaded_file}")
        downloaded_files.append((patch, downloaded_file))  # Store patch details and file path

    # Step 3: Apply patches
    patch_adopter = PatchAdopter(repo_path=XIAOMI_KERNEL_PATH, strip_level=1)
    for patch, downloaded_file in downloaded_files:
        print(f"🛠️ Applying {downloaded_file}...")

        success, failed_files = patch_adopter.apply_patch(downloaded_file)

        # Step 4: Extract affected files and functions
        file_changes = extract_code_from_diff(downloaded_file)

        # If failed_files is empty, but the patch failed, extract from diff
        if not failed_files and not success:
            print(f"⚠️ No failed files captured from patch output, extracting from diff file...")
            failed_files = {file: file_changes.get(file, {}).get("functions", []) for file in file_changes}

        # Extract all modified files from the patch
        all_modified_files = set(file_changes.keys())

        # Identify successfully applied files
        applied_files = {file: funcs for file, funcs in file_changes.items() if file not in failed_files and file in all_modified_files}

        # Failed files (as before)
        failed_files_report = {file: funcs for file, funcs in file_changes.items() if file in failed_files}

        # Update patch status determination
        if success and applied_files and not failed_files:
            status = "applied"
        elif success and failed_files:
            status = "partially_applied"
        else:
            status = "failed"


        status = "applied" if success and not failed_files else "partially_applied" if failed_files else "failed"

        # Store results
        patch_results.append({
            "patch_url": patch["patch_url"],
            "commit_id": extract_commit_id(patch["patch_url"]),
            "patch_file": downloaded_file,
            "status": status,
            "applied_files": applied_files if applied_files else {},
            "failed_files": failed_files_report if failed_files_report else {}
        })

        if status == "applied":
            print(f"✅ Successfully applied: {downloaded_file}")
        elif status == "partially_applied":
            print(f"⚠️ Partially applied: {downloaded_file} (some files/functions failed)")
        else:
            print(f"❌ Failed to apply: {downloaded_file}")


    # Step 5: Save JSON report
    final_report = {"patches": patch_results}
    with open(PATCH_REPORT_PATH, "w", encoding="utf-8") as json_file:
        json.dump(final_report, json_file, indent=4)

    print(f"\n📜 Patch application report saved to: {PATCH_REPORT_PATH}")
    print("\n🎉 Patch application process completed!")

if __name__ == "__main__":
    main()

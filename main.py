import os
import json
import re
import time
import requests
from bs4 import BeautifulSoup
from patch_adopter import PatchAdopter

def extract_patch_hash(patch_url):
    """
    Extracts the commit hash from a patch URL using regex.
    Constraint:  7 <= len(commit_hash) <= 40
    """
    match = re.search(r"([a-f0-9]{7,40})$", patch_url)  
    return match.group(1) if match else "N/A"

def extract_commit_hash(commit_url):
    """Extracts the commit hash from a Googlesource or CodeLinaro URL."""
    
    # Matches full 40-character SHA-1 hash (if present)
    full_hash_match = re.search(r'/([a-f0-9]{40})$', commit_url)
    if full_hash_match:
        return full_hash_match.group(1)
    
    # Matches shorter commit hashes (Googlesource sometimes uses short hashes)
    short_hash_match = re.search(r'/\+/(.*?)$', commit_url)
    if short_hash_match:
        return short_hash_match.group(1)

    print(f"⚠️ Could not extract commit hash from URL: {commit_url}")
    return None

def extract_diff(url, files_to_include):
    """Extracts and filters the diff content from the commit page for Android Googlesource."""
    
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract diff content from the correct HTML structure
    diff_sections = soup.find_all("pre", class_="u-pre u-monospace Diff-unified")
    diffs = [section.get_text() for section in diff_sections]

    # Extract file headers
    file_headers = soup.find_all("pre", class_="u-pre u-monospace Diff")
    headers = [header.get_text() for header in file_headers]

    # Combine headers and diffs
    filtered_diff = []
    found_files = set()
    for h, d in zip(headers, diffs):
        for file_path in files_to_include:
            if file_path in h:  # Check if file is in the header
                filtered_diff.append(h + d)
                found_files.add(file_path)
                break  # Avoid duplicate checks for the same file

    return "\n".join(filtered_diff) if filtered_diff else None

def fetch_patch(commit_url, files_to_include):
    """
    Fetches the diff for a given commit URL, filters it to only include relevant files, and saves it.

    Returns:
        str: The path to the saved formatted diff file.
    """

    # Extract commit hash from the URL
    commit_hash = extract_commit_hash(commit_url)
    if not commit_hash:
        return None

    # Determine source type and construct the diff URL
    if "android.googlesource.com" in commit_url:
        diff_url = commit_url + "^!"  # Googlesource requires ^! for diff
        is_codelinaro = False
    elif "git.codelinaro.org" in commit_url:
        diff_url = commit_url + ".diff"  # CodeLinaro requires .diff suffix
        is_codelinaro = True
    else:
        print(f"⚠️ Unsupported commit URL: {commit_url}")
        return None

    output_dir_diff = "outputs/fetched_diffs"
    os.makedirs(output_dir_diff, exist_ok=True)  
    output_filename = os.path.join(output_dir_diff, f"{commit_hash}.diff") 

    response = requests.get(diff_url)

    # Save raw .diff for CodeLinaro
    if is_codelinaro:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(response.text.strip() + "\n") 
        return output_filename

    # Extract and format diff content for Android Googlesource
    extracted_diff = extract_diff(diff_url, files_to_include)
    if not extracted_diff:
        return None

    # Save filtered diff
    with open(output_filename, "w", encoding="utf-8") as output_file:
        output_file.write(extracted_diff.strip() + "\n")  

    return output_filename

# TODO: make this into a class for mofularity (PatchParser)
def parse_vanir_report(file_path, output_path=None):
    """
    Parses the Vanir report into a structured format and writes it to a JSON file.
    Returns: output_path - the path where the parsed report is saved.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"{timestamp}_report.json"
    output_dir = os.path.abspath("outputs/parsed_reports")
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_path or os.path.join(output_dir, output_filename)

    with open(file_path, "r") as file:
        vanir_report = json.load(file)

    patch_list = []

    for patch_info in vanir_report.get("missing_patches", []):
        for detail in patch_info["details"]:
            patch_url = detail["patch"]
            file_function = detail["unpatched_code"]

            parts = file_function.split("::", 1)
            file_path = parts[0]
            function_name = parts[1] if len(parts) > 1 else None

            patch_hash = extract_patch_hash(patch_url)

            patch_entry = next((p for p in patch_list if p["patch_url"] == patch_url), None)

            if not patch_entry:
                patch_entry = {
                    "patch_url": patch_url,
                    "patch_file": f"{patch_hash}.diff" if patch_hash != "N/A" else "N/A",
                    "files": {}
                }
                patch_list.append(patch_entry)

            if file_path not in patch_entry["files"]:
                patch_entry["files"][file_path] = {"functions": []}

            if function_name and function_name not in patch_entry["files"][file_path]["functions"]:
                patch_entry["files"][file_path]["functions"].append(function_name)

            files_to_include = list(patch_entry["files"].keys())
            fetched_patch_path = fetch_patch(patch_url, files_to_include)
            patch_entry["patch_file"] = fetched_patch_path if fetched_patch_path else "N/A"

    reorganized_report = {"patches": patch_list}

    with open(output_path, "w") as file:
        json.dump(reorganized_report, file, indent=4)

    print(f"✅ Parsed report saved to {output_filename}")
    return output_path

def get_input_file_path():
    """Get and validate the path to the raw Vanir report file."""
    file_path = input("Enter the relative path to the raw Vanir report file (eg. inputs/raw_vanir_reports/xiaomi_flame.json): ").strip()
    if not os.path.exists(file_path):
        print("❌ Error: File does not exist.")
        return None
    return file_path

def get_kernel_path():
    """Get and validate the path to the kernel repository."""
    kernel_path = input("Enter the absolute path kernel repository path to apply the patches to: ").strip()
    if not os.path.exists(kernel_path):
        print("❌ Error: Kernel directory does not exist.")
        return None
    if not os.path.isdir(kernel_path):
        print(f"❌ Error: Kernel directory not found at {kernel_path}")
        return None
    return kernel_path

def process_patches(parsed_report, patcher, current_path, report_output_path):
    """
    Process all patches in the report, apply them, and save results.
    
    Args:
        parsed_report: The parsed report containing patches to apply
        patcher: The PatchAdopter instance
        current_path: The current working directory path
        report_output_path: Path where reports should be saved
    
    Returns:
        None
    """
    # Track rejected patches for LLM retry
    failed_patches = []

    # Iterate through patches
    for patch in parsed_report["patches"]:
        patch_file_path = os.path.join(current_path, patch["patch_file"])  
        print(f"\n🔍 Attempting to apply patch: {patch_file_path}")

        patch_result = patcher.apply_patch(patch_file_path, patch["patch_url"])
        patcher.patch_results["patches"].append(patch_result)

        if patch_result["status"] == "Rejected":
            failed_patches.append({
                "patch_file": patch_result["patch_file"],
                "patch_url": patch_result["patch_url"],
                "status": "Rejected",
                "rejected_files": patch_result["rejected_files"],
                "message_output": patch_result["message_output"]
            })

    # Save patch report
    report_output_dir = os.path.dirname(report_output_path)
    os.makedirs(report_output_dir, exist_ok=True) 
    patcher.save_report()

    # Define failed patches directory and save failed patches
    failed_patches_dir = os.path.join(current_path, "outputs/failed_patches")
    os.makedirs(failed_patches_dir, exist_ok=True) 
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    failed_patch_output_path = os.path.join(failed_patches_dir, f"failed_patches_{timestamp}.json")
    
    # Save failed patches for LLM
    with open(failed_patch_output_path, "w") as fail_file:
        json.dump({"patch": failed_patches}, fail_file, indent=4)
        print(f"📁 Failed patch list saved to: {failed_patch_output_path}")

    # Create directory for combined rejected hunks
    combined_hunks_dir = os.path.join(current_path, "outputs/combined_rejected_hunks")
    os.makedirs(combined_hunks_dir, exist_ok=True)

    # Generate a filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    combined_hunks_filename = f"combined_rejected_hunks_{timestamp}.rej"
    combined_hunks_path = os.path.join(combined_hunks_dir, combined_hunks_filename)

    # Get the combined rejected hunks, using the patch results for all info
    rejected_hunks_path = patcher.combine_rejected_hunks(
        combined_hunks_path,
        patch_results=patcher.patch_results["patches"]
    )
    if rejected_hunks_path:
        print(f"📁 Combined rejected hunks saved to: {rejected_hunks_path}")
    else:
        print("No rejected hunks found to combine.")


def main():
    file_path = get_input_file_path()
    if not file_path:
        return
    
    print("Compiling parsed Vanir report...")
    parsed_report_path = parse_vanir_report(file_path)
    
    # Get kernel path
    kernel_path = get_kernel_path()
    if not kernel_path:
        return
    
    # Setup paths
    current_path = os.getcwd()
    report_output_path = "outputs/application_reports"
    combined_current_path = os.path.join(current_path, report_output_path)
    
    print("Applying patches...")
    
    os.chdir(kernel_path) # Change to kernel directory
    
    with open(parsed_report_path, "r") as f: # Load parsed report
        parsed_report = json.load(f)
    
    # Instantiate patch adopter to get its methods
    patcher = PatchAdopter(kernel_path, combined_current_path) 
    
    process_patches(parsed_report, patcher, current_path, report_output_path)

    # TODO: implement 2 LLM-based approaches here (make it 2 different function)

if __name__ == "__main__":
    main()
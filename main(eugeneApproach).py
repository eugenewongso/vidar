import os
import json
import re
import time
import asyncio
import shutil
import requests
from bs4 import BeautifulSoup
from patch_adopter import PatchAdopter
from pydantic_agents.agent_v1 import PatchAgentV2

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

# TODO: make this into a class for modularity (PatchParser)
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
    if os.path.isdir(file_path):
        print("❌ Error: Path is a directory, not a file.")
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

def extract_hunks_from_rej(rej_file):
    """Extracts individual hunks from a .rej file and returns a list of hunks."""
    if not os.path.exists(rej_file):
        print(f"❌ Error: .rej file not found - {rej_file}")
        return []

    with open(rej_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Remove file headers like "--- file.c" and "+++ file.c"
    filtered_lines = []
    for line in lines:
        if not line.startswith(("--- ", "+++ ")):  # Skip file headers
            filtered_lines.append(line)

    rej_content = "".join(filtered_lines)

    # Split hunks using hunk headers (starts with @@ -<old>,<count> +<new>,<count> @@)
    hunks = re.split(r'(?=^@@ -)', rej_content, flags=re.MULTILINE)

    return [hunk.strip() for hunk in hunks if hunk.strip()]

# def extract_hunks_from_rej(rej_file):
#     """Extracts individual hunks from a .rej file and returns a list of hunks."""
#     if not os.path.exists(rej_file):
#         print(f"❌ Error: .rej file not found - {rej_file}")
#         return []

#     with open(rej_file, "r", encoding="utf-8") as f:
#         lines = f.readlines()

#     rej_content = "".join(lines[1:]) # Start from the second line

#     hunks = re.split(r'(?=^@@ -)', rej_content, flags=re.MULTILINE)  # Split by hunk header
#     return [hunk.strip() for hunk in hunks if hunk.strip()]

async def process_patches(parsed_report, patcher, current_path, patch_agent):
    """
    Process all patches in the report, apply them one by one, and handle rejections using AI.

    Args:
        parsed_report: The parsed report containing patches to apply.
        patcher: An instance of PatchAdopter.
        current_path: The working directory path.
        patch_agent: An instance of PatchAgentV2.
    """
    temp_directory = os.path.join(current_path, "outputs/temp")
    os.makedirs(temp_directory, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    resolved_patches_dir = os.path.join(current_path, f"outputs/resolved_patches_{timestamp}")
    os.makedirs(resolved_patches_dir, exist_ok=True)

    for index, patch in enumerate(parsed_report["patches"], start=1):  # Start index at 1
        patch_file_path = os.path.abspath(os.path.join(current_path, patch["patch_file"]))

        if not os.path.exists(patch_file_path):
            print(f"❌ Patch file not found: {patch_file_path}")
            continue  # Skip to the next patch

        # Create a unique resolved patches directory for each patch
        resolved_patches_dir = os.path.join(current_path, f"outputs/resolved_patches_{index}")
        os.makedirs(resolved_patches_dir, exist_ok=True)

        print(f"\n🔍 Attempting to apply patch {index}: {patch_file_path}")
        print(f"📁 Using directory: {resolved_patches_dir}")

        # Apply the patch
        patcher.apply_patch(patch_file_path)

        # Handle rejected hunks
        combined_rej = patcher.combine_rejected_hunks()
        if combined_rej:
            # Step 1: Extract hunks from the combined .rej file
            hunks = extract_hunks_from_rej(combined_rej)

            # Step 2: Extract inline merge conflicts from the file
            conflict_output_dict = patcher.generate_infile_merge_conflict(combined_rej)
            if not conflict_output_dict:
                print(f"✅ No merge conflicts detected for patch {index}.")
                continue

            # Step 3: Iterate through each merge conflict and corresponding hunk
            for conflict, hunk in zip(conflict_output_dict, hunks):
                print(f"\n🔄 Resolving Merge Conflict {conflict['conflict_id']} for Patch {index}...")

                # Prepare file paths for LLM processing
                conflict_id = conflict['conflict_id']
                raw_merge_conflict_path = os.path.join(temp_directory, f"merge_conflict_{conflict_id}.txt")
                code_context_path = os.path.join(temp_directory, f"code_context_{conflict_id}.txt")
                rejected_patch_path = os.path.join(temp_directory, f"rejected_patch_{conflict_id}.rej")
                output_file = os.path.join(resolved_patches_dir, f"resolved_patch_{conflict_id}.diff")

                # Step 4: Write contents to temporary files
                with open(raw_merge_conflict_path, "w", encoding="utf-8") as f:
                    f.write(conflict["conflict"])

                with open(code_context_path, "w", encoding="utf-8") as f:
                    f.write(f"{conflict['before']}\n{conflict['conflict']}\n{conflict['after']}")

                with open(rejected_patch_path, "w", encoding="utf-8") as f:
                    f.write(hunk)

                # Step 5: Call LLM PatchAgentV2 to generate a fixed patch
                fixed_patch_path = await patch_agent.generate_fixed_patch(
                    raw_merge_conflict_path, code_context_path, rejected_patch_path, output_file
                )

                if fixed_patch_path:
                    print(f"✅ Successfully resolved Merge Conflict {conflict_id} for Patch {index}!")
                else:
                    print(f"❌ Failed to resolve Merge Conflict {conflict_id} for Patch {index}.")

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
    patcher = PatchAdopter() 
    
    patch_agent = PatchAgentV2(model_name="gemini-2.0-pro-exp-02-05")
    # patch_agent = PatchAgentV2(model_name="openai:o1")

    # Run process_patches() asynchronously to process patch, applies them, and calls the LLM
    asyncio.run(process_patches(parsed_report, patcher, current_path, patch_agent))
     
if __name__ == "__main__":
    main()

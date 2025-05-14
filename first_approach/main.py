import os
import json
import asyncio
import re
import time
import requests
from bs4 import BeautifulSoup
from patch_adopter_2 import PatchAdopter
from pydantic_agents.agent_v2 import run_patch_porter_agent

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

# def extract_diff(url, files_to_include):
#     """
#     Extracts and filters the diff content from the commit page for Android Googlesource.
    
#     Args:
#         url (str): The URL of the commit page.
#         files_to_include (list): List of filenames to include in the diff.
    
#     Returns:
#         str or None: The filtered diff content, or None if extraction fails.
#     """
#     try:
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#     except requests.exceptions.RequestException as e:
#         # TODO: fix this, might want to add a delay (make sure to fetch all the diff files)
#         # print(f"❌ Failed to fetch diff from URL: {url}")
#         # print(f"   Error: {str(e)}")
#         return None

#     soup = BeautifulSoup(response.text, "html.parser")

#     # Extract diff content from the correct HTML structure
#     diff_sections = soup.find_all("pre", class_="u-pre u-monospace Diff-unified")
#     diffs = [section.get_text() for section in diff_sections]

#     # Extract file headers
#     file_headers = soup.find_all("pre", class_="u-pre u-monospace Diff")
#     headers = [header.get_text() for header in file_headers]

#     # Combine headers and diffs
#     filtered_diff = []
#     found_files = set()
#     for h, d in zip(headers, diffs):
#         for file_path in files_to_include:
#             if file_path in h:  # Check if file is in the header
#                 filtered_diff.append(h + d)
#                 found_files.add(file_path)
#                 break  # Avoid duplicate checks for the same file

#     return "\n".join(filtered_diff) if filtered_diff else None

def extract_diff(url, files_to_include, max_retries=5, initial_delay=2):
    """
    Extracts and filters the diff content from the commit page for Android Googlesource.
    Implements retry logic with exponential backoff for reliability.
    
    Args:
        url (str): The URL of the commit page.
        files_to_include (list): List of filenames to include in the diff.
        max_retries (int): Maximum number of retry attempts.
        initial_delay (int): Initial delay in seconds between retries.
    
    Returns:
        str or None: The filtered diff content, or None if extraction fails after all retries.
    """
    retry_count = 0
    delay = initial_delay
    
    while retry_count <= max_retries:
        try:
            if retry_count > 0:
                print(f"🔄 Retry attempt {retry_count}/{max_retries} for URL: {url}")
            
            response = requests.get(url, timeout=15)  # Increased timeout for reliability
            response.raise_for_status()
            
            # If request was successful, process the response
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

            # Check if we found anything
            if not filtered_diff:
                print(f"⚠️ No matching files found in diff at URL: {url}")
                print(f"   Looking for: {files_to_include}")
                print(f"   Found headers: {len(headers)}")
            
            return "\n".join(filtered_diff) if filtered_diff else None
            
        except requests.exceptions.RequestException as e:
            retry_count += 1
            
            # If we've reached the maximum retries, log the error and return None
            if retry_count > max_retries:
                print(f"❌ Failed to fetch diff after {max_retries} retries from URL: {url}")
                print(f"   Final error: {str(e)}")
                return None
            
            # Print information about the failure and retry
            print(f"⚠️ Error fetching diff (attempt {retry_count}/{max_retries}): {url}")
            print(f"   Error: {str(e)}")
            print(f"   Retrying in {delay} seconds...")
            
            # Wait with exponential backoff before the next retry
            time.sleep(delay)
            
            # Exponential backoff: double the delay for the next retry
            delay *= 2

def fetch_patch(commit_url, files_to_include, max_retries=5, initial_delay=2):
    # TODO: make sure to not fetch a diff file that is already fetched, to reduce redundances
    """
    Fetches the diff for a given commit URL, filters it to only include relevant files, and saves it.
    Implements retry logic for reliability.

    Args:
        commit_url (str): URL of the commit.
        files_to_include (list): List of files to include in the diff.
        max_retries (int): Maximum number of retry attempts.
        initial_delay (int): Initial delay in seconds between retries.
    
    Returns:
        str: The path to the saved formatted diff file, or None if fetching fails after all retries.
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

    # Check if we already have this diff file cached
    if os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
        print(f"✅ Using cached diff file: {output_filename}")
        return output_filename

    # Handle CodeLinaro URLs with retry logic
    if is_codelinaro:
        retry_count = 0
        delay = initial_delay
        
        while retry_count <= max_retries:
            try:
                if retry_count > 0:
                    print(f"🔄 Retry attempt {retry_count}/{max_retries} for CodeLinaro URL: {diff_url}")
                
                response = requests.get(diff_url, timeout=15)  # Increased timeout
                response.raise_for_status()
                
                # Save the raw diff
                with open(output_filename, "w", encoding="utf-8") as f:
                    f.write(response.text.strip() + "\n")
                
                print(f"✅ Successfully fetched and saved diff from CodeLinaro: {output_filename}")
                return output_filename
                
            except requests.exceptions.RequestException as e:
                retry_count += 1
                
                if retry_count > max_retries:
                    print(f"❌ Failed to fetch CodeLinaro diff after {max_retries} retries: {diff_url}")
                    print(f"   Final error: {str(e)}")
                    return None
                
                print(f"⚠️ Error fetching CodeLinaro diff (attempt {retry_count}/{max_retries}): {diff_url}")
                print(f"   Error: {str(e)}")
                print(f"   Retrying in {delay} seconds...")
                
                time.sleep(delay)
                delay *= 2
    
    # For Android Googlesource, use the extract_diff function which already has retry logic
    extracted_diff = extract_diff(diff_url, files_to_include, max_retries, initial_delay)
    if not extracted_diff:
        print(f"❌ Failed to extract diff from Googlesource after all retries: {diff_url}")
        return None

    # Save filtered diff
    with open(output_filename, "w", encoding="utf-8") as output_file:
        output_file.write(extracted_diff.strip() + "\n")
    
    print(f"✅ Successfully fetched, filtered and saved diff: {output_filename}")
    return output_filename
# def fetch_patch(commit_url, files_to_include):
#     """
#     Fetches the diff for a given commit URL, filters it to only include relevant files, and saves it.

#     Returns:
#         str: The path to the saved formatted diff file.
#     """

#     # Extract commit hash from the URL
#     commit_hash = extract_commit_hash(commit_url)
#     if not commit_hash:
#         return None

#     # Determine source type and construct the diff URL
#     if "android.googlesource.com" in commit_url:
#         diff_url = commit_url + "^!"  # Googlesource requires ^! for diff
#         is_codelinaro = False
#     elif "git.codelinaro.org" in commit_url:
#         diff_url = commit_url + ".diff"  # CodeLinaro requires .diff suffix
#         is_codelinaro = True
#     else:
#         print(f"⚠️ Unsupported commit URL: {commit_url}")
#         return None

#     output_dir_diff = "outputs/fetched_diffs"
#     os.makedirs(output_dir_diff, exist_ok=True)  
#     output_filename = os.path.join(output_dir_diff, f"{commit_hash}.diff") 

#     response = requests.get(diff_url)

#     # Save raw .diff for CodeLinaro
#     if is_codelinaro:
#         with open(output_filename, "w", encoding="utf-8") as f:
#             f.write(response.text.strip() + "\n") 
#         return output_filename

#     # Extract and format diff content for Android Googlesource
#     extracted_diff = extract_diff(diff_url, files_to_include)
#     if not extracted_diff:
#         return None

#     # Save filtered diff
#     with open(output_filename, "w", encoding="utf-8") as output_file:
#         output_file.write(extracted_diff.strip() + "\n")  

#     return output_filename

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

async def process_failed_patches_with_llm(failed_patches_path, kernel_path, current_path):
    """
    Process each failed patch through the LLM to resolve conflicts,
    and automatically update the kernel files with the resolved patches.
    
    Args:
        failed_patches_path (str): Path to the JSON file containing failed patches
        kernel_path (str): Path to the kernel repository
        current_path (str): Current working directory
        
    Returns:
        str: Path to the JSON file containing results of LLM processing
    """
    print(f"\n🤖 Starting LLM-based patch resolution with direct file updates...")
    
    # Load failed patches
    with open(failed_patches_path, 'r') as f:
        failed_patches_data = json.load(f)
    
    # Create output directories
    llm_output_dir = os.path.join(current_path, "outputs/llm_patched_files")
    os.makedirs(llm_output_dir, exist_ok=True)
    
    # Create output directory for agent results
    output_gpt_dir = os.path.join(current_path, "outputs/output_gpt_no_desc")
    os.makedirs(output_gpt_dir, exist_ok=True)
    
    # Track results
    llm_results = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "patches_processed": []
    }
    
    # Process each failed patch
    for patch in failed_patches_data.get("patch", []):
        patch_file = patch.get("patch_file")
        patch_url = patch.get("patch_url")
        
        print(f"\n📊 Processing patch: {patch_file}")
        print(f"   Source: {patch_url}")
        
        # Get full path to the patch file
        patch_diff_path = os.path.join(current_path, "outputs/fetched_diffs", patch_file)
        if not os.path.exists(patch_diff_path):
            print(f"⚠️ Patch file not found: {patch_diff_path}")
            # Try finding it in the current directory
            patch_diff_path = os.path.join(current_path, patch_file)
            if not os.path.exists(patch_diff_path):
                print(f"⚠️ Patch file not found in alternative location: {patch_diff_path}")
                continue
        
        patch_result = {
            "patch_file": patch_file,
            "patch_url": patch_url,
            "files_processed": []
        }
        
        # Process each rejected file
        for rejected_file_info in patch.get("rejected_files", []):
            source_file = rejected_file_info.get("failed_file")
            reject_file = rejected_file_info.get("reject_file")
            
            if not source_file or not reject_file:
                print(f"❌ Missing source or reject file information")
                continue
                
            source_file_path = os.path.join(kernel_path, source_file)
            
            # Check if files exist
            if not os.path.exists(source_file_path):
                print(f"❌ Source file not found: {source_file_path}")
                continue
                
            if not os.path.exists(reject_file):
                print(f"❌ Reject file not found: {reject_file}")
                continue
            
            print(f"📃 Processing source file: {source_file}")
            print(f"📃 Processing reject file: {reject_file}")
            
            # Use the actual source and patch files directly instead of copying them
            output_suffix = os.path.basename(source_file)
            
            # Call the LLM agent to resolve the patch
            try:
                print(f"🔄 Sending to LLM: {source_file}")
                
                # We need to set the working directory to the directory containing the input files
                original_dir = os.getcwd()
                os.chdir(current_path)
                
                # Run the agent with the actual file paths
                output_file = await run_patch_porter_agent(
                    patch_file_path=patch_diff_path,
                    vuln_file_path=source_file_path,
                    output_suffix=output_suffix
                )
                
                # Restore working directory
                os.chdir(original_dir)
                
                if output_file and os.path.exists(output_file):
                    # Read the patched file
                    with open(output_file, 'r', encoding='utf-8') as f:
                        patched_content = f.read()
                    
                    # Create a backup of the original file
                    backup_path = source_file_path + ".before_llm_patch"
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        with open(source_file_path, 'r', encoding='utf-8') as src:
                            f.write(src.read())
                    
                    # Update the original file with the patched content
                    with open(source_file_path, 'w', encoding='utf-8') as f:
                        f.write(patched_content)
                    
                    print(f"✅ Successfully patched file: {source_file_path}")
                    print(f"   (Backup saved at: {backup_path})")
                    
                    patch_result["files_processed"].append({
                        "file": source_file,
                        "status": "Success",
                        "backup_file": backup_path
                    })
                else:
                    print(f"⚠️ No output file returned from LLM agent")
                    patch_result["files_processed"].append({
                        "file": source_file,
                        "status": "Failed",
                        "reason": "No output from LLM agent"
                    })
            
            except Exception as e:
                print(f"❌ Error processing with LLM: {str(e)}")
                patch_result["files_processed"].append({
                    "file": source_file,
                    "status": "Error",
                    "reason": str(e)
                })
        
        llm_results["patches_processed"].append(patch_result)
    
    # Save results report
    results_path = os.path.join(current_path, "outputs/llm_results", f"llm_results_{llm_results['timestamp']}.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(llm_results, f, indent=4)
    
    print(f"\n📝 LLM processing results saved to: {results_path}")
    print(f"\n🔄 Total patches processed: {len(llm_results['patches_processed'])}")
    
    # Count successful and failed files
    success_count = 0
    failed_count = 0
    
    for patch in llm_results["patches_processed"]:
        for file in patch["files_processed"]:
            if file["status"] == "Success":
                success_count += 1
            else:
                failed_count += 1
    
    print(f"✅ Successfully patched files: {success_count}")
    print(f"❌ Failed to patch files: {failed_count}")
    
    return results_path

async def process_patches(parsed_report, patcher, current_path, report_output_path, kernel_path):
    """
    Process all patches in the report, apply them, and immediately fix rejections with LLM.
    
    Args:
        parsed_report: The parsed report containing patches to apply
        patcher: The PatchAdopter instance
        current_path: The current working directory path
        report_output_path: Path where reports should be saved
        kernel_path: Path to the kernel repository
    
    Returns:
        None
    """
    
    # Create output directory for LLM results
    llm_results_dir = os.path.join(current_path, "outputs/llm_results")
    os.makedirs(llm_results_dir, exist_ok=True)
    
    # Track all results
    llm_results = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "patches_processed": []
    }
    
    # Track success/failure counts
    success_count = 0
    failed_count = 0

    # Iterate through patches
    for patch in parsed_report["patches"]:
        patch_file_path = os.path.join(current_path, patch["patch_file"])  
        patch_url = patch["patch_url"]
        
        print(f"\n🔍 Attempting to apply patch: {patch_file_path}")

        # Apply the patch
        patch_result = patcher.apply_patch(patch_file_path, patch_url)
        patcher.patch_results["patches"].append(patch_result)
        
        # Create a patch result entry for LLM tracking
        llm_patch_result = {
            "patch_file": patch_result["patch_file"],
            "patch_url": patch_result["patch_url"],
            "status": patch_result["status"],
            "files_processed": []
        }

        # If patch was rejected (not already applied), use LLM to fix it
        if patch_result["status"] == "Rejected":
            print(f"📊 Patch was rejected. Attempting to fix with LLM.")
            
            # Save current directory and change to project root for LLM processing
            original_dir = os.getcwd()
            os.chdir(current_path)
            
            # Process each rejected file
            for rejected_file_info in patch_result["rejected_files"]:
                source_file = rejected_file_info.get("failed_file")
                reject_file = rejected_file_info.get("reject_file")
                
                if not source_file or not reject_file:
                    print(f"❌ Missing source or reject file information")
                    continue
                
                source_file_path = os.path.join(kernel_path, source_file)
                
                # Check if files exist
                if not os.path.exists(source_file_path):
                    print(f"❌ Source file not found: {source_file_path}")
                    continue
                    
                if not os.path.exists(reject_file):
                    print(f"❌ Reject file not found: {reject_file}")
                    continue
                
                print(f"📃 Processing source file: {source_file}")
                print(f"📃 Processing reject file: {reject_file}")
                
                # Call the LLM agent to resolve the patch
                try:
                    print(f"🔄 Sending to LLM: {source_file}")
                    
                    # Run the agent with the actual file paths
                    output_suffix = os.path.basename(source_file)
                    output_file = await run_patch_porter_agent(
                        patch_file_path=patch_file_path,
                        vuln_file_path=source_file_path,
                        output_suffix=output_suffix
                    )
                    
                    if output_file and os.path.exists(output_file):
                        # Read the patched file
                        with open(output_file, 'r', encoding='utf-8') as f:
                            patched_content = f.read()
                        
                        # Create a backup of the original file
                        backup_path = source_file_path + ".before_llm_patch"
                        with open(backup_path, 'w', encoding='utf-8') as f:
                            with open(source_file_path, 'r', encoding='utf-8') as src:
                                f.write(src.read())
                        
                        # Update the original file with the patched content
                        with open(source_file_path, 'w', encoding='utf-8') as f:
                            f.write(patched_content)
                        
                        print(f"✅ Successfully patched file: {source_file_path}")
                        print(f"   (Backup saved at: {backup_path})")
                        
                        llm_patch_result["files_processed"].append({
                            "file": source_file,
                            "status": "Success",
                            "backup_file": backup_path,
                            "patched_file": output_file
                        })
                        
                        success_count += 1
                    else:
                        print(f"⚠️ No output file returned from LLM agent")
                        llm_patch_result["files_processed"].append({
                            "file": source_file,
                            "status": "Failed",
                            "reason": "No output from LLM agent"
                        })
                        failed_count += 1
                
                except Exception as e:
                    print(f"❌ Error processing with LLM: {str(e)}")
                    llm_patch_result["files_processed"].append({
                        "file": source_file,
                        "status": "Error",
                        "reason": str(e)
                    })
                    failed_count += 1
            
            # Restore original directory
            os.chdir(original_dir)
        
        # Add this patch's result to the overall results
        llm_results["patches_processed"].append(llm_patch_result)

    # Save patch application report
    report_output_dir = os.path.dirname(report_output_path)
    os.makedirs(report_output_dir, exist_ok=True) 
    patcher.save_report()
    
    # Save LLM results report
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    llm_results_path = os.path.join(llm_results_dir, f"llm_results_{timestamp}.json")
    
    with open(llm_results_path, "w") as f:
        json.dump(llm_results, f, indent=4)
    
    # Print summary statistics
    print(f"\n📝 LLM processing results saved to: {llm_results_path}")
    print(f"\n🔄 Total patches processed: {len(llm_results['patches_processed'])}")
    print(f"✅ Successfully patched files: {success_count}")
    print(f"❌ Failed to patch files: {failed_count}")
    
    return llm_results_path

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
    
    # Process patches and apply LLM fixes immediately for rejections
    llm_results_path = asyncio.run(
        process_patches(
            parsed_report, 
            patcher, 
            current_path, 
            report_output_path,
            kernel_path
        )
    )
    
    # Create combined rejected hunks file for reference
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
    
    print(f"\n✅ Patch processing complete. Results saved to: {llm_results_path}")

if __name__ == "__main__":
    main()
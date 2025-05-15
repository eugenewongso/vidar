import json
import sys
import os
import requests
import re
import base64
import time
from pathlib import Path
from urllib.parse import urlparse

"""
Patch Fetcher Module

This module handles all aspects of downloading, filtering, and processing 
patch files from Git repositories (primarily Googlesource and CodeLinaro).

The module provides functionality for:
1. Extracting commit hashes from repository URLs
2. Handling rate limiting with exponential backoff
3. Downloading patches and processing their content
4. Filtering patches to include only relevant files
5. Batch processing of multiple patches from a Vanir report

Usage as a script:
    python patch_fetcher.py

Usage as a module:
    from patch_fetcher import fetch_patch, process_patches_from_report
    
    # To fetch a single patch
    patch_file = fetch_patch(commit_url, files_to_include)
    
    # To process all patches from a report
    results = process_patches_from_report()
"""

# --- UTILITY FUNCTIONS ---

def extract_commit_hash(commit_url):
    """
    Extracts the commit hash from a Googlesource or CodeLinaro URL.
    
    Parses the URL to identify the 40-character commit hash that uniquely
    identifies the patch in the repository.
    
    :param commit_url: URL to the commit in a Git repository
    :return: The commit hash extracted from the URL
    """
    path = urlparse(commit_url).path
    return path.split("/+/")[-1].rstrip("^!")

def get_with_backoff(url, retries=5):
    """
    Makes HTTP requests with exponential backoff for handling rate limiting.
    
    If a request fails with a 429 status (Too Many Requests), the function
    waits for an exponentially increasing amount of time before retrying.
    
    :param url: URL to request
    :param retries: Number of retry attempts
    :return: The HTTP response
    """
    for i in range(retries):
        response = requests.get(url)
        if response.status_code == 429:
            # Exponential backoff: wait 2^i seconds before retrying
            time.sleep(2 ** i)
            continue
        return response
    return response  # Return the last response even if all retries failed

# --- CORE PATCH FETCHING FUNCTIONALITY ---

def fetch_patch(commit_url, files_to_include):
    """
    Downloads and filters the diff for a given commit URL.
    
    This function:
    1. Extracts the commit hash from the URL
    2. Determines the repository type (Googlesource or CodeLinaro)
    3. Constructs the appropriate URL for downloading the patch
    4. Fetches the patch content
    5. Decodes it if necessary (Googlesource uses Base64 encoding)
    6. Filters the patch to include only the specified files
    7. Saves the filtered patch to disk
    
    :param commit_url: URL to the commit in a Git repository
    :param files_to_include: List of file paths to include in the filtered patch
    :return: Path to the saved patch file, or None if the operation failed
    """
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    commit_hash = extract_commit_hash(commit_url)
    if not commit_hash:
        print(f"⚠️ Could not extract commit hash from URL: {commit_url}")
        return None

    # Determine the repository type and construct the appropriate URL
    if "android.googlesource.com" in commit_url:
        diff_url = commit_url + "^!/?format=TEXT"  # Googlesource format for raw patch
        is_codelinaro = False
    elif "git.codelinaro.org" in commit_url:
        diff_url = commit_url + ".diff"  # CodeLinaro format for raw patch
        is_codelinaro = True
    else:
        print(f"⚠️ Unsupported commit URL: {commit_url}")
        return None

    print(f"🔍 Fetching diff from: {diff_url}")

    # Create output directory if it doesn't exist
    output_dir_diff = os.path.join(project_root, "..", "fetch_patch_output", "diff_output")
    os.makedirs(output_dir_diff, exist_ok=True)

    # Fetch the patch with backoff for rate limiting
    response = get_with_backoff(diff_url)

    if response.status_code != 200:
        print(f"❌ Failed to fetch diff for {commit_hash}. HTTP Status: {response.status_code}")
        return None

    output_filename = os.path.join(output_dir_diff, f"{commit_hash}.diff")

    # Handle CodeLinaro patches (directly save the text)
    if is_codelinaro:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(response.text.strip() + "\n")
        print(f"✅ CodeLinaro: Diff file saved as: {output_filename}")
        return output_filename

    # Handle Googlesource patches (decode from Base64)
    raw_diff = base64.b64decode(response.text)
    diff_text = raw_diff.decode("utf-8")

    # Normalize files_to_include to remove the first path segment (e.g., 'frameworks/', 'external/')
    normalized_files = [f.split("/")[-5:] for f in files_to_include]  # Use last 5 segments of the path

    filtered_diff = []
    current_file = None
    capture = False

    # Filter the patch to include only the specified files
    for line in diff_text.splitlines():
        if line.startswith("diff --git"):
            capture = False
            match = re.search(r" a/(.*?) ", line)
            if match:
                current_file = match.group(1)
                
            # Check if any file path matches
            for file_path in files_to_include:
                if file_path.endswith(current_file) or current_file.endswith(file_path):
                    capture = True
                    break

            if capture:
                filtered_diff.append(line)
        elif capture:
            filtered_diff.append(line)

    if not filtered_diff:
        print(f"❌ No matching diff content found for {commit_hash}")
        return None

    # Save the filtered patch to disk
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(filtered_diff).strip() + "\n")

    print(f"✅ Filtered diff file saved to: {output_filename}")
    return output_filename

# --- BATCH PROCESSING FUNCTIONALITY ---

def process_patches_from_report(report_path=None):
    """
    Processes all patches from a parsed Vanir report file.
    
    This function:
    1. Loads the parsed report data
    2. Processes each patch entry
    3. Tracks successful and failed operations
    4. Handles errors for individual patches
    5. Returns a summary of the operations
    
    :param report_path: Path to the parsed report JSON (default: auto-detect)
    :return: Dictionary with results of all patch fetching operations
    """
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Define path to parsed report if not provided
    if not report_path:
        report_path = os.path.join(project_root, "reports", "parsed_report.json")
    
    results = {
        "successful": [],
        "failed": []
    }
    
    # Load and validate the parsed report
    try:
        with open(report_path, "r") as f:
            parsed_report = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Error loading parsed report: {e}")
        return results

    print(f"📄 Processing patches from report: {report_path}")
    print(f"🔢 Total patches to process: {len(parsed_report.get('patches', []))}")

    # Process each patch in the report
    for patch in parsed_report.get("patches", []):
        patch_url = patch.get("patch_url")
        if not patch_url:
            continue
            
        files_to_include = list(patch.get("files", {}).keys())
        print(f"\n🔍 Processing patch: {patch_url}")
        print(f"   Filtering files: {files_to_include}")

        try:
            # Call fetch_patch to download and filter the patch
            diff_file = fetch_patch(patch_url, files_to_include)

            if diff_file:
                print(f"✅ Patch saved: {diff_file}")
                results["successful"].append({
                    "url": patch_url,
                    "file": diff_file,
                    "files_included": files_to_include
                })
            else:
                print(f"❌ Failed to fetch patch: {patch_url}")
                results["failed"].append({
                    "url": patch_url,
                    "reason": "Fetch failed",
                    "files_requested": files_to_include
                })

        except Exception as e:
            # Catch and report any exceptions, but continue processing other patches
            print(f"⚠️ Error processing {patch_url}: {e}")
            results["failed"].append({
                "url": patch_url,
                "reason": str(e),
                "files_requested": files_to_include
            })
    
    # Print a summary of the results
    print("\n===== Patch Fetching Summary =====")
    print(f"Total patches processed: {len(results['successful']) + len(results['failed'])}")
    print(f"Successfully fetched: {len(results['successful'])}")
    print(f"Failed to fetch: {len(results['failed'])}")
    
    return results

# --- MAIN ENTRY POINT ---

if __name__ == "__main__":
    """
    Main entry point for batch patch fetching.
    
    When executed directly, this script:
    1. Looks for the parsed Vanir report
    2. Processes all patches in the report
    3. Downloads and filters each patch
    4. Saves the patches to the output directory
    5. Provides a summary of operations
    
    Command line usage:
        python patch_fetcher.py
    """
    # Process all patches from the default report path
    results = process_patches_from_report()
    
    # Exit with an error code if any patches failed
    if len(results["failed"]) > 0:
        sys.exit(1)
    
    sys.exit(0)
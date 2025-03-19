import os
import requests
from bs4 import BeautifulSoup
import re

def extract_diff(url, files_to_include):
    # TODO: make this VCS agnostic
    """Extracts and filters the diff content from the commit page for Android Googlesource."""
    
    response = requests.get(url)
    # if response.status_code != 200:
    #     print(f"⚠️ Failed to fetch page: {url}")
    #     return None

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

    if not filtered_diff:  # Check if no files from the list were found in the diff
        # print(f"None of the specified files were found in the diff for {url}.")
        return None

    # Check for any specified files that were not found in the diffs
    # print("files_to_include", files_to_include)
    # print("found files", found_files)
    not_found_files = set(files_to_include) - found_files
    if not_found_files:
        print(f"The following specified files were not found in the diff: {', '.join(not_found_files)}")

    return "\n".join(filtered_diff) if filtered_diff else None

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

def fetch_patch(commit_url, files_to_include):
    """
    Fetches the diff for a given commit URL, filters it to only include relevant files, and saves it.

    Args:
        commit_url (str): The URL of the commit to fetch.
        files_to_include (list): List of file paths to include in the diff.

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

    # print(f"🔍 Fetching diff from: {diff_url}")

    # Define output directory
    output_dir_diff = "fetch_patch_output/diff_output"
    os.makedirs(output_dir_diff, exist_ok=True)

    # Fetch diff page
    response = requests.get(diff_url)

    # if response.status_code != 200:
    #     print(f"Failed to fetch diff for {commit_hash}. HTTP Status: {response.status_code}")
    #     return None

    # Set output filename using commit hash
    output_filename = os.path.join(output_dir_diff, f"{commit_hash}.diff")

    # Save raw .diff for CodeLinaro
    if is_codelinaro:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(response.text.strip() + "\n")  # Ensure exactly one empty line at the end
        # print(f"✅ CodeLinaro: Diff file saved as: {output_filename}")
        return output_filename

    # Extract and format diff content for Android Googlesource
    extracted_diff = extract_diff(diff_url, files_to_include)
    if not extracted_diff:
        # print(f"No matching diff content found for {commit_hash}")
        return None

    # Save filtered diff
    with open(output_filename, "w", encoding="utf-8") as output_file:
        output_file.write(extracted_diff.strip() + "\n")  # Ensure exactly one empty line at the end

    # print(f"Filtered diff file saved to: {output_filename}")
    return output_filename
import os
import requests
from bs4 import BeautifulSoup
import re

def extract_diff(url):
    """Extracts the diff content from the commit page for Android Googlesource."""
    
    response = requests.get(url)
    if response.status_code != 200:
        print(f"⚠️ Failed to fetch page: {url}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract diff content from the correct HTML structure
    diff_sections = soup.find_all("pre", class_="u-pre u-monospace Diff-unified")
    diffs = [section.get_text() for section in diff_sections]

    # Extract file headers
    file_headers = soup.find_all("pre", class_="u-pre u-monospace Diff")
    headers = [header.get_text() for header in file_headers]

    # Combine headers and diffs
    formatted_diff = "\n".join([h + d for h, d in zip(headers, diffs)])

    return formatted_diff

def fetch_patch(commit_url):
    """
    Fetches the diff for a given commit URL and saves it in a valid format.
    
    Args:
        commit_url (str): The URL of the commit to fetch.

    Returns:
        str: The path to the saved formatted diff file.
    """

    # Extract commit hash from URL
    commit_hash_match = re.search(r'/([a-f0-9]{40})$', commit_url)
    if not commit_hash_match:
        print(f"⚠️ Could not extract commit hash from URL: {commit_url}")
        return None

    commit_hash = commit_hash_match.group(1)

    # Determine source type
    if "android.googlesource.com" in commit_url:
        diff_url = commit_url + "^!"
        is_codelinaro = False
    elif "git.codelinaro.org" in commit_url:
        diff_url = commit_url + ".diff"
        is_codelinaro = True
    else:
        print(f"⚠️ Unsupported commit URL: {commit_url}")
        return None

    print(f"🔍 Fetching diff from: {diff_url}")

    # Define output directory
    output_dir_diff = "fetch_patch_output/diff_output"
    os.makedirs(output_dir_diff, exist_ok=True)

    # Fetch diff page
    response = requests.get(diff_url)

    if response.status_code != 200:
        print(f"❌ Failed to fetch diff for {commit_hash}. HTTP Status: {response.status_code}")
        return None

    # Save raw .diff for CodeLinaro
    if is_codelinaro:
        output_filename = os.path.join(output_dir_diff, f"{commit_hash}.diff")
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(response.text.strip() + "\n")  # Ensure exactly one empty line at the end
        print(f"✅ CodeLinaro: Diff file saved as: {output_filename}")
        return output_filename

    # Extract and format diff content for Android Googlesource
    extracted_diff = extract_diff(diff_url)
    if not extracted_diff:
        print(f"❌ Failed to extract diff content for {commit_hash}")
        return None

    # Save formatted diff
    output_filename = os.path.join(output_dir_diff, f"{commit_hash}.diff")
    with open(output_filename, "w", encoding="utf-8") as output_file:
        output_file.write(extracted_diff.strip() + "\n")  # Ensure exactly one empty line at the end

    print(f"✅ Formatted diff file saved to: {output_filename}")
    return output_filename

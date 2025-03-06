import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re

def extract_diff(url):
    # Fetch the webpage
    response = requests.get(url)
    
    if response.status_code != 200:
        print("Failed to fetch the page")
        return None

    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract diff content
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
    Fetches only the .diff file in a valid format that can be applied using `git apply patch.diff`.

    Args:
        commit_url (str): The URL of the commit to fetch.

    Returns:
        str: The path to the saved formatted diff file.
    """

    # Modify URL to fetch diff
    if "android.googlesource.com" in commit_url:
        diff_url = commit_url + "^!"
        is_codelinaro = False
    elif "git.codelinaro.org" in commit_url:
        diff_url = commit_url + ".diff"
        is_codelinaro = True
    else:
        raise ValueError("Unsupported commit URL.")

    print(f"Fetching diff from: {diff_url}")

    # Define output directory
    output_dir_diff = "fetch_patch_output/diff_output"
    os.makedirs(output_dir_diff, exist_ok=True)

    # Fetch diff page
    response = requests.get(diff_url)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if response.status_code != 200:
        raise Exception(f"Failed to fetch diff. HTTP Status: {response.status_code}")

    # **CodeLinaro Handling: Save raw .diff without parsing**
    if is_codelinaro:
        output_filename = os.path.join(output_dir_diff, f"{timestamp}.diff")
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(response.text.strip() + "\n")  # Ensure exactly one empty line at the end

        print(f"CodeLinaro: Raw diff file saved as: {output_filename}")
        return output_filename
    
    # **Android Googlesource Handling: Call extract_diff**
    extracted_diff = extract_diff(diff_url)
    if not extracted_diff:
        raise Exception("Failed to extract diff content from the page.")

    # Save the extracted and formatted diff
    output_filename = os.path.join(output_dir_diff, f"{timestamp}.diff")
    with open(output_filename, "w", encoding="utf-8") as output_file:
        output_file.write(extracted_diff.strip() + "\n")  # Ensure exactly one empty line at the end

    print(f"Formatted diff file saved to: {output_filename}")
    return output_filename

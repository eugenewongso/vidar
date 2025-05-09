import os
import requests
import re
import base64
import time
from urllib.parse import urlparse


def extract_commit_hash(commit_url):
    """Extracts the commit hash from a Googlesource or CodeLinaro URL."""
    path = urlparse(commit_url).path
    return path.split("/+/")[-1].rstrip("^!")

def get_with_backoff(url, retries=5):
    for i in range(retries):
        response = requests.get(url)
        if response.status_code == 429:
            time.sleep(2 ** i)
            continue
        return response
    return response

def fetch_patch(commit_url, files_to_include):
    """Fetches and filters the diff for a given commit URL."""
    commit_hash = extract_commit_hash(commit_url)
    if not commit_hash:
        print(f"⚠️ Could not extract commit hash from URL: {commit_url}")
        return None

    if "android.googlesource.com" in commit_url:
        diff_url = commit_url + "^!/?format=TEXT"
        is_codelinaro = False
    elif "git.codelinaro.org" in commit_url:
        diff_url = commit_url + ".diff"
        is_codelinaro = True
    else:
        print(f"⚠️ Unsupported commit URL: {commit_url}")
        return None

    print(f"🔍 Fetching diff from: {diff_url}")

    output_dir_diff = "fetch_patch_output/diff_output"
    os.makedirs(output_dir_diff, exist_ok=True)

    response = get_with_backoff(diff_url)

    if response.status_code != 200:
        print(f"❌ Failed to fetch diff for {commit_hash}. HTTP Status: {response.status_code}")
        return None

    output_filename = os.path.join(output_dir_diff, f"{commit_hash}.diff")

    if is_codelinaro:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(response.text.strip() + "\n")
        print(f"✅ CodeLinaro: Diff file saved as: {output_filename}")
        return output_filename

    raw_diff = base64.b64decode(response.text)
    diff_text = raw_diff.decode("utf-8")

    # Normalize files_to_include to remove the first path segment (e.g., 'frameworks/', 'external/')
    normalized_files = [f.split("/")[-5:] for f in files_to_include]  # Use last 5 segments of the path

    filtered_diff = []
    current_file = None
    capture = False

    for line in diff_text.splitlines():
        if line.startswith("diff --git"):
            capture = False
            match = re.search(r" a/(.*?) ", line)
            if match:
                current_file = match.group(1)
                
            # Fix: Use a simpler approach to check if any file path matches
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

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(filtered_diff).strip() + "\n")

    print(f"✅ Filtered diff file saved to: {output_filename}")
    return output_filename

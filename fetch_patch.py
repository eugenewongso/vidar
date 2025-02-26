import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re

def fetch_patch(commit_url):
    """
    Fetches a patch from a given commit URL, extracts metadata, and formats the diff content.

    Args:
        commit_url (str): The URL of the commit to fetch.

    Returns:
        str: The path to the saved formatted diff file.
    """

    # Modify URL to fetch diff
    if "android.googlesource.com" in commit_url:
        diff_url = commit_url + "^!"
    elif "git.codelinaro.org" in commit_url:
        diff_url = commit_url + ".diff"
    else:
        raise ValueError("Unsupported commit URL.")

    print(f"Fetching diff from: {diff_url}")

    # Define output directories
    output_dir_html = "fetch_patch_output/html_output"
    output_dir_diff = "fetch_patch_output/diff_output"

    os.makedirs(output_dir_html, exist_ok=True)
    os.makedirs(output_dir_diff, exist_ok=True)

    # Fetch diff page
    response = requests.get(diff_url)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")

        # Save raw HTML
        diff_filename_html = os.path.join(output_dir_html, f"{timestamp}.html")
        with open(diff_filename_html, "w", encoding="utf-8") as f:
            f.write(soup.prettify())
        print(f"Prettified HTML file saved as: {diff_filename_html}")

        # Extract commit information
        commit_metadata = {
            "commit_hash": "",
            "author": "",
            "author_email": "",
            "author_date": "",
            "committer": "",
            "committer_email": "",
            "committer_date": "",
            "parent": "",
            "commit_message": "",
            "bug": "",
            "test": "",
            "change_id": ""
        }

        # Extract text content for processing
        text_content = soup.get_text(separator="\n", strip=True)

        # Parse commit hash
        commit_match = re.search(r'commit\s+([a-f0-9]+)', text_content)
        if commit_match:
            commit_metadata["commit_hash"] = commit_match.group(1)

        # Parse author info
        author_match = re.search(r'author\s+(.*?)\s+<(.*?)>\s+(.*?)(?:-\d+)', text_content)
        if author_match:
            commit_metadata["author"] = author_match.group(1)
            commit_metadata["author_email"] = author_match.group(2)
            commit_metadata["author_date"] = author_match.group(3)

        # Parse committer info
        committer_match = re.search(r'committer\s+(.*?)\s+<(.*?)>\s+(.*?)(?:-\d+)', text_content)
        if committer_match:
            commit_metadata["committer"] = committer_match.group(1)
            commit_metadata["committer_email"] = committer_match.group(2)
            commit_metadata["committer_date"] = committer_match.group(3)

        # Parse parent commit
        parent_match = re.search(r'parent\s+([a-f0-9]+)', text_content)
        if parent_match:
            commit_metadata["parent"] = parent_match.group(1)

        # Parse commit message and metadata
        message_section = ""
        lines = text_content.split('\n')
        in_message = False

        for i, line in enumerate(lines):
            if "Change-Id:" in line:
                change_id_match = re.search(r'Change-Id:\s+(.*)', line)
                if change_id_match:
                    commit_metadata["change_id"] = change_id_match.group(1).strip()
                in_message = False
            elif "Bug:" in line:
                bug_match = re.search(r'Bug:\s+(.*)', line)
                if bug_match:
                    commit_metadata["bug"] = bug_match.group(1).strip()
            elif "Test:" in line:
                test_match = re.search(r'Test:\s+(.*)', line)
                if test_match:
                    commit_metadata["test"] = test_match.group(1).strip()
            elif "diff --git" in line:
                break
            elif in_message:
                message_section += line + "\n"
            elif "Update" in line and i > 10 and not in_message:
                in_message = True
                message_section = line + "\n"

        commit_metadata["commit_message"] = message_section.strip()

        # Format metadata in an appealing way
        formatted_metadata = [
            "==== COMMIT METADATA ====",
            f"Commit:     {commit_metadata['commit_hash']}",
            f"Parent:     {commit_metadata['parent']}",
            "",
            f"Author:     {commit_metadata['author']} <{commit_metadata['author_email']}>",
            f"Date:       {commit_metadata['author_date']}",
            "",
            f"Committer:  {commit_metadata['committer']} <{commit_metadata['committer_email']}>",
            f"Date:       {commit_metadata['committer_date']}",
            "",
            "==== COMMIT MESSAGE ====",
            commit_metadata["commit_message"],
            "",
            "==== ADDITIONAL INFO ====",
            f"Bug:        {commit_metadata['bug']}",
            f"Test:       {commit_metadata['test']}",
            f"Change-Id:  {commit_metadata['change_id']}",
            "",
        ]

        # Extract diff content
        diff_content = []
        in_diff = False
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            if line.startswith("diff --git"):
                in_diff = True
                if i + 2 < len(lines):  # Ensure we have at least two more lines for file paths
                    file_path_1 = lines[i + 1].strip()
                    file_path_2 = lines[i + 2].strip()
                    diff_content.append(f"diff --git {file_path_1} {file_path_2}")
                    i += 2  # Skip the next two lines
                else:
                    diff_content.append(line)  # Fallback if something is wrong

            elif in_diff:
                diff_content.append(line)

            i += 1

        # Combine metadata and diff
        formatted_output = "\n".join(formatted_metadata) + "\n" + "\n".join(diff_content)

        # Save formatted output
        output_filename = os.path.join(output_dir_diff, f"{timestamp}.diff")
        with open(output_filename, "w", encoding="utf-8") as output_file:
            output_file.write(formatted_output)

        print("Formatted content saved to:", output_filename)
        return output_filename  # Return the saved file path

    else:
        raise Exception(f"Failed to fetch diff. HTTP Status: {response.status_code}")

import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re
from unidiff import PatchSet

def validate_diff(diff_str):
    try:
        PatchSet(diff_str)
        return True, ""
    except Exception as e:
        return False, str(e)

def fetch_diff_from_url(diff_url, output_dir_html="Fetch_patch_output_html", output_dir_diff="Fetch_patch_output_diff"):
    """
    Fetches a diff from a given URL and parses the commit metadata.
    
    Args:
        diff_url (str): URL to the commit, such as 
                        "https://android.googlesource.com/platform/frameworks/base/+/cde345a7ee06db716e613e12a2c218ce248ad1c4"
        output_dir_html (str): Directory to save HTML output
        output_dir_diff (str): Directory to save formatted diff output
        
    Returns:
        tuple: (success, diff_content, metadata_dict, output_filename)
               - success (bool): Whether the operation was successful
               - diff_content (str): The raw diff content
               - metadata_dict (dict): Parsed commit metadata
               - output_filename (str): Path to the saved formatted diff file
    """
    # Format URL based on repository host
    original_url = diff_url
    if "android.googlesource.com" in diff_url:
        diff_url += "^!"
    elif "git.codelinaro.org" in diff_url:
        diff_url += ".diff"

    print(f"Fetching diff from: {diff_url}")

    # Create output directories if they don't exist
    os.makedirs(output_dir_html, exist_ok=True)
    os.makedirs(output_dir_diff, exist_ok=True)

    # Fetch diff page
    try:
        response = requests.get(diff_url)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if response.status_code != 200:
            return False, "", {}, f"Failed to fetch diff. HTTP Status: {response.status_code}"

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
            "change_id": "",
            "original_url": original_url
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
            elif "Update" in line and i > 10 and not in_message:  # Approximation for start of commit message
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
        raw_diff = []
        in_diff = False
        
        for line in lines:
            if in_diff:
                diff_content.append(line)
                raw_diff.append(line)
            elif "diff --git" in line:
                in_diff = True
                diff_content.append("==== DIFF CONTENT ====")
                diff_content.append(line)
                raw_diff.append(line)
        
        # Combine metadata and diff
        formatted_output = "\n".join(formatted_metadata) + "\n" + "\n".join(diff_content)
        raw_diff_content = "\n".join(raw_diff)
        
        # Save formatted output
        output_filename = os.path.join(output_dir_diff, f"{timestamp}.txt")
        with open(output_filename, "w", encoding="utf-8") as output_file:
            output_file.write(formatted_output)
        
        print("Formatted content saved to:", output_filename)
        
        # Validate the diff content
        is_valid, validation_error = validate_diff(raw_diff_content)
        if not is_valid:
            print(f"Warning: Diff validation failed: {validation_error}")
            commit_metadata["validation_error"] = validation_error
        
        return True, raw_diff_content, commit_metadata, output_filename
        
    except Exception as e:
        print(f"Error fetching diff: {str(e)}")
        return False, "", {}, str(e)

# Example usage
if __name__ == "__main__":
    url = "https://android.googlesource.com/platform/frameworks/base/+/cde345a7ee06db716e613e12a2c218ce248ad1c4"
    success, diff_content, metadata, output_file = fetch_diff_from_url(url)
    
    if success:
        print("Diff fetched successfully")
        print(f"Commit by: {metadata['author']}")
    else:
        print("Failed to fetch diff")
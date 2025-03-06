from fetch_patch import fetch_patch
from unidiff import PatchSet
from validate_diff import validate_diff
from call_gemini import patch_port_diff
from request_clean_patch import request_clean_patch

# commit_url = "https://android.googlesource.com/platform/frameworks/base/+/cde345a7ee06db716e613e12a2c218ce248ad1c4"
commit_url = "https://git.codelinaro.org/clo/la/kernel/msm-5.10/-/commit/c1a7b4b4a736fa175488122cca9743cff2ae72e8"
# commit_url = "https://android.googlesource.com/kernel/common/+/94c88f80ffddff00f0af448c02dfd8a3f3cdd692"
# commit_url = "https://android.googlesource.com/kernel/common/+/65e0a92c6d27d4cbaa0deef668df12b69853d65e"

diff_file_path = fetch_patch(commit_url) # only put in the url, followed by its commit hash; Do not add ^! or .diff at the end.

diff_file_path_string = "ori.diff"
with open(diff_file_path_string, "r", encoding="utf-8") as file:
    diff_content = file.read() # original diff content w/ conflicts

vulnerable_file_path = "kgsl_vbo.c"  
with open(vulnerable_file_path, "r", encoding="utf-8") as file:
    vulnerable_file_content = file.read() # vulnerable file content to read as context
    
error_message = """
    """

patched_diff = patch_port_diff(diff_content, vulnerable_file_content, error_message)
print(patched_diff)

patched_diff_file_path = "patched_output.diff"
with open(patched_diff_file_path, "w", encoding="utf-8") as file:
    file.write(patched_diff)

print("Patched diff saved to:", patched_diff_file_path)

# Validate the patched diff
is_valid, message = validate_diff(patched_diff_file_path)
print(f"Validation Result: {is_valid}, Message: {message}")
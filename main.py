from fetch_patch import fetch_patch
from unidiff import PatchSet
from validate_diff import validate_diff

if __name__ == "__main__":
    commit_url = "https://android.googlesource.com/platform/frameworks/base/+/cde345a7ee06db716e613e12a2c218ce248ad1c4"

    diff_file_path = fetch_patch(commit_url)

    is_valid, message = validate_diff(diff_file_path)
    print(f"Validation Result: {is_valid}, Message: {message}")

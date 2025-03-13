import json
import os

def load_patches(file_path):
    """Load patches from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return {patch["patch_url"]: patch for patch in data["patches"]}

def find_unique_patches(file1, file2):
    """Find patches that are unique to each file separately."""
    patches1 = load_patches(file1)
    patches2 = load_patches(file2)

    unique_in_file1 = {"patches": [patch for url, patch in patches1.items() if url not in patches2]}
    unique_in_file2 = {"patches": [patch for url, patch in patches2.items() if url not in patches1]}

    return unique_in_file1, unique_in_file2

if __name__ == "__main__":
    file1 = "2018-parsed-20250312215759.json"
    file2 = "2024-parsed-20250312222808.json"

    unique_in_file1, unique_in_file2 = find_unique_patches(file1, file2)

    # Extract base names without extensions
    base1 = os.path.splitext(file1)[0] 
    base2 = os.path.splitext(file2)[0]

    # Generate dynamic output filenames
    output_file1 = f"unique_in_{base1}.json"
    output_file2 = f"unique_in_{base2}.json"

    # Save outputs to separate files
    with open(output_file1, "w") as f:
        json.dump(unique_in_file1, f, indent=4)

    with open(output_file2, "w") as f:
        json.dump(unique_in_file2, f, indent=4)

    print(f"Unique patches in {file1} saved to {output_file1}")
    print(f"Unique patches in {file2} saved to {output_file2}")

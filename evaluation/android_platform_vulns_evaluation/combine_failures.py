import json

# File paths
file1_path = "reports/android_platform_vulnerability_report_2024_failures_2 copy.json"
file2_path = "reports/android_platform_vulnerability_report_2025_failures_3 copy.json"
output_path = "reports/20242025_combined_failures.json"

# Load JSON data from files
def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

# Save combined JSON data to a file
def save_json(data, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

# Main function to combine failures
def combine_failures():
    # Load data from both files
    data1 = load_json(file1_path)
    data2 = load_json(file2_path)

    # Combine failures
    combined_failures = data1.get("failures", []) + data2.get("failures", [])

    # Save the combined data
    save_json({"failures": combined_failures}, output_path)
    print(f"✅ Combined failures saved to {output_path}")

if __name__ == "__main__":
    combine_failures()
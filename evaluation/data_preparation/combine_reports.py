r"""Merges multiple JSON failure reports into a single consolidated report.

This script is a data preparation utility that takes one or more JSON files
containing lists of patch failures and combines them into a single file. This is
particularly useful for aggregating results from multiple test runs or across
different target environments (e.g., combining failure reports for Android 13
and Android 14 before filtering).

The process is as follows:
1.  Accepts a list of input JSON file paths and a single output file path via
    command-line arguments.
2.  Iterates through each input file and loads the JSON data.
3.  Extracts the list of failures from the `failures` key in each file.
4.  Concatenates all failure lists into a single master list.
5.  Saves a new JSON object with a single `failures` key containing the
    combined list to the specified output file.

Usage:
  python combine_reports.py \
      --inputs <report1.json> <report2.json> ... \
      --output <combined_report.json>
"""
import json
import argparse

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
r"""Enriches a filtered report with upstream patch data from a reference report.

This script performs a data enrichment task by merging information from two
separate reports. It takes a primary report (e.g., a filtered list of failures
for a specific Android version) and enriches it with detailed upstream patch
information (like the full patch content and commit hashes) from a more
comprehensive reference report (e.g., a combined report of all failures).

The process is as follows:
1.  Loads the main data file and the reference data file.
2.  Creates a fast lookup dictionary from the reference report, keyed by the
    vulnerability ID.
3.  Iterates through each entry in the main data file.
4.  Uses the vulnerability ID to find the corresponding entry in the reference
    report.
5.  If a match is found, it copies the `upstream_patch_content`,
    `upstream_commits`, and `upstream_patch_tokens` fields from the reference
    entry into the main entry.
6.  Saves the newly enriched data to a specified output file.

Usage:
  python extract_upstream_patch_data.py \
      --main-file <path_to_filtered_report.json> \
      --reference-file <path_to_comprehensive_report.json> \
      --output-file <path_to_enriched_report.json>
"""
import json
import argparse

# File paths
# Define the input and output file paths
main_file = "run_results/android_14/2024/android_14_2024_filtered_failures_with_ground_truth copy.json"  # Main data file
reference_file = "reports/android_platform_vulnerability_report_2024_failures_2 copy.json"  # Reference data file
output_file = "android_14_2024_filtered_failures_with_ground_truth_and_upstream_patch.json"  # Output file to save enriched data

# Load the main data file into a Python object
with open(main_file, "r") as f:
    main_data = json.load(f)

# Load the reference data file into a Python object
with open(reference_file, "r") as f:
    reference_data = json.load(f)

# Create a dictionary to quickly look up reference data by their 'id'
ref_lookup = {entry["id"]: entry for entry in reference_data}

# Enrich each main entry
# Iterate through each entry in the main data
for entry in main_data:
    entry_id = entry.get("id")  # Get the 'id' of the current entry
    ref = ref_lookup.get(entry_id)  # Find the corresponding reference entry by 'id'

    if ref:
        # Extract relevant fields from the reference entry
        upstream_patch_content = ref.get("upstream_patch_content")
        upstream_commits = ref.get("upstream_commits")
        upstream_patch_tokens = ref.get("upstream_patch_tokens")

        # If all required fields are present, enrich the main entry
        if upstream_patch_content and upstream_commits and upstream_patch_tokens:
            # Insert the new fields right after the 'severity' field
            new_entry = {}
            for key, value in entry.items():
                new_entry[key] = value
                if key == "severity":
                    # Add the new fields from the reference data
                    new_entry["upstream_patch_content"] = upstream_patch_content
                    new_entry["upstream_commits"] = upstream_commits
                    new_entry["upstream_patch_tokens"] = upstream_patch_tokens
            # Update the original entry with the enriched data
            entry.clear()
            entry.update(new_entry)

# Save the enriched file
# Write the enriched data back to the output file in JSON format
with open(output_file, "w") as f:
    json.dump(main_data, f, indent=2)

# Print a success message
print(f"✅ Enriched file saved to: {output_file}")

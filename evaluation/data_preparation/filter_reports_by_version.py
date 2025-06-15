r"""Filters a JSON failure report to include only entries for a specific version.

This script is a data preparation utility used to narrow down a comprehensive
failure report (e.g., one combining failures from multiple Android releases)
into a smaller report containing only the failures relevant to a single,
specific downstream version.

The process is as follows:
1.  Loads a JSON report containing a list of vulnerability failures.
2.  Accepts a target version number (e.g., "14") from the command line.
3.  Iterates through each vulnerability entry in the report.
4.  Within each entry, it inspects the list of individual patch failures and
    retains only those whose `downstream_version` field matches the target
    version.
5.  If any matching failures are found for a vulnerability, a new entry is
    created containing the original vulnerability metadata and the filtered
    list of failures.
6.  If an output path is provided, it saves the filtered list of vulnerabilities
    to a new file. The output format can be JSON or CSV.

Usage:
  python filter_reports_by_version.py \
      --file <path_to_combined_report.json> \
      --version <version_number> \
      --output <path_to_filtered_report.json>
"""
import json
import argparse
import pandas as pd

# Argument parsing
parser = argparse.ArgumentParser(description="Filter failures by downstream version")
parser.add_argument("--file", required=True, help="Path to the failures JSON file")
parser.add_argument("--version", required=True, help="Downstream version to filter for (e.g., '14')")
parser.add_argument("--output", help="Optional path to save the filtered results (JSON or CSV)")
args = parser.parse_args()

# Load data
with open(args.file, "r") as f:
    data = json.load(f)

# Filter entries
filtered = []
for entry in data.get("failures", []):
    matching_failures = [f for f in entry.get("failures", []) if f.get("downstream_version") == args.version]
    if matching_failures:
        new_entry = {
            "id": entry.get("id"),
            "vulnerability_url": entry.get("vulnerability_url"),
            "severity": entry.get("severity"),
            "failures": matching_failures
        }
        filtered.append(new_entry)

# Save output if requested
if args.output:
    if args.output.endswith(".json"):
        with open(args.output, "w") as f:
            json.dump(filtered, f, indent=2)
        print(f"💾 Saved filtered data to {args.output}")
    elif args.output.endswith(".csv"):
        # Flatten for CSV output
        df_flat = pd.json_normalize(filtered, 'failures', ['id', 'vulnerability_url', 'severity'])
        df_flat.to_csv(args.output, index=False)
        print(f"💾 Saved filtered CSV to {args.output}")
    else:
        print("⚠️ Unsupported output format. Use .json or .csv")

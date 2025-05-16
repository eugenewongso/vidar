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

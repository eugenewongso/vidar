import json
import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

# Load the JSON report from the specified file path
def load_report(path):
    with open(path, "r") as f:
        return json.load(f)

# Build a matrix to track patch success, failure, and error counts between versions
def build_cross_patch_matrix(report_data):
    matrix = defaultdict(lambda: defaultdict(lambda: {"success": 0, "fail": 0, "error": 0}))
    for entry in report_data:
        # Process main → downstream patch attempts
        for patch_summary in entry.get("patch_attempts", []):
            for attempt in patch_summary.get("patch_results", []):
                src = "main"  # Source version is always "main"
                tgt = attempt.get("downstream_version")  # Target version
                status = attempt.get("result")  # Result of the patch attempt
                if not tgt or not status:
                    continue
                if status == "success":
                    matrix[src][tgt]["success"] += 1
                elif status == "failure":
                    matrix[src][tgt]["fail"] += 1
                else:
                    matrix[src][tgt]["error"] += 1

        # Process cross-version patch attempts
        for result in entry.get("cross_patch_attempts", []):
            src = result.get("from")  # Source version
            tgt = result.get("to")  # Target version
            status = result.get("result")  # Result of the patch attempt
            if not src or not tgt:
                continue
            if status == "success":
                matrix[src][tgt]["success"] += 1
            elif status == "failure":
                matrix[src][tgt]["fail"] += 1
            else:
                matrix[src][tgt]["error"] += 1
    return matrix

# Convert the matrix into a DataFrame and pivot table for analysis and visualization
def matrix_to_dataframe(matrix):
    rows = []
    for src, tgts in matrix.items():
        for tgt, stats in tgts.items():
            total = stats["success"] + stats["fail"] + stats["error"]
            if total == 0:
                continue
            success_rate = round(stats["success"] / total * 100, 2)  # Calculate success rate
            rows.append({
                "From": src,
                "To": tgt,
                "Success Rate (%)": success_rate,
                "Successes": stats["success"],
                "Failures": stats["fail"],
                "Errors": stats["error"],
                "Total Attempts": total
            })
    df = pd.DataFrame(rows)  # Create a DataFrame from the rows
    pivot = df.pivot(index="From", columns="To", values="Success Rate (%)").fillna("-")  # Create a pivot table
    return df, pivot

# Save a heatmap visualization of the patch success rates
def save_heatmap(pivot, out_file, raw_counts):
    plt.figure(figsize=(10, 8))

    # Convert data to float for coloring
    annotated_data = pivot.replace("-", 0)
    annotated_data = annotated_data.applymap(
        lambda x: float(x) if isinstance(x, (int, float)) or str(x).replace(".", "", 1).isdigit() else 0
    )

    # Create annotation labels with success rate and total attempts
    annotations = annotated_data.astype("object")
    for i in annotations.index:
        for j in annotations.columns:
            rate = annotated_data.loc[i, j]
            total = raw_counts.get((i, j), 0)
            annotations.loc[i, j] = f"{rate:.1f}%\n({total})"

    # Plot the heatmap
    ax = sns.heatmap(
        annotated_data.astype(float),
        annot=annotations,
        fmt="",
        cmap="RdYlGn",
        cbar_kws={'label': 'Success Rate (%)'}
    )
    ax.invert_xaxis()  # Higher versions appear on the left
    plt.title("Patch Success Rate + Total Attempts")
    plt.xlabel("To Version")
    plt.ylabel("From Version")
    plt.tight_layout()
    plt.savefig(out_file)  # Save the heatmap to the specified file
    print(f"✅ Heatmap saved to {out_file}")

# Main function to parse arguments, process data, and generate outputs
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=str, required=True, help="Path to the JSON report file")
    parser.add_argument("--csv", type=str, help="Path to save the CSV output")
    parser.add_argument("--heatmap", type=str, help="Path to save the heatmap image")
    args = parser.parse_args()

    # Load the report data
    data = load_report(args.report)
    relevant_groups = [
        "vulnerabilities_with_all_failures",
        "vulnerabilities_with_partial_failures",
        "vulnerabilities_with_all_successful_patches"
    ]

    # Combine all relevant entries from the report
    all_entries = []
    for group in relevant_groups:
        all_entries.extend(data.get(group, []))

    # Build the patch success matrix
    matrix = build_cross_patch_matrix(all_entries)

    # Convert the matrix to a DataFrame and pivot table
    df, pivot = matrix_to_dataframe(matrix)

    print("\n📋 Cross-Version Patch Matrix:")
    print(pivot)

    # Save the DataFrame to a CSV file if specified
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"📄 CSV saved to {args.csv}")

    # Build a lookup for raw counts of total attempts
    raw_counts = {(row["From"], row["To"]): row["Total Attempts"] for row in df.to_dict("records")}

    # Save the heatmap visualization if specified
    if args.heatmap:
        total_attempts = len(df)
        unique_from_versions = df["From"].nunique()
        unique_to_versions = df["To"].nunique()
        print(f"\n📊 Total patch attempts: {total_attempts}")
        print(f"🔁 Unique 'From' versions: {unique_from_versions}")
        print(f"➡️ Unique 'To' versions: {unique_to_versions}")

        save_heatmap(pivot, args.heatmap, raw_counts)

# Entry point of the script
if __name__ == "__main__":
    main()

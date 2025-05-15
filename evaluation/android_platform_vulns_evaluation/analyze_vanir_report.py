import os
import json
import argparse
import urllib.request
import re

def extract_asb_ids(data):
    """Extract ASB IDs from the missing_patches field of the Vanir report"""
    asb_ids = set()
    missing_patches = data.get("missing_patches", [])
    for patch in missing_patches:
        if "ID" in patch and patch["ID"].startswith("ASB-A-"):
            asb_ids.add(patch["ID"])
    return asb_ids

def download_and_save_json(asb_id, save_path):
    url = f"https://api.osv.dev/v1/vulns/{asb_id}"
    try:
        with urllib.request.urlopen(url) as response:
            if response.status == 200:
                content = response.read().decode()
                with open(save_path, "w") as f:
                    f.write(content)
                return True
            else:
                return False
    except Exception as e:
        return False

def extract_cves_from_asb_files(directory):
    cves = set()
    pattern = re.compile(r"CVE-\d{4}-\d+")
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            try:
                with open(os.path.join(directory, filename), "r") as f:
                    data = json.load(f)
                    aliases = data.get("aliases", [])
                    for alias in aliases:
                        if pattern.match(alias):
                            cves.add(alias)
            except Exception:
                continue  # Skip any bad/malformed JSONs
    return cves


def analyze_vanir_report(report_path, osv_dir="osv_data_android"):
    with open(report_path, "r") as f:
        report = json.load(f)

    covered_cves = report.get("covered_cves", [])
    total_cves = len(covered_cves)
    asb_ids_in_report = extract_asb_ids(report)

    os.makedirs(osv_dir, exist_ok=True)
    osv_files = os.listdir(osv_dir)
    osv_asb_ids = {filename.replace(".json", "") for filename in osv_files if filename.endswith(".json")}

    unmatched_asb_ids = sorted(asb_ids_in_report - osv_asb_ids)
    for asb_id in unmatched_asb_ids:
        save_path = os.path.join(osv_dir, f"{asb_id}.json")
        download_and_save_json(asb_id, save_path)

    return sorted(asb_ids_in_report)  # ✅ Return ASBs to be used in vanir_runner


def main():
    parser = argparse.ArgumentParser(description="Analyze Vanir report and sync osv_data_android")
    parser.add_argument('--report', required=True, help="Path to Vanir JSON report")
    parser.add_argument('--osv_dir', default="osv_data_android", help="Directory containing ASB JSONs")
    parser.add_argument('--output', default="vanir_asb_ids.txt", help="Path to save ASB ID list")
    args = parser.parse_args()

    # Step 1: Analyze report and get ASB IDs
    asb_ids = analyze_vanir_report(args.report, args.osv_dir)

    # Step 2: Print ASB IDs
    print(f"\n📋 Identified {len(asb_ids)} ASB IDs:\n")
    for asb_id in asb_ids:
        print(f"- {asb_id}")

    # Step 3: Save ASB IDs to file
    with open(args.output, "w") as f:
        for asb_id in asb_ids:
            f.write(f"{asb_id}\n")

    print(f"\n✅ Saved to: {args.output}")

if __name__ == "__main__":
    main()

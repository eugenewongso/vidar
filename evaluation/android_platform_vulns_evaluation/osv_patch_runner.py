import json
import os
import requests
import argparse
from android_patch_manager import AndroidPatchManager
from datetime import datetime
import time
from collections import OrderedDict


DEFAULT_REPO_BASE = "android_repos"

def fetch_with_retry(url, method="GET", json_data=None, retries=3, delay=2):
    """
    Fetch a URL with retry logic for handling transient failures.

    Args:
        url (str): The URL to fetch.
        method (str): HTTP method to use ("GET" or "POST").
        json_data (dict, optional): JSON data to send with a POST request.
        retries (int): Number of retry attempts.
        delay (int): Delay in seconds between retries.

    Returns:
        requests.Response or None: The response object if successful, otherwise None.
    """
    for attempt in range(retries):
        try:
            if method == "POST":
                response = requests.post(url, json=json_data)
            else:
                response = requests.get(url)
                
            if response.status_code == 200:
                return response

            print(f"⚠️ Attempt {attempt + 1} failed with status {response.status_code}, retrying in {delay}s...")
        except requests.RequestException as e:
            print(f"⚠️ Attempt {attempt + 1} failed with error: {e}, retrying in {delay}s...")
        time.sleep(delay)
    print(f"❌ Failed to fetch {url} after {retries} attempts.")
    return None


def retry_process_vulnerability(vuln, repo_base, retries=2, allowed_downstream_versions=None):
    """
    Retry processing a vulnerability multiple times in case of failure.

    Args:
        vuln (dict): Vulnerability data to process.
        repo_base (str): Base directory for repositories.
        retries (int): Number of retry attempts.

    Returns:
        dict: Result of the vulnerability processing or an error message if retries are exhausted.
    """
    for attempt in range(retries):
        try:
            result = AndroidPatchManager.process_vulnerability(vuln, repo_base=repo_base, allowed_downstream_versions=allowed_downstream_versions)
            return result
        except Exception as e:
            print(f"⚠️ Processing failed for {vuln['id']} on attempt {attempt + 1}: {e}")
            time.sleep(2)
    print(f"❌ Skipping {vuln['id']} after {retries} failed attempts.")
    return {"vulnerability_url": f"https://api.osv.dev/v1/vulns/{vuln['id']}", "skipped": True, "error": "Retries exhausted"}


def load_local_android_vulns(data_dir, after_date=None, before_date=None):
    """
    Load Android vulnerabilities from local JSON files, applying filters.

    Args:
        data_dir (str): Directory containing vulnerability JSON files.
        after_date (datetime, optional): Include vulnerabilities published after this date.
        before_date (datetime, optional): Include vulnerabilities published before this date.

    Returns:
        list: Filtered list of vulnerabilities.
    """
    vulns = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            with open(os.path.join(data_dir, filename), "r") as f:
                vuln = json.load(f)

                for affected in vuln.get("affected", []):
                    package_name = affected.get("package", {}).get("name", "")
                    if not package_name.startswith("platform/"):
                        # Skip non-platform packages
                        continue

                    # Filter SPL to only those ending in "-01"
                    spl = affected.get("ecosystem_specific", {}).get("spl", "")
                    if not spl.endswith("-01"):
                        continue

                    # Check date filters
                    pub_date = datetime.fromisoformat(vuln['published'].replace("Z", "")).replace(tzinfo=None)
                    if after_date and pub_date < after_date:
                        continue
                    if before_date and pub_date > before_date:
                        continue

                    # Passed all filters, include this vulnerability
                    vulns.append(vuln)
                    break  # Move to the next vulnerability after finding a valid affected entry
    return vulns



def load_existing_report(report_path):
    """
    Load an existing report from a JSON file or initialize a new report structure.

    Args:
        report_path (str): Path to the report JSON file.

    Returns:
        dict: Report data, either loaded from the file or initialized.
    """
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            return json.load(f)
    return {
        "summary": {
            "total_vulnerabilities_tested": 0,  # Total number of vulnerabilities processed
            "total_downstream_versions_tested": 0,  # Total downstream versions tested
            "total_failed_patches": 0,  # Total number of failed patches
            "total_unique_downstream_versions_tested": 0,  # Unique downstream versions tested
            "total_unique_downstream_failed_patches": 0,  # Unique downstream versions with failed patches
            "vulnerabilities_with_all_failures": 0,  # Vulnerabilities where all patches failed
            "vulnerabilities_with_partial_failures": 0,  # Vulnerabilities with mixed patch results
            "vulnerabilities_with_all_successful_patches": 0,  # Vulnerabilities where all patches succeeded
            "vulnerabilities_skipped": 0,  # Vulnerabilities skipped during processing
            "vulnerabilities_with_commit_mismatch": 0,  # Vulnerabilities with mismatched commit counts
            "per_version_stats": {},  # Statistics per downstream version
            "total_tokens": {  # Token usage statistics
                "upstream_patch": {"gemini": 0, "openai": 0, "general_word": 0, "general_char": 0},
                "upstream_source": {"gemini": 0, "openai": 0, "general_word": 0, "general_char": 0},
                "downstream_patch": {"gemini": 0, "openai": 0, "general_word": 0, "general_char": 0},
                "downstream_source": {"gemini": 0, "openai": 0, "general_word": 0, "general_char": 0}
            }
        },
        "vulnerabilities_with_all_failures": [],  # List of vulnerabilities with all failed patches
        "vulnerabilities_with_partial_failures": [],  # List of vulnerabilities with mixed patch results
        "vulnerabilities_with_all_successful_patches": [],  # List of vulnerabilities with all successful patches
        "vulnerabilities_skipped": [],  # List of skipped vulnerabilities
        "commit_mismatch_vulnerabilities": []  # List of vulnerabilities with commit mismatches
    }


def compute_total_gemini_tokens(failure_entry):
    """
    Compute the total number of Gemini tokens used for a failure entry.

    Args:
        failure_entry (dict): A failure entry containing token data.

    Returns:
        int: Total number of Gemini tokens used.
    """
    total = 0
    total += failure_entry.get("upstream_patch_tokens", {}).get("gemini", 0)

    for result in failure_entry.get("failures", []):
        total += result.get("downstream_patch_tokens", {}).get("gemini", 0)

        for fc in result.get("file_conflicts", []):
            total += fc.get("upstream_file_tokens", {}).get("gemini", 0)
            total += fc.get("downstream_file_tokens", {}).get("gemini", 0)
            total += fc.get("rej_file_tokens", {}).get("gemini", 0)
            total += fc.get("inline_merge_token_summary", {}).get("gemini", 0)

    return total

def update_token_totals(summary_section: dict, token_data: dict):
    """
    Update a summary section with values from token_data.

    Args:
        summary_section (dict): The section of summary["total_tokens"] to update.
        token_data (dict): The token counts to add.
    """
    summary_section["gemini"] += token_data.get("gemini", 0)
    summary_section["openai"] += token_data.get("openai", 0)
    general = token_data.get("general", {})
    summary_section["general_word"] += general.get("word_based", 0)
    summary_section["general_char"] += general.get("char_based", 0)



def save_report(report_data, report_path, failures=None, strip_sensitive=True):
    """
    Save the report data to a JSON file, optionally stripping sensitive information.

    Args:
        report_data (dict): The report data to save.
        report_path (str): Path to the report JSON file.
        failures (list, optional): List of failure entries to save in a separate file.
        strip_sensitive (bool): Whether to strip sensitive information from the main report.
    """
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    import copy
    # Only strip in the main report — not the failures report
    report_to_save = copy.deepcopy(report_data)

    if strip_sensitive:
        # Strip upstream_file_content only from main report
        for vuln_group in ["vulnerabilities_with_partial_failures", "vulnerabilities_with_all_failures"]:
            for patch_attempt in report_to_save.get(vuln_group, []):
                for attempt in patch_attempt.get("patch_attempts", []):
                    for result in attempt.get("patch_results", []):
                        if "file_conflicts" in result:
                            for fc in result["file_conflicts"]:
                                fc.pop("upstream_file_content", None)

    # Save main report (possibly stripped)
    with open(report_path, "w") as f:
        json.dump(report_to_save, f, indent=4)
    print(f"💾 Report saved: {report_path}")

    if failures:
        failures_report_path = report_path.replace(".json", "_failures.json")

        # Compute and attach total gemini token usage for sorting
        for entry in failures:
            total_gemini = compute_total_gemini_tokens(entry)
            entry["_total_gemini_tokens"] = total_gemini  # for internal sorting


        failures.sort(key=lambda x: x["_total_gemini_tokens"], reverse=True)

        for i, entry in enumerate(failures):
            total_gemini = compute_total_gemini_tokens(entry)

            ordered_entry = OrderedDict()
            ordered_entry["id"] = entry.get("id")
            ordered_entry["total_gemini_token_usage"] = total_gemini
            ordered_entry["vulnerability_url"] = entry.get("vulnerability_url")
            ordered_entry["severity"] = entry.get("severity", "Unknown")
            ordered_entry["upstream_patch_content"] = entry.get("upstream_patch_content", "")
            ordered_entry["upstream_commits"] = entry.get("upstream_commits", [])
            ordered_entry["upstream_patch_tokens"] = entry.get("upstream_patch_tokens", {})

            # Reorder each downstream failure result
            ordered_failures = []
            for result in entry.get("failures", []):
                # Compute per-version gemini token usage
                usage = result.get("downstream_patch_tokens", {}).get("gemini", 0)
                for fc in result.get("file_conflicts", []):
                    usage += fc.get("upstream_file_tokens", {}).get("gemini", 0)
                    usage += fc.get("downstream_file_tokens", {}).get("gemini", 0)
                    usage += fc.get("rej_file_tokens", {}).get("gemini", 0)
                    usage += fc.get("inline_merge_token_summary", {}).get("gemini", 0)

                # Move gemini_token_usage to top
                reordered = OrderedDict()
                reordered["downstream_version"] = result.get("downstream_version")
                reordered["gemini_token_usage"] = usage
                for key, value in result.items():
                    if key not in reordered:
                        reordered[key] = value
                ordered_failures.append(reordered)

            # Sort by usage
            ordered_failures.sort(key=lambda r: r["gemini_token_usage"], reverse=True)
            ordered_entry["failures"] = ordered_failures
            ordered_entry["_total_gemini_tokens"] = total_gemini
            failures[i] = ordered_entry


        # Now sort all failures by overall total Gemini token usage
        failures.sort(key=lambda x: x["_total_gemini_tokens"])

        for entry in failures:
            entry.pop("_total_gemini_tokens", None)

        # Initialize accumulator
        total = {
            "upstream_patch": {"gemini": 0, "openai": 0, "general_word": 0, "general_char": 0},
            "downstream_patch": {"gemini": 0, "openai": 0, "general_word": 0, "general_char": 0},
            "upstream_source": {"gemini": 0, "openai": 0, "general_word": 0, "general_char": 0},
            "downstream_source": {"gemini": 0, "openai": 0, "general_word": 0, "general_char": 0},
            "rej_file": {"gemini": 0, "openai": 0, "general_word": 0, "general_char": 0},
            "inline_merge_conflict": {"gemini": 0, "openai": 0, "general_word": 0, "general_char": 0},
        }

        count = 0  # number of downstream versions (i.e., patch results)

        for entry in failures:
            # Top-level upstream patch tokens
            u_tokens = entry.get("upstream_patch_tokens", {})
            update_token_totals(total["upstream_patch"], u_tokens)

            for result in entry.get("failures", []):
                count += 1  # Count per patch result

                # Downstream patch
                d_tokens = result.get("downstream_patch_tokens", {})
                update_token_totals(total["downstream_patch"], d_tokens)

                for fc in result.get("file_conflicts", []):
                    for side_key, section in [
                        ("upstream_file_tokens", "upstream_source"),
                        ("downstream_file_tokens", "downstream_source"),
                        ("rej_file_tokens", "rej_file"),
                    ]:
                        tokens = fc.get(side_key, {})
                        update_token_totals(total[section], tokens)

                    # Inline merge conflicts
                    ims = fc.get("inline_merge_token_summary", {})
                    for k in ["gemini", "openai", "general_word", "general_char"]:
                        total["inline_merge_conflict"][k] += ims.get(k, 0)

        def avg_tokens(section):
            return {
                k: round(v / count, 2) if count else 0
                for k, v in section.items()
            }

        failure_summary = {
            "total_downstream_patch_failures": count,
            "average_tokens_per_downstream_version": {
                section: avg_tokens(data)
                for section, data in total.items()
            }
        }

        # Save final output
        failures_output = {
            "summary": failure_summary,
            "failures": failures
        }

        with open(failures_report_path, "w") as f:
            json.dump(failures_output, f, indent=4)
        print(f"💾 Failures-only report saved: {failures_report_path}")

# Fill in missing Gemini token counts for upstream patches and downstream patches
def fill_in_gemini_tokens(report_data):
    """
    Fill in missing Gemini token counts for upstream and downstream patches.

    Args:
        report_data (dict): The report data containing vulnerabilities and patch attempts.
    """
    for group in [
        "vulnerabilities_with_all_failures",
        "vulnerabilities_with_partial_failures",
        "vulnerabilities_with_all_successful_patches",
    ]:
        for entry in report_data.get(group, []):
            for attempt in entry.get("patch_attempts", []):
                # Upstream patch
                if "upstream_patch_tokens" in attempt and "gemini" not in attempt["upstream_patch_tokens"]:
                    content = attempt.get("upstream_patch_content", "")
                    attempt["upstream_patch_tokens"]["gemini"] = AndroidPatchManager.count_tokens_gemini(content, "neat-resolver-406722")

                for result in attempt.get("patch_results", []):
                    # Downstream patch
                    if "downstream_patch_tokens" in result and "gemini" not in result["downstream_patch_tokens"]:
                        content = result.get("downstream_patch_content", "")
                        result["downstream_patch_tokens"]["gemini"] = AndroidPatchManager.count_tokens_gemini(content, "neat-resolver-406722")

                    for fc in result.get("file_conflicts", []):
                        for side in ["upstream_file", "downstream_file", "rej_file"]:
                            key = f"{side}_tokens"
                            content = fc.get(f"{side}_content", "")
                            if isinstance(fc.get(key), dict) and "gemini" not in fc[key]:
                                fc[key]["gemini"] = AndroidPatchManager.count_tokens_gemini(content, "neat-resolver-406722")

                        # Inline merge conflict tokens
                        for conflict in fc.get("inline_merge_conflicts", []):
                            if "merge_conflict_tokens" in conflict and "gemini" not in conflict["merge_conflict_tokens"]:
                                content = conflict["merge_conflict"]
                                conflict["merge_conflict_tokens"]["gemini"] = AndroidPatchManager.count_tokens_gemini(content, "neat-resolver-406722")

                        # Sum gemini tokens into inline_merge_token_summary
                        if "inline_merge_token_summary" not in fc:
                            fc["inline_merge_token_summary"] = {"gemini": 0, "openai": 0, "general_word": 0, "general_char": 0}

                        for conflict in fc.get("inline_merge_conflicts", []):
                            tokens = conflict.get("merge_conflict_tokens", {})
                            fc["inline_merge_token_summary"]["gemini"] += tokens.get("gemini", 0)
                            fc["inline_merge_token_summary"]["openai"] += tokens.get("openai", 0)
                            fc["inline_merge_token_summary"]["general_word"] += tokens.get("general", {}).get("word_based", 0)
                            fc["inline_merge_token_summary"]["general_char"] += tokens.get("general", {}).get("char_based", 0)



def main():
    """
    Main function to process Android vulnerabilities, generate reports, and handle CLI arguments.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    DEFAULT_REPORT_PATH = f"reports/android_platform_vulnerability_report_{timestamp}.json"
    parser = argparse.ArgumentParser(description="Android Patch Automation")
    parser.add_argument("--limit", type=int, default=None, help="Max number of vulnerabilities to process")
    parser.add_argument("--start", type=int, default=0, help="Start index for vulnerabilities")
    parser.add_argument("--report", type=str, default=DEFAULT_REPORT_PATH, help="Path to the report JSON")
    parser.add_argument("--repo", type=str, default=DEFAULT_REPO_BASE, help="Directory to clone repos")
    parser.add_argument("--after", type=str, help="Only include vulns published after YYYY-MM-DD")
    parser.add_argument("--before", type=str, help="Only include vulns published before YYYY-MM-DD")
    parser.add_argument("--data_dir", type=str, default="osv_data_android", help="Directory containing OSV JSON files")
    parser.add_argument("--cve", type=str, help="Only process this specific CVE ID (e.g., CVE-2025-12345)")
    parser.add_argument("--asb_ids_file", type=str, help="Path to text file containing ASB IDs to filter by")
    parser.add_argument("--downstream_versions_file", type=str, help="Path to file containing downstream versions to include")


    args = parser.parse_args()

    allowed_downstream_versions = None
    if args.downstream_versions_file:
        with open(args.downstream_versions_file, "r") as f:
            allowed_downstream_versions = {line.strip() for line in f if line.strip()}
        print(f"🎯 Filtering to downstream versions: {allowed_downstream_versions}")


    asb_id_filter = set()
    if args.asb_ids_file:
        with open(args.asb_ids_file, "r") as f:
            asb_id_filter = {line.strip() for line in f if line.strip()}
        print(f"📄 Filtering to {len(asb_id_filter)} ASB IDs from {args.asb_ids_file}")

    report_data = load_existing_report(args.report)
    processed_ids = {entry["vulnerability_url"] for group in [
        "vulnerabilities_with_all_failures",
        "vulnerabilities_with_partial_failures",
        "vulnerabilities_with_all_successful_patches",
        "vulnerabilities_skipped"
    ] for entry in report_data[group]}

    after_date = datetime.fromisoformat(args.after) if args.after else None
    before_date = datetime.fromisoformat(args.before) if args.before else None

    vulns = load_local_android_vulns(
        data_dir=args.data_dir,
        after_date=after_date,
        before_date=before_date
    )


    # Filter to only specified ASB IDs if provided
    if asb_id_filter:
        vulns = [v for v in vulns if v["id"] in asb_id_filter]
        print(f"🎯 {len(vulns)} vulnerabilities matched ASB ID filter")

    if asb_id_filter:
        matched_ids = {v["id"] for v in vulns}
        unmatched_ids = asb_id_filter - matched_ids
        if unmatched_ids:
            print("⚠️ Unmatched ASB IDs:")
            for uid in sorted(unmatched_ids):
                print(f" - {uid}")
        else:
            print("✅ All ASB IDs matched.")



    # If a specific CVE is requested, filter down to just that one
    if args.cve:
        vulns = [v for v in vulns if v["id"] == args.cve or args.cve in v.get("aliases", [])]
        if not vulns:
            print(f"❌ No matching vulnerability found for ID: {args.cve}")
            return


    vulns_to_process = vulns[args.start:args.start + args.limit] if args.limit else vulns[args.start:]

    os.makedirs(args.repo, exist_ok=True)

    all_failures = []

    for vuln in vulns_to_process:
        vuln_url = f"https://api.osv.dev/v1/vulns/{vuln['id']}"
        if vuln_url in processed_ids:
            print(f"✅ Skipping already processed: {vuln['id']}")
            continue

        print(f"🔄 Processing {vuln['id']}")
        result = retry_process_vulnerability(
            vuln, repo_base=args.repo, retries=2, allowed_downstream_versions=allowed_downstream_versions
        )


        if result.get("skipped"):
            if result.get("commit_mismatch"):
                report_data["commit_mismatch_vulnerabilities"].append(result)
                report_data["summary"]["vulnerabilities_with_commit_mismatch"] += 1
            else:
                report_data["vulnerabilities_skipped"].append(result)
                report_data["summary"]["vulnerabilities_skipped"] += 1
        else:
            total_versions = sum([p["total_downstream_versions_tested"] for p in result["patch_attempts"]])
            total_success = sum([p["successful_patches"] for p in result["patch_attempts"]])
            total_failures = sum([p["failed_patches"] for p in result["patch_attempts"]])

            report_data["summary"]["total_vulnerabilities_tested"] += 1
            report_data["summary"]["total_downstream_versions_tested"] += total_versions
            report_data["summary"]["total_failed_patches"] += total_failures

            # Aggregate token counts
            for attempt in result.get("patch_attempts", []):
                patch_tokens = attempt.get("upstream_patch_tokens", {})
                if isinstance(patch_tokens, dict):
                    update_token_totals(report_data["summary"]["total_tokens"]["upstream_patch"], patch_tokens)

                    for res in attempt.get("patch_results", []):
                        d_tokens = res.get("downstream_patch_tokens", {})
                        if isinstance(d_tokens, dict):
                            update_token_totals(report_data["summary"]["total_tokens"]["downstream_patch"], d_tokens)

                        for fc in res.get("file_conflicts", []):
                            u_tokens = fc.get("upstream_file_tokens", {})
                            d_tokens = fc.get("downstream_file_tokens", {})

                            if isinstance(u_tokens, dict):
                                update_token_totals(report_data["summary"]["total_tokens"]["upstream_source"], u_tokens)

                            if isinstance(d_tokens, dict):
                                update_token_totals(report_data["summary"]["total_tokens"]["downstream_source"], d_tokens)


            unique_versions = set()
            unique_failed_versions = set()
            for attempt in result["patch_attempts"]:
                failures_only = [pr for pr in attempt["patch_results"] if pr["result"] == "failure"]
                if failures_only:
                    failure_entry = {
                        "id": result["id"],
                        "vulnerability_url": result["vulnerability_url"],
                        "severity": result.get("severity", "Unknown"),
                        "failures": failures_only,
                        "upstream_patch_content": result["patch_attempts"][0].get("upstream_patch_content", ""),
                        "upstream_commits": result["patch_attempts"][0].get("upstream_commits", []),
                        "upstream_patch_tokens": result["patch_attempts"][0].get("upstream_patch_tokens", {})  # ✅ Add this
                    }

                    
                    # Ensure all necessary token fields are present
                    fill_in_gemini_tokens({"vulnerabilities_with_all_failures": [failure_entry]})

                    all_failures.append(failure_entry)

                for res in attempt["patch_results"]:
                    version = res["downstream_version"]
                    if version not in report_data["summary"]["per_version_stats"]:
                        report_data["summary"]["per_version_stats"][version] = {
                            "total_downstreams_tested": 0,
                            "total_failed_patches": 0
                        }

                    report_data["summary"]["per_version_stats"][version]["total_downstreams_tested"] += 1
                    if res["result"] == "failure":
                        report_data["summary"]["per_version_stats"][version]["total_failed_patches"] += 1


            report_data["summary"]["total_unique_downstream_versions_tested"] += len(unique_versions)
            report_data["summary"]["total_unique_downstream_failed_patches"] += len(unique_failed_versions)

            if total_failures == total_versions:
                report_data["vulnerabilities_with_all_failures"].append(result)
                report_data["summary"]["vulnerabilities_with_all_failures"] += 1
            elif total_success > 0 and total_failures > 0:
                report_data["vulnerabilities_with_partial_failures"].append(result)
                report_data["summary"]["vulnerabilities_with_partial_failures"] += 1
            else:
                report_data["vulnerabilities_with_all_successful_patches"].append(result)
                report_data["summary"]["vulnerabilities_with_all_successful_patches"] += 1

        fill_in_gemini_tokens(report_data)
        
        # Compute average tokens per downstream version
        num_versions = report_data["summary"]["total_downstream_versions_tested"]

        
        def recompute_token_summary(report_data):
            summary = report_data["summary"]
            total = {
                "upstream_patch": {"gemini": 0, "openai": 0, "general_word": 0, "general_char": 0},
                "downstream_patch": {"gemini": 0, "openai": 0, "general_word": 0, "general_char": 0},
                "upstream_source": {"gemini": 0, "openai": 0, "general_word": 0, "general_char": 0},
                "downstream_source": {"gemini": 0, "openai": 0, "general_word": 0, "general_char": 0},
                "rej_file": {"gemini": 0, "openai": 0, "general_word": 0, "general_char": 0},
                "inline_merge_conflict": {"gemini": 0, "openai": 0, "general_word": 0, "general_char": 0},
            }

            for group in [
                "vulnerabilities_with_all_failures",
                "vulnerabilities_with_partial_failures",
                "vulnerabilities_with_all_successful_patches",
            ]:
                for entry in report_data.get(group, []):
                    for attempt in entry.get("patch_attempts", []):
                        for k, v in attempt.get("upstream_patch_tokens", {}).items():
                            if k == "general":
                                total["upstream_patch"]["general_word"] += v.get("word_based", 0)
                                total["upstream_patch"]["general_char"] += v.get("char_based", 0)
                            else:
                                total["upstream_patch"][k] += v

                        for res in attempt.get("patch_results", []):
                            for k, v in res.get("downstream_patch_tokens", {}).items():
                                if k == "general":
                                    total["downstream_patch"]["general_word"] += v.get("word_based", 0)
                                    total["downstream_patch"]["general_char"] += v.get("char_based", 0)
                                else:
                                    total["downstream_patch"][k] += v

                            for fc in res.get("file_conflicts", []):
                                for side, key in [("upstream_file", "upstream_source"), ("downstream_file", "downstream_source"), ("rej_file", "rej_file")]:
                                    token_dict = fc.get(f"{side}_tokens", {})
                                    for k, v in token_dict.items():
                                        if k == "general":
                                            total[key]["general_word"] += v.get("word_based", 0)
                                            total[key]["general_char"] += v.get("char_based", 0)
                                        else:
                                            total[key][k] += v

                                for conflict in fc.get("inline_merge_conflicts", []):
                                    token_dict = conflict.get("merge_conflict_tokens", {})
                                    for k, v in token_dict.items():
                                        if k == "general":
                                            total["inline_merge_conflict"]["general_word"] += v.get("word_based", 0)
                                            total["inline_merge_conflict"]["general_char"] += v.get("char_based", 0)
                                        else:
                                            total["inline_merge_conflict"][k] += v

            summary["total_tokens"] = total

        if num_versions > 0:
            recompute_token_summary(report_data)
            total_tokens = report_data["summary"]["total_tokens"]

            avg_tokens = {
                section: {
                    token_type: round(value / num_versions, 2)
                    for token_type, value in token_data.items()
                }
                for section, token_data in total_tokens.items()
            }

            avg_tokens["inline_merge_conflict"] = {
                token_type: round(report_data["summary"]["total_tokens"]["inline_merge_conflict"].get(token_type, 0) / num_versions, 2)
                for token_type in ["gemini", "openai", "general_word", "general_char"]
            }

            avg_tokens["rej_file"] = {
                token_type: round(report_data["summary"]["total_tokens"]["rej_file"].get(token_type, 0) / num_versions, 2)
                for token_type in ["gemini", "openai", "general_word", "general_char"]
            }


            report_data["summary"]["average_tokens_per_downstream_version"] = avg_tokens
        else:
            report_data["summary"]["average_tokens_per_downstream_version"] = {}

        save_report(report_data, args.report, failures=all_failures)


if __name__ == "__main__":
    main()

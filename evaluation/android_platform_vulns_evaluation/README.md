# Android Platform Vulnerability Patch Runner

This tool automates testing of Android platform vulnerability patches by loading OSV JSON files, applying upstream patches to downstream Android versions, and generating detailed reports on patch success, failure, merge conflicts, and severity.

---

## Project Structure

- `osv_patch_runner.py`: Orchestrates the vulnerability evaluation process and manages reporting.
- `android_patch_manager.py`: Handles repository cloning, patch generation (supports multi-commit), patch application, and conflict extraction.
- `reports/`: Auto-created directory for storing JSON reports.
- `android_repos/`: Auto-created directory for cloned Android repositories.

---

## Usage

### 1. Run with Default Settings

```bash
python osv_patch_runner.py
```

Processes **all local OSV JSON files** in the default directory `osv_data_android/`.

---

### 2. Limit the Number of Vulnerabilities

```bash
python osv_patch_runner.py --limit 10
```

Processes only the first 10 vulnerabilities.

---

### 3. Start from a Specific Index

```bash
python osv_patch_runner.py --start 20
```

Skips the first 20 vulnerabilities and processes from the 21st onward.

---

### 4. Filter by Publish Date

```bash
python osv_patch_runner.py --after 2025-01-01 --before 2025-12-31
```

Processes only vulnerabilities published in 2025.

---

### 5. Specify Custom Report or Repository Paths

```bash
python osv_patch_runner.py --report ./reports/custom_report.json --repo ./custom_repos
```

Saves the report and cloned repos to specified directories.

---

### 6. Combine Flags for Fine Control

```bash
python osv_patch_runner.py --limit 15 --start 5 --after 2024-01-01 --report ./reports/filtered_2024.json
```

---

## Report Format

### Summary Statistics:

- `total_vulnerabilities_tested`
- `total_versions_tested`
- `total_failed_patches`
- `total_unique_downstream_versions_tested`
- `total_unique_downstream_failed_patches`
- `vulnerabilities_with_all_failures`
- `vulnerabilities_with_partial_failures`
- `vulnerabilities_with_all_successful_patches`
- `vulnerabilities_skipped`
- `severity_counts` (Critical, High, Medium, Low, Unknown)

---

### Detailed Patch Results Structure:

```json
{
  "patch_attempts": [
    {
      "upstream_commits": ["abc123", "def456"],
      "upstream_patch_content": "Combined patch content for these commits",
      "upstream_branch_used": "main",
      "total_versions_tested": 3,
      "successful_patches": 2,
      "failed_patches": 1,
      "patch_results": [
        {
          "downstream_version": "12",
          "downstream_patch": "xyz789",
          "downstream_patch_content": "Patch content specific to version 12",
          "branch_used": "android12-release",
          "result": "success"
        },
        {
          "downstream_version": "11",
          "downstream_patch": "uvw345",
          "downstream_patch_content": "...",
          "branch_used": "android11-release",
          "result": "failure",
          "file_conflicts": [...]
        }
      ]
    }
  ]
}
```

---

## High-Level Workflow

1. Load CVE JSON files.
2. Filter for Android platform vulnerabilities.
3. Clone repositories as needed.
4. Generate and apply combined upstream patches.
5. Test patches against each downstream Android version.
6. Capture results (success, failure, conflicts).
7. Save and update JSON report.

---

## Helper Scripts

- **`evaluate_llm_patch_success.py`** – Compares LLM-generated patches with upstream patches by applying both, capturing file states, and recording success/failure metrics.
- **`clean_diff_text.py`** – Cleans diff output by removing headers and extracting hunk content using string-based or `unidiff` parsing.
- **`combine_failures.py`** – Merges multiple failure reports (e.g., from 2024 and 2025) into one.
- **`filter_failures_by_version.py`** – Filters failure reports by Android version and outputs filtered sets in JSON or CSV.
- **`extract_ground_truth_file_content.py`** – Extracts actual downstream file content at specific commits and computes token usage.
- **`extract_upstream_patch.py`** – Gathers upstream patch data and injects it into failure records for completeness.
- **`generate_patch_success_matrix.py`** – Generates heatmaps showing patch success rates and attempt counts across Android versions.
- **`analyze_vanir_report.py`** – Parses Vanir-format ASB reports, maps to OSV CVEs, and synchronizes relevant entries.

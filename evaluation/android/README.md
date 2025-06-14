# Android Platform Vulnerability Patch Evaluation

This tool automates the testing of Android platform vulnerability patches. It loads OSV JSON files, applies the relevant upstream patches to various downstream Android versions, and generates detailed JSON reports on patch success, failure, merge conflicts, and token usage.

## Key Components

*   `evaluate_android_patches.py`: The main script in this directory. It orchestrates the entire evaluation process.
*   `android_patch_manager.py`: A core utility library located in the parent `vidar/` directory. It handles all Git and patch operations.
*   `evaluation/reports/`: The default output directory for generated JSON reports. This can be changed via command-line arguments.
*   `android_repos/`: The default directory for cloning Android repositories.

## Prerequisites

The prerequisite is to install the required Python packages. From the `vidar/` root, run:
```bash
pip install -r evaluation/requirements.txt
```

The necessary OSV vulnerability data will be downloaded automatically to the `osv_data_android/` directory the first time you run the evaluation script.

## Usage

**Note:** All commands should be run from the root of the `vidar/` directory.

#### 1. Run the Evaluation
```bash
python evaluation/android/evaluate_android_patches.py
```
On the first run, this will automatically download the latest OSV data for Android before starting the evaluation. Subsequent runs will use the cached data.

#### 2. Limit the Number of Vulnerabilities
Processes only the first 10 vulnerabilities found.
```bash
python evaluation/android/evaluate_android_patches.py --limit 10
```

#### 3. Start from a Specific Index
Skips the first 20 vulnerabilities and processes from the 21st onward.
```bash
python evaluation/android/evaluate_android_patches.py --start 20
```

#### 4. Filter by Publish Date
Processes only vulnerabilities published in the year 2025.
```bash
python evaluation/android/evaluate_android_patches.py --after 2025-01-01 --before 2025-12-31
```

#### 5. Specify Custom Report or Repository Paths
Saves the report and cloned repos to custom directories.
```bash
python evaluation/android/evaluate_android_patches.py --report evaluation/reports/custom_report.json --repo ./custom_repos
```

#### 6. Combine Flags for Fine-Grained Control
```bash
python evaluation/android/evaluate_android_patches.py --limit 15 --start 5 --after 2024-01-01 --report evaluation/reports/filtered_2024.json
```

## Report Format

The script generates a detailed JSON report. The `summary` object provides high-level statistics, while the detailed results for each vulnerability are stored in separate lists based on their outcome.

#### High-Level Workflow
1. Load CVE JSON files from the data directory.
2. Filter for Android platform vulnerabilities.
3. Clone or update repositories as needed.
4. Generate a combined upstream patch from all relevant commits.
5. Test the patch against each specified downstream Android version.
6. Capture detailed results (success, failure, file conflicts, token counts).
7. Save and update the JSON report continuously.

#### Detailed Patch Results Structure
An individual entry in one of the `vulnerabilities_with_*` lists will look like this:
```json
{
  "patch_attempts": [
    {
      "upstream_commits": ["abc123", "def456"],
      "upstream_patch_content": "Combined patch content for these commits",
      "upstream_branch_used": "main",
      "total_downstream_versions_tested": 3,
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
          "file_conflicts": "[...]"
        }
      ]
    }
  ]
}
``` 
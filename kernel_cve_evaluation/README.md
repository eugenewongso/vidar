# Linux Kernel CVE Patch Runner

This tool automates the testing of Linux kernel CVE patches by fetching CVE JSON files, applying the related patches to historical commits, and generating a detailed report of success and failure rates.

---

## Project Structure

- `cve_patch_runner.py`: Orchestrates the CVE evaluation process and manages reporting.
- `kernel_cve_patch_manager.py`: Handles patch generation, application, and version checkout.
- `reports/`: Where reports are stored (created automatically).
- `patches/`: Where generated patch files are stored (created automatically).

---

## Usage

### 1. Run on a Directory of CVEs

```bash
python cve_patch_runner.py https://example.com/plain/cve/
```

This fetches and processes **all CVE JSON files** in the directory.

---

### 2. Start From a Specific CVE

```bash
python cve_patch_runner.py https://example.com/plain/cve/ --start=67
```

This skips the first 66 CVEs and starts processing from the 67th (1-indexed).

---

### 3. Continue from a Previous Run

```bash
python cve_patch_runner.py https://example.com/plain/cve/ --report ./reports/full_cve_report.json
```

This will:
- Load the existing report (if it exists)
- Skip any CVEs already in that report
- Append new results to the same file

You can combine it with `--start`:

```bash
python cve_patch_runner.py https://example.com/plain/cve/ --start=67 --report ./reports/full_cve_report.json
```

---

## Report Format

The report is saved in JSON and includes:

### Summary Statistics:
- `total_cves_tested`
- `total_versions_tested`
- `total_failed_patches`
- `total_unique_downstream_versions_tested`
- `total_unique_downstream_failed_patches`
- `cves_with_all_failures`
- `cves_with_partial_failures`
- `cves_with_all_successful_patches`
- `cves_skipped`

### Detailed Lists:
- `cves_with_all_failures`
- `cves_with_partial_failures`
- `cves_with_all_successful_patches`
- `cves_skipped`

#### Example:

```json
{
  "summary": {
    "total_cves_tested": 66,
    "total_versions_tested": 200,
    "total_failed_patches": 50,
    "total_unique_downstream_versions_tested": 140,
    "total_unique_downstream_failed_patches": 35,
    "cves_with_all_failures": 10,
    "cves_with_partial_failures": 20,
    "cves_with_all_successful_patches": 36,
    "cves_skipped": 5
  }
}
```

---

## Notes

- Each CVE JSON must contain a `repo` and a list of `versions` with a `lessThan` commit hash.
- You can use a single CVE JSON URL instead of a directory.
- The runner will generate patches from upstream commits and attempt to apply them to downstream versions.

---


# Linux Kernel CVE Patch Runner

This tool automates the testing of Linux kernel CVE patches by fetching or loading CVE JSON files, applying the related patches to historical commits, and generating a detailed report of success and failure rates.

---

## Project Structure

- `cve_patch_runner.py`: Orchestrates the CVE evaluation process and manages reporting.
- `kernel_cve_patch_manager.py`: Handles patch generation, application, and version checkout.
- `reports/`: Directory where reports are saved (created automatically).
- `patches/`: Directory where generated patch files are stored (created automatically).
- `vulns/`: Optional local clone of the Linux Kernel CVE repository.

---

## Usage

### 1. Run on a Remote Directory of CVEs

```bash
python cve_patch_runner.py https://example.com/plain/cve/
```

This fetches and processes **all CVE JSON files** from the given remote directory URL.

---

### 2. Run on a Local Directory of CVEs

```bash
python cve_patch_runner.py ./vulns/cve/published/2025 --local
```

This processes all `.json` files inside the specified **local** directory and its subdirectories.

> Tip: You can clone the official CVE repo from:
>
> ```bash
> git clone https://git.kernel.org/pub/scm/linux/security/vulns.git
> ```

---

### 3. Run a Single Local CVE JSON File (No `--local` Needed)

```bash
python cve_patch_runner.py ./vulns/cve/published/2025/CVE-2025-21629.json
```

You do **not** need the `--local` flag for a single file. The runner will detect it's a single local `.json` file and process it directly.

---

### 4. Start From a Specific CVE

```bash
python cve_patch_runner.py ./vulns/cve/published/2025 --local --start=67
```

This skips the first 66 CVEs and starts from the 67th file.

---

### 5. Continue from a Previous Run

```bash
python cve_patch_runner.py ./vulns/cve/published/2025 --local --report ./reports/full_cve_report.json
```

This will:

- Load the existing report (if it exists)
- Skip CVEs already listed
- Append new results

You can also combine it with `--start`:

```bash
python cve_patch_runner.py ./vulns/cve/published/2025 --local --start=67 --report ./reports/full_cve_report.json
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

- CVE JSON files must contain a `repo` and `versions` list with at least one `lessThan` commit hash.
- Works with both remote URLs and local filesystem inputs.
- Automatically skips CVEs already in the report.
- Patch success/failure is tracked per downstream commit and logged in detail.

# Android Kernel Patch Adopter

A tool for automating the processing, fetching, and application of security patches for Android kernels from Vanir security reports.

## Overview

This tool automates the workflow of applying security patches to Android kernels. It:

1. Parses Vanir security reports to identify missing patches
2. Fetches patch files from Android's Googlesource or CodeLinaro repositories
3. Applies patches to the target kernel
4. Generates detailed reports about successful and failed patch applications
5. Collects rejected patches for further analysis

## Prerequisites

- Python 3.6+
- Git
- GNU patch command (`patch` on Linux, `gpatch` on macOS)
- BeautifulSoup4 (`pip install beautifulsoup4`)
- Requests (`pip install requests`)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/kernel-patch-adopter.git
   cd kernel-patch-adopter
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
kernel-patch-adopter/
├── main.py                   # Main script for processing Vanir reports
├── patch_adopter.py          # PatchAdopter class for applying patches
├── inputs/                   # Directory for input files
│   └── raw_vanir_reports/    # Raw Vanir security reports
├── outputs/                  # Output directories
│   ├── fetched_diffs/        # Fetched patch files
│   ├── parsed_reports/       # Parsed Vanir reports
│   ├── application_reports/  # Patch application reports
│   ├── failed_patches/       # Lists of failed patches
│   └── combined_rejected_hunks/ # Combined rejected patch hunks
```

## Usage

### Basic Workflow

1. Place your raw Vanir security report in the `inputs/raw_vanir_reports/` directory.

2. Run the main script:
   ```bash
   python main.py
   ```

3. When prompted, enter:
   - The path to the raw Vanir report file (e.g., `inputs/raw_vanir_reports/xiaomi_flame.json`)
   - The absolute path to your kernel repository where patches will be applied

4. The tool will:
   - Parse the Vanir report
   - Fetch patches from the appropriate repositories
   - Apply patches to your kernel repository
   - Generate reports on the patch application status
   - Combine rejected hunks into a single file for review

### Output Files

After running the tool, you'll find:

- **Parsed Vanir Report**: `outputs/parsed_reports/YYYYMMDD_HHMMSS_report.json`
- **Fetched Patches**: `outputs/fetched_diffs/{commit_hash}.diff`
- **Patch Application Report**: `outputs/application_reports/patch_application_report_YYYYMMDD_HHMMSS.json`
- **Failed Patches List**: `outputs/failed_patches/failed_patches_YYYYMMDD_HHMMSS.json`
- **Combined Rejected Hunks**: `outputs/combined_rejected_hunks/combined_rejected_hunks_YYYYMMDD_HHMMSS.rej`

## Advanced Usage

### Processing Multiple Reports

To process multiple Vanir reports in sequence:

```bash
for report in inputs/raw_vanir_reports/*.json; do
    echo "Processing $report..."
    python main.py <<< "$report
/path/to/your/kernel"
done
```

### Manual Analysis of Rejected Patches

For patches that couldn't be applied automatically, you can review the combined rejected hunks:

1. Check `outputs/combined_rejected_hunks/combined_rejected_hunks_YYYYMMDD_HHMMSS.rej` for the rejected patches.
2. Review the `outputs/application_reports/patch_application_report_YYYYMMDD_HHMMSS.json` to identify which files failed.
3. Manually apply the patches to those files, modifying as necessary to resolve conflicts.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Your license information here]

## Troubleshooting

### Common Issues

1. **"Patch file not found"**: 
   - Ensure the Vanir report has valid patch URLs
   - Check your internet connection for fetching patches

2. **"Reversed (or previously applied) patch detected"**:
   - The patch may have already been applied to your kernel
   - You can safely ignore these

3. **Failed hunks**:
   - Common when applying patches to different kernel versions
   - Review the combined rejected hunks file for manual resolution

### Getting Help

If you encounter any issues, please [open an issue](https://github.com/yourusername/kernel-patch-adopter/issues) with:
- The command you ran
- The raw Vanir report (if possible)
- The error message or output
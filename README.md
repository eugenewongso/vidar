# Android Security Patch Adaptation Tools

This repository contains tools for adapting Android security patches across different versions using Large Language Models (LLMs), specifically Google's Gemini model.

## Directory Structure

```
.
├── android_repos/          # Cloned Android repositories
├── osv_data_android/      # OSV vulnerability data (.json files)
│   └── *.json     # Individual vulnerability metadata files
├── inputs/                # Input JSON files with patch failures
├── outputs/               # Generated outputs
│   └── report/           # Validation and processing reports
├── *.py                  # Python scripts
└── .env                  # Environment configuration
```

## Scripts Overview

### Core Scripts

#### 1. `android_patch_manager.py`
Core utility class managing Android repository operations:
- Repository cloning and management
- Patch application and validation
- Conflict detection and extraction
- Token counting utilities
```bash
# Not meant to be run directly - imported by other scripts
```

#### 2. `approach2.py`
Unified diff generation approach:
```bash
python approach2.py inputs/failures.json "14" [-o outputs/custom_output.json]
```
- Generates patch diffs only
- More efficient token usage
- Easier validation
- Better success rate

### Enhanced Variants

#### 3. `approach2_smart_retry.py`
Enhanced retry logic with detailed feedback:
```bash
python approach2_smart_retry.py inputs/failures.json "14" [-o outputs/custom_output.json]
```
- Includes validation feedback in retries
- Provides detailed error messages
- Maximum 3 retry attempts

#### 4. approach2_smart_retry_2_noguideline.py
Variant without explicit guidelines in prompts:
```bash
python approach2_smart_retry_2_noguideline.py inputs/failures.json "14" [-o outputs/custom_output.json]
```
- Iterative refinement
- Doesn't include: 5. Every line in each hunk must begin with a `+`, `-`, or space character.

#### 5. approach2_smart_retry_less_error_msg.py
Streamlined error messaging:
```bash
python approach2_smart_retry_less_error_msg.py inputs/failures.json "14" [-o outputs/custom_output.json]
```
- Iterative refinement
- Minimal error feedback (no detailed error output for format / patch error)

## Setup

1. **Environment Setup**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. **Required Directories**
```bash
mkdir -p android_repos osv_data_android inputs outputs/{approach1_results,approach2_results,report}
```

3. **Configure API Keys**
Create a `.env` file:
```env
GOOGLE_API_KEYS="your-api-key-1,your-api-key-2,..."
GCP_PROJECT="your-gcp-project-id"
```

4. **OSV Data**
Place Android vulnerability data files in `osv_data_android/`:
- Format: `ANDROID-CVE-YYYY-XXXXX.json`
- Contains vulnerability metadata
- Used for repository URL resolution

## Input Format Requirements

Input JSON files should contain:
```json
[
  {
    "id": "ANDROID-CVE-YYYY-XXXXX",
    "failures": [
      {
        "downstream_version": "14",
        "downstream_patch": "commit_sha",
        "file_conflicts": [
          {
            "file_name": "path/to/file.java",
            "rej_file_content": "... reject file content ...",
            "downstream_file_content": "... original file content ...",
            "downstream_file_content_patched_upstream_only": "... content after failed patch ..."
          }
        ]
      }
    ]
  }
]
```

## Output Structure

1. **Main Output JSON**
- Contains processed results
- Includes generated patches/diffs
- Validation results
- Token usage statistics

2. **Report JSON**
- Processing statistics
- Success/failure counts
- Validation details
- Runtime metrics

## Validation Process

Each generated patch goes through:
1. Format validation using `unidiff`
2. Applicability testing with GNU patch
3. Repository-based validation
4. Multiple retry attempts if needed

## Requirements

- Python 3.8+
- Google Cloud API access (Gemini model)
- GNU patch utility
- Git
- Sufficient disk space for Android repositories

## Notes

- Temporary files are cleaned up automatically
- API key rotation is supported
- Repository caching in `android_repos/` improves performance
- OSV data in `osv_data_android/` must match vulnerability IDs

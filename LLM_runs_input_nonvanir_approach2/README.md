## Setup Instructions

### 1. Install Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

---

### 2. Set Up Environment Variables

Create a .env file in the root of the LLM_runs_input_nonvanir_approach2 directory. Add your Google API keys as a comma-separated list. These keys will be used by the API key rotator:

```env
GOOGLE_API_KEYS="key1,key2,key3"
```

---

## Overview of Directory Components

This directory contains scripts and resources for "Approach 2" of an LLM-based patch generation process, focusing on different retry strategies.

*   **`android_patch_manager.py`**: Contains utilities or a class for managing Android patches. This likely involves functions for applying patches to source code, checking if patches apply cleanly (patch applicability), and validating the correctness of applied patches (patch validity).
*   **`approach2.py`**: The base script for "Approach 2". It processes entries by making a single call to the LLM to generate a patch, without iterative refinement or validation loops within this script itself.
*   **`approach2_blind_retry.py`**: Implements a blind retry strategy. If an LLM-generated patch fails validation (format or applicability), it retries the LLM call without providing specific error details from the failed attempt in the new prompt.
*   **`approach2_smart_retry.py`**: Implements a smart retry strategy. On patch failure, it retries the LLM call, providing detailed error messages and a snippet of the last failed diff to the LLM to guide correction.
*   **`approach2_smart_retry_less_error_msg.py`**: A variation of smart retry that provides less detailed (more generic) error types (e.g., "[Format Error]") to the LLM during retries, along with a snippet of the last failed diff.
*   **`approach2_smart_retry_noguideline.py`**: A smart retry variant that provides generic error types and a snippet of the last failed diff, but omits a specific formatting guideline (Guideline #5 regarding line prefixes `+`, `-`, ` `) from the retry prompt.
*   **`inputs/` (directory)**: Contains input data for the experiments, such as source code files needing patches, vulnerability information, or ground truth patches.
*   **`osv_data_android/` (directory)**: Stores data related to Android vulnerabilities, possibly sourced from OSV (Open Source Vulnerability) databases, used to identify vulnerabilities for patching.
*   **`outputs/` (directory)**: Where the scripts save their results, including generated patches, execution logs, evaluation metrics, and other artifacts.

---

## Script Variants

The primary difference between the retry scripts (`_blind_retry`, `_smart_retry`, `_smart_retry_less_error_msg`, `_smart_retry_noguideline`) lies in how they construct the prompt for the LLM on subsequent attempts after an initial failure. `approach2.py` does not implement this retry/validation loop.

### `approach2.py`
*   Performs a single LLM call per file conflict.
*   Does not include iterative refinement or validation loops within this script.

### `approach2_smart_retry.py`

*   Uses iterative refinement (up to `max_retries`).
*   On retry, provides detailed error messages from `format_error` and `apply_error` fields and a snippet of the previous invalid diff to the LLM.
*   Includes a line-formatting guideline in the prompt:
    *“Every line in each hunk must begin with a `+`, `-`, or space character.”*

### `approach2_smart_retry_noguideline.py`

*   Uses iterative refinement.
*   On retry, provides generic error types (e.g., "[Format Error]") and a snippet of the previous invalid diff.
*   Omits the specific line-formatting guideline (Guideline #5) from the retry prompt.

### `approach2_smart_retry_less_error_msg.py`

*   Uses iterative refinement.
*   On retry, provides generic error types (e.g., "[Format Error]", "[Apply Error]") and a snippet of the previous invalid diff.
*   Includes the line-formatting guideline.
*   The "less_error_msg" refers to providing generic error types instead of full error strings to the LLM.

### `approach2_blind_retry.py`

*   Uses a **blind retry strategy** (iterative refinement up to `max_retries` but without detailed error feedback to LLM).
*   On retry, provides general guidelines but does *not* include specific error messages from previous attempts or a snippet of the invalid diff in the prompt to the LLM.
*   Includes the line-formatting guideline.

---

## Input File

The scripts are designed to process a JSON file. Each item in the input JSON typically represents a vulnerability (CVE), which contains a list of `failures`. Each `failure` can have multiple `file_conflicts`. The scripts operate on these `file_conflict` objects.

An example input file is provided at:

```
./inputs/20242025_combined_failures_with_ground_truth_and_patched_upstream.json
```

This example includes patch failures for Android versions from **2024 and 2025**. You can replace this file with any other input file that follows the same structure.

---

## Output

Each script generates two main types of output:

1.  **An updated JSON file**: This file is a copy of the input JSON, but with additional fields added to each `file_conflict` object that was processed.
2.  **A separate report JSON file**: This summarizes the run, including counts for successes, failures, token usage, etc.

### 1. Updated JSON File Details

The original structure of the input JSON is preserved. The scripts add new key-value pairs to the `file_conflict` objects.

**For `approach2.py`:**
For each processed `file_conflict`, the following fields are added/updated:
*   **`downstream_llm_diff_output`**: (string) The raw unified diff string generated by the LLM.
*   **`runtime_seconds`**: (float) The time taken for the LLM call for this specific file conflict.
*   **`token_counts`**: (dict) Token counts for the `generated_diff`:
    *   `openai`: (int) Token count via Tiktoken.
    *   `general`: (dict) Estimated token counts (`word_based`, `char_based`).
    *   `gemini`: (int) Token count from Gemini API response's `usage_metadata`.

**For `approach2_blind_retry.py`, `approach2_smart_retry.py`, `approach2_smart_retry_less_error_msg.py`, and `approach2_smart_retry_noguideline.py`:**
These scripts add a more comprehensive set of fields due to their retry and validation logic:
*   **`downstream_llm_diff_output`**: (string or `null`) The final successfully validated LLM-generated diff, or `null` if all retries failed or if skipped.
*   **`llm_output_valid`**: (boolean) `True` if a patch was successfully generated and validated (format and applicability).
*   **`runtime_seconds`**: (float) Total runtime for processing this `file_conflict`, including all attempts and validations.
*   **`attempts_made`**: (int) Number of LLM call attempts made for this entry.
*   **`validation_results`**: (list of dicts) A list detailing each attempt:
    *   `attempt`: (int) The attempt number.
    *   `format_valid`: (boolean) Patch format validity for this attempt.
    *   `format_error`: (string) Message from format validation.
    *   `apply_valid`: (boolean) Patch applicability for this attempt.
    *   `apply_error`: (string) Message from applicability validation.
    *   `valid`: (boolean) Overall validity for this attempt (`format_valid && apply_valid`).
    *   `runtime_seconds`: (float) Runtime for this specific attempt.
    *   `error`: (string, optional) If an exception occurred during LLM generation for this attempt.
*   **`token_counts`**: (dict, optional) Token counts for the final successful `downstream_llm_diff_output` (if `llm_output_valid` is `True`). Structure:
    *   `openai`: (int)
    *   `general`: (dict - `word_based`, `char_based`)
    *   `gemini`: (int)
*   **`error`**: (string, optional) A top-level error message if all attempts failed or if the entry was skipped before LLM processing (e.g., "All validation attempts failed", "Empty or placeholder .rej File Content").
*   **`last_format_error`**: (string or `null`, optional) The `format_error` from the final attempt if all retries failed.
*   **`last_apply_error`**: (string or `null`, optional) The `apply_error` from the final attempt if all retries failed.

### 2. Report JSON File
All scripts also generate a separate summary report file (e.g., `report_diff_*.json` in `outputs/report/`). This file contains aggregate statistics about the run, such as:
*   `run_timestamp`
*   `target_downstream_version`
*   `input_json_file`
*   `main_output_json_file_with_diffs` (path to the updated JSON described above)
*   `summary`:
    *   `total_file_conflicts_matching_version`
    *   `files_attempted_for_llm_diff_generation`
    *   `files_with_llm_diff_successfully_generated`
    *   `files_skipped_pre_llm_call`
    *   `files_with_llm_diff_generation_errors_or_skipped_in_func`
    *   `successful_attempts_histogram` (for retry scripts, shows how many succeeded on 1st, 2nd, etc., attempt)
    *   `total_runtime_seconds_all`
    *   `total_runtime_seconds_successful`
*   `successfully_generated_diffs_log`: Log of successful generations.
*   `skipped_or_errored_diff_generation_log`: Log of skips or errors.

---

## How to Run

From the root of the `LLM_runs_input_nonvanir_approach2` directory, run one of the script variants using Python. 

Before running, ensure that:

* The `.env` file exists and includes your API keys.
* The input file (e.g., `./inputs/20242025_combined_failures_with_ground_truth_and_patched_upstream.json`) is available.

---

### Command-Line Arguments

Each script supports the following optional arguments to customize behavior:

| Argument                      | Description                                                                             |
| ----------------------------- | --------------------------------------------------------------------------------------- |
| `--input_json_file_path`      | Path to the input JSON file containing failed patches and references.                   |
| `--target_downstream_version` | Filter to only process CVEs for a specific Android version (e.g., `14`). |
| `--output_json_path`          | Path to save the output JSON file with generated patches and validation.                |

---

### Example: Running with All Arguments

```bash
python approach2_smart_retry.py \
  --input_json_file_path ./inputs/20242025_combined_failures_with_ground_truth_and_patched_upstream.json \
  --target_downstream_version 14 \
  --output_json_path ./outputs/android12_r3_output.json
```

This command:

* Loads the input file from the `./inputs` folder,
* Filters for CVEs specific to `android-12.0.0_r3`,
* Saves the result to `./outputs/android14_output.json`.

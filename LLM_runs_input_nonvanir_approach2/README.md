# LLM Runs Input Non-Vanir Approach 2 (Diff Output)

This script processes vulnerability data from a JSON file. Unlike Approach 1, this script uses a Generative AI model (Gemini) to generate a **unified diff** for applying patches, rather than outputting the entire patched file. It then outputs an updated JSON file containing these generated diffs, along with a summary report.

## Prerequisites

- Python 3.x
- Pip (Python package installer)
- Google API Key with access to Gemini models.

## Setup

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    The script uses `google-generativeai`, `python-dotenv`, and `logfire`. Ensure these are listed in a `requirements.txt` file or install them manually:
    ```bash
    pip install google-generativeai python-dotenv logfire
    ```
    If a `requirements.txt` file exists in the parent directory or this directory, you can use:
    ```bash
    pip install -r requirements.txt
    ```
    *(Adjust path to `requirements.txt` if necessary)*

4.  **Set up environment variables:**
    Create a `.env` file in the root directory of this script (or where it's executed from) and add your Google API key:
    ```env
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY" 
    ```
    *(Note: The script `approach2.py` uses `GOOGLE_API_KEY` as the variable name, ensure this matches your .env file or update the script.)*
    Replace `"YOUR_GOOGLE_API_KEY"` with your actual API key.

## Running the Script

The script `approach2.py` takes two mandatory arguments and one optional argument:

1.  `input_json_file_path`: (Required) Path to the input JSON file containing vulnerability data. This data should include the original source file content and the content of any `.rej` files from previous failed patch attempts.
2.  `target_downstream_version`: (Required) The downstream version string to filter by (e.g., "14", "13").
3.  `--output_json_path` or `-o`: (Optional) Path to save the output JSON file (which will contain the generated diffs). If not provided, it defaults to `outputs/approach2_results/approach2_output_diff_android_{version}_{timestamp}.json`.

**Command Syntax:**

```bash
python LLM_runs_input_nonvanir_approach2/approach2.py <input_json_file_path> <target_downstream_version> [-o <output_json_path>]
```

**Examples:**

1.  **Basic execution (output to default location):**
    This command will process `LLM_runs_input_nonvanir_approach2/inputs/filtered_failures_android_14_2025_with_context_3_5_10_20.json`, targeting entries with downstream version "14". The output JSON (containing generated diffs) and a report will be saved in the `outputs/approach2_results/` and `outputs/report/` directories respectively.

    ```bash
    python LLM_runs_input_nonvanir_approach2/approach2.py LLM_runs_input_nonvanir_approach2/inputs/filtered_failures_android_14_2025_with_context_3_5_10_20.json "14"
    ```

2.  **Specifying an output file path:**
    This command will process the same input file for version "14" but will save the main output JSON (with diffs) to `LLM_runs_input_nonvanir_approach2/outputs/custom_diff_output.json`. The report will still go to the default `outputs/report/` directory (named `report_diff_{version}_{timestamp}.json`).

    ```bash
    python LLM_runs_input_nonvanir_approach2/approach2.py LLM_runs_input_nonvanir_approach2/inputs/filtered_failures_android_14_2025_with_context_3_5_10_20.json "14" -o LLM_runs_input_nonvanir_approach2/outputs/custom_diff_output.json
    ```

## Input JSON Format

The script expects an input JSON file that is a list of vulnerability objects. Each object should have an `id` and a `failures` list. Each failure object can contain `downstream_version` and `file_conflicts`. Crucially for this approach, each `file_conflict` object must have:
-   `file_name`: The original name of the file that needs patching.
-   `rej_file_content`: The content of the `.rej` file (rejected hunks from a previous patch attempt).
-   `downstream_file_content`: The full original content of the source file to be patched.

Example structure:
```json
[
  {
    "id": "VULN-002",
    "failures": [
      {
        "downstream_version": "14",
        "downstream_patch": "commit_sha_def456",
        "repo_path": "path/to/another/repo",
        "file_conflicts": [
          {
            "file_name": "utils.java",
            "rej_file_content": "--- a/utils.java\n+++ b/utils.java\n@@ -10,7 +10,7 @@\n public class Utils {\n-    // old problematic line\n+    // intended new line from original patch\n     private static void helper() {\n",
            "downstream_file_content": "package com.example;\n\npublic class Utils {\n    // ... other code ...\n    // old problematic line\n    private static void helper() {\n        // ...\n    }\n    // ... more code ...\n}\n"
          }
          // ... more file conflicts
        ]
      }
      // ... more failures
    ]
  }
  // ... more vulnerabilities
]
```

## Output

The script generates two main outputs:

1.  **Processed JSON File (with Diffs):**
    -   Located at the path specified by `--output_json_path` or the default path (`outputs/approach2_results/approach2_output_diff_android_{version}_{timestamp}.json`).
    -   This file is a copy of the input JSON. For successfully processed `file_conflict` objects, it adds a `downstream_llm_diff_output` field. This field contains the **unified diff string** generated by the LLM, intended to patch the `downstream_file_content`.

2.  **Summary Report JSON File:**
    -   Located in the `outputs/report/` directory, with a filename like `report_diff_{version}_{timestamp}.json`.
    -   This file contains a summary of the run, including:
        -   Timestamp and input parameters.
        -   Counts of total file conflicts, files attempted for diff generation, successfully generated diffs, and skipped/errored files.
        -   Logs of successfully generated diffs (with a preview) and skipped/errored files with reasons.

## Script Logic Overview

1.  **Initialization:**
    -   Loads environment variables (especially `GOOGLE_API_KEY`).
    -   Configures the Google Generative AI client.
    -   Defines a `GeminiAgent` with a system prompt specifically instructing it to analyze `.rej` content and original source code to produce a **unified diff**.
    -   Sets up argument parsing.

2.  **Data Loading & Preparation:**
    -   Reads the input JSON file.
    -   Creates a deep copy of the input data.
    -   Initializes a `report_data` dictionary.

3.  **Processing Vulnerabilities & File Conflicts:**
    -   Iterates through vulnerabilities, their failures, and file conflicts, filtering by `target_downstream_version` (similar to Approach 1).
    -   For each relevant `file_conflict`:
        -   Retrieves `rej_file_content`, `downstream_file_content` (original source), and `file_name`.
        -   Performs validation.
        -   If valid, calls `process_single_entry`.
        -   `process_single_entry` constructs a prompt for the `GeminiAgent`, providing the original source, the `.rej` content, and the target filename (for `--- a/` and `+++ b/` lines in the diff).
        -   The agent's task is to generate a corrected unified diff.
        -   If successful, the `generated_diff` string is returned.
        -   This `generated_diff` is added as `downstream_llm_diff_output` to the `file_conflict` object in the output data.
        -   Updates `report_data` counters.

4.  **Output Generation:**
    -   Saves the modified data (with `downstream_llm_diff_output` fields) to the main output JSON file.
    -   Saves the `report_data` to the summary report JSON file (prefixed with `report_diff_`).

This approach focuses on leveraging the LLM to understand patch failures (via `.rej` files) and generate a corrected diff, which can then be applied using standard patching tools.

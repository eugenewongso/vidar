# Patch Evaluation Post-Processor

This project provides tools to evaluate code patches by comparing different versions of files extracted from a Git repository based on a JSON evaluation report.

## Workflow

1.  **`post_eval.py`** (Main Orchestrator):
    *   Parses command-line arguments for the repository path (`--repo`), evaluation report path (`--eval_report`), and optional output directories (`--upstream_dir`, `--downstream_dir`).
    *   Reads the specified JSON evaluation report.
    *   Iterates through CVE entries marked as failures in the report.
    *   For each relevant file path within a failed patch attempt, identifies the upstream and downstream commit hashes.
    *   Calls `extract_file_at_commit` from `post_eval_helpers.py` to fetch the file content for both the upstream and downstream commits from the specified Git repository.
    *   Saves the extracted files to the designated upstream and downstream directories.
    *   If both file versions are successfully extracted, it executes `main.py` as a subprocess, passing the paths to the extracted files.

2.  **`post_eval_helpers.py`** (Git Interaction):
    *   Provides the `extract_file_at_commit` function, which uses `git show` to retrieve a specific file version from a given commit hash within the repository.

3.  **`main.py`** (File Comparison):
    *   Accepts `--ground` (upstream) and `--candidate` (downstream) file paths via command-line arguments.
    *   Reads the content of the specified files.
    *   Calculates various comparison metrics:
        *   Relative Line Count Difference
        *   Token-Level Edit Distance
        *   Normalized Edit Distance
        *   CodeBERTScore (Precision, Recall, F1)
        *   Cosine Similarity using OpenAI embeddings (if token count is within limits).
    *   Prints the calculated metrics to standard output.

## Setup

1.  Clone the repository containing this code.
2.  Ensure you have Python 3 installed.
3.  Install required Python packages (you might need a `requirements.txt` file, but based on imports, you'll likely need libraries for `argparse`, `json`, potentially `torch` for CodeBERT, and `openai`).
    ```bash
    pip install openai # Add other necessary packages
    ```
4.  Set up your OpenAI API key, typically via an environment variable:
    ```bash
    export OPENAI_API_KEY='your-api-key'
    ```
    (Or use a `.env` file if the `openai` library is configured to read it).
5.  Ensure you have `git` installed and accessible in your PATH.
6.  Have a local clone of the target Git repository (e.g., the Linux kernel) accessible.

## Usage

Run the `post_eval.py` script, providing the necessary arguments.

### Required Arguments:

*   `--repo`: Path to the local clone of the Git repository (e.g., Linux kernel).
*   `--eval_report`: Path to the JSON file containing the evaluation report.

### Optional Arguments:

*   `--upstream_dir`: Directory to save the extracted upstream file versions (defaults to `input/upstream_commit`).
*   `--downstream_dir`: Directory to save the extracted downstream file versions (defaults to `input/downstream_commit`).

## Example Test Runs:

```bash
python3 post_eval.py \
  --repo /Volumes/Files/Google_Capstone/linux-stable \
  --eval_report input/report_modified.json
```

# Post eval-inline-direct (newest version april 29)
Example command:
```bash
python3 post_eval_inline_direct.py --json_input testing.json
```

## Notes:
Use file testing.json for testing purposes 
and use
- android_platform_vulnerability_report_2025_2_failures.json
as input.json as the real file being used. 

---

## LLM-Based Patch Processing (`process_json_vulns.py`)

This script processes a JSON vulnerability report, similar in structure to `android_platform_vulnerability_report_2025_2_failures.json`. For specified entries, it uses a Generative AI model (Google's Gemini) to attempt to generate patched code based on "reject file" content and "downstream file" content. The script then outputs a new JSON file identical to the input, but with an added `downstream_llm_output` field containing the LLM-generated patch for successfully processed items.

### Prerequisites

1.  **Python 3**: Ensure Python 3 is installed.
2.  **Python Packages**: Install the required packages. You can typically do this using pip:
    ```bash
    pip install google-generativeai python-dotenv logfire
    ```
    It's recommended to use a virtual environment.
3.  **API Key**: You need a Google API Key for the Gemini model. Create a `.env` file in the root of the project directory and add your API key:
    ```
    GOOGLE_API_KEY='your_google_api_key_here'
    ```
    The script uses `python-dotenv` to load this key.

### Input JSON Structure

The script expects an input JSON file with a structure that includes:
- A top-level list of vulnerability objects.
- Each vulnerability object should have an `id` and a `failures` list.
- Each `failure` object within the `failures` list should have:
    - `downstream_version`: (e.g., "14") - used for filtering.
    - `file_conflicts`: A list of objects, where each represents a file to be processed.
- Each `file_conflict` object should have:
    - `file_name`: (string) The name/path of the file.
    - `rej_file_content`: (string) The content of the .rej file (diff-like changes). This is used as the "patch content" for the LLM.
    - `downstream_file_content`: (string) The content of the vulnerable file. This is used as the "vulnerable codebase content" for the LLM.

### Output JSON Structure

The output is a new JSON file that:
- Mirrors the exact structure of the input JSON file.
- For each `file_conflict` object that was successfully processed by the LLM, an additional field `downstream_llm_output` is added. This field contains the string of the patched code generated by the LLM.
- If processing for a `file_conflict` is skipped (e.g., due to missing content) or fails, the `downstream_llm_output` field will not be added for that specific entry.

### Command-Line Usage

The script `process_json_vulns.py` accepts three command-line arguments:

1.  `json_file_path`: (Required) Path to the input JSON file.
2.  `output_json_path`: (Required) Path where the output JSON file (with LLM patches) will be saved.
3.  `target_downstream_version`: (Required) The specific `downstream_version` string (e.g., "14") to filter and process. Only failures matching this version will be considered.

**Example Command:**

```bash
python process_json_vulns.py input_vulnerabilities.json output_with_llm_patches.json 14
```

### Testing with Dummy Data

A sample input file with dummy data is provided as `test_input_for_script.json`. You can use this to test the script's functionality:

**Example Test Command:**

```bash
python process_json_vulns.py test_input_for_script.json test_output.json 14
```
After running, inspect `test_output.json` to see the `downstream_llm_output` fields added to the processed entries.

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
# Example with all arguments specified
python3 post_eval.py \
  --repo /Volumes/Files/Google_Capstone/linux-stable \
  --eval_report full_cve_report.json \
  --upstream_dir input/upstream_commit \
  --downstream_dir input/downstream_commit

# Example using default output directories
python3 post_eval.py \
  --repo /Volumes/Files/Google_Capstone/linux-stable \
  --eval_report report_modified.json

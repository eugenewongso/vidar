## Setup Instructions

### 1. Install Dependencies

Install the required Python packages using pip:

```bash
pip install google-generativeai python-dotenv logfire
```

---

### 2. Set Up Environment Variables

Create a .env file in the root of the LLM_runs_input_nonvanir_approach2 directory. Add your Google API keys as a comma-separated list. These keys will be used by the API key rotator:

```env
GOOGLE_API_KEYS="key1,key2,key3"
```

---

## Script Variants

There are three script versions you can choose from, each offering a variation of the iterative refinement approach:

### 3.a `approach2_smart_retry.py`

* Uses iterative refinement.
* Includes a line-formatting guideline in the prompt:
  *“Every line in each hunk must begin with a `+`, `-`, or space character.”*

### 3.b `approach2_smart_retry_noguideline.py`

* Uses iterative refinement.
* Omits the line-formatting guideline from the prompt.

### 3.c `approach2_smart_retry_less_error_msg.py`

* Uses iterative refinement.
* Does not include detailed error messages in validation (e.g., "hunk failed at line X").

---

To clearly explain in the README that the `20242025_combined_failures_with_ground_truth_and_patched_upstream.json` file is just an **example**, you can phrase it like this:

---

## Input File

The scripts are designed to process a JSON file containing patch failures along with ground truth and upstream patches.

An example input file is provided at:

```
./inputs/20242025_combined_failures_with_ground_truth_and_patched_upstream.json
```

This example includes patch failures for Android versions from **2024 and 2025**. You can replace this file with any other input file that follows the same structure.

---

## Output

Each script generates:

1. An updated JSON file with:

   * LLM-generated patches
   * Validation results
2. A report file summarizing:

   * Patch success and failure counts
   * Token usage
   * Error types encountered

---
Here's how you can update the **"How to Run"** section of your README to include explanations for the optional command-line arguments:

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
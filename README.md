# Vidar: AI-Powered Security Patch Automation

Vidar is an end-to-end orchestration system that automates the security patching process for downstream Android OEM maintainers. It ingests reports from patch detection tools like Vanir, applies the required patches, and uses a Large Language Model (LLM) to automatically correct patches that fail due to code divergence.

---
## Quick Start

1.  **(Prerequisite)** Ensure `git`, `python3`, and the GNU `patch` utility are installed.
2.  **(Prerequisite)** Create a `reports/` directory inside the `vidar/` directory and place your raw security report(s) (e.g., `vanir_output.json`) inside it.
3.  **Install dependencies** into a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
4.  **Configure your API Key** by creating a `.env` file:
    ```bash
    # Create the file and add your key
    echo "GOOGLE_API_KEYS=your_gemini_api_key_here" > .env
    ```
5.  **Run the pipeline:**
```bash
    # Point to the root of the source code you want to patch
    python pipeline_runner.py --source_path /path/to/your/android/source_code
```
6.  **Check the results** in the generated `reports/final_summary_report.json`.

---
## User Benefits

-   **Automated Conflict Resolution**: Vidar uses an LLM to automatically fix merge conflicts and structural divergences that plague manual patching, handling issues from simple line number mismatches to more complex semantic differences.
-   **Reduces Manual Effort**: Transforms a slow, manual, error-prone workflow into a scalable, AI-driven pipeline.
-   **Shortens Security Response Time**: Drastically reduces the time it takes for downstream maintainers to apply critical security patches, improving the security posture of their devices.

---
## Architectural Overview

Vidar operates as a multi-stage pipeline, orchestrated by `pipeline_runner.py`. Each step produces intermediate artifacts (mostly in the `reports/` directory) that feed into the next.

1.  **Parse Report**: Ingests all `vanir_output*.json` files and creates a de-duplicated master list of required patches.
2.  **Fetch Patches**: Downloads the `.diff` files for all patches from their respective Git repositories.
3.  **Apply Original Patches**: Makes a first attempt to apply the patches cleanly to the target source tree.
4.  **Prepare for LLM**: Identifies all failed patches from the previous step and bundles them with their error logs (`.rej` files) as input for the LLM.
5.  **Run LLM Correction**: For each failed patch, the LLM is prompted to analyze the original patch, the target code, and the specific error to generate a new, corrected patch.
6.  **Apply LLM Patches**: The newly generated patches are applied.
7.  **Generate Final Summary**: All results from all stages are aggregated into a final, comprehensive JSON report.

---
## User Guide

### Configuration
-   **Input Reports**: Place one or more `vanir_output*.json` files into the `vidar/reports/` directory. The parser will automatically find and process all of them. The `reports` directory will be created by the pipeline if it does not exist, but it must be created manually to hold the initial input files.
-   **Environment Variables**: The `GOOGLE_API_KEYS` variable must be set in a `.env` file in the `vidar/` directory. This file is ignored by Git.
-   **Pipeline Settings (`config.yaml`)**: Key pipeline behaviors can be modified in this file.

| Section | Key | Description |
| :--- | :--- | :--- |
| `llm_runner` | `model_name` | The specific Gemini model to use for patch correction. |
| | `temperature` | The creativity of the LLM's responses (0.0 is deterministic). |
| | `max_retries` | The number of self-correction attempts the LLM will make per failed patch. |
| | `concurrency` | The number of parallel API calls made to the LLM. |
| `patch_adopter` | `strip_level` | The `-p` level used by the `patch` command to strip leading path components from file paths in the patch. |
| | `patch_tool` | The executable command to use for applying patches (e.g., "patch"). |
| `paths` | `vanir_source_report` | **Input:** The path to the raw JSON report from Vanir. |
| | `parsed_vanir_report` | **Stage 1 Output:** The structured report for the patch fetcher. |
| | `fetched_patches_dir` | **Stage 2 Output:** Directory where downloaded `.diff` files are stored. |
| | `vanir_patch_application_report` | **Stage 3 Output:** The report on applying original patches. |
| | `llm_input_report` | **Stage 4 Output:** The filtered report of failed patches for the LLM. |
| | `llm_generated_patches_dir` | **Stage 5 Output:** Directory where new patches from the LLM are saved. |
| | `llm_successful_patches_report` | **Stage 5 Output:** Report listing successfully created LLM patches. |
| | `llm_patch_application_report` | **Stage 6 Output:** The report on applying the LLM-generated patches. |

### Understanding the Output
The most important output is `reports/final_summary_report.json`.

| Key (in `pipeline_summary`) | Description |
| :--- | :--- |
| `total_patches_from_vanir` | Total unique patches identified for processing. |
| `total_fetch_failures` | Patches that could not be downloaded. |
| `patches_applied_directly` | Patches that worked without any modification. |
| `patches_forwarded_to_llm` | Patches that failed and were sent for correction. |
| `patches_successfully_fixed_by_llm` | The number of patches the LLM successfully fixed. |
| `llm_application_failures` | Patches the LLM generated but which still failed to apply. |
| `llm_generation_failures` | Failures where the LLM could not produce a valid patch. |
| `llm_total_tokens` | The aggregate number of tokens consumed by the Gemini API across all correction attempts. |
| `llm_total_input_tokens` | The portion of tokens sent *to* the model in prompts. |
| `llm_total_output_tokens` | The portion of tokens received *from* the model in responses. |
| `total_successful_patches` | The final count of all patches applied to the source tree. |
| `pipeline_success_rate` | The end-to-end success percentage. |

The report also contains a `detailed_error_analysis` section with granular data on every failure at each stage of the pipeline.

---
## Core Components

The pipeline is a modular system orchestrated by `pipeline_runner.py`, which acts as the main CLI application. The other major components are now implemented as libraries that the pipeline runner imports and calls in sequence.

| File | Description |
| :--- | :--- |
| `pipeline_runner.py` | The main orchestrator and CLI entry point. It imports and executes the core logic from the other modules in the correct sequence, managing the overall workflow and displaying progress to the user. |
| `vanir_report_parser.py` | **(Stage 1)** A script that parses raw `vanir_output*.json` files. It extracts and de-duplicates patch information, creating a structured `parsed_report.json` for the next step. It is the only component run as a separate process by the pipeline. |
| `patch_fetcher.py` | **(Stage 2)** A library that reads the `parsed_report.json` and exposes a `run_fetcher_step` function to download all the required patch files from their upstream Git repositories. |
| `patch_adopter.py` | **(Stages 3 & 6)** A library that exposes a `run_adoption_step` function. It is called twice: first to apply the original patches, and a second time to apply patches generated by the LLM. It generates detailed reports on successes and failures. |
| `llm_patch_runner.py` | **(Stage 5)** A library exposing an async `run_llm_correction_step` function. It takes the list of failed patches and uses the Gemini LLM to generate corrected versions. It includes a self-correction loop to validate and refine the LLM's output. |
| `android_patch_manager.py` | A utility library of static methods used by other scripts. It contains helper functions for common tasks like cloning Git repos, applying patches with the `patch` command, and parsing patch content. It is not run directly. |
| `requirements.txt` | Lists the Python packages required to run the pipeline. |

---
## Guide to the `reports/` Directory
The pipeline generates several intermediate files. While the default locations are in the `reports/` directory, all paths are configurable in `config.yaml`. Here is a guide to the most important files, in the order they are typically generated.

| File | Description |
| :--- | :--- |
| `vanir_output.json` | **Input:** The raw security report from a tool like Vanir. You place this here before running the pipeline. |
| `parsed_report.json` | **Output of Stage 1.** A structured, de-duplicated list of unique patches to be processed. |
| `fetch_failures.json` | **Output of Stage 2.** A list of patches that could not be downloaded. |
| `patch_application_report.json` | **Output of Stage 3.** A detailed report on the attempt to apply the original, unaltered patches. |
| `failed_patch.json` | **Output of Stage 4.** A filtered list containing only the patches that were 'Rejected'. This file is the primary input for the LLM. |
| `llm_output_detailed.json` | **Output of Stage 5.** A comprehensive debug log from the LLM showing every self-correction attempt. |
| `successful_llm_patches.json` | **Output of Stage 5.** A clean list of only the patches that the LLM successfully generated. This is the input for the final patch application step. |
| `llm_patch_application_report.json` | **Output of Stage 6.** The report on the attempt to apply the newly generated LLM patches. |
| `final_summary_report.json` | **Output of Stage 7.** The final, consolidated report summarizing the results of the entire pipeline. This is the main file to check for results. |

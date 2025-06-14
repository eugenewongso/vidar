# Vidar Evaluation Framework

This directory contains the complete framework for evaluating the performance of LLM-generated Android security patches. It includes scripts for data preparation, running evaluations, and generating reports.

## How to Use the Scripts

Each Python script in the `data_preparation/` and `reporting/` directories is a standalone command-line tool.

1.  **View In-Script Documentation:** Every script now contains detailed documentation at the top of the file explaining its purpose, process, and usage.
2.  **Get Help from the Command Line:** You can get a list of all available command-line arguments and their descriptions by running any script with the `--help` flag.
    ```bash
    python evaluation/data_preparation/filter_reports_by_version.py --help
    ```

## Example Workflow

1.  **Install Dependencies**:
    ```bash
    pip install -r evaluation/requirements.txt
    ```
2.  **Prepare Data (Optional)**:
    If your raw data needs to be combined, filtered, or enriched, use the scripts in `data_preparation/`. This step is not necessary if your data is already in the correct format for evaluation.

3.  **Run Evaluation**:
    Execute the main evaluation script for your target.
    ```bash
    python evaluation/android/evaluate_llm_patch_success.py --input reports/augmented_v14.json --output results/evaluation_results_v14.json
    ```

4.  **Generate Reports**:
    Use the scripts in `reporting/` to analyze the results from the evaluation.
    ```bash
    python evaluation/reporting/generate_patch_success_matrix.py --report results/evaluation_results_v14.json --heatmap results/success_matrix_v14.png
    ```

## Directory Structure

```
evaluation/
├── README.md                           # This file.
├── requirements.txt                    # Python dependencies for all evaluation scripts.
├── android/                            # Scripts for evaluating Android platform patches.
│   ├── __init__.py
│   ├── evaluate_android_patches.py
│   └── evaluate_llm_patch_success.py
└── linux_kernel/                       # Scripts for evaluating Linux kernel patches. (NOTE: Outdated)
│   ├── __init__.py
│   └── evaluate_patches.py
├── data_preparation/                   # Shared utility scripts for preparing evaluation data.
│   ├── __init__.py
│   ├── analyze_vanir_report.py
│   ├── clean_diff_text.py
│   ├── combine_reports.py
│   ├── download_osv_data.py
│   ├── extract_ground_truth_file_content.py
│   ├── extract_upstream_patch_data.py
│   └── filter_reports_by_version.py
└── reporting/                          # Shared scripts for generating analysis reports.
    ├── __init__.py
    └── generate_patch_success_matrix.py
└── prompt_optimization/                # Framework for analyzing and improving LLM prompts.
    ├── __init__.py
    ├── analyze_llm_logs.py
    └── llm_meta_analyzer.py
```

## Script Descriptions

### Target-Specific Scripts (`android/`, `linux_kernel/`)

These directories contain the main evaluation scripts (e.g., `evaluate_llm_patch_success.py`). They orchestrate the end-to-end evaluation for a specific goal. For detailed usage instructions, please see the documentation within each script or the `README.md` file located inside the respective directory.

### Shared Utility Directories

*   **`data_preparation/`**: Contains standalone command-line tools for pre-processing, filtering, and enriching data *before* an evaluation is run.
*   **`reporting/`**: Contains standalone command-line tools for analysis and visualization that operate on the JSON output from an evaluation script.
*   **`prompt_optimization/`**: Contains a framework for meta-analysis of LLM failures to systematically improve prompt engineering.

*   **`data_preparation/`**: Contains scripts that are useful for both Android and Kernel evaluation, such as:
    *   `clean_diff_text.py`: Preprocesses `.diff` files.
    *   `combine_reports.py`: Merges multiple JSON reports.
    *   And other helpers you listed.
*   **`reporting/`**: Contains scripts for analysis and visualization that can operate on the output from any of the target-specific evaluation scripts, such as:
    *   `generate_patch_success_matrix.py`: Creates heatmap visualizations.
    *   `analyze_vanir_report.py`: Parses initial security reports. 
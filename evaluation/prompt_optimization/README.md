# Prompt Optimization Framework

To enhance Vidar's reliability and reduce repeated prompt debugging, we developed a Prompt Optimization Framework. This system treats the LLM not only as a patch generator but also as an agent that analyzes and improves its own prompts based on past failures.

The framework parses failure logs, identifies common prompt-related issues, and uses Gemini 2.5 Pro (`gemini-2.5-pro-preview-05-06`) to suggest changes to the `system_prompt` and `base_task_prompt`. Over time, this created a feedback loop that improved prompt clarity, patch format stability, and success rates.

## Background

Even with structured prompting and retry logic, we noticed persistent errors such as:
- Incorrect hunk length calculations in `@@ -x,y +a,b @@` headers.
- Extra or missing context lines.
- Inclusion of Markdown fences like ` ```diff `.
- Vague formatting instructions that led to inconsistent results.

Rather than continuing to adjust prompts manually, we built a system that uses logs to highlight failure patterns and passes that information to an LLM for analysis and recommendation.

## Framework Overview

The Prompt Optimization Framework consists of three stages:

### 1. Log Collection and Structuring

The `analyze_llm_logs.py` script processes JSON failure logs from past patch attempts and converts them into a structured CSV format. It captures:
- Prompt metadata (system and base prompts).
- File conflict identifiers.
- Patch statistics such as number of hunks, added/removed lines.
- Error messages and retry results.

### 2. Performance Summary Generation

Using `llm_meta_analyzer.py`, we compute:
- Success rates per unique prompt pair.
- Most frequent format and apply errors.
- Outlier prompt combinations with low success.

A textual summary of these findings is constructed for input to the LLM.

### 3. LLM-Powered Prompt Suggestions

Gemini 2.5 Pro analyzes the summary and suggests prompt modifications. The analysis includes:
- Key observations of what went wrong.
- Concrete, testable suggestions to improve prompts.
- Hypotheses for why certain prompt combinations performed poorly.

#### Example Gemini Output

Below is a short excerpt from one real Gemini 2.5 Pro meta analysis result, based only on the structured summary:

> "Okay, this is a great summary to work with. Here's an analysis based *only* on the provided information:
>
> ## Analysis of LLM Patch Generation Performance
>
> Here are my top observations, actionable suggestions, and hypotheses for the low-performing prompts:
>
> ### 1. Top 2-3 Most Critical Observations:
> …
> ### 2. Actionable Suggestions for Prompt Modification:
> …
> ### 3. Hypotheses for Low-Performing Combinations & Improvements:
> …
>
> By making the formatting requirements more explicit and example-driven, rather than just stating they are important, the LLM should have a clearer path to generating valid and correct patches."

This feedback helped inform the improvements made to our retry prompt scaffolding and validation strategy. 

### 4. Human-Readable Reporting

To complement the LLM-driven analysis, the `analyze_failed_vulns.py` script provides a human-centric view of failure patterns. This script aggregates multiple `vanir_patch_application_report.json` files and generates several outputs to help developers quickly identify the most problematic areas.

**Purpose:**
This tool answers the question: "Which vulnerabilities are causing the most failures across all our test runs?"

**Outputs:**
- **`failures_summary.json`**: A detailed JSON file that aggregates all failure data, including counts per vulnerability, failure reasons, and affected files.
- **`failures_report.html`**: An interactive HTML report with sortable tables, allowing developers to drill down into specific failures and view detailed context.
- **Charts (`/charts` directory)**:
    - A bar chart of the most frequently failing vulnerabilities (by CVE).
    - A bar chart of the most common failure reasons.
    - A treemap visualizing failure counts by file, making it easy to spot files that are consistently hard to patch.

Together, these three scripts form a robust framework for both automated, LLM-driven prompt improvement and manual, developer-led analysis of patch generation failures. 
import json
from collections import defaultdict

def summarize_results(filename):
    """
    Analyzes the output JSON from the LLM processing script and prints a summary report.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at '{filename}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{filename}'")
        return

    # Initialize summary statistics
    summary = {
        "total_file_conflicts_matching_version": 0,
        "files_attempted_for_llm_diff_generation": 0,
        "files_with_llm_diff_successfully_generated": 0,
        "files_with_llm_diff_generation_errors_or_skipped_in_func": 0,
        "successful_attempts_histogram": defaultdict(int),
        "total_runtime_seconds_all": 0,
        "total_runtime_seconds_successful": 0,
        "total_gemini_input_tokens": 0,
        "total_gemini_output_tokens": 0,
        "total_gemini_tokens": 0
    }

    # Iterate through the data to calculate stats
    for vuln in data:
        for failure in vuln.get("failures", []):
            for file_conflict in failure.get("file_conflicts", []):
                summary["total_file_conflicts_matching_version"] += 1
                
                # Check if an LLM attempt was made
                if "llm_output_valid" in file_conflict:
                    summary["files_attempted_for_llm_diff_generation"] += 1
                    
                    runtime = file_conflict.get("runtime_seconds", 0)
                    summary["total_runtime_seconds_all"] += runtime
                    
                    token_counts = file_conflict.get("token_counts", {})
                    if token_counts:
                        summary["total_gemini_input_tokens"] += token_counts.get("gemini_input_tokens", 0)
                        summary["total_gemini_output_tokens"] += token_counts.get("gemini_output_tokens", 0)
                        summary["total_gemini_tokens"] += token_counts.get("gemini_total_tokens", 0)
                        
                    if file_conflict["llm_output_valid"]:
                        summary["files_with_llm_diff_successfully_generated"] += 1
                        summary["total_runtime_seconds_successful"] += runtime
                        attempts = file_conflict.get("attempts_made", 0)
                        label = f"{attempts} run{'s' if attempts > 1 else ''}"
                        summary["successful_attempts_histogram"][label] += 1
                    else:
                        summary["files_with_llm_diff_generation_errors_or_skipped_in_func"] += 1

    # Print the summary in a nice format
    print("--- Summary Report ---")
    print(f"Total File Conflicts: {summary['total_file_conflicts_matching_version']}")
    print(f"Files Attempted by LLM: {summary['files_attempted_for_llm_diff_generation']}")
    print(f"  - Successful: {summary['files_with_llm_diff_successfully_generated']}")
    print(f"  - Failed/Errored: {summary['files_with_llm_diff_generation_errors_or_skipped_in_func']}")
    
    print("\nSuccessful Attempts Histogram:")
    for attempts, count in sorted(summary["successful_attempts_histogram"].items()):
        print(f"  - {attempts}: {count}")
        
    print("\nToken Usage (Gemini):")
    print(f"  - Input Tokens: {summary['total_gemini_input_tokens']:,}")
    print(f"  - Output Tokens: {summary['total_gemini_output_tokens']:,}")
    print(f"  - Total Tokens: {summary['total_gemini_tokens']:,}")

    print("\nRuntime:")
    print(f"  - Total (All Attempts): {summary['total_runtime_seconds_all']:.2f}s")
    print(f"  - Total (Successful Only): {summary['total_runtime_seconds_successful']:.2f}s")
    print("----------------------")


if __name__ == "__main__":
    summarize_results('top_11_vulns_2425_llm_processed_1.json') 
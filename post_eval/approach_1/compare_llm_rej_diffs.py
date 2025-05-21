import os # Ensure os is imported for path operations
import sys

# Add the parent directory (post_eval) to sys.path
# This allows imports from the sibling 'metrics' directory
current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir) # This should be 'post_eval'
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import json
import argparse
import difflib
# os is already imported above
import csv # Added for CSV output
from datetime import datetime

# Constants
MAX_TOKENS = 8192

# Helper functions (adapted from post_eval_inline_direct.py)
def count_tokens(text):
    if not isinstance(text, str):
        return 0
    return len(text.split())

def get_language_from_filename(filename):
    if not isinstance(filename, str):
        return "unknown"
    ext = filename.lower().rsplit('.', 1)[-1]
    ext_map = {
        "py": "python", "java": "java", "js": "javascript",
        "c": "c", "cpp": "cpp", "cc": "cpp", "cxx": "cpp",
        "h": "cpp", "hpp": "cpp", "cs": "csharp",
        # Add more mappings if needed for diffs (though diffs are not code)
        # For diffs, the language of the underlying code might be more relevant
        # but CodeBERT might not work well directly on diff text.
        # Defaulting to the extension itself if not in map.
    }
    return ext_map.get(ext, ext)

# Attempt to import metrics functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from metrics.line_metrics import relative_line_count_similarity
    from metrics.similarity.codeBERT import compute_codebert_score
    from metrics.similarity.openAI import compute_cosine_openai_embedding
    from metrics.distance.edit_distance import (
        token_level_edit_distance,
        normalized_edit_similarity,
    )
except ImportError as e:
    print(f"Warning: Could not import one or more metric functions: {e}. Ensure metrics module is in PYTHONPATH.")
    # Define placeholders for all if any import fails
    def relative_line_count_similarity(s1, s2): return "Error: Not implemented" # pragma: no cover
    def compute_codebert_score(s1, s2, lang): return {"error": "Not implemented"} # pragma: no cover
    def compute_cosine_openai_embedding(s1, s2): return "Error: Not implemented" # pragma: no cover
    def token_level_edit_distance(s1, s2): return "Error: Not implemented" # pragma: no cover
    def normalized_edit_similarity(s1, s2): return "Error: Not implemented" # pragma: no cover

def clean_diff_text(diff_text_str: str) -> str:
    """
    Removes standard diff headers (--- a/..., +++ b/..., --- original, +++ patched)
    and returns only the hunk content starting from the first '@@ '.
    If no '@@ ' is found, returns an empty string, as it implies no comparable hunk data.
    """
    if not isinstance(diff_text_str, str):
        return ""
    
    lines = diff_text_str.splitlines() # Work with lines without keepends for easier joining
    
    hunk_start_index = -1
    for i, line in enumerate(lines):
        if line.startswith("@@ "):
            hunk_start_index = i
            break
            
    if hunk_start_index != -1:
        # If a hunk header is found, take all lines from there and rejoin
        return "\n".join(lines[hunk_start_index:])
    else:
        # No hunk data found (e.g., empty diff, or diff only showed file mode changes)
        return ""

def compute_diff_metrics(rej_diff_str, llm_diff_str, file_name_for_lang):
    """
    Computes various metrics by comparing two diff strings.
    rej_diff_str is treated as the reference (s1/original).
    llm_diff_str is treated as the candidate (s2/modified).
    """
    metrics = {}

    # Ensure inputs are strings, default to empty string if None
    s1 = rej_diff_str if isinstance(rej_diff_str, str) else ""
    s2 = llm_diff_str if isinstance(llm_diff_str, str) else ""

    try:
        rls = relative_line_count_similarity(s2, s1) # Compares line counts of s2 vs s1
        metrics["relative_line_count_similarity"] = round(rls, 4) if isinstance(rls, float) else rls
    except Exception as e:
        metrics["relative_line_count_similarity"] = f"Error: {e}"

    try:
        nes = normalized_edit_similarity(s2, s1) # Similarity of s2 to s1
        metrics["normalized_edit_similarity"] = round(nes, 4) if isinstance(nes, float) else nes
    except Exception as e:
        metrics["normalized_edit_similarity"] = f"Error: {e}"
        
    try:
        tled = token_level_edit_distance(s2, s1) # Distance of s2 from s1
        metrics["token_level_edit_distance"] = round(tled, 4) if isinstance(tled, float) else tled
    except Exception as e:
        metrics["token_level_edit_distance"] = f"Error: {e}"

    language = get_language_from_filename(file_name_for_lang)
    # Note: Applying CodeBERT to diff text directly. Its effectiveness might vary.
    try:
        codebert = compute_codebert_score(s2, s1, language) # Semantic similarity of s2 to s1
        if isinstance(codebert, dict) and "error" not in codebert:
            metrics["codebert_score"] = {k: round(v, 4) if isinstance(v, (int, float)) else v for k, v in codebert.items()}
        else:
            metrics["codebert_score"] = codebert # Store error or unexpected result
    except Exception as e:
        metrics["codebert_score"] = {"error": f"Exception: {e}"}


    tokens_s1 = count_tokens(s1) # rej_diff_str
    tokens_s2 = count_tokens(s2) # llm_diff_str
    total_tokens = tokens_s1 + tokens_s2
    
    metrics["token_count_ref_diff"] = tokens_s1
    metrics["token_count_llm_diff"] = tokens_s2
    metrics["token_count_total"] = total_tokens
    
    try:
        if total_tokens > MAX_TOKENS:
            metrics["cosine_similarity_openai"] = "skipped_max_tokens"
        else:
            score = compute_cosine_openai_embedding(s2, s1) # Embedding similarity of s2 to s1
            metrics["cosine_similarity_openai"] = round(score, 4) if isinstance(score, float) else score
    except Exception as e:
        metrics["cosine_similarity_openai"] = f"Error: {e}"
        
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Compare LLM-generated diffs with original .rej diffs and compute metrics.")
    parser.add_argument("input_json_path", help="Path to the input JSON file (output from approach1.py)")
    args = parser.parse_args()

    try:
        with open(args.input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.input_json_path}")
        return

    all_evaluation_entries = []
    
    # For console summary
    conflicts_processed_count = 0
    conflicts_compared_count = 0
    conflicts_skipped_count = 0

    for vuln_item in data:
        vuln_id = vuln_item.get("id", "UnknownVulnerability")
        
        for failure_item in vuln_item.get("failures", []):
            downstream_version = failure_item.get("downstream_version", "UnknownVersion")

            for file_conflict_item in failure_item.get("file_conflicts", []):
                conflicts_processed_count += 1
                file_name = file_conflict_item.get("file_name", "UnknownFile")
                rej_content_raw = file_conflict_item.get("rej_file_content")
                llm_diff_content_raw = file_conflict_item.get("LLM_diff_content")
                
                llm_output_status = file_conflict_item.get("downstream_patched_file_llm_output", "")
                is_llm_skipped = isinstance(llm_output_status, str) and llm_output_status.startswith("skipped,")

                # Clean the diff texts
                cleaned_rej_content = clean_diff_text(rej_content_raw)
                cleaned_llm_diff_content = clean_diff_text(llm_diff_content_raw)

                # Prepare the base entry structure
                current_eval_entry = {
                    "cve_id": vuln_id,
                    "downstream_version": downstream_version,
                    "file_name": file_name,
                    "ref_diff_file": cleaned_rej_content, # Store cleaned content
                    "LLM_diff_file": cleaned_llm_diff_content, # Store cleaned content
                    "metrics": {} 
                }

                # Determine if metrics can be computed
                # Metrics computation depends on having non-empty cleaned diffs,
                # and LLM not being skipped (as LLM_diff_content might be irrelevant then).
                # An empty rej_content is valid (means no changes expected by .rej).
                # An empty llm_diff_content is valid (means LLM proposed no changes or identical file).
                # The key is whether both are available and LLM wasn't skipped for other reasons.

                can_compute_metrics = True
                skip_note = "Metrics "
                
                if is_llm_skipped:
                    can_compute_metrics = False
                    skip_note += f"not computed due to: LLM processing was skipped ({llm_output_status})."
                # If rej_content_raw was None or not a string, cleaned_rej_content will be empty.
                # If llm_diff_content_raw was None or not a string (and not an LLM skip case), cleaned_llm_diff_content will be empty.
                # The compute_diff_metrics function handles empty strings for s1 and s2.
                # We only truly skip if the LLM itself was skipped for a processing reason.
                # Otherwise, we attempt to compute metrics even if one of the diffs is empty (e.g. rej is empty, LLM made changes).

                if can_compute_metrics:
                    conflicts_compared_count +=1
                    metrics_result = compute_diff_metrics(cleaned_rej_content, cleaned_llm_diff_content, file_name)
                    current_eval_entry["metrics"] = metrics_result
                    current_eval_entry["metrics_status"] = "computed"
                    # Add raw diffs if needed for inspection, or keep them out of the main record
                    # current_eval_entry["raw_ref_diff_file"] = rej_content_raw 
                    # current_eval_entry["raw_LLM_diff_file"] = llm_diff_content_raw
                else:
                    conflicts_skipped_count +=1
                    current_eval_entry["metrics_status"] = skip_note.strip()
                    # Ensure metrics dict is empty if not computed
                    current_eval_entry["metrics"] = {} 
                
                all_evaluation_entries.append(current_eval_entry)

    # Save the list of evaluation entries to a JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join("comparison_reports_diff")
    os.makedirs(report_dir, exist_ok=True)
    
    base_input_filename = os.path.splitext(os.path.basename(args.input_json_path))[0]
    report_filename = os.path.join(report_dir, f"comparison_metrics_{base_input_filename}_{timestamp}.json")

    try:
        with open(report_filename, 'w', encoding='utf-8') as f_out:
            json.dump(all_evaluation_entries, f_out, indent=4)
        print(f"✅ Comparison metrics report saved to: {report_filename}")
    except IOError as e:
        print(f"Error writing comparison report to '{report_filename}': {e}")
    
    print(f"\nSummary of processing:")
    print(f"  Total file conflicts encountered: {conflicts_processed_count}")
    print(f"  File conflicts where metrics were computed: {conflicts_compared_count}")
    print(f"  File conflicts skipped for metrics computation: {conflicts_skipped_count}")

    # Write to CSV
    csv_report_filename = os.path.join(report_dir, f"comparison_metrics_{base_input_filename}_{timestamp}.csv")
    if all_evaluation_entries:
        # Prepare data for CSV (flattened)
        csv_data = []
        # Dynamically determine fieldnames from the first entry's metrics, plus base fields
        # This assumes all metric dicts will have mostly the same keys if computed.
        # Fallback if metrics is empty or not a dict.
        
        # Define base field names
        base_fieldnames = ["cve_id", "downstream_version", "file_name", "metrics_status"]
        
        # Attempt to get metric field names from the first entry with computed metrics
        metric_fieldnames = []
        for entry in all_evaluation_entries:
            if entry.get("metrics_status") == "computed" and isinstance(entry.get("metrics"), dict):
                for k in entry["metrics"].keys():
                    if k == "codebert_score" and isinstance(entry["metrics"][k], dict):
                        for cb_k in entry["metrics"][k].keys():
                            metric_fieldnames.append(f"codebert_{cb_k}")
                    else:
                        metric_fieldnames.append(k)
                break # Got fieldnames from the first valid entry
        
        # If no metrics were computed, metric_fieldnames will be empty.
        # Ensure a default set if all were skipped or errored to avoid empty header.
        if not metric_fieldnames: # Fallback if no metrics computed across all entries
            metric_fieldnames = [
                "relative_line_count_similarity", "normalized_edit_similarity", 
                "token_level_edit_distance", "codebert_precision", "codebert_recall", 
                "codebert_f1", "codebert_f3", "token_count_ref_diff", 
                "token_count_llm_diff", "token_count_total", "cosine_similarity_openai"
            ]


        all_fieldnames = base_fieldnames + sorted(list(set(metric_fieldnames))) # Ensure uniqueness and order

        for entry in all_evaluation_entries:
            flat_row = {
                "cve_id": entry.get("cve_id"),
                "downstream_version": entry.get("downstream_version"),
                "file_name": entry.get("file_name"),
                "metrics_status": entry.get("metrics_status")
            }
            metrics_dict = entry.get("metrics", {})
            if isinstance(metrics_dict, dict):
                for key, value in metrics_dict.items():
                    if key == "codebert_score" and isinstance(value, dict):
                        for cb_key, cb_value in value.items():
                            flat_row[f"codebert_{cb_key}"] = cb_value
                    else:
                        flat_row[key] = value
            csv_data.append(flat_row)

        try:
            with open(csv_report_filename, 'w', newline='', encoding='utf-8') as cf:
                writer = csv.DictWriter(cf, fieldnames=all_fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(csv_data)
            print(f"✅ Comparison metrics CSV report saved to: {csv_report_filename}")
        except IOError as e:
            print(f"Error writing comparison CSV report to '{csv_report_filename}': {e}")
    else:
        print("No evaluation entries to write to CSV.")

if __name__ == "__main__":
    main()

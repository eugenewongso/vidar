import json
import argparse
import os
import csv
from datetime import datetime
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Constants
MAX_TOKENS = 8192

# Helper functions
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
    }
    return ext_map.get(ext, ext)

def clean_code(code_str: str) -> str:
    """
    Basic cleaning for code content, e.g., removing markdown backticks.
    """
    if not isinstance(code_str, str):
        return ""
    code = code_str.strip()
    if code.startswith("```"):
        # Remove the first line if it's just ``` or ```<lang>
        lines = code.splitlines()
        if lines[0].strip() == "```" or lines[0].strip().startswith("```"):
            code = "\n".join(lines[1:])
    if code.endswith("```"):
        code = code.rsplit("```", 1)[0].strip()
    return code.strip()

# Attempt to import metrics functions
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
    # Define placeholders
    def relative_line_count_similarity(s1, s2): return "Error: Not implemented" # pragma: no cover
    def compute_codebert_score(s1, s2, lang): return {"error": "Not implemented"} # pragma: no cover
    def compute_cosine_openai_embedding(s1, s2): return "Error: Not implemented" # pragma: no cover
    def token_level_edit_distance(s1, s2): return "Error: Not implemented" # pragma: no cover
    def normalized_edit_similarity(s1, s2): return "Error: Not implemented" # pragma: no cover


def compute_code_comparison_metrics(ground_truth_code_str, llm_patched_code_str, file_name_for_lang):
    """
    Computes various metrics by comparing two code strings.
    ground_truth_code_str is s1 (reference).
    llm_patched_code_str is s2 (candidate).
    """
    metrics = {}

    s1 = clean_code(ground_truth_code_str)
    s2 = clean_code(llm_patched_code_str)

    try:
        rls = relative_line_count_similarity(s2, s1)
        metrics["relative_line_count_similarity"] = round(rls, 4) if isinstance(rls, float) else rls
    except Exception as e:
        metrics["relative_line_count_similarity"] = f"Error: {e}"

    try:
        nes = normalized_edit_similarity(s2, s1)
        metrics["normalized_edit_similarity"] = round(nes, 4) if isinstance(nes, float) else nes
    except Exception as e:
        metrics["normalized_edit_similarity"] = f"Error: {e}"
        
    try:
        tled = token_level_edit_distance(s2, s1)
        metrics["token_level_edit_distance"] = round(tled, 4) if isinstance(tled, float) else tled
    except Exception as e:
        metrics["token_level_edit_distance"] = f"Error: {e}"

    language = get_language_from_filename(file_name_for_lang)
    try:
        codebert = compute_codebert_score(s2, s1, language)
        if isinstance(codebert, dict) and "error" not in codebert:
            metrics["codebert_score"] = {k: round(v, 4) if isinstance(v, (int, float)) else v for k, v in codebert.items()}
        else:
            metrics["codebert_score"] = codebert
    except Exception as e:
        metrics["codebert_score"] = {"error": f"Exception: {e}"}

    tokens_s1 = count_tokens(s1)
    tokens_s2 = count_tokens(s2)
    total_tokens = tokens_s1 + tokens_s2
    
    metrics["token_count_ground_truth"] = tokens_s1
    metrics["token_count_llm_output"] = tokens_s2
    metrics["token_count_total"] = total_tokens
    
    try:
        if total_tokens > MAX_TOKENS:
            metrics["cosine_similarity_openai"] = "skipped_max_tokens"
        else:
            score = compute_cosine_openai_embedding(s2, s1)
            metrics["cosine_similarity_openai"] = round(score, 4) if isinstance(score, float) else score
    except Exception as e:
        metrics["cosine_similarity_openai"] = f"Error: {e}"
        
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Compare LLM-patched code against ground truth code and compute metrics.")
    parser.add_argument("input_json_path", help="Path to the input JSON file containing ground truth and LLM output.")
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
    
    conflicts_processed_count = 0
    conflicts_evaluated_count = 0
    conflicts_skipped_evaluation_count = 0

    for vuln_item in data:
        vuln_id = vuln_item.get("id", "UnknownVulnerability")
        
        for failure_item in vuln_item.get("failures", []):
            downstream_version = failure_item.get("downstream_version", "UnknownVersion")

            for file_conflict_item in failure_item.get("file_conflicts", []):
                conflicts_processed_count += 1
                file_name = file_conflict_item.get("file_name", "UnknownFile")
                
                ground_truth_content = file_conflict_item.get("downstream_file_content_ground_truth")
                llm_patched_content = file_conflict_item.get("downstream_patched_file_llm_output")
                
                # llm_output_status is the string like "skipped, <reason>" or the actual code
                # is_llm_output_code helps distinguish between actual code and a skip message
                is_llm_output_code = not (isinstance(llm_patched_content, str) and llm_patched_content.startswith("skipped,"))

                current_eval_entry = {
                    "cve_id": vuln_id,
                    "downstream_version": downstream_version,
                    "file_name": file_name,
                    "ground_truth_codebase": clean_code(ground_truth_content),
                    "llm_patched_codebase": clean_code(llm_patched_content) if is_llm_output_code else llm_patched_content, # Store skip message as is
                    "metrics": {} 
                }

                can_compute_metrics = True
                skip_reason_for_metrics = "Metrics "

                if ground_truth_content is None:
                    can_compute_metrics = False
                    skip_reason_for_metrics += "not computed: 'downstream_file_content_ground_truth' is missing. "
                elif not is_llm_output_code:
                    can_compute_metrics = False
                    skip_reason_for_metrics += f"not computed: LLM output was '{llm_patched_content}'. "
                elif llm_patched_content is None : # Should be caught by is_llm_output_code if it's a skip message
                    can_compute_metrics = False
                    skip_reason_for_metrics += "not computed: 'downstream_patched_file_llm_output' is missing. "
                
                if can_compute_metrics:
                    conflicts_evaluated_count += 1
                    # Pass raw content to clean_code inside compute_code_comparison_metrics
                    metrics_result = compute_code_comparison_metrics(ground_truth_content, llm_patched_content, file_name)
                    current_eval_entry["metrics"] = metrics_result
                    current_eval_entry["metrics_status"] = "computed"
                else:
                    conflicts_skipped_evaluation_count += 1
                    current_eval_entry["metrics_status"] = skip_reason_for_metrics.strip()
                    current_eval_entry["metrics"] = {} # Ensure metrics is empty
                
                all_evaluation_entries.append(current_eval_entry)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join("comparison_reports_diff") # New directory
    os.makedirs(report_dir, exist_ok=True)
    
    base_input_filename = os.path.splitext(os.path.basename(args.input_json_path))[0]
    json_report_filename = os.path.join(report_dir, f"gt_comparison_metrics_{base_input_filename}_{timestamp}.json")
    csv_report_filename = os.path.join(report_dir, f"gt_comparison_metrics_{base_input_filename}_{timestamp}.csv")

    try:
        with open(json_report_filename, 'w', encoding='utf-8') as f_out:
            json.dump(all_evaluation_entries, f_out, indent=4)
        print(f"✅ Ground truth comparison JSON report saved to: {json_report_filename}")
    except IOError as e:
        print(f"Error writing JSON report: {e}")
    
    print(f"\nSummary of processing:")
    print(f"  Total file conflicts processed: {conflicts_processed_count}")
    print(f"  File conflicts evaluated for metrics: {conflicts_evaluated_count}")
    print(f"  File conflicts skipped for metrics evaluation: {conflicts_skipped_evaluation_count}")

    if all_evaluation_entries:
        csv_data = []
        base_fieldnames = ["cve_id", "downstream_version", "file_name", "metrics_status"]
        metric_fieldnames_set = set()

        for entry in all_evaluation_entries:
            if entry.get("metrics_status") == "computed" and isinstance(entry.get("metrics"), dict):
                for k in entry["metrics"].keys():
                    if k == "codebert_score" and isinstance(entry["metrics"][k], dict):
                        for cb_k in entry["metrics"][k].keys():
                            metric_fieldnames_set.add(f"codebert_{cb_k}")
                    else:
                        metric_fieldnames_set.add(k)
        
        # Fallback default metric names if none were computed
        if not metric_fieldnames_set:
             metric_fieldnames_set.update([
                "relative_line_count_similarity", "normalized_edit_similarity", 
                "token_level_edit_distance", "codebert_precision", "codebert_recall", 
                "codebert_f1", "codebert_f3", "token_count_ground_truth", 
                "token_count_llm_output", "token_count_total", "cosine_similarity_openai"
            ])

        all_fieldnames = base_fieldnames + sorted(list(metric_fieldnames_set))

        for entry in all_evaluation_entries:
            flat_row = {bf: entry.get(bf) for bf in base_fieldnames}
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
            print(f"✅ Ground truth comparison CSV report saved to: {csv_report_filename}")
        except IOError as e:
            print(f"Error writing CSV report: {e}")
    else:
        print("No evaluation entries to write to CSV.")

if __name__ == "__main__":
    main()

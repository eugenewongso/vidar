r"""Processes raw LLM evaluation JSON logs into a detailed CSV file.

This script is the first stage in the Prompt Optimization Framework. It iterates
through a directory of JSON logs, where each log represents the output of an
LLM patch generation run. It parses these complex, nested logs to extract a
wealth of data about each patch attempt.

The process for each JSON file is as follows:
1.  Extracts global metadata like the system and base prompts used for the run.
2.  Traverses the list of vulnerabilities and their associated file conflicts.
3.  For each file conflict, it analyzes the rejected patch (`.rej` file) and the
    ground truth patch to calculate metrics like hunk counts and line changes.
4.  It then unnests the LLM's individual validation attempts, recording the
    outcome, format errors, and apply errors for each retry.
5.  All of this structured data is written row-by-row into a single, comprehensive
    CSV file, which serves as the input for the next stage of analysis.
"""

import json
import csv
import os
import io
import re
from unidiff import PatchSet

# --- Helper Functions ---

def clean_patch_content(patch_content: str, target_file: str) -> str:
    """
    Extracts the specific patch section for the target_file from a larger patch_content.
    It now also handles and strips git commit message headers before parsing.
    """
    if not patch_content or not isinstance(patch_content, str):
        return "" # Return empty string if no content or not a string

    # Attempt to find the start of the actual diff content
    # Common patch formats start with "diff --git" or "--- a/"
    # We look for "diff --git" which is common in full commit patches.
    # Ensure we're looking for it at the start of a line.
    processed_patch_content = patch_content
    diff_start_marker = "\\ndiff --git"
    marker_pos = patch_content.find(diff_start_marker)
    if marker_pos != -1:
        # If found, take everything from this marker (inclusive of the newline for split)
        # but PatchSet expects the "diff --git" line itself.
        processed_patch_content = patch_content[marker_pos+1:] # +1 to exclude the leading \\n
    else:
        # Fallback: if "diff --git" is not found, maybe it's a "--- a/" style patch already
        # or a patch without the "diff --git" prefix. Unidiff might handle it.
        # Or, if the "diff --git" is at the very beginning.
        if patch_content.startswith("diff --git"):
            processed_patch_content = patch_content
        # If neither, we proceed with original content but unidiff might struggle
        # if there's still a header unidiff doesn't like.

    def normalize(p):
        if not p: return ""
        return p.strip().lstrip("ab/")

    try:
        patch_set = PatchSet(io.StringIO(processed_patch_content))
        for patched_file_obj in patch_set: # Renamed to avoid conflict
            if (
                normalize(patched_file_obj.path) == normalize(target_file)
                or normalize(patched_file_obj.source_file) == normalize(target_file)
                or normalize(patched_file_obj.target_file) == normalize(target_file)
            ):
                return str(patched_file_obj).strip()
        return "" # Return empty if no matching file found in patch set
    except Exception as e:
        print(f"⚠️ Error parsing patch with unidiff for target '{target_file}': {e}")
        print(f"   Problematic patch content snippet (processed): {processed_patch_content[:200]}")
        return "" # Return empty on error


def clean_diff_text(diff_text_str: str) -> str:
    """
    Removes standard diff headers (--- a/..., +++ b/..., --- original, +++ patched)
    and returns only the hunk content starting from the first '@@ '.
    If no '@@ ' is found, returns an empty string, as it implies no comparable hunk data.
    """

    # print(diff_text_str)
    if not isinstance(diff_text_str, str):
        return ""
    
    lines = diff_text_str.splitlines() # Work with lines without keepends for easier joining
    # print("lines", lines)
    
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

# --- New Diff Analysis Helper Functions ---

def count_hunks_in_diff(diff_content: str) -> int:
    """
    Counts the number of hunks (lines starting with '@@ ') in a diff string.
    Assumes diff_content is hunk-only data or a full diff.
    """
    if not diff_content or not isinstance(diff_content, str):
        return 0
    return len(re.findall(r"^@@ .* @@", diff_content, re.MULTILINE))

def count_lines_in_hunks(diff_content: str) -> tuple[int, int, int]:
    """
    Counts total, added (+), and removed (-) lines within hunks of a diff string.
    Assumes diff_content contains only hunk data (e.g., after clean_diff_text).
    """
    if not diff_content or not isinstance(diff_content, str):
        return 0, 0, 0
        
    print(f"DEBUG_count_lines: Input diff_content (first 300 chars): {diff_content[:300]}")
    total_lines_in_hunks = 0
    added_lines = 0
    removed_lines = 0
    
    in_hunk_body = False
    lines = diff_content.splitlines() # Use splitlines to handle newlines consistently

    print(f"DEBUG_count_lines: Processing {len(lines)} lines total for counting.")

    for i, line in enumerate(lines):
        # print(f"DEBUG_count_lines: Line #{i+1}: '{line}' (in_hunk_body: {in_hunk_body})")
        if line.startswith("@@ "):
            in_hunk_body = True # Entered a hunk, subsequent lines are part of its body
            # Don't count the "@@ " line itself as part of total/added/removed lines in body
            continue 
        
        if in_hunk_body:
            # If a line (that is not an @@ line) starts with +, -, or space, count it.
            if line.startswith('+'):
                added_lines += 1
                total_lines_in_hunks +=1
                # print(f"DEBUG_count_lines: Counted ADD: {line}")
            elif line.startswith('-'):
                removed_lines += 1
                total_lines_in_hunks += 1
                # print(f"DEBUG_count_lines: Counted REMOVE: {line}")
            elif line.startswith(' '): # Context line
                total_lines_in_hunks += 1
                # print(f"DEBUG_count_lines: Counted CONTEXT: {line}")
            # Lines within a hunk body that don't start with +, -, or space (e.g., \\ No newline) are ignored.
            
    print(f"DEBUG_count_lines: Final counts - TotalInHunks: {total_lines_in_hunks}, Added: {added_lines}, Removed: {removed_lines}")
    return total_lines_in_hunks, added_lines, removed_lines


def strip_markdown_code_block(content: str) -> str:
    """
    Removes the ```diff ... ``` markdown fences if present.
    """
    if not content or not isinstance(content, str):
        return ""
    
    if content.startswith("```diff\n") and content.endswith("\n```"):
        return content[len("```diff\n"):-len("\n```")]
    elif content.startswith("```") and content.endswith("```"): # Generic case
        # Find first newline after ```
        first_newline = content.find('\n')
        if first_newline != -1:
            return content[first_newline+1:-len("\n```")]
    return content


# --- Main Processing Logic ---

def process_json_file(json_file_path: str, csv_writer, global_prompts: dict):
    """
    Processes a single JSON log file and writes its data to the CSV.
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading or parsing JSON file {json_file_path}: {e}")
        return

    system_prompt = global_prompts.get('system_prompt', "MISSING_IN_SUMMARY")
    base_task_prompt = global_prompts.get('base_task_prompt', "MISSING_IN_SUMMARY")

    all_vulnerabilities = data.get("failures", []) # Changed from "vulnerability_details" to "failures" as per JSON structure
    if not isinstance(all_vulnerabilities, list):
        print(f"⚠️ 'failures' key not found or not a list in {json_file_path}. Skipping.")
        all_vulnerabilities = []


    for vuln_item in all_vulnerabilities:
        if not isinstance(vuln_item, dict):
            # print(f"⚠️ Skipping non-dictionary item in top-level 'failures' list in {json_file_path}")
            continue
        
        vuln_id = vuln_item.get("id", "UnknownVulnerabilityID")
        
        # Ground truth patch content for this vulnerability (applies to all its file conflicts)
        # This is the 'upstream_patch_content' from the main vulnerability item
        vuln_ground_truth_patch_full = vuln_item.get("upstream_patch_content", "")


        downstream_contexts = vuln_item.get("failures", []) # This is the nested "failures" array
        if not isinstance(downstream_contexts, list):
            # print(f"⚠️ Nested 'failures' key not found or not a list for vuln {vuln_id} in {json_file_path}. Skipping this vulnerability.")
            continue

        for ds_context in downstream_contexts:
            if not isinstance(ds_context, dict):
                # print(f"⚠️ Skipping non-dictionary item in nested 'failures' for vuln {vuln_id} in {json_file_path}")
                continue

            downstream_version = ds_context.get("downstream_version", "N/A")
            
            # This is the ground truth patch that *was* applied in the downstream version successfully.
            # The existing 'ground_truth_*' fields will use this if available, or fallback to upstream.
            actual_ds_patch_content = ds_context.get("downstream_patch_content", "")

            downstream_successful_patch_content_full = actual_ds_patch_content
            if not downstream_successful_patch_content_full: # Fallback to upstream if specific downstream is empty
                downstream_successful_patch_content_full = vuln_ground_truth_patch_full


            file_conflicts = ds_context.get("file_conflicts", [])
            if not isinstance(file_conflicts, list):
                # print(f"⚠️ 'file_conflicts' not found or not a list for vuln {vuln_id}, version {downstream_version} in {json_file_path}. Skipping.")
                continue
                
            for fc_item in file_conflicts:
                if not isinstance(fc_item, dict):
                    # print(f"⚠️ Skipping non-dictionary item in 'file_conflicts' for vuln {vuln_id}, version {downstream_version} in {json_file_path}")
                    continue

                file_name = fc_item.get("file_name", "UnknownFile")
                
                # .rej characteristics
                original_patch_total_hunks_for_file = fc_item.get("total_hunks", 0) # total hunks for this file in original patch
                original_patch_failed_hunk_numbers = fc_item.get("failed_hunks", [])
                rej_file_reported_hunk_count = len(original_patch_failed_hunk_numbers)
                
                raw_rej_content = fc_item.get("rej_file_content", "")
                stripped_rej_content = strip_markdown_code_block(raw_rej_content)
                cleaned_rej_hunks_only = clean_diff_text(stripped_rej_content)
                print(f"DEBUG_MAIN: Vuln {vuln_id}, File {file_name}, cleaned_rej_hunks_only (first 300): {cleaned_rej_hunks_only[:300]}") # Keep this from user
                
                # Print count_hunks_in_diff for rej data
                rej_actual_hunks = count_hunks_in_diff(cleaned_rej_hunks_only)
                print(f"DEBUG_MAIN: Vuln {vuln_id}, File {file_name}, count_hunks_in_diff(cleaned_rej_hunks_only) = {rej_actual_hunks}")

                rej_file_actual_hunk_count = rej_actual_hunks # Storing the already calculated value
                rej_total_lines, rej_added, rej_removed = count_lines_in_hunks(cleaned_rej_hunks_only)

                # Ground truth characteristics for THIS file
                # Use the downstream_successful_patch_content_full which might be the specific downstream one or the fallback upstream one
                # print(f"DEBUG: Vuln {vuln_id}, File {file_name}, Raw GT Patch Content (first 500 chars):\\n{downstream_successful_patch_content_full[:500]}")
                gt_patch_for_file_raw = clean_patch_content(downstream_successful_patch_content_full, file_name)
                # print(f"DEBUG: Vuln {vuln_id}, File {file_name}, clean_patch_content output (first 500 chars):\\n{gt_patch_for_file_raw[:500]}")
                gt_patch_for_file_hunks_only = clean_diff_text(gt_patch_for_file_raw)
                print(f"DEBUG_MAIN: Vuln {vuln_id}, File {file_name}, gt_patch_for_file_hunks_only (first 300): {gt_patch_for_file_hunks_only[:300]}")
                
                # Print count_hunks_in_diff for GT data
                gt_actual_hunks = count_hunks_in_diff(gt_patch_for_file_hunks_only)
                print(f"DEBUG_MAIN: Vuln {vuln_id}, File {file_name}, count_hunks_in_diff(gt_patch_for_file_hunks_only) = {gt_actual_hunks}")
                
                gt_hunk_count = gt_actual_hunks # Storing the already calculated value
                gt_total_lines, gt_added, gt_removed = count_lines_in_hunks(gt_patch_for_file_hunks_only)

                # --- Downstream-Specific Ground Truth Characteristics ---
                ds_spec_gt_cleaned_hunks = ""
                ds_spec_gt_hunk_count = 0
                ds_spec_gt_total_lines = 0
                ds_spec_gt_added_lines = 0
                ds_spec_gt_removed_lines = 0

                if actual_ds_patch_content: # Only process if there's actual downstream patch content
                    ds_spec_gt_patch_for_file_raw = clean_patch_content(actual_ds_patch_content, file_name)
                    ds_spec_gt_cleaned_hunks = clean_diff_text(ds_spec_gt_patch_for_file_raw)
                    print(f"DEBUG_MAIN: Vuln {vuln_id}, File {file_name}, downstream_specific_gt_cleaned_hunks (first 300): {ds_spec_gt_cleaned_hunks[:300]}")
                    
                    ds_spec_gt_actual_hunks = count_hunks_in_diff(ds_spec_gt_cleaned_hunks)
                    print(f"DEBUG_MAIN: Vuln {vuln_id}, File {file_name}, count_hunks_in_diff(downstream_specific_gt_cleaned_hunks) = {ds_spec_gt_actual_hunks}")
                    
                    ds_spec_gt_hunk_count = ds_spec_gt_actual_hunks
                    ds_spec_gt_total_lines, ds_spec_gt_added_lines, ds_spec_gt_removed_lines = count_lines_in_hunks(ds_spec_gt_cleaned_hunks)


                # Overall outcome for this file_conflict
                fc_llm_output_valid = fc_item.get("llm_output_valid", False)
                fc_attempts_made_overall = fc_item.get("attempts_made", 0)
                fc_error_overall = fc_item.get("error", "")
                fc_last_format_error_overall = fc_item.get("last_format_error", "")
                fc_last_apply_error_overall = fc_item.get("last_apply_error", "")
                fc_runtime_overall = fc_item.get("runtime_seconds", 0.0)
                
                # Individual attempt details
                validation_results = fc_item.get("validation_results", [])
                if not isinstance(validation_results, list) or not validation_results: # No attempts made or results missing
                    row = {
                        "json_file_source": os.path.basename(json_file_path),
                        "system_prompt": system_prompt,
                        "base_task_prompt": base_task_prompt,
                        "vulnerability_id": vuln_id,
                        "downstream_version": downstream_version,
                        "target_file_name": file_name,
                        "original_patch_total_hunks_for_file": original_patch_total_hunks_for_file,
                        "original_patch_failed_hunk_numbers": ','.join(map(str, original_patch_failed_hunk_numbers)),
                        "rej_file_reported_hunk_count": rej_file_reported_hunk_count,
                        "rej_file_actual_hunk_count": rej_file_actual_hunk_count,
                        "rej_file_total_lines": rej_total_lines,
                        "rej_file_added_lines": rej_added,
                        "rej_file_removed_lines": rej_removed,
                        "ground_truth_hunk_count": gt_hunk_count,
                        "ground_truth_total_lines": gt_total_lines,
                        "ground_truth_added_lines": gt_added,
                        "ground_truth_removed_lines": gt_removed,
                        "downstream_specific_gt_cleaned_hunks": ds_spec_gt_cleaned_hunks,
                        "downstream_specific_gt_hunk_count": ds_spec_gt_hunk_count,
                        "downstream_specific_gt_total_lines": ds_spec_gt_total_lines,
                        "downstream_specific_gt_added_lines": ds_spec_gt_added_lines,
                        "downstream_specific_gt_removed_lines": ds_spec_gt_removed_lines,
                        "fc_llm_output_valid_overall": fc_llm_output_valid,
                        "fc_attempts_made_overall": fc_attempts_made_overall,
                        "fc_error_overall": fc_error_overall,
                        "fc_last_format_error_overall": fc_last_format_error_overall,
                        "fc_last_apply_error_overall": fc_last_apply_error_overall,
                        "fc_runtime_total_sec": fc_runtime_overall,
                        "attempt_number": 0, # Indicates no attempt or pre-LLM skip
                        "attempt_format_valid": False,
                        "attempt_format_error": "No LLM attempt made or logged",
                        "attempt_apply_valid": False,
                        "attempt_apply_error": "No LLM attempt made or logged",
                        "attempt_valid_overall": False,
                        "attempt_runtime_sec": 0.0
                    }
                    csv_writer.writerow(row)
                else:
                    for attempt_detail in validation_results:
                        if not isinstance(attempt_detail, dict):
                            # print(f"⚠️ Skipping non-dictionary item in 'validation_results' for {file_name} in {vuln_id}")
                            continue
                        row = {
                            "json_file_source": os.path.basename(json_file_path),
                            "system_prompt": system_prompt,
                            "base_task_prompt": base_task_prompt,
                            "vulnerability_id": vuln_id,
                            "downstream_version": downstream_version,
                            "target_file_name": file_name,
                            "original_patch_total_hunks_for_file": original_patch_total_hunks_for_file,
                            "original_patch_failed_hunk_numbers": ','.join(map(str, original_patch_failed_hunk_numbers)),
                            "rej_file_reported_hunk_count": rej_file_reported_hunk_count,
                            "rej_file_actual_hunk_count": rej_file_actual_hunk_count,
                            "rej_file_total_lines": rej_total_lines,
                            "rej_file_added_lines": rej_added,
                            "rej_file_removed_lines": rej_removed,
                            "ground_truth_hunk_count": gt_hunk_count,
                            "ground_truth_total_lines": gt_total_lines,
                            "ground_truth_added_lines": gt_added,
                            "ground_truth_removed_lines": gt_removed,
                            "downstream_specific_gt_cleaned_hunks": ds_spec_gt_cleaned_hunks,
                            "downstream_specific_gt_hunk_count": ds_spec_gt_hunk_count,
                            "downstream_specific_gt_total_lines": ds_spec_gt_total_lines,
                            "downstream_specific_gt_added_lines": ds_spec_gt_added_lines,
                            "downstream_specific_gt_removed_lines": ds_spec_gt_removed_lines,
                            "fc_llm_output_valid_overall": fc_llm_output_valid, # Overall success for this file_conflict
                            "fc_attempts_made_overall": fc_attempts_made_overall,
                            "fc_error_overall": fc_error_overall,
                            "fc_last_format_error_overall": fc_last_format_error_overall,
                            "fc_last_apply_error_overall": fc_last_apply_error_overall,
                            "fc_runtime_total_sec": fc_runtime_overall,
                            "attempt_number": attempt_detail.get("attempt", 0),
                            "attempt_format_valid": attempt_detail.get("format_valid", False),
                            "attempt_format_error": attempt_detail.get("format_error", ""),
                            "attempt_apply_valid": attempt_detail.get("apply_valid", False),
                            "attempt_apply_error": attempt_detail.get("apply_error", ""),
                            "attempt_valid_overall": attempt_detail.get("valid", False),
                            "attempt_runtime_sec": attempt_detail.get("runtime_seconds", 0.0)
                        }
                        csv_writer.writerow(row)


def main_parser(input_directory: str, output_csv_file: str):
    """
    Main function to parse all JSON files in a directory and create a summary CSV.
    """
    csv_headers = [
        "json_file_source", "system_prompt", "base_task_prompt", "vulnerability_id", 
        "downstream_version", "target_file_name", 
        "original_patch_total_hunks_for_file", "original_patch_failed_hunk_numbers",
        "rej_file_reported_hunk_count", "rej_file_actual_hunk_count", 
        "rej_file_total_lines", "rej_file_added_lines", "rej_file_removed_lines",
        "ground_truth_hunk_count", "ground_truth_total_lines", 
        "ground_truth_added_lines", "ground_truth_removed_lines",
        "downstream_specific_gt_cleaned_hunks", "downstream_specific_gt_hunk_count",
        "downstream_specific_gt_total_lines", "downstream_specific_gt_added_lines",
        "downstream_specific_gt_removed_lines",
        "fc_llm_output_valid_overall", "fc_attempts_made_overall", 
        "fc_error_overall", "fc_last_format_error_overall", "fc_last_apply_error_overall",
        "fc_runtime_total_sec",
        "attempt_number", "attempt_format_valid", "attempt_format_error",
        "attempt_apply_valid", "attempt_apply_error", "attempt_valid_overall", 
        "attempt_runtime_sec"
    ]

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_csv_file)
    if output_dir: # Only call makedirs if there's a directory part
        os.makedirs(output_dir, exist_ok=True)

    with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader()

        json_files_processed_count = 0
        for filename in os.listdir(input_directory):
            if filename.endswith(".json"):
                file_path = os.path.join(input_directory, filename)
                print(f"Processing file: {file_path}...")
                
                # First, extract global prompts from this specific file's summary
                try:
                    with open(file_path, 'r', encoding='utf-8') as f_prompts:
                        data_for_prompts = json.load(f_prompts)
                    summary_section = data_for_prompts.get("summary", {})
                    global_prompts_for_file = {
                        'system_prompt': summary_section.get('system_prompt', 'NOT_IN_SUMMARY'),
                        'base_task_prompt': summary_section.get('base_task_prompt', 'NOT_IN_SUMMARY')
                    }
                except Exception as e_prompt:
                    print(f"⚠️ Could not extract prompts from summary of {file_path}: {e_prompt}")
                    global_prompts_for_file = {
                        'system_prompt': 'ERROR_EXTRACTING_PROMPT',
                        'base_task_prompt': 'ERROR_EXTRACTING_PROMPT'
                    }
                
                process_json_file(file_path, writer, global_prompts_for_file)
                json_files_processed_count +=1
        
        print(f"\n✅ CSV processing complete. Processed {json_files_processed_count} JSON files.")
        print(f"Output CSV saved to: {output_csv_file}")

if __name__ == '__main__':
    # --- Configuration ---
    INPUT_JSON_DIRECTORY = "./inputs" # Example: current directory
    
    OUTPUT_CSV = "meta_prompting_data.csv"

    # --- Run the parser ---
    main_parser(INPUT_JSON_DIRECTORY, OUTPUT_CSV) 

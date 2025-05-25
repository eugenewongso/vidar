import os
import json
import argparse
import csv
from tqdm import tqdm  # type: ignore
from metrics.line_metrics import relative_line_count_similarity
from metrics.similarity.codeBERT import compute_codebert_score
from metrics.similarity.openAI import compute_cosine_openai_embedding
from metrics.distance.edit_distance import (
    token_level_edit_distance,
    normalized_edit_similarity,
)
from datetime import datetime
import io
from unidiff import PatchSet
import re

MAX_TOKENS = 8192

def count_tokens(text):
    return len(text.split())

def clean_code(code):
    code = code.strip()
    if code.startswith("```"):
        code = code.split('\n', 1)[-1]
    if code.endswith("```"):
        code = code.rsplit('```', 1)[0]
    return code.strip()

# def clean_patch_content(patch_content, file_name):
#     # Pattern to match: diff --git a/file_name b/file_name
#     pattern = f'diff --git a/{file_name} b/{file_name}'
    
#     # Find the start of our target diff
#     match = re.search(pattern, patch_content)
#     if not match:
#         return None
    
#     start_pos = match.start()
    
#     # Find the next diff --git or use the end of string
#     next_diff = re.search(r'\ndiff --git', patch_content[start_pos + 1:])
#     if next_diff:
#         end_pos = start_pos + next_diff.start()
#         cleaned_patch = patch_content[start_pos:end_pos]
#     else:
#         cleaned_patch = patch_content[start_pos:]
    
#     return cleaned_patch.strip()

# def clean_patch_content(patch_content: str, target_file: str) -> str:
#     """
#     Extracts the unified diff for a specific file from a combined patch using unidiff.
    
#     Parameters:
#         patch_content (str): The full patch content (from git diff or similar).
#         target_file (str): The file name to extract the patch for (e.g., 'src/foo.cpp').

#     Returns:
#         str: The diff for the specified file, or None if not found.
#     """
#     try:
#         patch_set = PatchSet(io.StringIO(patch_content))
#         for patched_file in patch_set:
#             # Match either the old or new path
#             if patched_file.path == target_file or patched_file.source_file == f"a/{target_file}" or patched_file.target_file == f"b/{target_file}":
#                 return str(patched_file).strip()
#         return None
#     except Exception as e:
#         print(f"⚠️ Error parsing patch with unidiff: {e}")
#         return None

def clean_patch_content(patch_content: str, target_file: str) -> str:
    def normalize(p):
        return p.strip().lstrip("ab/")

    try:
        patch_set = PatchSet(io.StringIO(patch_content))
        for patched_file in patch_set:
            if (
                normalize(patched_file.path) == normalize(target_file)
                or normalize(patched_file.source_file) == normalize(target_file)
                or normalize(patched_file.target_file) == normalize(target_file)
            ):
                return str(patched_file).strip()
        return None
    except Exception as e:
        print(f"⚠️ Error parsing patch with unidiff: {e}")
        return None


# def clean_diff_text(diff_text_str: str) -> str:
#     """
#     Uses unidiff to extract and return only the hunk content from a diff string.
#     """
#     if not isinstance(diff_text_str, str):
#         return ""

#     try:
#         patch = PatchSet(io.StringIO(diff_text_str))
#         cleaned_hunks = [str(hunk) for patched_file in patch for hunk in patched_file]
#         return "\n".join(cleaned_hunks)
#     except Exception as e:
#         print(f"Error parsing diff: {e}")
#         return ""

def clean_diff_text(diff_text_str: str) -> str:
    """
    Removes standard diff headers (--- a/..., +++ b/..., --- original, +++ patched)
    and returns only the hunk content starting from the first '@@ '.
    If no '@@ ' is found, returns an empty string, as it implies no comparable hunk data.
    """

    print(diff_text_str)
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


def get_language_from_filename(filename):
    ext = filename.lower().rsplit('.', 1)[-1]
    ext_map = {
        "py": "python",
        "java": "java",
        "js": "javascript",
        "c": "c",
        "cpp": "cpp",
        "cc": "cpp",
        "cxx": "cpp",
        "h": "cpp",
        "hpp": "cpp",
        "cs": "csharp",
    }
    return ext_map.get(ext, ext)  # fallback to extension

def compute_metrics(upstream_code, downstream_code, file_name) -> dict:
    metrics = {}

    rls = relative_line_count_similarity(downstream_code, upstream_code)
    metrics["relative_line_count_similarity"] = round(rls, 4) if isinstance(rls, float) else rls

    nes = normalized_edit_similarity(downstream_code, upstream_code)
    metrics["normalized_edit_similarity"] = round(nes, 4) if isinstance(nes, float) else nes

    tled = token_level_edit_distance(downstream_code, upstream_code)
    metrics["token_level_edit_distance"] = round(tled, 4) if isinstance(tled, float) else tled

    language = get_language_from_filename(file_name)
    codebert = compute_codebert_score(downstream_code, upstream_code, language)
    if "error" in codebert:
        metrics["codebert_score"] = codebert
    else:
        metrics["codebert_score"] = {k: round(v, 4) for k, v in codebert.items()}

    total_tokens_upstream = count_tokens(upstream_code)
    total_tokens_downstream = count_tokens(downstream_code)
    total_tokens = total_tokens_upstream + total_tokens_downstream

    metrics["token_count_upstream"] = total_tokens_upstream
    metrics["token_count_downstream"] = total_tokens_downstream
    metrics["token_count_total"] = total_tokens
    print(f"🧠 Token usage for {file_name}:")
    print(f"- Upstream: {total_tokens_upstream}")
    print(f"- Downstream: {total_tokens_downstream}")
    print(f"- Total: {total_tokens} (limit = {MAX_TOKENS})")
    if total_tokens > MAX_TOKENS:
        metrics["cosine_similarity_openai"] = "skipped"
    else:
        score = compute_cosine_openai_embedding(upstream_code, downstream_code)
        metrics["cosine_similarity_openai"] = round(score, 4) if isinstance(score, float) else score

    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate patch comparisons and save results incrementally.")
    parser.add_argument("--json_input", required=True, help="Path to input JSON file")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_json = f"results/summary_{timestamp}.json"
    results_csv = f"results/summary_{timestamp}.csv"
    cleaned_inputs_json = f"results/cleaned_inputs_{timestamp}.json"
    cleaned_inputs = []  # Collect cleaned content separately


    os.makedirs(os.path.dirname(results_json), exist_ok=True)
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)

    with open(args.json_input, "r") as f:
        report = json.load(f)

    total_conflicts = sum(
        len(failure.get("file_conflicts", []))
        for vuln in report
        for failure in vuln.get("failures", [])
    )

    all_results = []
    header_written = os.path.exists(results_csv)

    with tqdm(total=total_conflicts, desc="Evaluating file conflicts") as pbar:
        for vuln in report:
            cve_id = vuln.get("id")
            failures = vuln.get("failures", [])
            if not failures:
                print(f"⚠️ Skipping {cve_id} — no failures listed.")
                continue

            for failure in failures:
                downstream_version = failure.get("downstream_version", "unknown")
                file_conflicts = failure.get("file_conflicts", [])
                if not file_conflicts:
                    print(f"⚠️ Skipping {cve_id} — no file_conflicts for {downstream_version}.")
                    continue

                for conflict in file_conflicts:
                    file_name = conflict.get("file_name", "unknown")
                    runtime_seconds = conflict.get("llm_time_taken_seconds", None)

                    # Extract and clean the specific file's diff from the downstream patch
                    raw_patch = failure.get("downstream_patch_content", "")
                    file_specific_patch = clean_patch_content(raw_patch, file_name)
                    if file_specific_patch is None:
                        print(f"⚠️ No matching diff found for {file_name} in downstream_patch_content. Skipping.")
                        continue
                    upstream_content = clean_diff_text(file_specific_patch)
                    downstream_content = clean_diff_text(conflict.get("LLM_diff_content", ""))



                    print(f"\n🔍 Comparing LLM patch vs ground truth patch for {file_name} in {cve_id} ({downstream_version})")
                    metrics = compute_metrics(upstream_content, downstream_content, file_name)

                    cleaned_inputs.append({
                        "cve_id": cve_id,
                        "downstream_version": downstream_version,
                        "file_name": file_name,
                        "runtime_seconds": runtime_seconds,
                        "ground_truth_diff": file_specific_patch,
                        "upstream_plus_llm_generated_patch": conflict.get("LLM_diff_content", ""),
                        "cleaned_ground_truth": upstream_content,
                        "cleaned_upstream_plus_llm": downstream_content,
                    })


                    result_entry = {
                        "cve_id": cve_id,
                        "downstream_version": downstream_version,
                        "file_name": file_name,
                        "runtime_seconds": runtime_seconds,
                        "cleaned_ground_truth": upstream_content,
                        "cleaned_upstream_plus_llm": downstream_content,
                        "metrics": metrics,
                    }


                    all_results.append(result_entry)
                    pbar.update(1)

                    # Write updated JSON
                    with open(results_json, "w") as jf:
                        json.dump(all_results, jf, indent=2)

                    # Flatten for CSV
                    flat = {
                        "cve_id": cve_id,
                        "downstream_version": downstream_version,
                        "file_name": file_name,
                        "runtime_seconds": runtime_seconds,
                        "relative_line_count_similarity": metrics.get("relative_line_count_similarity"),
                        "token_level_edit_distance": metrics.get("token_level_edit_distance"),
                        "normalized_edit_similarity": metrics.get("normalized_edit_similarity"),
                        "cosine_similarity_openai": metrics.get("cosine_similarity_openai"),
                        "total_tokens": metrics.get("token_count_total"),
                        "token_count_upstream": metrics.get("token_count_upstream"),
                        "token_count_downstream": metrics.get("token_count_downstream"),
                    }

                    if "codebert_score" in metrics:
                        codebert_dict = metrics["codebert_score"]
                        if isinstance(codebert_dict, dict):
                            for k, v in codebert_dict.items():
                                flat[f"codebert_{k}"] = v

                    # Write to CSV
                    with open(results_csv, "a", newline="") as cf:
                        writer = csv.DictWriter(cf, fieldnames=flat.keys())
                        if not header_written:
                            writer.writeheader()
                            header_written = True
                        writer.writerow(flat)

    with open(cleaned_inputs_json, "w") as cf:
        json.dump(cleaned_inputs, cf, indent=2)

    print(f"\n📄 Cleaned comparison inputs saved to: {cleaned_inputs_json}")

    print(f"\n✅ Incremental results saved to:\n  JSON: {results_json}\n  CSV: {results_csv}")

if __name__ == "__main__":
    main()

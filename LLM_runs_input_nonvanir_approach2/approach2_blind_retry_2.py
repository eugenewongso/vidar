import asyncio
import os
import datetime
import sys
import json
import argparse
import copy # Added for deepcopy
import logfire
from dataclasses import dataclass
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Optional, List, Dict, Any
import re
import time
import tiktoken
import subprocess
import tempfile
from unidiff import PatchSet
from android_patch_manager import AndroidPatchManager
from pathlib import Path

class APIKeyRotator:
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.index = 0

    def get_current_key(self):
        return self.api_keys[self.index]

    def rotate_key(self):
        self.index = (self.index + 1) % len(self.api_keys)
        print(f"🔄 Rotating to new API key index {self.index}")
        return self.get_current_key()



# Load environment variables from .env file
load_dotenv()

# Configure Logfire for logging (if token is available)
logfire.configure(send_to_logfire='if-token-present')

api_keys = os.getenv("GOOGLE_API_KEYS", "").split(",")
if not api_keys or api_keys == [""]:
    raise ValueError("Missing API keys in GOOGLE_API_KEYS")

key_rotator = APIKeyRotator(api_keys)

def get_repo_url_from_osv(vuln_id: str, osv_dir: str = "osv_data_android") -> Optional[str]:
    """
    Load OSV metadata JSON and construct repo URL from package name.
    """
    try:
        osv_path = os.path.join(osv_dir, f"{vuln_id}.json")
        with open(osv_path, "r", encoding="utf-8") as f:
            osv_data = json.load(f)
        affected = osv_data.get("affected", [])
        if affected and "package" in affected[0]:
            package_name = affected[0]["package"]["name"]  # e.g., "platform/frameworks/base"
            return f"https://android.googlesource.com/{package_name}"
    except Exception as e:
        print(f"❌ Failed to load repo_url from OSV for {vuln_id}: {e}")
    return None


def save_partial_output(path: str, data: Any):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"💾 Partial output saved to: {path}")
    except Exception as e:
        print(f"⚠️ Failed to save partial output to {path}: {e}")

def count_tokens_general(text: str):
    """
    Estimate token count based on word and character counts.

    Args:
        text (str): Input text.

    Returns:
        dict: Estimated token counts based on words and characters.
    """
    # Rough estimate: ~1 token = 0.75 words or ~4 chars/token
    word_count = len(re.findall(r'\w+', text))
    char_estimate = len(text) // 4
    return {
        "word_based": word_count,
        "char_based": char_estimate
    }


def count_tokens_tiktoken(text: str, model: str = "gpt-3.5-turbo"):
    """
    Count tokens using the Tiktoken library.

    Args:
        text (str): Input text.
        model (str): Model name.

    Returns:
        int: Total token count.
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def get_all_token_counts(text: str, gemini_token_count: Optional[int] = None):
    result = {
        "openai": count_tokens_tiktoken(text),
        "general": count_tokens_general(text),
    }
    if gemini_token_count is not None:
        result["gemini"] = gemini_token_count
    return result

@dataclass
class SupportDependencies: 
    rej_file_content: str # Renamed from diff_content for clarity
    original_source_file_content: str # Renamed from vulnerable_code_content

class GeminiAgent:
    def __init__(self, model_name: str, system_prompt: str, key_rotator: APIKeyRotator):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.key_rotator = key_rotator
        self._configure_genai()

    def _configure_genai(self):
        key = self.key_rotator.get_current_key()
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.system_prompt
        )

    async def run(self, prompt: str, deps: Optional[SupportDependencies] = None):
        for attempt in range(len(self.key_rotator.api_keys)):
            try:
                response = self.model.generate_content(prompt)
                token_count = None
                if hasattr(response, "usage_metadata"):
                    token_count = getattr(response.usage_metadata, "total_token_count", None)
                return type("Result", (), {"data": response.text, "token_count": token_count})
            except Exception as e:
                error_message = str(e).lower()
                if "quota" in error_message or "rate limit" in error_message or "internal error" in error_message:
                    print(f"⚠️ API error encountered: {e}")
                    self.key_rotator.rotate_key()
                    self._configure_genai()
                else:
                    raise e
        raise RuntimeError("All API keys exhausted or failed.")

def validate_patch_format(patch_content: str) -> tuple[bool, str]:
    """
    Validate patch format using unidiff library.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        PatchSet.from_string(patch_content)
        return True, "Valid patch format"
    except Exception as e:
        return False, f"Invalid patch format: {str(e)}"

def validate_patch_applicability(patch_content: str, original_file_content: str, target_filename: str) -> tuple[bool, str]:
    """
    Test if patch can be applied using GNU patch dry run.
    
    Returns:
        tuple: (can_apply, error_message)
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create original file
            original_file_path = os.path.join(temp_dir, target_filename)
            with open(original_file_path, 'w', encoding='utf-8') as f:
                f.write(original_file_content)
            
            # Create patch file
            patch_file_path = os.path.join(temp_dir, "test.patch")
            with open(patch_file_path, 'w', encoding='utf-8') as f:
                f.write(patch_content)
            
            # Test patch application with dry run
            result = subprocess.run(
                ['patch', '--dry-run', '-p1', '-i', patch_file_path],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return True, "Patch can be applied successfully"
            else:
                return False, f"Patch cannot be applied: {result.stderr}"
                
    except subprocess.TimeoutExpired:
        return False, "Patch validation timed out"
    except Exception as e:
        return False, f"Error during patch validation: {str(e)}"


def validate_patch_applicability_in_repo(patch_content: str, repo_path: str) -> tuple[bool, str]:
    """
    Test if patch can be applied using GNU patch dry run in actual repo.
    
    Returns:
        tuple: (can_apply, error_message)
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".diff", mode="w") as f:
            f.write(patch_content)
            patch_file_path = f.name

        result = subprocess.run(
            ['patch', '--dry-run', '-p1', '-i', patch_file_path],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            return True, "Patch applies cleanly in repo"
        else:
            return False, f"Patch failed in repo: {(result.stdout + result.stderr).strip()}"
    except subprocess.TimeoutExpired:
        return False, "Patch validation timed out"
    except Exception as e:
        return False, f"Error during patch validation: {str(e)}"


async def process_single_entry_with_retry(
    rej_content: str,
    original_source_content: str,
    target_filename_for_diff: str,
    vulnerability_id: str,
    failure_details: Dict[str, Any],
    upstream_patch_content: str,
    patch_porter_agent: GeminiAgent,
    max_retries: int = 3
):
    """
    Processes a single vulnerability entry with retry logic for validation.
    """
    # Type checks for content fields
    if not isinstance(rej_content, str):
        print(f"Skipping entry for {target_filename_for_diff} in {vulnerability_id} because .rej File Content is not a string.")
        return {
            "llm_output_valid": False,
            "runtime_seconds": 0,
            "attempts_made": 0,
            "validation_results": [],
            "error": ".rej File Content is not a string",
            "last_format_error": None,
            "last_apply_error": None
        }

    if not isinstance(original_source_content, str):
        print(f"Skipping entry for {target_filename_for_diff} in {vulnerability_id} because Original Source File content is not a string.")
        return {
            "llm_output_valid": False,
            "runtime_seconds": 0,
            "attempts_made": 0,
            "validation_results": [],
            "error": "Original Source File content is not a string",
            "last_format_error": None,
            "last_apply_error": None
        }


    # Existing checks for empty/placeholder content
    if not original_source_content or original_source_content.strip() == "```" or original_source_content.strip() == "":
        print(f"Skipping entry for {target_filename_for_diff} in {vulnerability_id} due to empty or placeholder Original Source File content.")
        return {
            "llm_output_valid": False,
            "runtime_seconds": 0,
            "attempts_made": 0,
            "validation_results": [],
            "error": "Empty or placeholder Original Source File content",
            "last_format_error": None,
            "last_apply_error": None
        }

    if not rej_content or rej_content.strip() == "```" or rej_content.strip() == "":
        print(f"Skipping entry for {target_filename_for_diff} in {vulnerability_id} due to empty or placeholder .rej File Content. A .rej file is expected.")
        return {
            "llm_output_valid": False,
            "runtime_seconds": 0,
            "attempts_made": 0,
            "validation_results": [],
            "error": "Empty or placeholder .rej File Content",
            "last_format_error": None,
            "last_apply_error": None
        }


    dependencies = SupportDependencies(
        rej_file_content=rej_content, 
        original_source_file_content=original_source_content,
    )

    base_task_prompt = f"""You are an expert software patch generation assistant. Your task is to resolve the conflicts in the provided `.rej` file by generating a corrected unified diff that applies cleanly to the 'Original Source File'.
Adhere strictly to your system instructions regarding format and content, especially the requirement for a single ```diff markdown block output.

Inputs:
1.  **Original Source File**: The content of the file that needs to be patched.
    ```text
    {dependencies.original_source_file_content}
    ```

2.  **.rej File Content**: The rejected hunks from a previous patch attempt. These are the changes you MUST adapt and correct. Pay close attention to the `@@ ... @@` headers in this content to determine the correct number of lines for your output hunk.
    ```text
    {dependencies.rej_file_content}
    ```

3.  **Target Filename**: Use this exact name for the `--- a/` and `+++ b/` lines in your diff.
    `{target_filename_for_diff}`

Your Specific Task:
Generate a corrected unified diff.
*   The diff must take the changes described in the '.rej File Content' and make them apply cleanly to the 'Original Source File'.
*   Address line number offsets or context mismatches.
*   Preserve the original intent of the security patch.
*   Strictly adhere to line counts and structure indicated by the `.rej` file's hunk headers.
*   Ensure all lines within your generated hunks start with ' ' (space), '+', or '-'.

Output Requirements (Strictly Enforced as per System Instructions):
1.  **ONLY THE DIFF in a ```diff block**: Your entire response must be *exclusively* a single markdown ```diff code block, starting with ```diff\\n and ending with ```.
2.  **Correct Headers**: Start the diff content (inside the block) with `--- a/{target_filename_for_diff}` and `+++ b/{target_filename_for_diff}`.
3.  **Valid Hunks**: Ensure all hunk headers (`@@ -old,count +new,count @@`) are correct and accurately reflect the changes and context, matching the structure derived from the input `.rej` file.

Example of Expected Output Format (Illustrative - actual content will depend on inputs):
```diff
--- a/{target_filename_for_diff}
+++ b/{target_filename_for_diff}
@@ -12,5 +12,5 @@ // Example: this header must match the logic of the resolved .rej hunk
 context line 1 (unchanged)
 context line 2 (unchanged)
-old line to be removed
+new line to be added
 context line 3 (unchanged)
```
"""

    print(f"Processing for diff generation: {vulnerability_id} - {target_filename_for_diff}")
    
    total_start_time = time.monotonic()
    validation_results = []
    
    for attempt in range(max_retries):
        print(f"🔄 Attempt {attempt + 1} of {max_retries} for {target_filename_for_diff}")
        
        # For a truly blind retry with consistent guidelines, the task_prompt is always the base_task_prompt.
        task_prompt = base_task_prompt
        
        if attempt > 0:
            # Log previous errors to console for debugging, but they are not added to the LLM prompt.
            prev_errors = []
            for r in validation_results:
                if not r["valid"]:
                    if not r["format_valid"]:
                        prev_errors.append(f"[Format Error] {r['format_error']}")
                    elif not r["apply_valid"]:
                        prev_errors.append(f"[Apply Error] {r['apply_error']}")
            
            if prev_errors:
                print("Previous errors (for console, not added to LLM prompt for blind retry):", prev_errors)
            else:
                print("Previous attempt failed, retrying with the same prompt (blind retry).")

        try:
            start_time = time.monotonic()
            result = await patch_porter_agent.run(task_prompt)
            end_time = time.monotonic()
            generated_diff = result.data.strip()

            # # Print LLM output for debugging
            # print("🔍 LLM-Generated Unified Diff:")
            # print("-" * 60)
            # print(generated_diff[:2000])  # Limit to 2000 chars to avoid console flooding
            # print("-" * 60)


            # Validation 1: Check patch format
            format_valid, format_error = validate_patch_format(generated_diff)
            
            # Validation 2: Check if patch can be applied (only if format is valid)
            apply_valid, apply_error = False, "Skipped due to format error"
            if format_valid:
                repo_path_str = failure_details.get("repo_path")
                downstream_version = failure_details.get("downstream_version")
                downstream_patch_sha = failure_details.get("downstream_patch")

                if not repo_path_str or not downstream_version or not downstream_patch_sha or not upstream_patch_content:
                    print(f"⚠️ Missing required repo metadata for validation: {vulnerability_id} - {target_filename_for_diff}")
                    return None

                repo_path = Path(repo_path_str)

                # Clean repo
                AndroidPatchManager.clean_repo(repo_path)

                # Checkout branch
                AndroidPatchManager.checkout_downstream_branch(repo_path, downstream_version)

                # Reset to the commit before the downstream patch
                AndroidPatchManager.checkout_commit(repo_path, f"{downstream_patch_sha}^")

                # Save upstream patch to a temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".patch", mode="w") as temp_patch_file:
                    temp_patch_file.write(upstream_patch_content)
                    temp_patch_path = temp_patch_file.name

                # Actually apply the upstream patch (allow partial success)
                success, message, stdout, stderr = AndroidPatchManager.apply_patch(repo_path, temp_patch_path)

                if not success:
                    print(f"⚠️ Partial application of upstream patch (some hunks rejected) for {vulnerability_id} - {target_filename_for_diff}")
                    # Continue — this is expected!
                else:
                    print(f"✅ Upstream patch applied (fully or partially) for {vulnerability_id} - {target_filename_for_diff}")


                apply_valid, apply_error = validate_patch_applicability_in_repo(
                    generated_diff,
                    str(repo_path) 
                )

            
            validation_result = {
                "attempt": attempt + 1,
                "format_valid": format_valid,
                "format_error": format_error,
                "apply_valid": apply_valid,
                "apply_error": apply_error,
                "valid": format_valid and apply_valid,
                "runtime_seconds": round(end_time - start_time, 2)
            }
            validation_results.append(validation_result)
            
            print(f"📊 Validation results for attempt {attempt + 1}:")
            print(f"  - Format valid: {format_valid}")
            print(f"  - Can apply: {apply_valid}")
            
            if validation_result["valid"]:
                total_end_time = time.monotonic()
                print(f"✅ LLM diff generation and validation successful for: {vulnerability_id} - {target_filename_for_diff}")
                return {
                    "downstream_llm_diff_output": generated_diff,
                    "llm_output_valid": True,
                    "runtime_seconds": round(total_end_time - total_start_time, 2),
                    "attempts_made": attempt + 1,
                    "validation_results": validation_results,
                    "token_counts": get_all_token_counts(generated_diff, gemini_token_count=result.token_count)
                }
            else:
                print(f"⚠️ Validation failed for attempt {attempt + 1}, retrying...")
                if attempt == max_retries - 1:
                    print(f"❌ All {max_retries} attempts failed validation for {target_filename_for_diff}")

        except Exception as e:
            validation_results.append({
                "attempt": attempt + 1,
                "error": f"Exception during generation: {str(e)}",
                "valid": False,
                "format_valid": False,
                "format_error": "Skipped due to exception",
                "apply_valid": False,
                "apply_error": "Skipped due to exception"
            })

            print(f"❌ Error on attempt {attempt + 1} for {target_filename_for_diff}: {e}")
            if attempt == max_retries - 1:
                print(f"❌ All {max_retries} attempts failed for {target_filename_for_diff}")

    # If we get here, all attempts failed
    total_end_time = time.monotonic()
    return {
        "downstream_llm_diff_output": None,
        "runtime_seconds": round(total_end_time - total_start_time, 2),
        "llm_output_valid": False,
        "attempts_made": max_retries,
        "validation_results": validation_results,
        "error": "All validation attempts failed" if validation_results else "Exception occurred before any validation",
        "last_format_error": validation_results[-1]["format_error"] if validation_results else None,
        "last_apply_error": validation_results[-1]["apply_error"] if validation_results else None
    }

async def main():
    parser = argparse.ArgumentParser(description="Process vulnerability JSON, generate corrected diffs using an LLM, and output an updated JSON.")
    parser.add_argument("input_json_file_path", help="Path to the input JSON file.")
    parser.add_argument(
        "--target_downstream_version", "-v",
        help="(Optional) Filter by specific downstream_version (e.g., '14'). If not provided, all versions will be processed.",
        default=None
    )
    parser.add_argument("--output_json_path", "-o", help="Path to save the output JSON file (with generated diffs). Defaults to outputs/approach2_output_diff_android_{version}_{timestamp}.json", default=None)
    parser.add_argument(
        "--resume_from_id",
        help="(Optional) Vulnerability ID to resume from. All earlier entries will be skipped.",
        default=None
    )

    
    args = parser.parse_args()

    system_prompt="""You are an expert software patch generation assistant.
Your primary task is to generate *perfectly formatted* unified diffs, enclosed in a ```diff markdown block, to resolve software conflicts presented in `.rej` files.
Adherence to the unified diff format and the specified output structure is paramount.

Key Objectives:
1.  **Analyze Rejects**: Carefully examine the provided '.rej File Content' and 'Original Source File'.
2.  **Generate Corrected Diff**: Create a new unified diff that incorporates the intended changes from the .rej file, adjusted to apply successfully to the 'Original Source File'.
3.  **Minimal Changes**: Modify *only* what is absolutely necessary within the rejected hunks to ensure they apply correctly and preserve the original intent of the patch.
4.  **Security Focus**: While ensuring correct formatting, the goal is to apply a security patch. Ensure your corrections maintain or achieve the intended security remediation.

STRICT FORMATTING AND OUTPUT REQUIREMENTS:
*   **Markdown Diff Block**: Your entire response MUST be a single markdown code block of type 'diff'. Start with ```diff (with a newline after it) and end with ```.
*   **No Extra Text**: Absolutely NO explanations, NO comments, NO apologies, NO summaries, or any other text outside this single ```diff ... ``` block.
*   **Diff Content - Headers**: Inside the block, the diff MUST start with `--- a/FILENAME` and `+++ b/FILENAME` lines, where FILENAME is the target filename provided to you in the task.
*   **Diff Content - Hunk Structure and Line Counts**:
    *   You MUST generate hunks that precisely match the line counts implied by the input `.rej` file's hunk headers (e.g., `@@ -old_start,old_lines +new_start,new_lines @@`).
    *   The output hunk should contain `old_lines` from the original (prefixed with ' ' for context, '-' for removed) and `new_lines` for the patched version (prefixed with ' ' for context, '+' for added).
    *   Do NOT add extraneous context lines beyond what is necessary to make the patch apply cleanly and match the implied line counts.
*   **Diff Content - Line Prefixes**: Every content line *within* a diff hunk (after the `@@ ... @@` header) MUST start with a ' ' (space for context), '-' (for removed lines), or '+' (for added lines).
*   **No Nested Code Blocks**: Do NOT use further markdown code fences (like ```) *inside* the main diff content.

Example of the complete expected output structure:
```diff
--- a/example.txt
+++ b/example.txt
@@ -1,3 +1,3 @@
 context line
-old line
+new line
 context line
```
"""

    patch_porter_agent = GeminiAgent(
        model_name="gemini-2.5-pro-preview-05-06",
        system_prompt=system_prompt,
        key_rotator=key_rotator
    )


    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    version_label = args.target_downstream_version or "all_versions"


    output_json_path_to_use = args.output_json_path
    if output_json_path_to_use is None:
        outputs_dir = "outputs/approach2_results" 
        os.makedirs(outputs_dir, exist_ok=True)
        output_json_path_to_use = os.path.join(
            outputs_dir,
            f"approach2_output_diff_android_{version_label}_{timestamp_str}.json"
        )


    report_data = {
        "run_timestamp": timestamp_str,
        "target_downstream_version": args.target_downstream_version or "all_versions",
        "input_json_file": args.input_json_file_path,
        "main_output_json_file_with_diffs": output_json_path_to_use,
        "summary": {
            "total_file_conflicts_matching_version": 0,
            "files_attempted_for_llm_diff_generation": 0,
            "files_with_llm_diff_successfully_generated": 0, 
            "files_skipped_pre_llm_call": 0,
            "files_with_llm_diff_generation_errors_or_skipped_in_func": 0,
            "successful_attempts_histogram": {},
            "total_runtime_seconds_all": 0,
            "total_runtime_seconds_successful": 0
        },
        "successfully_generated_diffs_log": [],
        "skipped_or_errored_diff_generation_log": []
    }

    try:
        with open(args.input_json_file_path, "r", encoding="utf-8") as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input JSON file not found at '{args.input_json_file_path}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{args.input_json_file_path}'")
        return

    if not isinstance(input_data, list):
        print("Error: Expected a list of vulnerabilities in the input JSON file.")
        return

    output_data = copy.deepcopy(input_data)

    resume_from_id = args.resume_from_id
    resume_mode = resume_from_id is not None

    if resume_mode:
        resume_index = next((i for i, item in enumerate(output_data) if item.get("id") == resume_from_id), None)
        if resume_index is None:
            print(f"❌ Error: resume_from_id '{resume_from_id}' not found in input.")
            return
        output_data = output_data[resume_index:]
        print(f"⏩ Resuming from vulnerability ID: {resume_from_id}")


    report_dir = os.path.join("outputs", "report")
    os.makedirs(report_dir, exist_ok=True)
    report_filename = os.path.join(
        report_dir,
        f"report_diff_{version_label}_{timestamp_str}.json"
    )

    for vulnerability_item in output_data:
        vulnerability_id = vulnerability_item.get("id", "unknown_vuln_id")
        failures = vulnerability_item.get("failures", [])

        for failure in failures:
            if args.target_downstream_version is None or failure.get("downstream_version") == args.target_downstream_version:
                file_conflicts = failure.get("file_conflicts", [])
                for file_conflict in file_conflicts:
                    report_data["summary"]["total_file_conflicts_matching_version"] += 1
                    
                    rej_content_for_llm = file_conflict.get("rej_file_content")
                    original_source_for_llm = file_conflict.get("downstream_file_content_patched_upstream_only")    # This is the "original source" for this task
                    target_filename = file_conflict.get("file_name")
                    
                    skip_reason = None
                    if not target_filename: skip_reason = "Missing 'file_name' (target filename for diff)"
                    elif not rej_content_for_llm: skip_reason = "Missing 'rej_file_content' (rejected hunks)"
                    elif not original_source_for_llm: skip_reason = "Missing 'downstream_file_content' (original source file)"

                    if skip_reason:
                        print(f"Skipping file_conflict in {vulnerability_id} (file: {target_filename or 'N/A'}) due to: {skip_reason}.")
                        report_data["summary"]["files_skipped_pre_llm_call"] += 1
                        report_data["skipped_or_errored_diff_generation_log"].append({
                            "vulnerability_id": vulnerability_id, "file_name": target_filename or "Unknown",
                            "patch_sha": failure.get('downstream_patch', 'N/A'), "reason": skip_reason
                        })
                        continue
                    
                    report_data["summary"]["files_attempted_for_llm_diff_generation"] += 1

                    REPO_BASE = Path("android_repos")

                    vuln_id = vulnerability_item.get("id")
                    repo_url = get_repo_url_from_osv(vuln_id)

                    repo_path_str = failure.get("repo_path")
                    repo_path = Path(repo_path_str) if repo_path_str else None

                    if not repo_path_str:
                        if not repo_url:
                            print(f"⚠️ Skipping due to missing repo_path and repo_url in failure entry for {vulnerability_id}")
                            report_data["summary"]["files_skipped_pre_llm_call"] += 1
                            report_data["skipped_or_errored_diff_generation_log"].append({
                                "vulnerability_id": vulnerability_id,
                                "file_name": target_filename or "Unknown",
                                "patch_sha": failure.get('downstream_patch', 'N/A'),
                                "reason": "Missing repo_path and repo_url"
                            })
                            continue
                        # Clone the repo into android_repos/
                        repo_path = AndroidPatchManager.clone_repo(repo_url, REPO_BASE)
                        failure["repo_path"] = str(repo_path)  # Save for use in downstream
                    else:
                        repo_path = Path(repo_path_str)
                        if not repo_path.exists():
                            if repo_url:
                                print(f"📦 Repo not found at {repo_path}, cloning from {repo_url}")
                                repo_path = AndroidPatchManager.clone_repo(repo_url, REPO_BASE)
                                failure["repo_path"] = str(repo_path)
                            else:
                                print(f"⚠️ Repo path does not exist and repo_url missing for {vulnerability_id}")
                                continue
                        else:
                            print(f"📂 Using existing repo at {repo_path}")

                    
                    # Use the new retry function
                    generated_diff = await process_single_entry_with_retry(
                        rej_content=rej_content_for_llm,
                        original_source_content=original_source_for_llm,
                        target_filename_for_diff=target_filename,
                        vulnerability_id=vulnerability_id,
                        failure_details=failure,
                        upstream_patch_content=vulnerability_item.get("upstream_patch_content"),
                        patch_porter_agent=patch_porter_agent,
                        max_retries=3
                    )

                    runtime = generated_diff.get("runtime_seconds", 0)
                    report_data["summary"]["total_runtime_seconds_all"] = (
                        report_data["summary"].get("total_runtime_seconds_all", 0) + runtime
                    )

                    if generated_diff.get("llm_output_valid"):
                        report_data["summary"]["total_runtime_seconds_successful"] = (
                            report_data["summary"].get("total_runtime_seconds_successful", 0) + runtime
                        )

                    if generated_diff is not None:
                        file_conflict.update(generated_diff) # Update the current file_conflict in output_data
                        
                        # Incrementally save the entire output_data after this file_conflict has been processed
                        # and output_data has been updated.
                        # Ensure vulnerability_item is accessible for the print message.
                        # It should be, as file_conflict is nested within vulnerability_item's loops.
                        current_fc_name_for_save = file_conflict.get("file_name", "unknown_file")
                        vuln_id_for_save = vulnerability_item.get("id", "unknown_vuln_id")
                        print(f"💾 Incrementally saving output after processing file: {current_fc_name_for_save} for vuln: {vuln_id_for_save}")
                        save_partial_output(output_json_path_to_use, output_data)

                        # Incrementally save the report_data
                        print(f"📊 Incrementally saving report data after processing file: {current_fc_name_for_save} for vuln: {vuln_id_for_save}")
                        save_partial_output(report_filename, report_data)
                        
                        if generated_diff.get("llm_output_valid"):
                            report_data["summary"]["files_with_llm_diff_successfully_generated"] += 1
                            # Track number of attempts in histogram (e.g., "1 run", "2 runs")
                            attempts = generated_diff.get("attempts_made", 0)
                            label = f"{attempts} run" if attempts == 1 else f"{attempts} runs"
                            histogram = report_data["summary"]["successful_attempts_histogram"]
                            histogram[label] = histogram.get(label, 0) + 1

                            report_data["successfully_generated_diffs_log"].append({
                                "vulnerability_id": vulnerability_id,
                                "file_name": target_filename,
                                "patch_sha": failure.get('downstream_patch', 'N/A'),
                                "downstream_version": failure.get("downstream_version", "N/A"),
                                "diff_preview": generated_diff["downstream_llm_diff_output"][:100] + "..." if generated_diff.get("downstream_llm_diff_output") else "None"
                            })
                        else:
                            report_data["summary"]["files_with_llm_diff_generation_errors_or_skipped_in_func"] += 1
                            report_data["skipped_or_errored_diff_generation_log"].append({
                                "vulnerability_id": vulnerability_id,
                                "file_name": target_filename,
                                "patch_sha": failure.get('downstream_patch', 'N/A'),
                                "reason": generated_diff.get("error", "Validation failed"),
                                "last_format_error": generated_diff.get("last_format_error"),
                                "last_apply_error": generated_diff.get("last_apply_error")
                            })

    
    try:
        with open(output_json_path_to_use, "w", encoding="utf-8") as f_out:
            json.dump(output_data, f_out, indent=4)
        print(f"✅ Main processed JSON (with diffs) successfully saved to '{output_json_path_to_use}'")
    except IOError as e:
        print(f"Error writing main output JSON to '{output_json_path_to_use}': {e}")

    try:
        with open(report_filename, "w", encoding="utf-8") as f_report:
            json.dump(report_data, f_report, indent=4)
        print(f"✅ Summary diff generation report successfully saved to '{report_filename}'")
    except IOError as e:
        print(f"Error writing summary report to '{report_filename}': {e}")

if __name__ == '__main__':
    asyncio.run(main())

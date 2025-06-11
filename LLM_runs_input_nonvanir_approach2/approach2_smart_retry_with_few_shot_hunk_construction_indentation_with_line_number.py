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

def add_line_numbers(content: str) -> str:
    """
    Add line numbers to content to help LLM with hunk header calculation.
    
    Args:
        content (str): Original file content
        
    Returns:
        str: Content with line numbers added
    """
    lines = content.split('\n')
    numbered_lines = [f"{i+1:4d}: {line}" for i, line in enumerate(lines)]
    return '\n'.join(numbered_lines)

def get_all_token_counts(text: str, gemini_token_counts: Optional[Dict[str, int]] = None):
    result = {
        "openai": count_tokens_tiktoken(text),
        "general": count_tokens_general(text),
    }
    if gemini_token_counts is not None:
        result["gemini"] = gemini_token_counts
    return result

@dataclass
class SupportDependencies: 
    rej_file_content: str # Renamed from diff_content for clarity
    original_source_file_content: str # Renamed from vulnerable_code_content

class GeminiAgent:
    def __init__(self, model_name: str, system_prompt: str, key_rotator: APIKeyRotator, temperature: float = 0.0):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.key_rotator = key_rotator
        self.temperature = temperature
        self._configure_genai()

    def _configure_genai(self):
        key = self.key_rotator.get_current_key()
        genai.configure(api_key=key)

        generation_config = genai.types.GenerationConfig(
            temperature=self.temperature
        )

        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.system_prompt,
            generation_config=generation_config
        )

    async def run(self, prompt: str, deps: Optional[SupportDependencies] = None):
        for attempt in range(len(self.key_rotator.api_keys)):
            try:
                response = self.model.generate_content(prompt)
                token_counts = None
                if hasattr(response, "usage_metadata"):
                    token_counts = {
                        "input": getattr(response.usage_metadata, "prompt_token_count", 0),
                        "output": getattr(response.usage_metadata, "candidates_token_count", 0),
                        "total": getattr(response.usage_metadata, "total_token_count", 0)
                    }
                return {"data": response.text, "token_counts": token_counts}
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

    base_task_prompt = f"""Resolve the conflicts in the provided `.rej` file by generating a corrected unified diff.
The generated diff must apply cleanly to the 'Original Source File' and strictly follow all formatting and hunk construction rules from your system guidelines.

**Inputs:**

1.  **Original Source File Content**: The full content of the file to be patched, with line numbers added for precision. Use this to find the correct context for the changes and calculate accurate hunk headers.
    ```text
    {add_line_numbers(dependencies.original_source_file_content)}
    ```

2.  **.rej File Content**: The rejected hunks from a patch. You MUST adapt and correct these changes. The `@@ ... @@` headers in this content are your primary guide for the changes and line counts.
    ```text
    {dependencies.rej_file_content}
    ```

3.  **Target Filename**: Use this exact name for the `--- a/` and `+++ b/` lines of your diff.
    `{target_filename_for_diff}`

**Your Task:**
Generate a corrected unified diff that:
1.  Takes the changes described in the '.rej File Content' and correctly applies them to the 'Original Source File Content'.
2.  Addresses any line number offsets or context mismatches that caused the original rejection.
3.  Uses the provided line numbers to calculate precise hunk headers (`@@ ... @@`) as explained in your system guidelines.
4.  Strictly adheres to the hunk header line counts (`@@ ... @@`) as explained in your system guidelines.
5.  **IMPORTANT**: The diff content itself must NOT include the line number prefixes. Use line numbers for calculation only, but generate clean diff content.

Generate the corrected unified diff now:
"""

    print(f"Processing for diff generation: {vulnerability_id} - {target_filename_for_diff}")
    
    total_start_time = time.monotonic()
    validation_results = []
    previous_generated_diff: Optional[str] = None # Store the last generated diff
    last_format_error: Optional[str] = None # Store last format error
    last_apply_error: Optional[str] = None # Store last apply error
    total_gemini_token_counts = {"input": 0, "output": 0, "total": 0}
    
    for attempt in range(max_retries):
        print(f"🔄 Attempt {attempt + 1} of {max_retries} for {target_filename_for_diff}")
        
        task_prompt: str
        if attempt == 0:
            task_prompt = base_task_prompt
        else:
            # Construct feedback prompt for subsequent attempts
            error_feedback_summary = []
            if last_format_error and last_format_error != "Skipped due to exception":
                error_feedback_summary.append(f"- Format Error: {last_format_error}")
            if last_apply_error and last_apply_error != "Skipped due to exception" and last_apply_error != "Skipped due to format error":
                error_feedback_summary.append(f"- Apply Error: {last_apply_error}")
            
            if not error_feedback_summary: # Should not happen if previous attempt failed validation
                error_feedback_summary.append("An unspecified validation error occurred in the previous attempt.")

            error_feedback_str = "\\n".join(error_feedback_summary)

            # It's crucial that previous_generated_diff is not None here if attempt > 0 and there was a validation failure
            # We might need to ensure it's always set after an attempt.
            if previous_generated_diff is None:
                 # This case should ideally be avoided by ensuring previous_generated_diff is updated.
                 # Falling back to base_task_prompt if something went wrong with capturing previous output.
                print("⚠️ Warning: previous_generated_diff is None on a retry attempt. Falling back to base_task_prompt.")
                task_prompt = base_task_prompt
            else:
                task_prompt = f"""Your previous attempt to generate the diff failed.
Review the errors and your previous output, then provide a corrected unified diff.
Adhere strictly to all system guidelines for unified diff format, hunk construction, and output requirements.

Previous Validation Errors:
{error_feedback_str}

Your Previous (Incorrect) Diff Output:
```diff
{previous_generated_diff}
```

Original Inputs for reference:
1.  **Original Source File Content** (with line numbers for precision):
    ```text
    {add_line_numbers(dependencies.original_source_file_content)}
    ```
2.  **.rej File Content**: The rejected hunks.
    ```text
    {dependencies.rej_file_content}
    ```
3.  **Target Filename**:
    `{target_filename_for_diff}`

Your Specific Task:
Generate a new, corrected unified diff that addresses the reported errors and successfully applies to the 'Original Source File Content'.
Use the provided line numbers to calculate precise hunk headers and ensure the new diff strictly follows all formatting and hunk construction rules from your system guidelines.
**IMPORTANT**: The diff content itself must NOT include the line number prefixes - use them for calculation only.
"""
        
        # if attempt > 0: # This block was for blind retry, now handled by iterative feedback
        #     # Log previous errors to console for debugging, but they are not added to the LLM prompt.
        #     prev_errors = []
        #     for r in validation_results:
        #         if not r["valid"]:
        #             if not r["format_valid"]:
        #                 prev_errors.append(f"[Format Error] {r['format_error']}")
        #             elif not r["apply_valid"]:
        #                 prev_errors.append(f"[Apply Error] {r['apply_error']}")
            
        #     if prev_errors:
        #         print("Previous errors (for console, not added to LLM prompt for blind retry):", prev_errors)
        #     else:
        #         print("Previous attempt failed, retrying with the same prompt (blind retry).")

        try:
            start_time = time.monotonic()
            result = await patch_porter_agent.run(task_prompt) # Pass the dynamically chosen prompt
            end_time = time.monotonic()
            generated_diff = result["data"].strip()
            previous_generated_diff = generated_diff # Store for next iteration's feedback

            attempt_token_counts = result["token_counts"] or {"input": 0, "output": 0, "total": 0}
            total_gemini_token_counts["input"] += attempt_token_counts.get("input", 0)
            total_gemini_token_counts["output"] += attempt_token_counts.get("output", 0)
            total_gemini_token_counts["total"] += attempt_token_counts.get("total", 0)

            # # Print LLM output for debugging
            # print("🔍 LLM-Generated Unified Diff:")
            # print("-" * 60)
            # print(generated_diff[:2000])  # Limit to 2000 chars to avoid console flooding
            # print("-" * 60)


            # Validation 1: Check patch format
            format_valid, format_error = validate_patch_format(generated_diff)
            last_format_error = format_error # Store for feedback
            
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
            last_apply_error = apply_error # Store for feedback
            
            validation_result = {
                "attempt": attempt + 1,
                "format_valid": format_valid,
                "format_error": format_error,
                "apply_valid": apply_valid,
                "apply_error": apply_error,
                "valid": format_valid and apply_valid,
                "runtime_seconds": round(end_time - start_time, 2),
                "token_counts": get_all_token_counts(task_prompt, gemini_token_counts=attempt_token_counts)
            }
            validation_results.append(validation_result)
            
            print(f"📊 Validation results for attempt {attempt + 1}:")
            print(f"  - Format valid: {format_valid} (Error: {format_error if not format_valid else 'N/A'})")
            print(f"  - Can apply: {apply_valid} (Error: {apply_error if not apply_valid else 'N/A'})")
            
            if validation_result["valid"]:
                total_end_time = time.monotonic()
                print(f"✅ LLM diff generation and validation successful for: {vulnerability_id} - {target_filename_for_diff}")
                return {
                    "downstream_llm_diff_output": generated_diff,
                    "llm_output_valid": True,
                    "runtime_seconds": round(total_end_time - total_start_time, 2),
                    "attempts_made": attempt + 1,
                    "validation_results": validation_results,
                    "token_counts": get_all_token_counts(generated_diff, gemini_token_counts=total_gemini_token_counts),
                    "final_prompt_used": task_prompt # Log the prompt that led to success
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
            # Update last_format_error and last_apply_error in case of exception too, so feedback prompt doesn't use stale data
            last_format_error = "Exception during generation, format validation skipped."
            last_apply_error = "Exception during generation, applicability validation skipped."
            if attempt == max_retries - 1:
                print(f"❌ All {max_retries} attempts failed for {target_filename_for_diff}")

    # If we get here, all attempts failed
    total_end_time = time.monotonic()
    final_error_prompt_summary = "All validation attempts failed"
    if validation_results:
        # Try to get the last actual validation errors if possible
        last_recorded_format_error = validation_results[-1].get("format_error", "N/A")
        last_recorded_apply_error = validation_results[-1].get("apply_error", "N/A")
        if last_recorded_format_error != "Skipped due to exception" and last_recorded_apply_error != "Skipped due to exception":
             final_error_prompt_summary = f"Last Format Error: {last_recorded_format_error}, Last Apply Error: {last_recorded_apply_error}"
        elif "error" in validation_results[-1]:
             final_error_prompt_summary = validation_results[-1]["error"]


    return {
        "downstream_llm_diff_output": previous_generated_diff, # Return the last attempted diff
        "runtime_seconds": round(total_end_time - total_start_time, 2),
        "llm_output_valid": False,
        "attempts_made": max_retries,
        "validation_results": validation_results,
        "error": final_error_prompt_summary,
        "last_format_error": validation_results[-1].get("format_error") if validation_results else "No validation ran",
        "last_apply_error": validation_results[-1].get("apply_error") if validation_results else "No validation ran",
        "token_counts": get_all_token_counts(previous_generated_diff or "", gemini_token_counts=total_gemini_token_counts),
        "final_prompt_used": task_prompt # Log the last prompt used, even if it failed
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
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.0,
        help="Set the temperature for the LLM. Defaults to 0.0."
    )

    
    args = parser.parse_args()

    system_prompt = '''You are an expert software patch generation assistant.
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
*   **Diff Content - Line Prefixes**: Every content line *within* a diff hunk (after the `@@ ... @@` header) MUST start with a ' ' (space for context), '-' (for removed lines), or '+' (for added lines).
*   **No Nested Code Blocks**: Do NOT use further markdown code fences (like ```) *inside* the main diff content.

Crucial Details for Hunk Construction:
*   **Hunk Header Calculation is Critical**: The `@@ -old_start,old_lines +new_start,new_lines @@` header must be perfectly accurate.
    *   `old_lines` is the total count of all lines for the original file snippet within that hunk (lines starting with ' ' or '-').
    *   `new_lines` is the total count of all lines for the new file snippet within that hunk (lines starting with ' ' or '+').
*   **Match Line Counts Exactly**: The number of lines in the hunk body *must* exactly match the `old_lines` and `new_lines` counts in the `@@` header. The most common error is a mismatch where the hunk body is longer than specified.
*   **Use `.rej` File for Guidance**: The `@@ ... @@` header from the input `.rej` file is your primary guide for what the changes *should have been*. Adapt its context and line counts to fit the Original Source File. Do NOT add extraneous context lines beyond what is necessary to make the patch apply cleanly.
*   **Use Line Numbers for Precision**: The Original Source File Content includes line numbers to help you calculate hunk headers accurately. Use these numbers to determine the exact `@@ -start,count +start,count @@` values. When you see line numbers like `  42: some code`, that means line 42 contains "some code".
*   **CRITICAL - Do NOT Include Line Numbers in Diff**: The line numbers are ONLY for your reference and calculation. Your generated diff content must contain the actual code WITHOUT line numbers. For example, if you see `  42: private UsbHandler mHandler;`, your diff should contain `private UsbHandler mHandler;` (without the `  42: ` prefix).

Strategy for Generating Correct Diffs:
*   **Replace Entire Blocks**: When editing a function, method, loop, or any multi-line code block, it is much safer to replace the *entire* block. Do this by deleting the entire existing version with `-` lines and then adding the new, updated version with `+` lines. This helps generate correct code and avoids complex context issues that can cause a patch to fail.
*   **Moving Code**: To move code within a file, use two separate hunks: one to delete it from its original location, and another to add it to its new location.
*   **Skip Useless Hunks**: Only output hunks that contain code changes (`+` or `-` lines). Do NOT generate hunks that only contain context lines (` `).
*   **Indentation is Critical**: Pay very close attention to indentation. Incorrect indentation will cause the patch to fail.

Here is an example of a successful transformation. Pay close attention to how the `.rej` file is resolved against the original source snippets to produce the "Expected Output Diff".

**Few-Shot Example:**

**Target Filename:**
`services/usb/java/com/android/server/usb/UsbDeviceManager.java`

**Original Source (services/usb/java/com/android/server/usb/UsbDeviceManager.java):**
```java
   1: /*
   2:  * Copyright (C) 2011 The Android Open Source Project
   3:  *
   4:  * Licensed under the Apache License, Version 2.0 (the "License");
   5:  * you may not use this file except in compliance with the License.
   6:  * You may obtain a copy of the License at
   7:  *
   8:  *      http://www.apache.org/licenses/LICENSE-2.0
   9:  *
  10:  * Unless required by applicable law or agreed to in writing, software
  11:  * distributed under the License is distributed on an "AS IS" BASIS,
  12:  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  13:  * See the License for the specific language governing permissions an
  14:  * limitations under the License.
  15:  */
  16: 
  17: package com.android.server.usb;
  18: 
  19: import static android.hardware.usb.UsbPortStatus.DATA_ROLE_DEVICE;
  20: import static android.hardware.usb.UsbPortStatus.DATA_ROLE_HOST;
  21: import static android.hardware.usb.UsbPortStatus.MODE_AUDIO_ACCESSORY;
  22: import static android.hardware.usb.UsbPortStatus.POWER_ROLE_SINK;
  23: import static android.hardware.usb.UsbPortStatus.POWER_ROLE_SOURCE;
  24: 
  25: import static com.android.internal.usb.DumpUtils.writeAccessory;
  26: import static com.android.internal.util.dump.DumpUtils.writeStringIfNotNull;
  27: 
  28: import android.app.ActivityManager;
  29: import android.app.KeyguardManager;
  30: import android.app.Notification;
  31: import android.app.NotificationChannel;
  32: import android.app.NotificationManager;
  33: import android.app.PendingIntent;
  34: import android.content.BroadcastReceiver;
  35: import android.content.ComponentName;
  36: import android.content.ContentResolver;
  37: import android.content.Context;
  38: import android.content.Intent;
  39: import android.content.IntentFilter;
  40: import android.content.SharedPreferences;
  41: import android.content.pm.PackageManager;
  42: import android.content.res.Resources;
  43: import android.debug.AdbManagerInternal;
  44: import android.debug.AdbNotifications;
  45: import android.debug.AdbTransportType;
  46: import android.debug.IAdbTransport;
  47: import android.hardware.usb.ParcelableUsbPort;
  48: import android.hardware.usb.UsbAccessory;
  49: import android.hardware.usb.UsbConfiguration;
  50: import android.hardware.usb.UsbConstants;
  51: import android.hardware.usb.UsbDevice;
  52: import android.hardware.usb.UsbInterface;
  53: import android.hardware.usb.UsbManager;
  54: import android.hardware.usb.UsbPort;
  55: import android.hardware.usb.UsbPortStatus;
  56: import android.hardware.usb.gadget.V1_0.GadgetFunction;
  57: import android.hardware.usb.gadget.V1_0.IUsbGadget;
  58: import android.hardware.usb.gadget.V1_0.IUsbGadgetCallback;
  59: import android.hardware.usb.gadget.V1_0.Status;
  60: import android.hidl.manager.V1_0.IServiceManager;
  61: import android.hidl.manager.V1_0.IServiceNotification;
  62: import android.os.BatteryManager;
  63: import android.os.Environment;
  64: import android.os.FileUtils;
  65: import android.os.Handler;
  66: import android.os.HwBinder;
  67: import android.os.Looper;
  68: import android.os.Message;
  69: import android.os.ParcelFileDescriptor;
  70: import android.os.RemoteException;
  71: import android.os.SystemClock;
  72: import android.os.SystemProperties;
  73: import android.os.UEventObserver;
  74: import android.os.UserHandle;
  75: import android.os.UserManager;
  76: import android.os.storage.StorageManager;
  77: import android.os.storage.StorageVolume;
  78: import android.provider.Settings;
  79: import android.service.usb.UsbDeviceManagerProto;
  80: import android.service.usb.UsbHandlerProto;
  81: import android.util.Pair;
  82: import android.util.Slog;
  83: 
  84: import com.android.internal.annotations.GuardedBy;
  85: import com.android.internal.logging.MetricsLogger;
  86: import com.android.internal.logging.nano.MetricsProto.MetricsEvent;
  87: import com.android.internal.messages.nano.SystemMessageProto.SystemMessage;
  88: import com.android.internal.notification.SystemNotificationChannels;
  89: import com.android.internal.os.SomeArgs;
  90: import com.android.internal.util.dump.DualDumpOutputStream;
  91: import com.android.server.FgThread;
  92: import com.android.server.LocalServices;
  93: import com.android.server.wm.ActivityTaskManagerInternal;
  94: 
  95: import java.io.File;
  96: import java.io.FileDescriptor;
  97: import java.io.FileNotFoundException;
  98: import java.io.IOException;
  99: import java.util.HashMap;
 100: import java.util.HashSet;
 101: import java.util.Iterator;
 102: import java.util.Locale;
 103: import java.util.Map;
 104: import java.util.NoSuchElementException;
 105: import java.util.Scanner;
 106: import java.util.Set;
 107: 
 108: /**
 109:  * UsbDeviceManager manages USB state in device mode.
 110:  */
 111: public class UsbDeviceManager implements ActivityTaskManagerInternal.ScreenObserver {
 112: 
 113:     private static final String TAG = UsbDeviceManager.class.getSimpleName();
 114:     private static final boolean DEBUG = false;
 115: 
 116:     /**
 117:      * The name of the xml file in which screen unlocked functions are stored.
 118:      */
 119:     private static final String USB_PREFS_XML = "UsbDeviceManagerPrefs.xml";
 120: 
 121:     /**
 122:      * The SharedPreference setting per user that stores the screen unlocked functions between
 123:      * sessions.
 124:      */
 125:     static final String UNLOCKED_CONFIG_PREF = "usb-screen-unlocked-config-%d";
 126: 
 127:     /**
 128:      * ro.bootmode value when phone boots into usual Android.
 129:      */
 130:     private static final String NORMAL_BOOT = "normal";
 131: 
 132:     private static final String USB_STATE_MATCH =
 133:             "DEVPATH=/devices/virtual/android_usb/android0";
 134:     private static final String ACCESSORY_START_MATCH =
 135:             "DEVPATH=/devices/virtual/misc/usb_accessory";
 136:     private static final String FUNCTIONS_PATH =
 137:             "/sys/class/android_usb/android0/functions";
 138:     private static final String STATE_PATH =
 139:             "/sys/class/android_usb/android0/state";
 140:     private static final String RNDIS_ETH_ADDR_PATH =
 141:             "/sys/class/android_usb/android0/f_rndis/ethaddr";
 142:     private static final String AUDIO_SOURCE_PCM_PATH =
 143:             "/sys/class/android_usb/android0/f_audio_source/pcm";
 144:     private static final String MIDI_ALSA_PATH =
 145:             "/sys/class/android_usb/android0/f_midi/alsa";
 146: 
 147:     private static final int MSG_UPDATE_STATE = 0;
 148:     private static final int MSG_ENABLE_ADB = 1;
 149:     private static final int MSG_SET_CURRENT_FUNCTIONS = 2;
 150:     private static final int MSG_SYSTEM_READY = 3;
 151:     private static final int MSG_BOOT_COMPLETED = 4;
 152:     private static final int MSG_USER_SWITCHED = 5;
 153:     private static final int MSG_UPDATE_USER_RESTRICTIONS = 6;
 154:     private static final int MSG_UPDATE_PORT_STATE = 7;
 155:     private static final int MSG_ACCESSORY_MODE_ENTER_TIMEOUT = 8;
 156:     private static final int MSG_UPDATE_CHARGING_STATE = 9;
 157:     private static final int MSG_UPDATE_HOST_STATE = 10;
 158:     private static final int MSG_LOCALE_CHANGED = 11;
 159:     private static final int MSG_SET_SCREEN_UNLOCKED_FUNCTIONS = 12;
 160:     private static final int MSG_UPDATE_SCREEN_LOCK = 13;
 161:     private static final int MSG_SET_CHARGING_FUNCTIONS = 14;
 162:     private static final int MSG_SET_FUNCTIONS_TIMEOUT = 15;
 163:     private static final int MSG_GET_CURRENT_USB_FUNCTIONS = 16;
 164:     private static final int MSG_FUNCTION_SWITCH_TIMEOUT = 17;
 165:     private static final int MSG_GADGET_HAL_REGISTERED = 18;
 166:     private static final int MSG_RESET_USB_GADGET = 19;
 167: 
 168:     private static final int AUDIO_MODE_SOURCE = 1;
 169: 
 170:     // Delay for debouncing USB disconnects.
 171:     // We often get rapid connect/disconnect events when enabling USB functions,
 172:     // which need debouncing.
 173:     private static final int UPDATE_DELAY = 1000;
 174: 
 175:     // Timeout for entering USB request mode.
 176:     // Request is cancelled if host does not configure device within 10 seconds.
 177:     private static final int ACCESSORY_REQUEST_TIMEOUT = 10 * 1000;
 178: 
 179:     private static final String BOOT_MODE_PROPERTY = "ro.bootmode";
 180: 
 181:     private static final String ADB_NOTIFICATION_CHANNEL_ID_TV = "usbdevicemanager.adb.tv";
 182:     private UsbHandler mHandler;
 183: 
 184:     private final Object mLock = new Object();
 185: 
 186:     private final Context mContext;
 187:     private final ContentResolver mContentResolver;
 188:     @GuardedBy("mLock")
 189:     private UsbProfileGroupSettingsManager mCurrentSettings;
 190:     private final boolean mHasUsbAccessory;
 191:     @GuardedBy("mLock")
 192:     private String[] mAccessoryStrings;
 193:     private final UEventObserver mUEventObserver;
 194: 
 195:     private static Set<Integer> sBlackListedInterfaces;
 196:     private HashMap<Long, FileDescriptor> mControlFds;
 197: 
 198:     static {
 199:         sBlackListedInterfaces = new HashSet<>();
 200:         sBlackListedInterfaces.add(UsbConstants.USB_CLASS_AUDIO);
 201:         sBlackListedInterfaces.add(UsbConstants.USB_CLASS_COMM);
 202:         sBlackListedInterfaces.add(UsbConstants.USB_CLASS_HID);
 203:         sBlackListedInterfaces.add(UsbConstants.USB_CLASS_PRINTER);
 204:         sBlackListedInterfaces.add(UsbConstants.USB_CLASS_MASS_STORAGE);
 205:         sBlackListedInterfaces.add(UsbConstants.USB_CLASS_HUB);
 206:         sBlackListedInterfaces.add(UsbConstants.USB_CLASS_CDC_DATA);
 207:         sBlackListedInterfaces.add(UsbConstants.USB_CLASS_CSCID);
 208:         sBlackListedInterfaces.add(UsbConstants.USB_CLASS_CONTENT_SEC);
 209:         sBlackListedInterfaces.add(UsbConstants.USB_CLASS_VIDEO);
 210:         sBlackListedInterfaces.add(UsbConstants.USB_CLASS_WIRELESS_CONTROLLER);
 211:     }
 212: 
 213:     /*
 214:      * Listens for uevent messages from the kernel to monitor the USB state
 215:      */
 216:     private final class UsbUEventObserver extends UEventObserver {
 217:         @Override
 218:         public void onUEvent(UEventObserver.UEvent event) {
 219:             if (DEBUG) Slog.v(TAG, "USB UEVENT: " + event.toString());
 220: 
 221:             String state = event.get("USB_STATE");
 222:             String accessory = event.get("ACCESSORY");
 223:             if (state != null) {
 224:                 mHandler.updateState(state);
 225:             } else if ("START".equals(accessory)) {
 226:                 if (DEBUG) Slog.d(TAG, "got accessory start");
 227:                 startAccessoryMode();
 228:             }
 229:         }
 230:     }
 231: 
 232:     @Override
 233:     public void onKeyguardStateChanged(boolean isShowing) {
 234:         int userHandle = ActivityManager.getCurrentUser();
 235:         boolean secure = mContext.getSystemService(KeyguardManager.class)
 236:                 .isDeviceSecure(userHandle);
 237:         if (DEBUG) {
 238:             Slog.v(TAG, "onKeyguardStateChanged: isShowing:" + isShowing + " secure:" + secure
 239:                     + " user:" + userHandle);
 240:         }
 241:         // We are unlocked when the keyguard is down or non-secure.
 242:         mHandler.sendMessage(MSG_UPDATE_SCREEN_LOCK, (isShowing && secure));
 243:     }
 244: 
 245:     @Override
 246:     public void onAwakeStateChanged(boolean isAwake) {
 247:         // ignore
 248:     }
 249: 
 250:     /** Called when a user is unlocked. */
 251:     public void onUnlockUser(int userHandle) {
 252:         onKeyguardStateChanged(false);
 253:     }
 254: 
 255:     public UsbDeviceManager(Context context, UsbAlsaManager alsaManager,
 256:             UsbSettingsManager settingsManager, UsbPermissionManager permissionManager) {
 257:         mContext = context;
 258:         mContentResolver = context.getContentResolver();
 259:         PackageManager pm = mContext.getPackageManager();
 260:         mHasUsbAccessory = pm.hasSystemFeature(PackageManager.FEATURE_USB_ACCESSORY);
 261:         initRndisAddress();
 262: 
 263:         boolean halNotPresent = false;
 264:         try {
 265:             IUsbGadget.getService(true);
 266:         } catch (RemoteException e) {
 267:             Slog.e(TAG, "USB GADGET HAL present but exception thrown", e);
 268:         } catch (NoSuchElementException e) {
 269:             halNotPresent = true;
 270:             Slog.i(TAG, "USB GADGET HAL not present in the device", e);
 271:         }
 272: 
 273:         mControlFds = new HashMap<>();
 274:         FileDescriptor mtpFd = nativeOpenControl(UsbManager.USB_FUNCTION_MTP);
 275:         if (mtpFd == null) {
 276:             Slog.e(TAG, "Failed to open control for mtp");
 277:         }
 278:         mControlFds.put(UsbManager.FUNCTION_MTP, mtpFd);
 279:         FileDescriptor ptpFd = nativeOpenControl(UsbManager.USB_FUNCTION_PTP);
 280:         if (ptpFd == null) {
 281:             Slog.e(TAG, "Failed to open control for ptp");
 282:         }
 283:         mControlFds.put(UsbManager.FUNCTION_PTP, ptpFd);
 284: 
 285:         if (halNotPresent) {
 286:             /**
 287:              * Initialze the legacy UsbHandler
 288:              */
 289:             mHandler = new UsbHandlerLegacy(FgThread.get().getLooper(), mContext, this,
 290:                     alsaManager, permissionManager);
 291:         } else {
 292:             /**
 293:              * Initialize HAL based UsbHandler
 294:              */
 295:             mHandler = new UsbHandlerHal(FgThread.get().getLooper(), mContext, this,
 296:                     alsaManager, permissionManager);
 297:         }
 298: 
 299:         if (nativeIsStartRequested()) {
 300:             if (DEBUG) Slog.d(TAG, "accessory attached at boot");
 301:             startAccessoryMode();
 302:         }
 303: 
 304:         BroadcastReceiver portReceiver = new BroadcastReceiver() {
 305:             @Override
 306:             public void onReceive(Context context, Intent intent) {
 307:                 ParcelableUsbPort port = intent.getParcelableExtra(UsbManager.EXTRA_PORT);
 308:                 UsbPortStatus status = intent.getParcelableExtra(UsbManager.EXTRA_PORT_STATUS);
 309:                 mHandler.updateHostState(
 310:                         port.getUsbPort(context.getSystemService(UsbManager.class)), status);
 311:             }
 312:         };
 313: 
 314:         BroadcastReceiver chargingReceiver = new BroadcastReceiver() {
 315:             @Override
 316:             public void onReceive(Context context, Intent intent) {
 317:                 int chargePlug = intent.getIntExtra(BatteryManager.EXTRA_PLUGGED, -1);
 318:                 boolean usbCharging = chargePlug == BatteryManager.BATTERY_PLUGGED_USB;
 319:                 mHandler.sendMessage(MSG_UPDATE_CHARGING_STATE, usbCharging);
 320:             }
 321:         };
 322: 
 323:         BroadcastReceiver hostReceiver = new BroadcastReceiver() {
 324:             @Override
 325:             public void onReceive(Context context, Intent intent) {
 326:                 Iterator devices = ((UsbManager) context.getSystemService(Context.USB_SERVICE))
 327:                         .getDeviceList().entrySet().iterator();
 328:                 if (intent.getAction().equals(UsbManager.ACTION_USB_DEVICE_ATTACHED)) {
 329:                     mHandler.sendMessage(MSG_UPDATE_HOST_STATE, devices, true);
 330:                 } else {
 331:                     mHandler.sendMessage(MSG_UPDATE_HOST_STATE, devices, false);
 332:                 }
 333:             }
 334:         };
 335: 
 336:         BroadcastReceiver languageChangedReceiver = new BroadcastReceiver() {
 337:             @Override
 338:             public void onReceive(Context context, Intent intent) {
 339:                 mHandler.sendEmptyMessage(MSG_LOCALE_CHANGED);
 340:             }
 341:         };
 342: 
 343:         mContext.registerReceiver(portReceiver,
 344:                 new IntentFilter(UsbManager.ACTION_USB_PORT_CHANGED));
 345:         mContext.registerReceiver(chargingReceiver,
 346:                 new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
 347: 
 348:         IntentFilter filter =
 349:                 new IntentFilter(UsbManager.ACTION_USB_DEVICE_ATTACHED);
 350:         filter.addAction(UsbManager.ACTION_USB_DEVICE_DETACHED);
 351:         mContext.registerReceiver(hostReceiver, filter);
 352: 
 353:         mContext.registerReceiver(languageChangedReceiver,
 354:                 new IntentFilter(Intent.ACTION_LOCALE_CHANGED));
 355: 
 356:         // Watch for USB configuration changes
 357:         mUEventObserver = new UsbUEventObserver();
 358:         mUEventObserver.startObserving(USB_STATE_MATCH);
 359:         mUEventObserver.startObserving(ACCESSORY_START_MATCH);
 360:     }
 361: 
 362:     UsbProfileGroupSettingsManager getCurrentSettings() {
 363:         synchronized (mLock) {
 364:             return mCurrentSettings;
 365:         }
 366:     }
 367: 
 368:     String[] getAccessoryStrings() {
 369:         synchronized (mLock) {
 370:             return mAccessoryStrings;
 371:         }
 372:     }
 373: 
 374:     public void systemReady() {
 375:         if (DEBUG) Slog.d(TAG, "systemReady");
 376: 
 377:         LocalServices.getService(ActivityTaskManagerInternal.class).registerScreenObserver(this);
 378: 
 379:         mHandler.sendEmptyMessage(MSG_SYSTEM_READY);
 380:     }
 381: 
 382:     public void bootCompleted() {
 383:         if (DEBUG) Slog.d(TAG, "boot completed");
 384:         mHandler.sendEmptyMessage(MSG_BOOT_COMPLETED);
 385:     }
 386: 
 387:     public void setCurrentUser(int newCurrentUserId, UsbProfileGroupSettingsManager settings) {
 388:         synchronized (mLock) {
 389:             mCurrentSettings = settings;
 390:             mHandler.obtainMessage(MSG_USER_SWITCHED, newCurrentUserId, 0).sendToTarget();
 391:         }
 392:     }
 393: 
 394:     public void updateUserRestrictions() {
 395:         mHandler.sendEmptyMessage(MSG_UPDATE_USER_RESTRICTIONS);
 396:     }
 397: 
 398:     private void startAccessoryMode() {
 399:         if (!mHasUsbAccessory) return;
 400: 
 401:         mAccessoryStrings = nativeGetAccessoryStrings();
 402:         boolean enableAudio = (nativeGetAudioMode() == AUDIO_MODE_SOURCE);
 403:         // don't start accessory mode if our mandatory strings have not been set
 404:         boolean enableAccessory = (mAccessoryStrings != null &&
 405:                 mAccessoryStrings[UsbAccessory.MANUFACTURER_STRING] != null &&
 406:                 mAccessoryStrings[UsbAccessory.MODEL_STRING] != null);
 407: 
 408:         long functions = UsbManager.FUNCTION_NONE;
 409:         if (enableAccessory) {
 410:             functions |= UsbManager.FUNCTION_ACCESSORY;
 411:         }
 412:         if (enableAudio) {
 413:             functions |= UsbManager.FUNCTION_AUDIO_SOURCE;
 414:         }
 415: 
 416:         if (functions != UsbManager.FUNCTION_NONE) {
 417:             mHandler.sendMessageDelayed(mHandler.obtainMessage(MSG_ACCESSORY_MODE_ENTER_TIMEOUT),
 418:                     ACCESSORY_REQUEST_TIMEOUT);
 419:             setCurrentFunctions(functions);
 420:         }
 421:     }
 422: 
 423:     private static void initRndisAddress() {
 424:         // configure RNDIS ethernet address based on our serial number using the same algorithm
 425:         // we had been previously using in kernel board files
 426:         final int ETH_ALEN = 6;
 427:         int address[] = new int[ETH_ALEN];
 428:         // first byte is 0x02 to signify a locally administered address
 429:         address[0] = 0x02;
 430: 
 431:         String serial = SystemProperties.get("ro.serialno", "1234567890ABCDEF");
 432:         int serialLength = serial.length();
 433:         // XOR the USB serial across the remaining 5 bytes
 434:         for (int i = 0; i < serialLength; i++) {
 435:             address[i % (ETH_ALEN - 1) + 1] ^= (int) serial.charAt(i);
 436:         }
 437:         String addrString = String.format(Locale.US, "%02X:%02X:%02X:%02X:%02X:%02X",
 438:                 address[0], address[1], address[2], address[3], address[4], address[5]);
 439:         try {
 440:             FileUtils.stringToFile(RNDIS_ETH_ADDR_PATH, addrString);
 441:         } catch (IOException e) {
 442:             Slog.e(TAG, "failed to write to " + RNDIS_ETH_ADDR_PATH);
 443:         }
 444:     }
 445: 
 446:     abstract static class UsbHandler extends Handler {
 447: 
 448:         // current USB state
 449:         private boolean mHostConnected;
 450:         private boolean mSourcePower;
 451:         private boolean mSinkPower;
 452:         private boolean mConfigured;
 453:         private boolean mAudioAccessoryConnected;
 454:         private boolean mAudioAccessorySupported;
 455: 
 456:         private UsbAccessory mCurrentAccessory;
 457:         private int mUsbNotificationId;
 458:         private boolean mAdbNotificationShown;
 459:         private boolean mUsbCharging;
 460:         private boolean mHideUsbNotification;
 461:         private boolean mSupportsAllCombinations;
 462:         private boolean mScreenLocked;
 463:         private boolean mSystemReady;
 464:         private Intent mBroadcastedIntent;
 465:         private boolean mPendingBootBroadcast;
 466:         private boolean mAudioSourceEnabled;
 467:         private boolean mMidiEnabled;
 468:         private int mMidiCard;
 469:         private int mMidiDevice;
 470: 
 471:         private final Context mContext;
 472:         private final UsbAlsaManager mUsbAlsaManager;
 473:         private final UsbPermissionManager mPermissionManager;
 474:         private NotificationManager mNotificationManager;
 475: 
 476:         protected boolean mConnected;
 477:         protected long mScreenUnlockedFunctions;
 478:         protected boolean mBootCompleted;
 479:         protected boolean mCurrentFunctionsApplied;
 480:         protected boolean mUseUsbNotification;
 481:         protected long mCurrentFunctions;
 482:         protected final UsbDeviceManager mUsbDeviceManager;
 483:         protected final ContentResolver mContentResolver;
 484:         protected SharedPreferences mSettings;
 485:         protected int mCurrentUser;
 486:         protected boolean mCurrentUsbFunctionsReceived;
 487: 
 488:         /**
 489:          * The persistent property which stores whether adb is enabled or not.
 490:          * May also contain vendor-specific default functions for testing purposes.
 491:          */
 492:         protected static final String USB_PERSISTENT_CONFIG_PROPERTY = "persist.sys.usb.config";
 493: 
 494:         UsbHandler(Looper looper, Context context, UsbDeviceManager deviceManager,
 495:                 UsbAlsaManager alsaManager, UsbPermissionManager permissionManager) {
 496:             super(looper);
 497:             mContext = context;
 498:             mUsbDeviceManager = deviceManager;
 499:             mUsbAlsaManager = alsaManager;
 500:             mPermissionManager = permissionManager;
 501:             mContentResolver = context.getContentResolver();
 502: 
 503:             mCurrentUser = ActivityManager.getCurrentUser();
 504:             mScreenLocked = true;
 505: 
 506:             mSettings = getPinnedSharedPrefs(mContext);
 507:             if (mSettings == null) {
 508:                 Slog.e(TAG, "Couldn't load shared preferences");
 509:             } else {
 510:                 mScreenUnlockedFunctions = UsbManager.usbFunctionsFromString(
 511:                         mSettings.getString(
 512:                                 String.format(Locale.ENGLISH, UNLOCKED_CONFIG_PREF, mCurrentUser),
 513:                                 ""));
 514:             }
 515: 
 516:             // We do not show the USB notification if the primary volume supports mass storage.
 517:             // The legacy mass storage UI will be used instead.
 518:             final StorageManager storageManager = StorageManager.from(mContext);
 519:             final StorageVolume primary =
 520:                     storageManager != null ? storageManager.getPrimaryVolume() : null;
 521: 
 522:             boolean massStorageSupported = primary != null && primary.allowMassStorage();
 523:             mUseUsbNotification = !massStorageSupported && mContext.getResources().getBoolean(
 524:                     com.android.internal.R.bool.config_usbChargingMessage);
 525:         }
 526: 
 527:         public void sendMessage(int what, boolean arg) {
 528:             removeMessages(what);
 529:             Message m = Message.obtain(this, what);
 530:             m.arg1 = (arg ? 1 : 0);
 531:             sendMessage(m);
 532:         }
 533: 
 534:         public void sendMessage(int what, Object arg) {
 535:             removeMessages(what);
 536:             Message m = Message.obtain(this, what);
 537:             m.obj = arg;
 538:             sendMessage(m);
 539:         }
 540: 
 541:         public void sendMessage(int what, Object arg, boolean arg1) {
 542:             removeMessages(what);
 543:             Message m = Message.obtain(this, what);
 544:             m.obj = arg;
 545:             m.arg1 = (arg1 ? 1 : 0);
 546:             sendMessage(m);
 547:         }
 548: 
 549:         public void sendMessage(int what, boolean arg1, boolean arg2) {
 550:             removeMessages(what);
 551:             Message m = Message.obtain(this, what);
 552:             m.arg1 = (arg1 ? 1 : 0);
 553:             m.arg2 = (arg2 ? 1 : 0);
 554:             sendMessage(m);
 555:         }
 556: 
 557:         public void sendMessageDelayed(int what, boolean arg, long delayMillis) {
 558:             removeMessages(what);
 559:             Message m = Message.obtain(this, what);
 560:             m.arg1 = (arg ? 1 : 0);
 561:             sendMessageDelayed(m, delayMillis);
 562:         }
 563: 
 564:         public void updateState(String state) {
 565:             int connected, configured;
 566: 
 567:             if ("DISCONNECTED".equals(state)) {
 568:                 connected = 0;
 569:                 configured = 0;
 570:             } else if ("CONNECTED".equals(state)) {
 571:                 connected = 1;
 572:                 configured = 0;
 573:             } else if ("CONFIGURED".equals(state)) {
 574:                 connected = 1;
 575:                 configured = 1;
 576:             } else {
 577:                 Slog.e(TAG, "unknown state " + state);
 578:                 return;
 579:             }
 580:             removeMessages(MSG_UPDATE_STATE);
 581:             if (connected == 1) removeMessages(MSG_FUNCTION_SWITCH_TIMEOUT);
 582:             Message msg = Message.obtain(this, MSG_UPDATE_STATE);
 583:             msg.arg1 = connected;
 584:             msg.arg2 = configured;
 585:             // debounce disconnects to avoid problems bringing up USB tethering
 586:             sendMessageDelayed(msg, (connected == 0) ? UPDATE_DELAY : 0);
 587:         }
 588: 
 589:         public void updateHostState(UsbPort port, UsbPortStatus status) {
 590:             if (DEBUG) {
 591:                 Slog.i(TAG, "updateHostState " + port + " status=" + status);
 592:             }
 593: 
 594:             SomeArgs args = SomeArgs.obtain();
 595:             args.arg1 = port;
 596:             args.arg2 = status;
 597: 
 598:             removeMessages(MSG_UPDATE_PORT_STATE);
 599:             Message msg = obtainMessage(MSG_UPDATE_PORT_STATE, args);
 600:             // debounce rapid transitions of connect/disconnect on type-c ports
 601:             sendMessageDelayed(msg, UPDATE_DELAY);
 602:         }
 603: 
 604:         private void setAdbEnabled(boolean enable) {
 605:             if (DEBUG) Slog.d(TAG, "setAdbEnabled: " + enable);
 606: 
 607:             if (enable) {
 608:                 setSystemProperty(USB_PERSISTENT_CONFIG_PROPERTY, UsbManager.USB_FUNCTION_ADB);
 609:             } else {
 610:                 setSystemProperty(USB_PERSISTENT_CONFIG_PROPERTY, "");
 611:             }
 612: 
 613:             setEnabledFunctions(mCurrentFunctions, true);
 614:             updateAdbNotification(false);
 615:         }
 616: 
 617:         protected boolean isUsbTransferAllowed() {
 618:             UserManager userManager = (UserManager) mContext.getSystemService(Context.USER_SERVICE);
 619:             return !userManager.hasUserRestriction(UserManager.DISALLOW_USB_FILE_TRANSFER);
 620:         }
 621: 
 622:         private void updateCurrentAccessory() {
 623:             // We are entering accessory mode if we have received a request from the host
 624:             // and the request has not timed out yet.
 625:             boolean enteringAccessoryMode = hasMessages(MSG_ACCESSORY_MODE_ENTER_TIMEOUT);
 626: 
 627:             if (mConfigured && enteringAccessoryMode) {
 628:                 // successfully entered accessory mode
 629:                 String[] accessoryStrings = mUsbDeviceManager.getAccessoryStrings();
 630:                 if (accessoryStrings != null) {
 631:                     UsbSerialReader serialReader = new UsbSerialReader(mContext, mPermissionManager,
 632:                             accessoryStrings[UsbAccessory.SERIAL_STRING]);
 633: 
 634:                     mCurrentAccessory = new UsbAccessory(
 635:                             accessoryStrings[UsbAccessory.MANUFACTURER_STRING],
 636:                             accessoryStrings[UsbAccessory.MODEL_STRING],
 637:                             accessoryStrings[UsbAccessory.DESCRIPTION_STRING],
 638:                             accessoryStrings[UsbAccessory.VERSION_STRING],
 639:                             accessoryStrings[UsbAccessory.URI_STRING],
 640:                             serialReader);
 641: 
 642:                     serialReader.setDevice(mCurrentAccessory);
 643: 
 644:                     Slog.d(TAG, "entering USB accessory mode: " + mCurrentAccessory);
 645:                     // defer accessoryAttached if system is not ready
 646:                     if (mBootCompleted) {
 647:                         mUsbDeviceManager.getCurrentSettings().accessoryAttached(mCurrentAccessory);
 648:                     } // else handle in boot completed
 649:                 } else {
 650:                     Slog.e(TAG, "nativeGetAccessoryStrings failed");
 651:                 }
 652:             } else {
 653:                 if (!enteringAccessoryMode) {
 654:                     notifyAccessoryModeExit();
 655:                 } else if (DEBUG) {
 656:                     Slog.v(TAG, "Debouncing accessory mode exit");
 657:                 }
 658:             }
 659:         }
 660: 
 661:         private void notifyAccessoryModeExit() {
 662:             // make sure accessory mode is off
 663:             // and restore default functions
 664:             Slog.d(TAG, "exited USB accessory mode");
 665:             setEnabledFunctions(UsbManager.FUNCTION_NONE, false);
 666: 
 667:             if (mCurrentAccessory != null) {
 668:                 if (mBootCompleted) {
 669:                     mPermissionManager.usbAccessoryRemoved(mCurrentAccessory);
 670:                 }
 671:                 mCurrentAccessory = null;
 672:             }
 673:         }
 674: 
 675:         protected SharedPreferences getPinnedSharedPrefs(Context context) {
 676:             final File prefsFile = new File(
 677:                     Environment.getDataSystemDeDirectory(UserHandle.USER_SYSTEM), USB_PREFS_XML);
 678:             return context.createDeviceProtectedStorageContext()
 679:                     .getSharedPreferences(prefsFile, Context.MODE_PRIVATE);
 680:         }
 681: 
 682:         private boolean isUsbStateChanged(Intent intent) {
 683:             final Set<String> keySet = intent.getExtras().keySet();
 684:             if (mBroadcastedIntent == null) {
 685:                 for (String key : keySet) {
 686:                     if (intent.getBooleanExtra(key, false)) {
 687:                         return true;
 688:                     }
 689:                 }
 690:             } else {
 691:                 if (!keySet.equals(mBroadcastedIntent.getExtras().keySet())) {
 692:                     return true;
 693:                 }
 694:                 for (String key : keySet) {
 695:                     if (intent.getBooleanExtra(key, false) !=
 696:                             mBroadcastedIntent.getBooleanExtra(key, false)) {
 697:                         return true;
 698:                     }
 699:                 }
 700:             }
 701:             return false;
 702:         }
 703: 
 704:         protected void updateUsbStateBroadcastIfNeeded(long functions) {
 705:             // send a sticky broadcast containing current USB state
 706:             Intent intent = new Intent(UsbManager.ACTION_USB_STATE);
 707:             intent.addFlags(Intent.FLAG_RECEIVER_REPLACE_PENDING
 708:                     | Intent.FLAG_RECEIVER_INCLUDE_BACKGROUND
 709:                     | Intent.FLAG_RECEIVER_FOREGROUND);
 710:             intent.putExtra(UsbManager.USB_CONNECTED, mConnected);
 711:             intent.putExtra(UsbManager.USB_HOST_CONNECTED, mHostConnected);
 712:             intent.putExtra(UsbManager.USB_CONFIGURED, mConfigured);
 713:             intent.putExtra(UsbManager.USB_DATA_UNLOCKED,
 714:                     isUsbTransferAllowed() && isUsbDataTransferActive(mCurrentFunctions));
 715: 
 716:             long remainingFunctions = functions;
 717:             while (remainingFunctions != 0) {
 718:                 intent.putExtra(UsbManager.usbFunctionsToString(
 719:                         Long.highestOneBit(remainingFunctions)), true);
 720:                 remainingFunctions -= Long.highestOneBit(remainingFunctions);
 721:             }
 722: 
 723:             // send broadcast intent only if the USB state has changed
 724:             if (!isUsbStateChanged(intent)) {
 725:                 if (DEBUG) {
 726:                     Slog.d(TAG, "skip broadcasting " + intent + " extras: " + intent.getExtras());
 727:                 }
 728:                 return;
 729:             }
 730: 
 731:             if (DEBUG) Slog.d(TAG, "broadcasting " + intent + " extras: " + intent.getExtras());
 732:             sendStickyBroadcast(intent);
 733:             mBroadcastedIntent = intent;
 734:         }
 735: 
 736:         protected void sendStickyBroadcast(Intent intent) {
 737:             mContext.sendStickyBroadcastAsUser(intent, UserHandle.ALL);
 738:         }
 739: 
 740:         private void updateUsbFunctions() {
 741:             updateMidiFunction();
 742:         }
 743: 
 744:         private void updateMidiFunction() {
 745:             boolean enabled = (mCurrentFunctions & UsbManager.FUNCTION_MIDI) != 0;
 746:             if (enabled != mMidiEnabled) {
 747:                 if (enabled) {
 748:                     Scanner scanner = null;
 749:                     try {
 750:                         scanner = new Scanner(new File(MIDI_ALSA_PATH));
 751:                         mMidiCard = scanner.nextInt();
 752:                         mMidiDevice = scanner.nextInt();
 753:                     } catch (FileNotFoundException e) {
 754:                         Slog.e(TAG, "could not open MIDI file", e);
 755:                         enabled = false;
 756:                     } finally {
 757:                         if (scanner != null) {
 758:                             scanner.close();
 759:                         }
 760:                     }
 761:                 }
 762:                 mMidiEnabled = enabled;
 763:             }
 764:             mUsbAlsaManager.setPeripheralMidiState(
 765:                     mMidiEnabled && mConfigured, mMidiCard, mMidiDevice);
 766:         }
 767: 
 768:         private void setScreenUnlockedFunctions() {
 769:             setEnabledFunctions(mScreenUnlockedFunctions, false);
 770:         }
 771: 
 772:         private static class AdbTransport extends IAdbTransport.Stub {
 773:             private final UsbHandler mHandler;
 774: 
 775:             AdbTransport(UsbHandler handler) {
 776:                 mHandler = handler;
 777:             }
 778: 
 779:             @Override
 780:             public void onAdbEnabled(boolean enabled, byte transportType) {
 781:                 if (transportType == AdbTransportType.USB) {
 782:                     mHandler.sendMessage(MSG_ENABLE_ADB, enabled);
 783:                 }
 784:             }
 785:         }
 786: 
 787:         /**
 788:          * Returns the functions that are passed down to the low level driver once adb and
 789:          * charging are accounted for.
 790:          */
 791:         long getAppliedFunctions(long functions) {
 792:             if (functions == UsbManager.FUNCTION_NONE) {
 793:                 return getChargingFunctions();
 794:             }
 795:             if (isAdbEnabled()) {
 796:                 return functions | UsbManager.FUNCTION_ADB;
 797:             }
 798:             return functions;
 799:         }
 800: 
 801:         @Override
 802:         public void handleMessage(Message msg) {
 803:             switch (msg.what) {
 804:                 case MSG_UPDATE_STATE:
 805:                     mConnected = (msg.arg1 == 1);
 806:                     mConfigured = (msg.arg2 == 1);
 807: 
 808:                     updateUsbNotification(false);
 809:                     updateAdbNotification(false);
 810:                     if (mBootCompleted) {
 811:                         updateUsbStateBroadcastIfNeeded(getAppliedFunctions(mCurrentFunctions));
 812:                     }
 813:                     if ((mCurrentFunctions & UsbManager.FUNCTION_ACCESSORY) != 0) {
 814:                         updateCurrentAccessory();
 815:                     }
 816:                     if (mBootCompleted) {
 817:                         if (!mConnected && !hasMessages(MSG_ACCESSORY_MODE_ENTER_TIMEOUT)
 818:                                 && !hasMessages(MSG_FUNCTION_SWITCH_TIMEOUT)) {
 819:                             // restore defaults when USB is disconnected
 820:                             if (!mScreenLocked
 821:                                     && mScreenUnlockedFunctions != UsbManager.FUNCTION_NONE) {
 822:                                 setScreenUnlockedFunctions();
 823:                             } else {
 824:                                 setEnabledFunctions(UsbManager.FUNCTION_NONE, false);
 825:                             }
 826:                         }
 827:                         updateUsbFunctions();
 828:                     } else {
 829:                         mPendingBootBroadcast = true;
 830:                     }
 831:                     break;
 832:                 case MSG_UPDATE_PORT_STATE:
 833:                     SomeArgs args = (SomeArgs) msg.obj;
 834:                     boolean prevHostConnected = mHostConnected;
 835:                     UsbPort port = (UsbPort) args.arg1;
 836:                     UsbPortStatus status = (UsbPortStatus) args.arg2;
 837:                     mHostConnected = status.getCurrentDataRole() == DATA_ROLE_HOST;
 838:                     mSourcePower = status.getCurrentPowerRole() == POWER_ROLE_SOURCE;
 839:                     mSinkPower = status.getCurrentPowerRole() == POWER_ROLE_SINK;
 840:                     mAudioAccessoryConnected = (status.getCurrentMode() == MODE_AUDIO_ACCESSORY);
 841:                     mAudioAccessorySupported = port.isModeSupported(MODE_AUDIO_ACCESSORY);
 842:                     // Ideally we want to see if PR_SWAP and DR_SWAP is supported.
 843:                     // But, this should be suffice, since, all four combinations are only supported
 844:                     // when PR_SWAP and DR_SWAP are supported.
 845:                     mSupportsAllCombinations = status.isRoleCombinationSupported(
 846:                             POWER_ROLE_SOURCE, DATA_ROLE_HOST)
 847:                             && status.isRoleCombinationSupported(POWER_ROLE_SINK, DATA_ROLE_HOST)
 848:                             && status.isRoleCombinationSupported(POWER_ROLE_SOURCE,
 849:                             DATA_ROLE_DEVICE)
 850:                             && status.isRoleCombinationSupported(POWER_ROLE_SINK, DATA_ROLE_DEVICE);
 851: 
 852:                     args.recycle();
 853:                     updateUsbNotification(false);
 854:                     if (mBootCompleted) {
 855:                         if (mHostConnected || prevHostConnected) {
 856:                             updateUsbStateBroadcastIfNeeded(getAppliedFunctions(mCurrentFunctions));
 857:                         }
 858:                     } else {
 859:                         mPendingBootBroadcast = true;
 860:                     }
 861:                     break;
 862:                 case MSG_UPDATE_CHARGING_STATE:
 863:                     mUsbCharging = (msg.arg1 == 1);
 864:                     updateUsbNotification(false);
 865:                     break;
 866:                 case MSG_UPDATE_HOST_STATE:
 867:                     Iterator devices = (Iterator) msg.obj;
 868:                     boolean connected = (msg.arg1 == 1);
 869: 
 870:                     if (DEBUG) {
 871:                         Slog.i(TAG, "HOST_STATE connected:" + connected);
 872:                     }
 873: 
 874:                     mHideUsbNotification = false;
 875:                     while (devices.hasNext()) {
 876:                         Map.Entry pair = (Map.Entry) devices.next();
 877:                         if (DEBUG) {
 878:                             Slog.i(TAG, pair.getKey() + " = " + pair.getValue());
 879:                         }
 880:                         UsbDevice device = (UsbDevice) pair.getValue();
 881:                         int configurationCount = device.getConfigurationCount() - 1;
 882:                         while (configurationCount >= 0) {
 883:                             UsbConfiguration config = device.getConfiguration(configurationCount);
 884:                             configurationCount--;
 885:                             int interfaceCount = config.getInterfaceCount() - 1;
 886:                             while (interfaceCount >= 0) {
 887:                                 UsbInterface intrface = config.getInterface(interfaceCount);
 888:                                 interfaceCount--;
 889:                                 if (sBlackListedInterfaces.contains(intrface.getInterfaceClass())) {
 890:                                     mHideUsbNotification = true;
 891:                                     break;
 892:                                 }
 893:                             }
 894:                         }
 895:                     }
 896:                     updateUsbNotification(false);
 897:                     break;
 898:                 case MSG_ENABLE_ADB:
 899:                     setAdbEnabled(msg.arg1 == 1);
 900:                     break;
 901:                 case MSG_SET_CURRENT_FUNCTIONS:
 902:                     long functions = (Long) msg.obj;
 903:                     setEnabledFunctions(functions, false);
 904:                     break;
 905:                 case MSG_SET_SCREEN_UNLOCKED_FUNCTIONS:
 906:                     mScreenUnlockedFunctions = (Long) msg.obj;
 907:                     if (mSettings != null) {
 908:                         SharedPreferences.Editor editor = mSettings.edit();
 909:                         editor.putString(String.format(Locale.ENGLISH, UNLOCKED_CONFIG_PREF,
 910:                                 mCurrentUser),
 911:                                 UsbManager.usbFunctionsToString(mScreenUnlockedFunctions));
 912:                         editor.commit();
 913:                     }
 914:                     if (!mScreenLocked && mScreenUnlockedFunctions != UsbManager.FUNCTION_NONE) {
 915:                         // If the screen is unlocked, also set current functions.
 916:                         setScreenUnlockedFunctions();
 917:                     } else {
 918:                         setEnabledFunctions(UsbManager.FUNCTION_NONE, false);
 919:                     }
 920:                     break;
 921:                 case MSG_UPDATE_SCREEN_LOCK:
 922:                     if (msg.arg1 == 1 == mScreenLocked) {
 923:                         break;
 924:                     }
 925:                     mScreenLocked = msg.arg1 == 1;
 926:                     if (!mBootCompleted) {
 927:                         break;
 928:                     }
 929:                     if (mScreenLocked) {
 930:                         if (!mConnected) {
 931:                             setEnabledFunctions(UsbManager.FUNCTION_NONE, false);
 932:                         }
 933:                     } else {
 934:                         if (mScreenUnlockedFunctions != UsbManager.FUNCTION_NONE
 935:                                 && mCurrentFunctions == UsbManager.FUNCTION_NONE) {
 936:                             // Set the screen unlocked functions if current function is charging.
 937:                             setScreenUnlockedFunctions();
 938:                         }
 939:                     }
 940:                     break;
 941:                 case MSG_UPDATE_USER_RESTRICTIONS:
 942:                     // Restart the USB stack if USB transfer is enabled but no longer allowed.
 943:                     if (isUsbDataTransferActive(mCurrentFunctions) && !isUsbTransferAllowed()) {
 944:                         setEnabledFunctions(UsbManager.FUNCTION_NONE, true);
 945:                     }
 946:                     break;
 947:                 case MSG_SYSTEM_READY:
 948:                     mNotificationManager = (NotificationManager)
 949:                             mContext.getSystemService(Context.NOTIFICATION_SERVICE);
 950: 
 951:                     LocalServices.getService(
 952:                             AdbManagerInternal.class).registerTransport(new AdbTransport(this));
 953: 
 954:                     // Ensure that the notification channels are set up
 955:                     if (isTv()) {
 956:                         // TV-specific notification channel
 957:                         mNotificationManager.createNotificationChannel(
 958:                                 new NotificationChannel(ADB_NOTIFICATION_CHANNEL_ID_TV,
 959:                                         mContext.getString(
 960:                                                 com.android.internal.R.string
 961:                                                         .adb_debugging_notification_channel_tv),
 962:                                         NotificationManager.IMPORTANCE_HIGH));
 963:                     }
 964:                     mSystemReady = true;
 965:                     finishBoot();
 966:                     break;
 967:                 case MSG_LOCALE_CHANGED:
 968:                     updateAdbNotification(true);
 969:                     updateUsbNotification(true);
 970:                     break;
 971:                 case MSG_BOOT_COMPLETED:
 972:                     mBootCompleted = true;
 973:                     finishBoot();
 974:                     break;
 975:                 case MSG_USER_SWITCHED: {
 976:                     if (mCurrentUser != msg.arg1) {
 977:                         if (DEBUG) {
 978:                             Slog.v(TAG, "Current user switched to " + msg.arg1);
 979:                         }
 980:                         mCurrentUser = msg.arg1;
 981:                         mScreenLocked = true;
 982:                         mScreenUnlockedFunctions = UsbManager.FUNCTION_NONE;
 983:                         if (mSettings != null) {
 984:                             mScreenUnlockedFunctions = UsbManager.usbFunctionsFromString(
 985:                                     mSettings.getString(String.format(Locale.ENGLISH,
 986:                                             UNLOCKED_CONFIG_PREF, mCurrentUser), ""));
 987:                         }
 988:                         setEnabledFunctions(UsbManager.FUNCTION_NONE, false);
 989:                     }
 990:                     break;
 991:                 }
 992:                 case MSG_ACCESSORY_MODE_ENTER_TIMEOUT: {
 993:                     if (DEBUG) {
 994:                         Slog.v(TAG, "Accessory mode enter timeout: " + mConnected);
 995:                     }
 996:                     if (!mConnected || (mCurrentFunctions & UsbManager.FUNCTION_ACCESSORY) == 0) {
 997:                         notifyAccessoryModeExit();
 998:                     }
 999:                     break;
1000:                 }
1001:             }
1002:         }
1003: 
1004:         protected void finishBoot() {
1005:             if (mBootCompleted && mCurrentUsbFunctionsReceived && mSystemReady) {
1006:                 if (mPendingBootBroadcast) {
1007:                     updateUsbStateBroadcastIfNeeded(getAppliedFunctions(mCurrentFunctions));
1008:                     mPendingBootBroadcast = false;
1009:                 }
1010:                 if (!mScreenLocked
1011:                         && mScreenUnlockedFunctions != UsbManager.FUNCTION_NONE) {
1012:                     setScreenUnlockedFunctions();
1013:                 } else {
1014:                     setEnabledFunctions(UsbManager.FUNCTION_NONE, false);
1015:                 }
1016:                 if (mCurrentAccessory != null) {
1017:                     mUsbDeviceManager.getCurrentSettings().accessoryAttached(mCurrentAccessory);
1018:                 }
1019: 
1020:                 updateUsbNotification(false);
1021:                 updateAdbNotification(false);
1022:                 updateUsbFunctions();
1023:             }
1024:         }
1025: 
1026:         protected boolean isUsbDataTransferActive(long functions) {
1027:             return (functions & UsbManager.FUNCTION_MTP) != 0
1028:                     || (functions & UsbManager.FUNCTION_PTP) != 0;
1029:         }
1030: 
1031:         public UsbAccessory getCurrentAccessory() {
1032:             return mCurrentAccessory;
1033:         }
1034: 
1035:         protected void updateUsbNotification(boolean force) {
1036:             if (mNotificationManager == null || !mUseUsbNotification
1037:                     || ("0".equals(getSystemProperty("persist.charging.notify", "")))) {
1038:                 return;
1039:             }
1040: 
1041:             // Dont show the notification when connected to a USB peripheral
1042:             // and the link does not support PR_SWAP and DR_SWAP
1043:             if (mHideUsbNotification && !mSupportsAllCombinations) {
1044:                 if (mUsbNotificationId != 0) {
1045:                     mNotificationManager.cancelAsUser(null, mUsbNotificationId,
1046:                             UserHandle.ALL);
1047:                     mUsbNotificationId = 0;
1048:                     Slog.d(TAG, "Clear notification");
1049:                 }
1050:                 return;
1051:             }
1052: 
1053:             int id = 0;
1054:             int titleRes = 0;
1055:             Resources r = mContext.getResources();
1056:             CharSequence message = r.getText(
1057:                     com.android.internal.R.string.usb_notification_message);
1058:             if (mAudioAccessoryConnected && !mAudioAccessorySupported) {
1059:                 titleRes = com.android.internal.R.string.usb_unsupported_audio_accessory_title;
1060:                 id = SystemMessage.NOTE_USB_AUDIO_ACCESSORY_NOT_SUPPORTED;
1061:             } else if (mConnected) {
1062:                 if (mCurrentFunctions == UsbManager.FUNCTION_MTP) {
1063:                     titleRes = com.android.internal.R.string.usb_mtp_notification_title;
1064:                     id = SystemMessage.NOTE_USB_MTP;
1065:                 } else if (mCurrentFunctions == UsbManager.FUNCTION_PTP) {
1066:                     titleRes = com.android.internal.R.string.usb_ptp_notification_title;
1067:                     id = SystemMessage.NOTE_USB_PTP;
1068:                 } else if (mCurrentFunctions == UsbManager.FUNCTION_MIDI) {
1069:                     titleRes = com.android.internal.R.string.usb_midi_notification_title;
1070:                     id = SystemMessage.NOTE_USB_MIDI;
1071:                 } else if (mCurrentFunctions == UsbManager.FUNCTION_RNDIS) {
1072:                     titleRes = com.android.internal.R.string.usb_tether_notification_title;
1073:                     id = SystemMessage.NOTE_USB_TETHER;
1074:                 } else if (mCurrentFunctions == UsbManager.FUNCTION_ACCESSORY) {
1075:                     titleRes = com.android.internal.R.string.usb_accessory_notification_title;
1076:                     id = SystemMessage.NOTE_USB_ACCESSORY;
1077:                 }
1078:                 if (mSourcePower) {
1079:                     if (titleRes != 0) {
1080:                         message = r.getText(
1081:                                 com.android.internal.R.string.usb_power_notification_message);
1082:                     } else {
1083:                         titleRes = com.android.internal.R.string.usb_supplying_notification_title;
1084:                         id = SystemMessage.NOTE_USB_SUPPLYING;
1085:                     }
1086:                 } else if (titleRes == 0) {
1087:                     titleRes = com.android.internal.R.string.usb_charging_notification_title;
1088:                     id = SystemMessage.NOTE_USB_CHARGING;
1089:                 }
1090:             } else if (mSourcePower) {
1091:                 titleRes = com.android.internal.R.string.usb_supplying_notification_title;
1092:                 id = SystemMessage.NOTE_USB_SUPPLYING;
1093:             } else if (mHostConnected && mSinkPower && mUsbCharging) {
1094:                 titleRes = com.android.internal.R.string.usb_charging_notification_title;
1095:                 id = SystemMessage.NOTE_USB_CHARGING;
1096:             }
1097:             if (id != mUsbNotificationId || force) {
1098:                 // clear notification if title needs changing
1099:                 if (mUsbNotificationId != 0) {
1100:                     mNotificationManager.cancelAsUser(null, mUsbNotificationId,
1101:                             UserHandle.ALL);
1102:                     Slog.d(TAG, "Clear notification");
1103:                     mUsbNotificationId = 0;
1104:                 }
1105:                 // Not relevant for automotive.
1106:                 if (mContext.getPackageManager().hasSystemFeature(
1107:                         PackageManager.FEATURE_AUTOMOTIVE)
1108:                         && id == SystemMessage.NOTE_USB_CHARGING) {
1109:                     mUsbNotificationId = 0;
1110:                     return;
1111:                 }
1112: 
1113:                 if (id != 0) {
1114:                     CharSequence title = r.getText(titleRes);
1115:                     PendingIntent pi;
1116:                     String channel;
1117: 
1118:                     if (titleRes
1119:                             != com.android.internal.R.string
1120:                             .usb_unsupported_audio_accessory_title) {
1121:                         Intent intent = Intent.makeRestartActivityTask(
1122:                                 new ComponentName("com.android.settings",
1123:                                         "com.android.settings.Settings$UsbDetailsActivity"));
1124:                         pi = PendingIntent.getActivityAsUser(mContext, 0,
1125:                                 intent, 0, null, UserHandle.CURRENT);
1126:                         channel = SystemNotificationChannels.USB;
1127:                     } else {
1128:                         final Intent intent = new Intent();
1129:                         intent.setClassName("com.android.settings",
1130:                                 "com.android.settings.HelpTrampoline");
1131:                         intent.putExtra(Intent.EXTRA_TEXT,
1132:                                 "help_url_audio_accessory_not_supported");
1133: 
1134:                         if (mContext.getPackageManager().resolveActivity(intent, 0) != null) {
1135:                             pi = PendingIntent.getActivity(mContext, 0, intent, 0);
1136:                         } else {
1137:                             pi = null;
1138:                         }
1139: 
1140:                         channel = SystemNotificationChannels.ALERTS;
1141:                         message = r.getText(
1142:                                 com.android.internal.R.string
1143:                                         .usb_unsupported_audio_accessory_message);
1144:                     }
1145: 
1146:                     Notification.Builder builder = new Notification.Builder(mContext, channel)
1147:                             .setSmallIcon(com.android.internal.R.drawable.stat_sys_adb)
1148:                             .setWhen(0)
1149:                             .setOngoing(true)
1150:                             .setTicker(title)
1151:                             .setDefaults(0)  // please be quiet
1152:                             .setColor(mContext.getColor(
1153:                                     com.android.internal.R.color
1154:                                             .system_notification_accent_color))
1155:                             .setContentTitle(title)
1156:                             .setContentText(message)
1157:                             .setContentIntent(pi)
1158:                             .setVisibility(Notification.VISIBILITY_PUBLIC);
1159: 
1160:                     if (titleRes
1161:                             == com.android.internal.R.string
1162:                             .usb_unsupported_audio_accessory_title) {
1163:                         builder.setStyle(new Notification.BigTextStyle()
1164:                                 .bigText(message));
1165:                     }
1166:                     Notification notification = builder.build();
1167: 
1168:                     mNotificationManager.notifyAsUser(null, id, notification,
1169:                             UserHandle.ALL);
1170:                     Slog.d(TAG, "push notification:" + title);
1171:                     mUsbNotificationId = id;
1172:                 }
1173:             }
1174:         }
1175: 
1176:         protected boolean isAdbEnabled() {
1177:             return LocalServices.getService(AdbManagerInternal.class)
1178:                     .isAdbEnabled(AdbTransportType.USB);
1179:         }
1180: 
1181:         protected void updateAdbNotification(boolean force) {
1182:             if (mNotificationManager == null) return;
1183:             final int id = SystemMessage.NOTE_ADB_ACTIVE;
1184: 
1185:             if (isAdbEnabled() && mConnected) {
1186:                 if ("0".equals(getSystemProperty("persist.adb.notify", ""))) return;
1187: 
1188:                 if (force && mAdbNotificationShown) {
1189:                     mAdbNotificationShown = false;
1190:                     mNotificationManager.cancelAsUser(null, id, UserHandle.ALL);
1191:                 }
1192: 
1193:                 if (!mAdbNotificationShown) {
1194:                     Notification notification = AdbNotifications.createNotification(mContext,
1195:                             AdbTransportType.USB);
1196:                     mAdbNotificationShown = true;
1197:                     mNotificationManager.notifyAsUser(null, id, notification, UserHandle.ALL);
1198:                 }
1199:             } else if (mAdbNotificationShown) {
1200:                 mAdbNotificationShown = false;
1201:                 mNotificationManager.cancelAsUser(null, id, UserHandle.ALL);
1202:             }
1203:         }
1204: 
1205:         private boolean isTv() {
1206:             return mContext.getPackageManager().hasSystemFeature(PackageManager.FEATURE_LEANBACK);
1207:         }
1208: 
1209:         protected long getChargingFunctions() {
1210:             // if ADB is enabled, reset functions to ADB
1211:             // else enable MTP as usual.
1212:             if (isAdbEnabled()) {
1213:                 return UsbManager.FUNCTION_ADB;
1214:             } else {
1215:                 return UsbManager.FUNCTION_MTP;
1216:             }
1217:         }
1218: 
1219:         protected void setSystemProperty(String prop, String val) {
1220:             SystemProperties.set(prop, val);
1221:         }
1222: 
1223:         protected String getSystemProperty(String prop, String def) {
1224:             return SystemProperties.get(prop, def);
1225:         }
1226: 
1227:         protected void putGlobalSettings(ContentResolver contentResolver, String setting, int val) {
1228:             Settings.Global.putInt(contentResolver, setting, val);
1229:         }
1230: 
1231:         public long getEnabledFunctions() {
1232:             return mCurrentFunctions;
1233:         }
1234: 
1235:         public long getScreenUnlockedFunctions() {
1236:             return mScreenUnlockedFunctions;
1237:         }
1238: 
1239:         /**
1240:          * Dump a functions mask either as proto-enums (if dumping to proto) or a string (if dumping
1241:          * to a print writer)
1242:          */
1243:         private void dumpFunctions(DualDumpOutputStream dump, String idName, long id,
1244:                 long functions) {
1245:             // UsbHandlerProto.UsbFunction matches GadgetFunction
1246:             for (int i = 0; i < 63; i++) {
1247:                 if ((functions & (1L << i)) != 0) {
1248:                     if (dump.isProto()) {
1249:                         dump.write(idName, id, 1L << i);
1250:                     } else {
1251:                         dump.write(idName, id, GadgetFunction.toString(1L << i));
1252:                     }
1253:                 }
1254:             }
1255:         }
1256: 
1257:         public void dump(DualDumpOutputStream dump, String idName, long id) {
1258:             long token = dump.start(idName, id);
1259: 
1260:             dumpFunctions(dump, "current_functions", UsbHandlerProto.CURRENT_FUNCTIONS,
1261:                     mCurrentFunctions);
1262:             dump.write("current_functions_applied", UsbHandlerProto.CURRENT_FUNCTIONS_APPLIED,
1263:                     mCurrentFunctionsApplied);
1264:             dumpFunctions(dump, "screen_unlocked_functions",
1265:                     UsbHandlerProto.SCREEN_UNLOCKED_FUNCTIONS, mScreenUnlockedFunctions);
1266:             dump.write("screen_locked", UsbHandlerProto.SCREEN_LOCKED, mScreenLocked);
1267:             dump.write("connected", UsbHandlerProto.CONNECTED, mConnected);
1268:             dump.write("configured", UsbHandlerProto.CONFIGURED, mConfigured);
1269:             if (mCurrentAccessory != null) {
1270:                 writeAccessory(dump, "current_accessory", UsbHandlerProto.CURRENT_ACCESSORY,
1271:                         mCurrentAccessory);
1272:             }
1273:             dump.write("host_connected", UsbHandlerProto.HOST_CONNECTED, mHostConnected);
1274:             dump.write("source_power", UsbHandlerProto.SOURCE_POWER, mSourcePower);
1275:             dump.write("sink_power", UsbHandlerProto.SINK_POWER, mSinkPower);
1276:             dump.write("usb_charging", UsbHandlerProto.USB_CHARGING, mUsbCharging);
1277:             dump.write("hide_usb_notification", UsbHandlerProto.HIDE_USB_NOTIFICATION,
1278:                     mHideUsbNotification);
1279:             dump.write("audio_accessory_connected", UsbHandlerProto.AUDIO_ACCESSORY_CONNECTED,
1280:                     mAudioAccessoryConnected);
1281: 
1282:             try {
1283:                 writeStringIfNotNull(dump, "kernel_state", UsbHandlerProto.KERNEL_STATE,
1284:                         FileUtils.readTextFile(new File(STATE_PATH), 0, null).trim());
1285:             } catch (Exception e) {
1286:                 Slog.e(TAG, "Could not read kernel state", e);
1287:             }
1288: 
1289:             try {
1290:                 writeStringIfNotNull(dump, "kernel_function_list",
1291:                         UsbHandlerProto.KERNEL_FUNCTION_LIST,
1292:                         FileUtils.readTextFile(new File(FUNCTIONS_PATH), 0, null).trim());
1293:             } catch (Exception e) {
1294:                 Slog.e(TAG, "Could not read kernel function list", e);
1295:             }
1296: 
1297:             dump.end(token);
1298:         }
1299: 
1300:         /**
1301:          * Evaluates USB function policies and applies the change accordingly.
1302:          */
1303:         protected abstract void setEnabledFunctions(long functions, boolean forceRestart);
1304:     }
1305: 
1306:     private static final class UsbHandlerLegacy extends UsbHandler {
1307:         /**
1308:          * The non-persistent property which stores the current USB settings.
1309:          */
1310:         private static final String USB_CONFIG_PROPERTY = "sys.usb.config";
1311: 
1312:         /**
1313:          * The non-persistent property which stores the current USB actual state.
1314:          */
1315:         private static final String USB_STATE_PROPERTY = "sys.usb.state";
1316: 
1317:         private HashMap<String, HashMap<String, Pair<String, String>>> mOemModeMap;
1318:         private String mCurrentOemFunctions;
1319:         private String mCurrentFunctionsStr;
1320:         private boolean mUsbDataUnlocked;
1321: 
1322:         UsbHandlerLegacy(Looper looper, Context context, UsbDeviceManager deviceManager,
1323:                 UsbAlsaManager alsaManager, UsbPermissionManager permissionManager) {
1324:             super(looper, context, deviceManager, alsaManager, permissionManager);
1325:             try {
1326:                 readOemUsbOverrideConfig(context);
1327:                 // Restore default functions.
1328:                 mCurrentOemFunctions = getSystemProperty(getPersistProp(false),
1329:                         UsbManager.USB_FUNCTION_NONE);
1330:                 if (isNormalBoot()) {
1331:                     mCurrentFunctionsStr = getSystemProperty(USB_CONFIG_PROPERTY,
1332:                             UsbManager.USB_FUNCTION_NONE);
1333:                     mCurrentFunctionsApplied = mCurrentFunctionsStr.equals(
1334:                             getSystemProperty(USB_STATE_PROPERTY, UsbManager.USB_FUNCTION_NONE));
1335:                 } else {
1336:                     mCurrentFunctionsStr = getSystemProperty(getPersistProp(true),
1337:                             UsbManager.USB_FUNCTION_NONE);
1338:                     mCurrentFunctionsApplied = getSystemProperty(USB_CONFIG_PROPERTY,
1339:                             UsbManager.USB_FUNCTION_NONE).equals(
1340:                             getSystemProperty(USB_STATE_PROPERTY, UsbManager.USB_FUNCTION_NONE));
1341:                 }
1342:                 mCurrentFunctions = UsbManager.FUNCTION_NONE;
1343:                 mCurrentUsbFunctionsReceived = true;
1344: 
1345:                 String state = FileUtils.readTextFile(new File(STATE_PATH), 0, null).trim();
1346:                 updateState(state);
1347:             } catch (Exception e) {
1348:                 Slog.e(TAG, "Error initializing UsbHandler", e);
1349:             }
1350:         }
1351: 
1352:         private void readOemUsbOverrideConfig(Context context) {
1353:             String[] configList = context.getResources().getStringArray(
1354:                     com.android.internal.R.array.config_oemUsbModeOverride);
1355: 
1356:             if (configList != null) {
1357:                 for (String config : configList) {
1358:                     String[] items = config.split(":");
1359:                     if (items.length == 3 || items.length == 4) {
1360:                         if (mOemModeMap == null) {
1361:                             mOemModeMap = new HashMap<>();
1362:                         }
1363:                         HashMap<String, Pair<String, String>> overrideMap =
1364:                                 mOemModeMap.get(items[0]);
1365:                         if (overrideMap == null) {
1366:                             overrideMap = new HashMap<>();
1367:                             mOemModeMap.put(items[0], overrideMap);
1368:                         }
1369: 
1370:                         // Favoring the first combination if duplicate exists
1371:                         if (!overrideMap.containsKey(items[1])) {
1372:                             if (items.length == 3) {
1373:                                 overrideMap.put(items[1], new Pair<>(items[2], ""));
1374:                             } else {
1375:                                 overrideMap.put(items[1], new Pair<>(items[2], items[3]));
1376:                             }
1377:                         }
1378:                     }
1379:                 }
1380:             }
1381:         }
1382: 
1383:         private String applyOemOverrideFunction(String usbFunctions) {
1384:             if ((usbFunctions == null) || (mOemModeMap == null)) {
1385:                 return usbFunctions;
1386:             }
1387: 
1388:             String bootMode = getSystemProperty(BOOT_MODE_PROPERTY, "unknown");
1389:             Slog.d(TAG, "applyOemOverride usbfunctions=" + usbFunctions + " bootmode=" + bootMode);
1390: 
1391:             Map<String, Pair<String, String>> overridesMap =
1392:                     mOemModeMap.get(bootMode);
1393:             // Check to ensure that the oem is not overriding in the normal
1394:             // boot mode
1395:             if (overridesMap != null && !(bootMode.equals(NORMAL_BOOT)
1396:                     || bootMode.equals("unknown"))) {
1397:                 Pair<String, String> overrideFunctions =
1398:                         overridesMap.get(usbFunctions);
1399:                 if (overrideFunctions != null) {
1400:                     Slog.d(TAG, "OEM USB override: " + usbFunctions
1401:                             + " ==> " + overrideFunctions.first
1402:                             + " persist across reboot "
1403:                             + overrideFunctions.second);
1404:                     if (!overrideFunctions.second.equals("")) {
1405:                         String newFunction;
1406:                         if (isAdbEnabled()) {
1407:                             newFunction = addFunction(overrideFunctions.second,
1408:                                     UsbManager.USB_FUNCTION_ADB);
1409:                         } else {
1410:                             newFunction = overrideFunctions.second;
1411:                         }
1412:                         Slog.d(TAG, "OEM USB override persisting: " + newFunction + "in prop: "
1413:                                 + getPersistProp(false));
1414:                         setSystemProperty(getPersistProp(false), newFunction);
1415:                     }
1416:                     return overrideFunctions.first;
1417:                 } else if (isAdbEnabled()) {
1418:                     String newFunction = addFunction(UsbManager.USB_FUNCTION_NONE,
1419:                             UsbManager.USB_FUNCTION_ADB);
1420:                     setSystemProperty(getPersistProp(false), newFunction);
1421:                 } else {
1422:                     setSystemProperty(getPersistProp(false), UsbManager.USB_FUNCTION_NONE);
1423:                 }
1424:             }
1425:             // return passed in functions as is.
1426:             return usbFunctions;
1427:         }
1428: 
1429:         private boolean waitForState(String state) {
1430:             // wait for the transition to complete.
1431:             // give up after 1 second.
1432:             String value = null;
1433:             for (int i = 0; i < 20; i++) {
1434:                 // State transition is done when sys.usb.state is set to the new configuration
1435:                 value = getSystemProperty(USB_STATE_PROPERTY, "");
1436:                 if (state.equals(value)) return true;
1437:                 SystemClock.sleep(50);
1438:             }
1439:             Slog.e(TAG, "waitForState(" + state + ") FAILED: got " + value);
1440:             return false;
1441:         }
1442: 
1443:         private void setUsbConfig(String config) {
1444:             if (DEBUG) Slog.d(TAG, "setUsbConfig(" + config + ")");
1445:             /**
1446:              * set the new configuration
1447:              * we always set it due to b/23631400, where adbd was getting killed
1448:              * and not restarted due to property timeouts on some devices
1449:              */
1450:             setSystemProperty(USB_CONFIG_PROPERTY, config);
1451:         }
1452: 
1453:         @Override
1454:         protected void setEnabledFunctions(long usbFunctions, boolean forceRestart) {
1455:             boolean usbDataUnlocked = isUsbDataTransferActive(usbFunctions);
1456:             if (DEBUG) {
1457:                 Slog.d(TAG, "setEnabledFunctions functions=" + usbFunctions + ", "
1458:                         + "forceRestart=" + forceRestart + ", usbDataUnlocked=" + usbDataUnlocked);
1459:             }
1460: 
1461:             if (usbDataUnlocked != mUsbDataUnlocked) {
1462:                 mUsbDataUnlocked = usbDataUnlocked;
1463:                 updateUsbNotification(false);
1464:                 forceRestart = true;
1465:             }
1466: 
1467:             /**
1468:              * Try to set the enabled functions.
1469:              */
1470:             final long oldFunctions = mCurrentFunctions;
1471:             final boolean oldFunctionsApplied = mCurrentFunctionsApplied;
1472:             if (trySetEnabledFunctions(usbFunctions, forceRestart)) {
1473:                 return;
1474:             }
1475: 
1476:             /**
1477:              * Didn't work.  Try to revert changes.
1478:              * We always reapply the policy in case certain constraints changed such as
1479:              * user restrictions independently of any other new functions we were
1480:              * trying to activate.
1481:              */
1482:             if (oldFunctionsApplied && oldFunctions != usbFunctions) {
1483:                 Slog.e(TAG, "Failsafe 1: Restoring previous USB functions.");
1484:                 if (trySetEnabledFunctions(oldFunctions, false)) {
1485:                     return;
1486:                 }
1487:             }
1488: 
1489:             /**
1490:              * Still didn't work.  Try to restore the default functions.
1491:              */
1492:             Slog.e(TAG, "Failsafe 2: Restoring default USB functions.");
1493:             if (trySetEnabledFunctions(UsbManager.FUNCTION_NONE, false)) {
1494:                 return;
1495:             }
1496: 
1497:             /**
1498:              * Now we're desperate.  Ignore the default functions.
1499:              * Try to get ADB working if enabled.
1500:              */
1501:             Slog.e(TAG, "Failsafe 3: Restoring empty function list (with ADB if enabled).");
1502:             if (trySetEnabledFunctions(UsbManager.FUNCTION_NONE, false)) {
1503:                 return;
1504:             }
1505: 
1506:             /**
1507:              * Ouch.
1508:              */
1509:             Slog.e(TAG, "Unable to set any USB functions!");
1510:         }
1511: 
1512:         private boolean isNormalBoot() {
1513:             String bootMode = getSystemProperty(BOOT_MODE_PROPERTY, "unknown");
1514:             return bootMode.equals(NORMAL_BOOT) || bootMode.equals("unknown");
1515:         }
1516: 
1517:         protected String applyAdbFunction(String functions) {
1518:             // Do not pass null pointer to the UsbManager.
1519:             // There isn't a check there.
1520:             if (functions == null) {
1521:                 functions = "";
1522:             }
1523:             if (isAdbEnabled()) {
1524:                 functions = addFunction(functions, UsbManager.USB_FUNCTION_ADB);
1525:             } else {
1526:                 functions = removeFunction(functions, UsbManager.USB_FUNCTION_ADB);
1527:             }
1528:             return functions;
1529:         }
1530: 
1531:         private boolean trySetEnabledFunctions(long usbFunctions, boolean forceRestart) {
1532:             String functions = null;
1533:             if (usbFunctions != UsbManager.FUNCTION_NONE) {
1534:                 functions = UsbManager.usbFunctionsToString(usbFunctions);
1535:             }
1536:             mCurrentFunctions = usbFunctions;
1537:             if (functions == null || applyAdbFunction(functions)
1538:                     .equals(UsbManager.USB_FUNCTION_NONE)) {
1539:                 functions = UsbManager.usbFunctionsToString(getChargingFunctions());
1540:             }
1541:             functions = applyAdbFunction(functions);
1542: 
1543:             String oemFunctions = applyOemOverrideFunction(functions);
1544: 
1545:             if (!isNormalBoot() && !mCurrentFunctionsStr.equals(functions)) {
1546:                 setSystemProperty(getPersistProp(true), functions);
1547:             }
1548: 
1549:             if ((!functions.equals(oemFunctions)
1550:                     && !mCurrentOemFunctions.equals(oemFunctions))
1551:                     || !mCurrentFunctionsStr.equals(functions)
1552:                     || !mCurrentFunctionsApplied
1553:                     || forceRestart) {
1554:                 Slog.i(TAG, "Setting USB config to " + functions);
1555:                 mCurrentFunctionsStr = functions;
1556:                 mCurrentOemFunctions = oemFunctions;
1557:                 mCurrentFunctionsApplied = false;
1558: 
1559:                 /**
1560:                  * Kick the USB stack to close existing connections.
1561:                  */
1562:                 setUsbConfig(UsbManager.USB_FUNCTION_NONE);
1563: 
1564:                 if (!waitForState(UsbManager.USB_FUNCTION_NONE)) {
1565:                     Slog.e(TAG, "Failed to kick USB config");
1566:                     return false;
1567:                 }
1568: 
1569:                 /**
1570:                  * Set the new USB configuration.
1571:                  */
1572:                 setUsbConfig(oemFunctions);
1573: 
1574:                 if (mBootCompleted
1575:                         && (containsFunction(functions, UsbManager.USB_FUNCTION_MTP)
1576:                         || containsFunction(functions, UsbManager.USB_FUNCTION_PTP))) {
1577:                     /**
1578:                      * Start up dependent services.
1579:                      */
1580:                     updateUsbStateBroadcastIfNeeded(getAppliedFunctions(mCurrentFunctions));
1581:                 }
1582: 
1583:                 if (!waitForState(oemFunctions)) {
1584:                     Slog.e(TAG, "Failed to switch USB config to " + functions);
1585:                     return false;
1586:                 }
1587: 
1588:                 mCurrentFunctionsApplied = true;
1589:             }
1590:             return true;
1591:         }
1592: 
1593:         private String getPersistProp(boolean functions) {
1594:             String bootMode = getSystemProperty(BOOT_MODE_PROPERTY, "unknown");
1595:             String persistProp = USB_PERSISTENT_CONFIG_PROPERTY;
1596:             if (!(bootMode.equals(NORMAL_BOOT) || bootMode.equals("unknown"))) {
1597:                 if (functions) {
1598:                     persistProp = "persist.sys.usb." + bootMode + ".func";
1599:                 } else {
1600:                     persistProp = "persist.sys.usb." + bootMode + ".config";
1601:                 }
1602:             }
1603:             return persistProp;
1604:         }
1605: 
1606:         private static String addFunction(String functions, String function) {
1607:             if (UsbManager.USB_FUNCTION_NONE.equals(functions)) {
1608:                 return function;
1609:             }
1610:             if (!containsFunction(functions, function)) {
1611:                 if (functions.length() > 0) {
1612:                     functions += ",";
1613:                 }
1614:                 functions += function;
1615:             }
1616:             return functions;
1617:         }
1618: 
1619:         private static String removeFunction(String functions, String function) {
1620:             String[] split = functions.split(",");
1621:             for (int i = 0; i < split.length; i++) {
1622:                 if (function.equals(split[i])) {
1623:                     split[i] = null;
1624:                 }
1625:             }
1626:             if (split.length == 1 && split[0] == null) {
1627:                 return UsbManager.USB_FUNCTION_NONE;
1628:             }
1629:             StringBuilder builder = new StringBuilder();
1630:             for (int i = 0; i < split.length; i++) {
1631:                 String s = split[i];
1632:                 if (s != null) {
1633:                     if (builder.length() > 0) {
1634:                         builder.append(",");
1635:                     }
1636:                     builder.append(s);
1637:                 }
1638:             }
1639:             return builder.toString();
1640:         }
1641: 
1642:         static boolean containsFunction(String functions, String function) {
1643:             int index = functions.indexOf(function);
1644:             if (index < 0) return false;
1645:             if (index > 0 && functions.charAt(index - 1) != ',') return false;
1646:             int charAfter = index + function.length();
1647:             if (charAfter < functions.length() && functions.charAt(charAfter) != ',') return false;
1648:             return true;
1649:         }
1650:     }
1651: 
1652:     private static final class UsbHandlerHal extends UsbHandler {
1653: 
1654:         /**
1655:          * Proxy object for the usb gadget hal daemon.
1656:          */
1657:         @GuardedBy("mGadgetProxyLock")
1658:         private IUsbGadget mGadgetProxy;
1659: 
1660:         private final Object mGadgetProxyLock = new Object();
1661: 
1662:         /**
1663:          * Cookie sent for usb gadget hal death notification.
1664:          */
1665:         private static final int USB_GADGET_HAL_DEATH_COOKIE = 2000;
1666: 
1667:         /**
1668:          * Keeps track of the latest setCurrentUsbFunctions request number.
1669:          */
1670:         private int mCurrentRequest = 0;
1671: 
1672:         /**
1673:          * The maximum time for which the UsbDeviceManager would wait once
1674:          * setCurrentUsbFunctions is called.
1675:          */
1676:         private static final int SET_FUNCTIONS_TIMEOUT_MS = 3000;
1677: 
1678:         /**
1679:          * Conseration leeway to make sure that the hal callback arrives before
1680:          * SET_FUNCTIONS_TIMEOUT_MS expires. If the callback does not arrive
1681:          * within SET_FUNCTIONS_TIMEOUT_MS, UsbDeviceManager retries enabling
1682:          * default functions.
1683:          */
1684:         private static final int SET_FUNCTIONS_LEEWAY_MS = 500;
1685: 
1686:         /**
1687:          * While switching functions, a disconnect is excpect as the usb gadget
1688:          * us torn down and brought back up. Wait for SET_FUNCTIONS_TIMEOUT_MS +
1689:          * ENUMERATION_TIME_OUT_MS before switching back to default fumctions when
1690:          * switching functions.
1691:          */
1692:         private static final int ENUMERATION_TIME_OUT_MS = 2000;
1693: 
1694:         /**
1695:          * Gadget HAL fully qualified instance name for registering for ServiceNotification.
1696:          */
1697:         protected static final String GADGET_HAL_FQ_NAME =
1698:                 "android.hardware.usb.gadget@1.0::IUsbGadget";
1699: 
1700:         protected boolean mCurrentUsbFunctionsRequested;
1701: 
1702:         UsbHandlerHal(Looper looper, Context context, UsbDeviceManager deviceManager,
1703:                 UsbAlsaManager alsaManager, UsbPermissionManager permissionManager) {
1704:             super(looper, context, deviceManager, alsaManager, permissionManager);
1705:             try {
1706:                 ServiceNotification serviceNotification = new ServiceNotification();
1707: 
1708:                 boolean ret = IServiceManager.getService()
1709:                         .registerForNotifications(GADGET_HAL_FQ_NAME, "", serviceNotification);
1710:                 if (!ret) {
1711:                     Slog.e(TAG, "Failed to register usb gadget service start notification");
1712:                     return;
1713:                 }
1714: 
1715:                 synchronized (mGadgetProxyLock) {
1716:                     mGadgetProxy = IUsbGadget.getService(true);
1717:                     mGadgetProxy.linkToDeath(new UsbGadgetDeathRecipient(),
1718:                             USB_GADGET_HAL_DEATH_COOKIE);
1719:                     mCurrentFunctions = UsbManager.FUNCTION_NONE;
1720:                     mCurrentUsbFunctionsRequested = true;
1721:                     mGadgetProxy.getCurrentUsbFunctions(new UsbGadgetCallback());
1722:                 }
1723:                 String state = FileUtils.readTextFile(new File(STATE_PATH), 0, null).trim();
1724:                 updateState(state);
1725:             } catch (NoSuchElementException e) {
1726:                 Slog.e(TAG, "Usb gadget hal not found", e);
1727:             } catch (RemoteException e) {
1728:                 Slog.e(TAG, "Usb Gadget hal not responding", e);
1729:             } catch (Exception e) {
1730:                 Slog.e(TAG, "Error initializing UsbHandler", e);
1731:             }
1732:         }
1733: 
1734: 
1735:         final class UsbGadgetDeathRecipient implements HwBinder.DeathRecipient {
1736:             @Override
1737:             public void serviceDied(long cookie) {
1738:                 if (cookie == USB_GADGET_HAL_DEATH_COOKIE) {
1739:                     Slog.e(TAG, "Usb Gadget hal service died cookie: " + cookie);
1740:                     synchronized (mGadgetProxyLock) {
1741:                         mGadgetProxy = null;
1742:                     }
1743:                 }
1744:             }
1745:         }
1746: 
1747:         final class ServiceNotification extends IServiceNotification.Stub {
1748:             @Override
1749:             public void onRegistration(String fqName, String name, boolean preexisting) {
1750:                 Slog.i(TAG, "Usb gadget hal service started " + fqName + " " + name);
1751:                 if (!fqName.equals(GADGET_HAL_FQ_NAME)) {
1752:                     Slog.e(TAG, "fqName does not match");
1753:                     return;
1754:                 }
1755: 
1756:                 sendMessage(MSG_GADGET_HAL_REGISTERED, preexisting);
1757:             }
1758:         }
1759: 
1760:         @Override
1761:         public void handleMessage(Message msg) {
1762:             switch (msg.what) {
1763:                 case MSG_SET_CHARGING_FUNCTIONS:
1764:                     setEnabledFunctions(UsbManager.FUNCTION_NONE, false);
1765:                     break;
1766:                 case MSG_SET_FUNCTIONS_TIMEOUT:
1767:                     Slog.e(TAG, "Set functions timed out! no reply from usb hal");
1768:                     if (msg.arg1 != 1) {
1769:                         // Set this since default function may be selected from Developer options
1770:                         setEnabledFunctions(mScreenUnlockedFunctions, false);
1771:                     }
1772:                     break;
1773:                 case MSG_GET_CURRENT_USB_FUNCTIONS:
1774:                     Slog.e(TAG, "prcessing MSG_GET_CURRENT_USB_FUNCTIONS");
1775:                     mCurrentUsbFunctionsReceived = true;
1776: 
1777:                     if (mCurrentUsbFunctionsRequested) {
1778:                         Slog.e(TAG, "updating mCurrentFunctions");
1779:                         // Mask out adb, since it is stored in mAdbEnabled
1780:                         mCurrentFunctions = ((Long) msg.obj) & ~UsbManager.FUNCTION_ADB;
1781:                         Slog.e(TAG,
1782:                                 "mCurrentFunctions:" + mCurrentFunctions + "applied:" + msg.arg1);
1783:                         mCurrentFunctionsApplied = msg.arg1 == 1;
1784:                     }
1785:                     finishBoot();
1786:                     break;
1787:                 case MSG_FUNCTION_SWITCH_TIMEOUT:
1788:                     /**
1789:                      * Dont force to default when the configuration is already set to default.
1790:                      */
1791:                     if (msg.arg1 != 1) {
1792:                         // Set this since default function may be selected from Developer options
1793:                         setEnabledFunctions(mScreenUnlockedFunctions, false);
1794:                     }
1795:                     break;
1796:                 case MSG_GADGET_HAL_REGISTERED:
1797:                     boolean preexisting = msg.arg1 == 1;
1798:                     synchronized (mGadgetProxyLock) {
1799:                         try {
1800:                             mGadgetProxy = IUsbGadget.getService();
1801:                             mGadgetProxy.linkToDeath(new UsbGadgetDeathRecipient(),
1802:                                     USB_GADGET_HAL_DEATH_COOKIE);
1803:                             if (!mCurrentFunctionsApplied && !preexisting) {
1804:                                 setEnabledFunctions(mCurrentFunctions, false);
1805:                             }
1806:                         } catch (NoSuchElementException e) {
1807:                             Slog.e(TAG, "Usb gadget hal not found", e);
1808:                         } catch (RemoteException e) {
1809:                             Slog.e(TAG, "Usb Gadget hal not responding", e);
1810:                         }
1811:                     }
1812:                     break;
1813:                 case MSG_RESET_USB_GADGET:
1814:                     synchronized (mGadgetProxyLock) {
1815:                         if (mGadgetProxy == null) {
1816:                             Slog.e(TAG, "reset Usb Gadget mGadgetProxy is null");
1817:                             break;
1818:                         }
1819: 
1820:                         try {
1821:                             android.hardware.usb.gadget.V1_1.IUsbGadget gadgetProxy =
1822:                                     android.hardware.usb.gadget.V1_1.IUsbGadget
1823:                                             .castFrom(mGadgetProxy);
1824:                             gadgetProxy.reset();
1825:                         } catch (RemoteException e) {
1826:                             Slog.e(TAG, "reset Usb Gadget failed", e);
1827:                         }
1828:                     }
1829:                     break;
1830:                 default:
1831:                     super.handleMessage(msg);
1832:             }
1833:         }
1834: 
1835:         private class UsbGadgetCallback extends IUsbGadgetCallback.Stub {
1836:             int mRequest;
1837:             long mFunctions;
1838:             boolean mChargingFunctions;
1839: 
1840:             UsbGadgetCallback() {
1841:             }
1842: 
1843:             UsbGadgetCallback(int request, long functions,
1844:                     boolean chargingFunctions) {
1845:                 mRequest = request;
1846:                 mFunctions = functions;
1847:                 mChargingFunctions = chargingFunctions;
1848:             }
1849: 
1850:             @Override
1851:             public void setCurrentUsbFunctionsCb(long functions,
1852:                     int status) {
1853:                 /**
1854:                  * Callback called for a previous setCurrenUsbFunction
1855:                  */
1856:                 if ((mCurrentRequest != mRequest) || !hasMessages(MSG_SET_FUNCTIONS_TIMEOUT)
1857:                         || (mFunctions != functions)) {
1858:                     return;
1859:                 }
1860: 
1861:                 removeMessages(MSG_SET_FUNCTIONS_TIMEOUT);
1862:                 Slog.e(TAG, "notifyCurrentFunction request:" + mRequest + " status:" + status);
1863:                 if (status == Status.SUCCESS) {
1864:                     mCurrentFunctionsApplied = true;
1865:                 } else if (!mChargingFunctions) {
1866:                     Slog.e(TAG, "Setting default fuctions");
1867:                     sendEmptyMessage(MSG_SET_CHARGING_FUNCTIONS);
1868:                 }
1869:             }
1870: 
1871:             @Override
1872:             public void getCurrentUsbFunctionsCb(long functions,
1873:                     int status) {
1874:                 sendMessage(MSG_GET_CURRENT_USB_FUNCTIONS, functions,
1875:                         status == Status.FUNCTIONS_APPLIED);
1876:             }
1877:         }
1878: 
1879:         private void setUsbConfig(long config, boolean chargingFunctions) {
1880:             if (true) Slog.d(TAG, "setUsbConfig(" + config + ") request:" + ++mCurrentRequest);
1881:             /**
1882:              * Cancel any ongoing requests, if present.
1883:              */
1884:             removeMessages(MSG_FUNCTION_SWITCH_TIMEOUT);
1885:             removeMessages(MSG_SET_FUNCTIONS_TIMEOUT);
1886:             removeMessages(MSG_SET_CHARGING_FUNCTIONS);
1887: 
1888:             synchronized (mGadgetProxyLock) {
1889:                 if (mGadgetProxy == null) {
1890:                     Slog.e(TAG, "setUsbConfig mGadgetProxy is null");
1891:                     return;
1892:                 }
1893:                 try {
1894:                     if ((config & UsbManager.FUNCTION_ADB) != 0) {
1895:                         /**
1896:                          * Start adbd if ADB function is included in the configuration.
1897:                          */
1898:                         LocalServices.getService(AdbManagerInternal.class)
1899:                                 .startAdbdForTransport(AdbTransportType.USB);
1900:                     } else {
1901:                         /**
1902:                          * Stop adbd otherwise
1903:                          */
1904:                         LocalServices.getService(AdbManagerInternal.class)
1905:                                 .stopAdbdForTransport(AdbTransportType.USB);
1906:                     }
1907:                     UsbGadgetCallback usbGadgetCallback = new UsbGadgetCallback(mCurrentRequest,
1908:                             config, chargingFunctions);
1909:                     mGadgetProxy.setCurrentUsbFunctions(config, usbGadgetCallback,
1910:                             SET_FUNCTIONS_TIMEOUT_MS - SET_FUNCTIONS_LEEWAY_MS);
1911:                     sendMessageDelayed(MSG_SET_FUNCTIONS_TIMEOUT, chargingFunctions,
1912:                             SET_FUNCTIONS_TIMEOUT_MS);
1913:                     if (mConnected) {
1914:                         // Only queue timeout of enumeration when the USB is connected
1915:                         sendMessageDelayed(MSG_FUNCTION_SWITCH_TIMEOUT, chargingFunctions,
1916:                                 SET_FUNCTIONS_TIMEOUT_MS + ENUMERATION_TIME_OUT_MS);
1917:                     }
1918:                     if (DEBUG) Slog.d(TAG, "timeout message queued");
1919:                 } catch (RemoteException e) {
1920:                     Slog.e(TAG, "Remoteexception while calling setCurrentUsbFunctions", e);
1921:                 }
1922:             }
1923:         }
1924: 
1925:         @Override
1926:         protected void setEnabledFunctions(long functions, boolean forceRestart) {
1927:             if (DEBUG) {
1928:                 Slog.d(TAG, "setEnabledFunctions functions=" + functions + ", "
1929:                         + "forceRestart=" + forceRestart);
1930:             }
1931:             if (mCurrentFunctions != functions
1932:                     || !mCurrentFunctionsApplied
1933:                     || forceRestart) {
1934:                 Slog.i(TAG, "Setting USB config to " + UsbManager.usbFunctionsToString(functions));
1935:                 mCurrentFunctions = functions;
1936:                 mCurrentFunctionsApplied = false;
1937:                 // set the flag to false as that would be stale value
1938:                 mCurrentUsbFunctionsRequested = false;
1939: 
1940:                 boolean chargingFunctions = functions == UsbManager.FUNCTION_NONE;
1941:                 functions = getAppliedFunctions(functions);
1942: 
1943:                 // Set the new USB configuration.
1944:                 setUsbConfig(functions, chargingFunctions);
1945: 
1946:                 if (mBootCompleted && isUsbDataTransferActive(functions)) {
1947:                     // Start up dependent services.
1948:                     updateUsbStateBroadcastIfNeeded(functions);
1949:                 }
1950:             }
1951:         }
1952:     }
1953: 
1954:     /* returns the currently attached USB accessory */
1955:     public UsbAccessory getCurrentAccessory() {
1956:         return mHandler.getCurrentAccessory();
1957:     }
1958: 
1959:     /**
1960:      * opens the currently attached USB accessory.
1961:      *
1962:      * @param accessory accessory to be openened.
1963:      * @param uid Uid of the caller
1964:      */
1965:     public ParcelFileDescriptor openAccessory(UsbAccessory accessory,
1966:             UsbUserPermissionManager permissions, int uid) {
1967:         UsbAccessory currentAccessory = mHandler.getCurrentAccessory();
1968:         if (currentAccessory == null) {
1969:             throw new IllegalArgumentException("no accessory attached");
1970:         }
1971:         if (!currentAccessory.equals(accessory)) {
1972:             String error = accessory.toString()
1973:                     + " does not match current accessory "
1974:                     + currentAccessory;
1975:             throw new IllegalArgumentException(error);
1976:         }
1977:         permissions.checkPermission(accessory, uid);
1978:         return nativeOpenAccessory();
1979:     }
1980: 
1981:     public long getCurrentFunctions() {
1982:         return mHandler.getEnabledFunctions();
1983:     }
1984: 
1985:     /**
1986:      * Returns a dup of the control file descriptor for the given function.
1987:      */
1988:     public ParcelFileDescriptor getControlFd(long usbFunction) {
1989:         FileDescriptor fd = mControlFds.get(usbFunction);
1990:         if (fd == null) {
1991:             return null;
1992:         }
1993:         try {
1994:             return ParcelFileDescriptor.dup(fd);
1995:         } catch (IOException e) {
1996:             Slog.e(TAG, "Could not dup fd for " + usbFunction);
1997:             return null;
1998:         }
1999:     }
2000: 
2001:     public long getScreenUnlockedFunctions() {
2002:         return mHandler.getScreenUnlockedFunctions();
2003:     }
2004: 
2005:     /**
2006:      * Adds function to the current USB configuration.
2007:      *
2008:      * @param functions The functions to set, or empty to set the charging function.
2009:      */
2010:     public void setCurrentFunctions(long functions) {
2011:         if (DEBUG) {
2012:             Slog.d(TAG, "setCurrentFunctions(" + UsbManager.usbFunctionsToString(functions) + ")");
2013:         }
2014:         if (functions == UsbManager.FUNCTION_NONE) {
2015:             MetricsLogger.action(mContext, MetricsEvent.ACTION_USB_CONFIG_CHARGING);
2016:         } else if (functions == UsbManager.FUNCTION_MTP) {
2017:             MetricsLogger.action(mContext, MetricsEvent.ACTION_USB_CONFIG_MTP);
2018:         } else if (functions == UsbManager.FUNCTION_PTP) {
2019:             MetricsLogger.action(mContext, MetricsEvent.ACTION_USB_CONFIG_PTP);
2020:         } else if (functions == UsbManager.FUNCTION_MIDI) {
2021:             MetricsLogger.action(mContext, MetricsEvent.ACTION_USB_CONFIG_MIDI);
2022:         } else if (functions == UsbManager.FUNCTION_RNDIS) {
2023:             MetricsLogger.action(mContext, MetricsEvent.ACTION_USB_CONFIG_RNDIS);
2024:         } else if (functions == UsbManager.FUNCTION_ACCESSORY) {
2025:             MetricsLogger.action(mContext, MetricsEvent.ACTION_USB_CONFIG_ACCESSORY);
2026:         }
2027:         mHandler.sendMessage(MSG_SET_CURRENT_FUNCTIONS, functions);
2028:     }
2029: 
2030:     /**
2031:      * Sets the functions which are set when the screen is unlocked.
2032:      *
2033:      * @param functions Functions to set.
2034:      */
2035:     public void setScreenUnlockedFunctions(long functions) {
2036:         if (DEBUG) {
2037:             Slog.d(TAG, "setScreenUnlockedFunctions("
2038:                     + UsbManager.usbFunctionsToString(functions) + ")");
2039:         }
2040:         mHandler.sendMessage(MSG_SET_SCREEN_UNLOCKED_FUNCTIONS, functions);
2041:     }
2042: 
2043:     /**
2044:      * Resets the USB Gadget.
2045:      */
2046:     public void resetUsbGadget() {
2047:         if (DEBUG) {
2048:             Slog.d(TAG, "reset Usb Gadget");
2049:         }
2050: 
2051:         mHandler.sendMessage(MSG_RESET_USB_GADGET, null);
2052:     }
2053: 
2054:     private void onAdbEnabled(boolean enabled) {
2055:         mHandler.sendMessage(MSG_ENABLE_ADB, enabled);
2056:     }
2057: 
2058:     /**
2059:      * Write the state to a dump stream.
2060:      */
2061:     public void dump(DualDumpOutputStream dump, String idName, long id) {
2062:         long token = dump.start(idName, id);
2063: 
2064:         if (mHandler != null) {
2065:             mHandler.dump(dump, "handler", UsbDeviceManagerProto.HANDLER);
2066:         }
2067: 
2068:         dump.end(token);
2069:     }
2070: 
2071:     private native String[] nativeGetAccessoryStrings();
2072: 
2073:     private native ParcelFileDescriptor nativeOpenAccessory();
2074: 
2075:     private native FileDescriptor nativeOpenControl(String usbFunction);
2076: 
2077:     private native boolean nativeIsStartRequested();
2078: 
2079:     private native int nativeGetAudioMode();
2080: }
```

**.rej File Content:**
```diff
--- services/usb/java/com/android/server/usb/UsbDeviceManager.java
+++ services/usb/java/com/android/server/usb/UsbDeviceManager.java
@@ -80,9 +80,9 @@ import android.os.storage.StorageVolume;
 import android.provider.Settings;
 import android.service.usb.UsbDeviceManagerProto;
 import android.service.usb.UsbHandlerProto;
+import android.text.TextUtils;
 import android.util.Pair;
 import android.util.Slog;
-import android.text.TextUtils;
 
 import com.android.internal.R;
 import com.android.internal.annotations.GuardedBy;
@@ -880,7 +880,7 @@ public class UsbDeviceManager implements ActivityTaskManagerInternal.ScreenObser
             }
         }
 
-        private void notifyAccessoryModeExit(int operationId) {
+        protected void notifyAccessoryModeExit(int operationId) {
             // make sure accessory mode is off
             // and restore default functions
             Slog.d(TAG, "exited USB accessory mode");
@@ -2313,8 +2313,13 @@ public class UsbDeviceManager implements ActivityTaskManagerInternal.ScreenObser
                      */
                     operationId = sUsbOperationCount.incrementAndGet();
                     if (msg.arg1 != 1) {
-                        // Set this since default function may be selected from Developer options
-                        setEnabledFunctions(mScreenUnlockedFunctions, false, operationId);
+                        if (mCurrentFunctions == UsbManager.FUNCTION_ACCESSORY) {
+                            notifyAccessoryModeExit(operationId);
+                        } else {
+                            // Set this since default function may be selected from Developer
+                            // options
+                            setEnabledFunctions(mScreenUnlockedFunctions, false, operationId);
+                        }
                     }
                     break;
                 case MSG_GADGET_HAL_REGISTERED:
```

**Expected Output Diff:**
```diff
--- a/services/usb/java/com/android/server/usb/UsbDeviceManager.java
+++ b/services/usb/java/com/android/server/usb/UsbDeviceManager.java
@@ -719,7 +719,7 @@ public class UsbDeviceManager implements ActivityTaskManagerInternal.ScreenObser
             }
         }
 
-        private void notifyAccessoryModeExit() {
+        protected void notifyAccessoryModeExit() {
             // make sure accessory mode is off
             // and restore default functions
             Slog.d(TAG, "exited USB accessory mode");
@@ -1957,8 +1957,13 @@ public class UsbDeviceManager implements ActivityTaskManagerInternal.ScreenObser
                      * Dont force to default when the configuration is already set to default.
                      */
                     if (msg.arg1 != 1) {
-                        // Set this since default function may be selected from Developer options
-                        setEnabledFunctions(mScreenUnlockedFunctions, false);
+                        if (mCurrentFunctions == UsbManager.FUNCTION_ACCESSORY) {
+                            notifyAccessoryModeExit();
+                        } else {
+                            // Set this since default function may be selected from Developer
+                            // options
+                            setEnabledFunctions(mScreenUnlockedFunctions, false);
+                        }
                     }
                     break;
                 case MSG_GADGET_HAL_REGISTERED:
```
'''

    patch_porter_agent = GeminiAgent(
        model_name="gemini-2.5-pro-preview-05-06",
        system_prompt=system_prompt,
        key_rotator=key_rotator,
        temperature=args.temperature
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
        "temperature_setting": args.temperature,
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
            "total_runtime_seconds_successful": 0,
            "total_gemini_input_tokens": 0,
            "total_gemini_output_tokens": 0,
            "total_gemini_tokens": 0
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

                    entry_token_counts = generated_diff.get("token_counts", {}).get("gemini", {})
                    if entry_token_counts:
                        report_data["summary"]["total_gemini_input_tokens"] += entry_token_counts.get("input", 0)
                        report_data["summary"]["total_gemini_output_tokens"] += entry_token_counts.get("output", 0)
                        report_data["summary"]["total_gemini_tokens"] += entry_token_counts.get("total", 0)

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
                                "diff_preview": generated_diff["downstream_llm_diff_output"][:100] + "..." if generated_diff.get("downstream_llm_diff_output") else "None",
                                "token_counts": generated_diff.get("token_counts", {}).get("gemini")
                            })
                        else:
                            report_data["summary"]["files_with_llm_diff_generation_errors_or_skipped_in_func"] += 1
                            report_data["skipped_or_errored_diff_generation_log"].append({
                                "vulnerability_id": vulnerability_id,
                                "file_name": target_filename,
                                "patch_sha": failure.get('downstream_patch', 'N/A'),
                                "reason": generated_diff.get("error", "Validation failed"),
                                "last_format_error": generated_diff.get("last_format_error"),
                                "last_apply_error": generated_diff.get("last_apply_error"),
                                "token_counts": generated_diff.get("token_counts", {}).get("gemini")
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

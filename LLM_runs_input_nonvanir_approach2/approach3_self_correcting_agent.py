import argparse
import asyncio
import copy
import json
import os
import random
import re
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from android_patch_manager import AndroidPatchManager
from unidiff import PatchSet
from dotenv import load_dotenv

import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory

# Load environment variables from .env file
load_dotenv()

# Load and validate API keys from environment
api_keys_str = os.getenv("GOOGLE_API_KEYS")
if not api_keys_str:
    raise ValueError("The GOOGLE_API_KEYS environment variable is not set. Please set it in your environment or in a .env file.")

api_keys = [key.strip() for key in api_keys_str.split(',') if key.strip()]
if not api_keys:
    raise ValueError("No valid API keys found in the GOOGLE_API_KEYS environment variable.")

class APIKeyRotator:
    """Rotates through a list of API keys."""

    def __init__(self, api_keys: List[str]):
        if not api_keys:
            raise ValueError("API keys list cannot be empty.")
        self.api_keys = api_keys
        self.index = 0

    def get_current_key(self):
        """Returns the current API key."""
        return self.api_keys[self.index]

    def rotate_key(self):
        """Rotates to the next API key."""
        self.index = (self.index + 1) % len(self.api_keys)
        print(f"🔄 Rotating to new API key index {self.index}")
        return self.get_current_key()

key_rotator = APIKeyRotator(api_keys)

def add_line_numbers(content: str) -> str:
    """Adds line numbers to each line of a string."""
    lines = content.splitlines()
    return "\n".join(f"{i+1}: {line}" for i, line in enumerate(lines))

def get_all_token_counts(prompt: str, gemini_token_counts: Dict[str, int]) -> Dict[str, int]:
    """Provides a comprehensive token count including the prompt."""
    return {
        "prompt_tokens": len(prompt.split()),  # A rough estimate
        "gemini_input_tokens": gemini_token_counts.get("input", 0),
        "gemini_output_tokens": gemini_token_counts.get("output", 0),
        "gemini_total_tokens": gemini_token_counts.get("total", 0)
    }

def save_partial_output(path: str, data: Any):
    """Saves data to a JSON file, useful for iterative saving."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"💾 Partial output saved to: {path}")
    except Exception as e:
        print(f"⚠️ Failed to save partial output to {path}: {e}")

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

class GeminiAgent:
    """A class to interact with the Gemini API, with support for key rotation on quota errors."""

    def __init__(self, key_rotator: APIKeyRotator, model_name: str, system_prompt: str, temperature: float = 0.0):
        self.key_rotator = key_rotator
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self._configure_genai()

    def _configure_genai(self):
        api_key = self.key_rotator.get_current_key()
        genai.configure(api_key=api_key)
        
        self.generation_config = GenerationConfig(temperature=self.temperature)
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        self.genai_model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
            system_instruction=self.system_prompt
        )

    async def run(self, prompt: str) -> Dict[str, Any]:
        """Sends a prompt to the Gemini model, with retries and key rotation for API errors."""
        for _ in range(len(self.key_rotator.api_keys)):
            try:
                response = await self.genai_model.generate_content_async(prompt)
                return {
                    "data": response.text,
                    "token_counts": {
                        "input": response.usage_metadata.prompt_token_count,
                        "output": response.usage_metadata.candidates_token_count,
                        "total": response.usage_metadata.total_token_count,
                    }
                }
            except Exception as e:
                error_message = str(e).lower()
                if "quota" in error_message or "rate limit" in error_message or "internal error" in error_message:
                    print(f"⚠️ API error encountered: {e}")
                    self.key_rotator.rotate_key()
                    self._configure_genai()  # Reconfigure with the new key
                else:
                    # For other errors, we don't retry, just raise
                    raise e
        raise RuntimeError("All API keys failed. No more keys to try.")

class SelfCorrectingAgent:
    """
    An agent that analyzes its own failures and refines its prompts to achieve a correct result.
    It encapsulates the logic for a single, complex, multi-step generation task.
    """

    def __init__(self, gemini_agent: GeminiAgent, original_inputs: dict, max_retries: int = 3, repo_lock: asyncio.Lock = None):
        self.gemini_agent = gemini_agent
        self.original_inputs = original_inputs
        self.max_retries = max_retries
        self.attempts = []
        self.repo_lock = repo_lock

    def _create_initial_prompt(self) -> str:
        # This is the original `base_task_prompt`
        return f"""Resolve the conflicts in the provided `.rej` file by generating a corrected unified diff.
The generated diff must apply cleanly to the 'Original Source File' and strictly follow all formatting and hunk construction rules from your system guidelines.

**IMPORTANT CONTEXT**: The "Original Source File Content" you see is from a repository state where some hunks of a patch may have already been successfully applied. The ".rej File Content" contains the hunks that FAILED. Your task is to generate a NEW patch containing ONLY the corrected versions of the REJECTED hunks. Do NOT include changes that were already successfully applied.

**Inputs:**

1.  **Original Source File Content**: The full content of the file to be patched, with line numbers added for precision. This is the state of the file AFTER a partial patch attempt.
    ```text
    {add_line_numbers(self.original_inputs['original_source_content'])}
    ```

2.  **.rej File Content**: The rejected hunks from a patch. You MUST adapt and correct these changes. The `@@ ... @@` headers in this content are your primary guide for the changes and line counts.
    ```text
    {self.original_inputs['rej_content']}
    ```

3.  **Upstream Patch Content**: This is the full original patch that failed to apply. Use this to understand the developer's original intent. The goal is to create a new patch that achieves the same goal but applies cleanly.
    ```diff
    {self.original_inputs['upstream_patch_content']}
    ```

4.  **Target Filename**: Use this exact name for the `--- a/` and `+++ b/` lines of your diff.
    `{self.original_inputs['target_filename']}`

**Your Task:**
Generate a corrected unified diff that:
1.  Contains ONLY the fixed versions of the hunks from the ".rej File Content".
2.  Correctly applies to the "Original Source File Content".
3.  Addresses any line number offsets or context mismatches that caused the original rejection.
4.  Uses the provided line numbers to calculate precise hunk headers (`@@ ... @@`) as explained in your system guidelines.
5.  Strictly adheres to the hunk header line counts (`@@ ... @@`) as explained in your system guidelines.
6.  **IMPORTANT**: The diff content itself must NOT include the line number prefixes. Use line numbers for calculation only, but generate clean diff content.

Generate the corrected unified diff now:
"""

    def _analyze_failure(self, format_error: Optional[str], apply_error: Optional[str]) -> str:
        analysis_parts = ["I have reviewed your last attempt, and it failed. Here is my analysis and prescription for your next attempt:"]

        if format_error and format_error != "Valid patch format":
            analysis_parts.append(f"**1. Formatting Error:** `{format_error}`")
            error_detail = """**Expert Diagnosis:** This is a severe error. It means your output did not conform to the basic unified diff format. The most common mistakes are:
1.  Including explanatory text or comments outside the `diff markdown block.
2.  Forgetting or miswriting the `--- a/` and `+++ b/` headers.
3.  Errors in the `@@ ... @@` hunk headers.
**Prescription:** You MUST re-read the 'STRICT FORMATTING AND OUTPUT REQUIREMENTS' section of your system instructions. Your *entire* response must be a single markdown code block starting with ```diff and containing a valid unified diff. Do not write anything else."""
            analysis_parts.append(error_detail)

        if apply_error and apply_error != "Patch applies cleanly in repo":
            analysis_parts.append(f"**2. Application Error:** `{apply_error}`")
            error_detail = ""
            if "hunk #" in apply_error.lower() or "failed" in apply_error.lower():
                error_detail = """**Expert Diagnosis:** This is the most common failure. It means the context or line numbers for a specific hunk were incorrect. The most likely cause is an error in calculating the `@@ -start,count +start,count @@` values.
**Prescription:**
1.  **Re-read the 'Crucial Details for Hunk Construction' section of your system instructions.** Pay extreme attention to the rules for calculating line counts.
2.  **Double-check your line counting.** Compare the `.rej` file content with the line-numbered 'Original Source File Content' very carefully. Your calculations must be perfect.
3.  **Favor larger hunks.** When in doubt, it is safer to replace an entire function or logical block rather than attempting a small, surgical change. This provides more stable context for the `patch` command."""
            else:
                error_detail = """**Expert Diagnosis:** The patch could not be applied due to reasons other than a specific hunk failure. This may be due to incorrect context lines or subtle errors in the changes.
**Prescription:** Carefully re-examine the original source and the `.rej` file. Ensure the changes make sense and that the context lines around the changes are exactly correct. Verify the filenames in the `--- a/` and `+++ b/` headers."""
            analysis_parts.append(error_detail)

        if len(analysis_parts) == 1:
            return "**Analysis:** An unspecified validation error occurred. Please review your entire output and the system guidelines carefully and try again. Be extremely meticulous."

        return "\n\n".join(analysis_parts)

    def _create_retry_prompt(self) -> str:
        last_attempt = self.attempts[-1]
        analysis = self._analyze_failure(
            last_attempt["validation_result"]["format_error"],
            last_attempt["validation_result"]["apply_error"]
        )

        return f"""You are an expert developer in a self-correction cycle. Your previous attempt to generate a patch was incorrect. I am acting as your mentor and have analyzed the failure.

**Expert Feedback on Your Last Attempt:**
{analysis}

**Your Task (Recap):**
Your goal remains the same: generate a corrected unified diff that resolves the conflicts from the `.rej` file and applies cleanly. You must internalize the feedback above and avoid repeating the same mistakes.

**REMINDER**: You are creating a patch for ONLY the rejected hunks. Do NOT include hunks from the original patch that already applied successfully. The "Reversed patch" error means you are likely including already-applied changes.

**Original Inputs (for your reference):**
1.  **Original Source File Content**:
    ```text
    {add_line_numbers(self.original_inputs['original_source_content'])}
    ```
2.  **.rej File Content**:
    ```text
    {self.original_inputs['rej_content']}
    ```
3.  **Upstream Patch Content**:
    ```diff
    {self.original_inputs['upstream_patch_content']}
    ```
4.  **Target Filename**: `{self.original_inputs['target_filename']}`

**Your Previous (Incorrect) Diff Output:**
```diff
{last_attempt["generated_diff"]}
```

Now, apply the feedback and generate a new, corrected unified diff. Be precise and meticulous.
"""

    async def _validate_applicability_in_repo(self, patch_content: str) -> tuple[bool, str]:
        """
        Helper method to encapsulate the complex in-repo validation logic.
        This method now uses a lock to ensure only one task modifies the repo at a time.
        """
        if not self.repo_lock:
            return False, "Repo lock not provided"

        async with self.repo_lock:
            failure_details = self.original_inputs["failure_details"]
            upstream_patch_content = self.original_inputs["upstream_patch_content"]
            vulnerability_id = self.original_inputs["vulnerability_id"]
            target_filename_for_diff = self.original_inputs["target_filename"]
            
            repo_path_str = failure_details.get("repo_path")
            downstream_version = failure_details.get("downstream_version")
            downstream_patch_sha = failure_details.get("downstream_patch")

            if not all([repo_path_str, downstream_version, downstream_patch_sha, upstream_patch_content]):
                msg = f"⚠️ Missing required repo metadata for validation: {vulnerability_id} - {target_filename_for_diff}"
                print(msg)
                return False, msg

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
            success, _, _, _ = AndroidPatchManager.apply_patch(repo_path, temp_patch_path)

            if not success:
                print(f"⚠️ Partial application of upstream patch (some hunks rejected) for {vulnerability_id} - {target_filename_for_diff}")
            else:
                print(f"✅ Upstream patch applied (fully or partially) for {vulnerability_id} - {target_filename_for_diff}")

            return validate_patch_applicability_in_repo(patch_content, str(repo_path))

    async def run(self):
        total_start_time = time.monotonic()
        total_gemini_token_counts = {"input": 0, "output": 0, "total": 0}

        for attempt_num in range(self.max_retries):
            print(f"🔄 Attempt {attempt_num + 1} of {self.max_retries} for {self.original_inputs['target_filename']}")
            
            prompt = self._create_initial_prompt() if attempt_num == 0 else self._create_retry_prompt()
            
            validation_result = {}
            generated_diff = ""
            
            try:
                start_time = time.monotonic()
                result = await self.gemini_agent.run(prompt)
                end_time = time.monotonic()
                
                generated_diff = result["data"].strip()
                attempt_token_counts = result["token_counts"] or {"input": 0, "output": 0, "total": 0}
                total_gemini_token_counts["input"] += attempt_token_counts.get("input", 0)
                total_gemini_token_counts["output"] += attempt_token_counts.get("output", 0)
                total_gemini_token_counts["total"] += attempt_token_counts.get("total", 0)

                format_valid, format_error = validate_patch_format(generated_diff)
                
                apply_valid, apply_error = False, "Skipped due to format error"
                if format_valid:
                    apply_valid, apply_error = await self._validate_applicability_in_repo(generated_diff)
                
                validation_result = {
                    "attempt": attempt_num + 1,
                    "format_valid": format_valid,
                    "format_error": format_error,
                    "apply_valid": apply_valid,
                    "apply_error": apply_error,
                    "valid": format_valid and apply_valid,
                    "runtime_seconds": round(end_time - start_time, 2),
                    "token_counts": get_all_token_counts(prompt, gemini_token_counts=attempt_token_counts)
                }
                
                print(f"📊 Validation results for attempt {attempt_num + 1}:")
                print(f"  - Format valid: {format_valid} (Error: {format_error if not format_valid else 'N/A'})")
                print(f"  - Can apply: {apply_valid} (Error: {apply_error if not apply_valid else 'N/A'})")

            except Exception as e:
                print(f"❌ Error on attempt {attempt_num + 1} for {self.original_inputs['target_filename']}: {e}")
                validation_result = {
                    "attempt": attempt_num + 1,
                    "error": f"Exception during generation: {str(e)}",
                    "valid": False,
                    "format_valid": False,
                    "format_error": "Skipped due to exception",
                    "apply_valid": False,
                    "apply_error": "Skipped due to exception"
                }

            self.attempts.append({
                "generated_diff": generated_diff,
                "validation_result": validation_result,
                "prompt": prompt
            })

            if validation_result.get("valid"):
                total_end_time = time.monotonic()
                print(f"✅ LLM diff generation and validation successful for: {self.original_inputs['vulnerability_id']} - {self.original_inputs['target_filename']}")
                return {
                    "downstream_llm_diff_output": generated_diff,
                    "llm_output_valid": True,
                    "runtime_seconds": round(total_end_time - total_start_time, 2),
                    "attempts_made": attempt_num + 1,
                    "validation_results": [a["validation_result"] for a in self.attempts],
                    "token_counts": get_all_token_counts(generated_diff, gemini_token_counts=total_gemini_token_counts),
                    "final_prompt_used": prompt
                }
            else:
                 print(f"⚠️ Validation failed for attempt {attempt_num + 1}, retrying...")

        # If loop finishes, all retries failed
        total_end_time = time.monotonic()
        final_attempt = self.attempts[-1]
        final_validation = final_attempt["validation_result"]

        return {
            "downstream_llm_diff_output": final_attempt["generated_diff"],
            "runtime_seconds": round(total_end_time - total_start_time, 2),
            "llm_output_valid": False,
            "attempts_made": self.max_retries,
            "validation_results": [a["validation_result"] for a in self.attempts],
            "error": f"All {self.max_retries} attempts failed.",
            "last_format_error": final_validation.get("format_error"),
            "last_apply_error": final_validation.get("apply_error"),
            "token_counts": get_all_token_counts(final_attempt["generated_diff"] or "", gemini_token_counts=total_gemini_token_counts),
            "final_prompt_used": final_attempt["prompt"]
        }


async def process_single_entry_with_retry(
    rej_content: str,
    original_source_content: str,
    target_filename_for_diff: str,
    vulnerability_id: str,
    failure_details: Dict[str, Any],
    upstream_patch_content: str,
    patch_porter_agent: GeminiAgent,
    max_retries: int = 3,
    repo_lock: asyncio.Lock = None
):
    """
    Processes a single vulnerability entry by orchestrating the SelfCorrectingAgent.
    """
    # Pre-flight checks for content validity
    if not isinstance(rej_content, str) or not rej_content.strip():
        return {"llm_output_valid": False, "error": "Empty or invalid .rej File Content"}
    if not isinstance(original_source_content, str) or not original_source_content.strip():
        return {"llm_output_valid": False, "error": "Empty or invalid Original Source File content"}

    print(f"Processing for diff generation: {vulnerability_id} - {target_filename_for_diff}")

    original_inputs = {
        "rej_content": rej_content,
        "original_source_content": original_source_content,
        "target_filename": target_filename_for_diff,
        "vulnerability_id": vulnerability_id,
        "failure_details": failure_details,
        "upstream_patch_content": upstream_patch_content
    }
    
    agent = SelfCorrectingAgent(
        gemini_agent=patch_porter_agent,
        original_inputs=original_inputs,
        max_retries=max_retries,
        repo_lock=repo_lock
    )

    result = await agent.run()
    # Return identifiers along with the result for proper placement
    return vulnerability_id, target_filename_for_diff, result


async def run_with_semaphore(semaphore, coro):
    async with semaphore:
        return await coro


async def main():
    parser = argparse.ArgumentParser(description="Process vulnerability JSON, generate corrected diffs using an LLM, and output an updated JSON.")
    parser.add_argument("json_file", help="Path to the input JSON file.")
    parser.add_argument("--output-file", help="Path to the output JSON file. If not provided, it will be generated based on the input filename.")
    parser.add_argument("--model-name", default="gemini-2.5-pro-preview-05-06", help="Name of the Gemini model to use.")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for the LLM.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for the LLM generation.")
    parser.add_argument(
        "--target_downstream_version", "-v",
        help="(Optional) Filter by specific downstream_version (e.g., '14'). If not provided, all versions will be processed.",
        default=None
    )
    parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent API calls.")

    args = parser.parse_args()
    script_start_time = time.monotonic()

    # API key loading is now handled at the top of the script using python-dotenv.

    # Load the JSON data
    with open(args.json_file, 'r') as f:
        input_data = json.load(f)

    if not isinstance(input_data, list):
        print("Error: Expected a list of vulnerabilities in the input JSON file.")
        return
    
    output_data = copy.deepcopy(input_data)
    semaphore = asyncio.Semaphore(args.concurrency)
    repo_lock = asyncio.Lock()

    # --- Initialize Report ---
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_label = args.target_downstream_version or "all_versions"
    
    report_dir = "report"
    os.makedirs(report_dir, exist_ok=True)
    report_filename = os.path.join(report_dir, f"report_diff_{version_label}_{timestamp_str}.json")

    report_data = {
        "run_timestamp": timestamp_str,
        "target_downstream_version": version_label,
        "concurrency": args.concurrency,
        "temperature_setting": args.temperature,
        "input_json_file": args.json_file,
        "summary": {
            "total_file_conflicts_to_process": 0,
            "files_attempted_for_llm_diff_generation": 0,
            "files_with_llm_diff_successfully_generated": 0,
            "files_skipped_pre_llm_call": 0,
            "files_with_llm_diff_generation_errors_or_skipped_in_func": 0,
            "successful_attempts_histogram": {},
            "total_runtime_seconds_all": 0,
            "total_runtime_seconds_successful": 0,
            "total_script_wall_clock_runtime_seconds": 0,
            "total_gemini_input_tokens": 0,
            "total_gemini_output_tokens": 0,
            "total_gemini_tokens": 0
        },
        "successfully_generated_diffs_log": [],
        "skipped_or_errored_diff_generation_log": []
    }

    # Initialize the Gemini Agent
    # The system prompt is now managed inside the SelfCorrectingAgent, so we pass a generic one here or none.
    # For simplicity, let's define a minimal system prompt that can be used.
    system_prompt = """You are an expert Android build system and Linux kernel patch developer. Your task is to resolve conflicts in `.rej` files by generating a corrected unified diff. You must be meticulous and precise.

**STRICT FORMATTING AND OUTPUT REQUIREMENTS:**
1.  **Unified Diff Format ONLY**: Your *entire* response must be a single markdown code block starting with ```diff. Do NOT include any other text, explanations, or apologies before or after the code block.
2.  **Headers**: The diff must start with `--- a/path/to/file` and `+++ b/path/to/file`. You will be given the correct `path/to/file`.
3.  **Hunk Headers**: Each change must be enclosed in a hunk, starting with `@@ -old_start,old_lines +new_start,new_lines @@`. These must be calculated perfectly.
4.  **Line Prefixes**:
    -   Lines that are unchanged must start with a single space: ` context`.
    -   Lines that are removed must start with a minus sign: `- removed`.
    -   Lines that are added must start with a plus sign: `+ added`.
5.  **No Line Numbers**: The line number column from the input (`1: ...`) is for your reference only. It must NOT be included in your output diff.

**Crucial Details for Hunk Construction:**
-   `old_start`: The starting line number in the *original* file for this hunk.
-   `old_lines`: The total number of lines (added, removed, context) shown for the *original* file in this hunk.
-   `new_start`: The starting line number in the *new* file for this hunk.
-   `new_lines`: The total number of lines (added, removed, context) shown for the *new* file in this hunk.
-   **Context Lines**: You MUST include 3 lines of unchanged context before and after each change. If the changes are closer than 3 lines, the hunks should be merged.

**Critical Pitfalls to Avoid:**
-   **Incorrect Hunk Calculations**: Double-check your line counts for the `@@ ... @@` header. An off-by-one error will cause the patch to fail. Use the provided line-numbered source file as your ground truth.
-   **Mixing up added/removed lines**: Be careful with `+` and `-`.
-   **Extra Text**: Do not add any text outside the ```diff block.

Now, await the task."""

    patch_porter_agent = GeminiAgent(
        key_rotator=key_rotator,
        model_name=args.model_name,
        system_prompt=system_prompt,
        temperature=args.temperature
    )

    tasks = []
    file_conflict_refs = []

    for vulnerability_item in output_data:
        vulnerability_id = vulnerability_item.get("id", "unknown_vuln_id")
        failures = vulnerability_item.get("failures", [])

        for failure in failures:
            if args.target_downstream_version and failure.get("downstream_version") != args.target_downstream_version:
                continue
            
            file_conflicts = failure.get("file_conflicts", [])
            for file_conflict in file_conflicts:
                rej_content = file_conflict.get("rej_file_content")
                original_source_content = file_conflict.get("downstream_file_content_patched_upstream_only")
                target_filename_for_diff = file_conflict.get("file_name")
                upstream_patch_content = vulnerability_item.get("upstream_patch_content")

                if not all([target_filename_for_diff, rej_content, original_source_content, upstream_patch_content]):
                    print(f"Skipping due to missing data for {vulnerability_id} - {target_filename_for_diff}")
                    file_conflict["llm_result"] = {"llm_output_valid": False, "error": "Missing required data fields for processing."}
                    report_data["summary"]["files_skipped_pre_llm_call"] += 1
                    report_data["skipped_or_errored_diff_generation_log"].append({
                        "vulnerability_id": vulnerability_id, "file_name": target_filename_for_diff or "Unknown",
                        "reason": "Missing required data fields for processing."
                    })
                    continue
                
                report_data["summary"]["total_file_conflicts_to_process"] += 1
                coro = process_single_entry_with_retry(
                    rej_content=rej_content,
                    original_source_content=original_source_content,
                    target_filename_for_diff=target_filename_for_diff,
                    vulnerability_id=vulnerability_id,
                    failure_details=failure,
                    upstream_patch_content=upstream_patch_content,
                    patch_porter_agent=patch_porter_agent,
                    max_retries=args.max_retries,
                    repo_lock=repo_lock
                )
                task = run_with_semaphore(semaphore, coro)
                tasks.append(task)
                file_conflict_refs.append(file_conflict)

    # Determine output filename early
    if args.output_file:
        output_filename = args.output_file
    else:
        base_filename = Path(args.json_file).stem
        output_filename = f"{base_filename}_llm_processed.json"

    # Process tasks as they complete and save iteratively
    for future in asyncio.as_completed(tasks):
        vuln_id, filename, result = await future
        
        # --- Update Report ---
        report_data["summary"]["files_attempted_for_llm_diff_generation"] += 1
        runtime = result.get("runtime_seconds", 0)
        report_data["summary"]["total_runtime_seconds_all"] += runtime
        
        entry_token_counts = result.get("token_counts", {}).get("gemini_total_tokens", 0)
        if entry_token_counts:
            report_data["summary"]["total_gemini_input_tokens"] += result["token_counts"].get("gemini_input_tokens", 0)
            report_data["summary"]["total_gemini_output_tokens"] += result["token_counts"].get("gemini_output_tokens", 0)
            report_data["summary"]["total_gemini_tokens"] += result["token_counts"].get("gemini_total_tokens", 0)

        if result.get("llm_output_valid"):
            report_data["summary"]["files_with_llm_diff_successfully_generated"] += 1
            report_data["summary"]["total_runtime_seconds_successful"] += runtime
            attempts = result.get("attempts_made", 0)
            label = f"{attempts} run{'s' if attempts > 1 else ''}"
            histogram = report_data["summary"]["successful_attempts_histogram"]
            histogram[label] = histogram.get(label, 0) + 1
            report_data["successfully_generated_diffs_log"].append({
                "vulnerability_id": vuln_id,
                "file_name": filename,
                "diff_preview": result.get("downstream_llm_diff_output", "")[:150] + "...",
                "token_counts": result.get("token_counts")
            })
        else:
            report_data["summary"]["files_with_llm_diff_generation_errors_or_skipped_in_func"] += 1
            report_data["skipped_or_errored_diff_generation_log"].append({
                "vulnerability_id": vuln_id,
                "file_name": filename,
                "reason": result.get("error", "Validation failed"),
                "last_format_error": result.get("last_format_error"),
                "last_apply_error": result.get("last_apply_error"),
                "token_counts": result.get("token_counts")
            })

        # Find the correct file_conflict to update
        for vuln_item in output_data:
            if vuln_item.get("id") == vuln_id:
                for failure in vuln_item.get("failures", []):
                    for file_conflict in failure.get("file_conflicts", []):
                        if file_conflict.get("file_name") == filename:
                            file_conflict.update(result)
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
        
        # Save the entire updated data structure after each result
        save_partial_output(output_filename, output_data)
        save_partial_output(report_filename, report_data)

    script_end_time = time.monotonic()
    total_script_runtime = script_end_time - script_start_time
    report_data["summary"]["total_script_wall_clock_runtime_seconds"] = total_script_runtime
    
    # Save final report with wall-clock time
    save_partial_output(report_filename, report_data)

    print(f"\nProcessing complete. Final results saved to {output_filename}")
    print(f"Final report saved to {report_filename}")
    print(f"Total script wall-clock runtime: {total_script_runtime:.2f}s")


if __name__ == "__main__":
    asyncio.run(main()) 
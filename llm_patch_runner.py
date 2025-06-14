r"""Uses an LLM to correct patches that failed to apply.

This module is the fourth step in the patch management pipeline. It is triggered
if the `patch_adopter.py` script fails to apply one or more of the original
patches.

This script reads a `failed_patch.json` file, which contains the details of
the rejected patches. For each failed patch, it uses a large language model
(LLM) with a self-correction mechanism to generate a new, corrected patch.

The process for each patch is as follows:
1.  A detailed prompt is constructed, including the original source file
    content, the rejected hunks (`.rej` file), and the original patch.
2.  The LLM attempts to generate a corrected patch.
3.  The generated patch is validated for format and applicability.
4.  If validation fails, the script analyzes the failure, provides detailed
    feedback to the LLM, and prompts it to try again.
5.  This cycle continues for a configurable number of retries.

Successfully generated patches are saved to the `patch_adoption/generated_patches`
directory, and a detailed JSON report is created.

Usage:
  python llm_patch_runner.py
"""

import os
import json
import asyncio
import tempfile
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys
import argparse
import logging
import yaml

from dotenv import load_dotenv
from unidiff import PatchSet

import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
from android_patch_manager import AndroidPatchManager

# Get a logger for this module
logger = logging.getLogger(__name__)

# --- Load Configuration ---
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

LLM_CONFIG = config.get("llm_runner", {})
PATHS_CONFIG = config.get("paths", {})
# --- End Load Configuration ---

# --- Argument Parser ---
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Uses an LLM to correct patches that failed to apply."
    )
    return parser.parse_args()

# --- Constants ---
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables
load_dotenv()
api_keys_str = os.getenv("GOOGLE_API_KEYS")
if not api_keys_str:
    raise ValueError("The GOOGLE_API_KEYS environment variable is not set.")
api_keys = [key.strip() for key in api_keys_str.split(',') if key.strip()]
if not api_keys:
    raise ValueError("No valid API keys found in GOOGLE_API_KEYS.")

class APIKeyRotator:
    """Rotates through a list of API keys to handle quota limits."""

    def __init__(self, api_keys: List[str]):
        if not api_keys:
            raise ValueError("API keys list cannot be empty.")
        self._api_keys = api_keys
        self._index = 0

    def get_current_key(self) -> str:
        """Returns the current API key."""
        return self._api_keys[self._index]

    def rotate_key(self):
        """Rotates to the next API key."""
        self._index = (self._index + 1) % len(self._api_keys)
        logger.info(f"Rotating to new API key index {self._index}")
        return self.get_current_key()

key_rotator = APIKeyRotator(api_keys)

def _add_line_numbers(content: str) -> str:
    """Adds line numbers to each line of a string for context in prompts."""
    lines = content.splitlines()
    return "\n".join(f"{i+1}: {line}" for i, line in enumerate(lines))

def get_all_token_counts(prompt: str, gemini_token_counts: Dict[str, int]) -> Dict[str, int]:
    """Provides a comprehensive token count including the prompt."""
    return {
        "prompt_tokens": len(prompt.split()),
        "gemini_input_tokens": gemini_token_counts.get("input", 0),
        "gemini_output_tokens": gemini_token_counts.get("output", 0),
        "gemini_total_tokens": gemini_token_counts.get("total", 0)
    }

def validate_patch_format(patch_content: str) -> tuple[bool, str]:
    """Validate patch format using unidiff library."""
    try:
        PatchSet.from_string(patch_content)
        return True, "Valid patch format"
    except Exception as e:
        return False, f"Invalid patch format: {str(e)}"

def validate_patch_applicability_in_repo(patch_content: str, repo_path: str) -> tuple[bool, str]:
    """Test if patch can be applied using GNU patch dry run in actual repo."""
    return AndroidPatchManager.check_patch_applicability(patch_content, repo_path)

class GeminiAgent:
    """A client for interacting with the Gemini API.

    This class encapsulates the logic for sending prompts to the Gemini model,
    handling API errors, and managing API key rotation for quota-related issues.
    """

    def __init__(self, key_rotator: APIKeyRotator, model_name: str,
                 system_prompt: str, temperature: float = 0.0):
        """Initializes the GeminiAgent.

        Args:
            key_rotator: An instance of APIKeyRotator.
            model_name: The name of the Gemini model to use.
            system_prompt: The system-level prompt to guide the model's behavior.
            temperature: The temperature for LLM generation (0.0 for deterministic).
        """
        self.key_rotator = key_rotator
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self._configure_genai()

    def _configure_genai(self):
        """Configures the `genai` library with the current API key."""
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
        """Sends a prompt to the Gemini model with retry and key rotation logic.

        If an API error related to quota or rate limits occurs, it rotates to
        the next API key and retries.

        Args:
            prompt: The user prompt to send to the model.

        Returns:
            A dictionary containing the model's response text and token counts.
        
        Raises:
            RuntimeError: If all API keys fail.
        """
        for _ in range(len(self.key_rotator._api_keys)):
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
                if ("quota" in error_message or "rate limit" in error_message or
                    "internal error" in error_message):
                    logger.warning(f"API error encountered: {e}")
                    self.key_rotator.rotate_key()
                    self._configure_genai()  # Reconfigure with the new key.
                else:
                    # For other errors, raise the exception without retrying.
                    raise e
        raise RuntimeError("All API keys failed. No more keys to try.")

class PatchCorrectionAgent:
    """An agent that uses an LLM to correct failed patches via self-correction.
    
    This agent manages a multi-step process of generating, validating, and
    refining a patch until it is correct or a maximum number of retries is
    reached.
    """
    def __init__(self, gemini_agent: GeminiAgent, original_inputs: dict,
                 max_retries: int = 3, repo_lock: asyncio.Lock = None):
        """Initializes the PatchCorrectionAgent.

        Args:
            gemini_agent: An instance of the GeminiAgent.
            original_inputs: A dictionary containing all necessary data for the task.
            max_retries: The maximum number of times to attempt correction.
            repo_lock: An asyncio.Lock to prevent race conditions during Git operations.
        """
        self.gemini_agent = gemini_agent
        self.original_inputs = original_inputs
        self.max_retries = max_retries
        self.attempts = []
        self.repo_lock = repo_lock

    def _create_initial_prompt(self) -> str:
        """Creates the first prompt sent to the LLM for a given task."""
        # This prompt provides all context and asks for the first attempt.
        return f"""Resolve the conflicts in the provided `.rej` file by generating a corrected unified diff.
The generated diff must apply cleanly to the 'Original Source File' and strictly follow all formatting and hunk construction rules from your system guidelines.

**IMPORTANT CONTEXT**: The "Original Source File Content" you see is from a repository state where some hunks of a patch may have already been successfully applied. The ".rej File Content" contains the hunks that FAILED. Your task is to generate a NEW patch containing ONLY the corrected versions of the REJECTED hunks. Do NOT include changes that were already successfully applied.

**Inputs:**

1.  **Original Source File Content**: The full content of the file to be patched, with line numbers added for precision. This is the state of the file AFTER a partial patch attempt.
    ```text
    {_add_line_numbers(self.original_inputs['original_source_content'])}
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

    def _analyze_failure(self, format_error: Optional[str],
                         apply_error: Optional[str]) -> str:
        """Analyzes the failure of the previous attempt and generates feedback."""
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
            return ("**Analysis:** An unspecified validation error occurred. Please "
                    "review your entire output and the system guidelines "
                    "carefully and try again. Be extremely meticulous.")

        return "\n\n".join(analysis_parts)

    def _create_retry_prompt(self) -> str:
        """Creates a refined prompt for subsequent attempts after a failure."""
        last_attempt = self.attempts[-1]
        analysis = self._analyze_failure(
            last_attempt["validation_result"].get("format_error"),
            last_attempt["validation_result"].get("apply_error")
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
    {_add_line_numbers(self.original_inputs['original_source_content'])}
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
        Validates patch applicability in the current state of the repo using a
        non-destructive dry run.
        """
        if not self.repo_lock:
            return False, "Repo lock not provided"

        async with self.repo_lock:
            repo_path_str = self.original_inputs["repo_path"]
            return validate_patch_applicability_in_repo(
                patch_content, str(repo_path_str)
            )

    async def run(self) -> dict:
        """Runs the full self-correction loop for a single patch task."""
        total_gemini_token_counts = {"input": 0, "output": 0, "total": 0}

        for attempt_num in range(self.max_retries):
            logger.info(f"  -> Attempt {attempt_num + 1}/{self.max_retries} for "
                        f"{self.original_inputs['target_filename']}")
            prompt = (self._create_initial_prompt() if attempt_num == 0
                      else self._create_retry_prompt())
            
            generated_diff = ""
            try:
                result = await self.gemini_agent.run(prompt)
                
                generated_diff = result["data"].strip()
                attempt_token_counts = result["token_counts"] or {}
                for key in total_gemini_token_counts:
                    total_gemini_token_counts[key] += attempt_token_counts.get(key, 0)

                format_valid, format_error = validate_patch_format(generated_diff)
                
                apply_valid, apply_error = (False, "Skipped due to format error")
                if format_valid:
                    apply_valid, apply_error = await self._validate_applicability_in_repo(generated_diff)
                
                validation_result = {
                    "attempt": attempt_num + 1,
                    "format_valid": format_valid, "format_error": format_error,
                    "apply_valid": apply_valid, "apply_error": apply_error,
                    "valid": format_valid and apply_valid
                    }
            except Exception as e:
                logger.error(f"  -> ❌ Error on attempt {attempt_num + 1}: {e}")
                validation_result = {
                    "attempt": attempt_num + 1, "error": str(e),
                    "valid": False, "format_error": "N/A", "apply_error": str(e)
                }

            self.attempts.append({"generated_diff": generated_diff,
                                  "validation_result": validation_result})

            if validation_result.get("valid"):
                logger.info(f"  -> ✅ Validation successful on attempt {attempt_num + 1}.")
                return {
                    "success": True,
                    "patch_hash": self.original_inputs["commit_hash"],
                    "file": self.original_inputs["target_filename"],
                    "output_path": self.original_inputs["output_path"],
                    "validation_results": [a["validation_result"] for a in self.attempts],
                    "downstream_llm_diff_output": generated_diff,
                    "token_counts": total_gemini_token_counts
                }
            else:
                 logger.warning(f"  -> ❌ Validation failed on attempt {attempt_num + 1}.")

        # If the loop finishes, all retries have failed.
        logger.error(f"  -> ❌ All {self.max_retries} attempts failed.")
        return {
            "success": False,
            "patch_hash": self.original_inputs["commit_hash"],
            "file": self.original_inputs["target_filename"],
            "validation_results": [a["validation_result"] for a in self.attempts],
            "token_counts": total_gemini_token_counts
        }


async def process_patch_entry(patch: dict, repo_lock: asyncio.Lock) -> list[dict]:
    """Processes a single entry from the `failed_patch.json` file.

    This function sets up the repository and then iterates through each file
    conflict within that patch entry, triggering the PatchCorrectionAgent for each.

    Args:
        patch: A dictionary representing one patch entry from the input file.
        repo_lock: An asyncio.Lock to manage concurrent Git operations.

    Returns:
        A list of result dictionaries, one for each file conflict processed.
    """
    patch_hash = Path(patch["patch_file"]).stem
    
    target_source_path = os.getenv("TARGET_SOURCE_PATH")
    if not target_source_path:
        logger.error("TARGET_SOURCE_PATH environment variable is not set in process_patch_entry.")
        # Return an error for all files in this patch entry.
        return [{
            "file": f.get("failed_file", "Unknown"),
            "error": "TARGET_SOURCE_PATH not set.",
            "success": False
        } for f in patch.get("rejected_files", [{"failed_file": "Unknown"}])]

    user_provided_repo_path = Path(target_source_path)
    project_rel_path = patch.get("project")

    if not project_rel_path:
        logger.error(f"Patch entry for {patch_hash} is missing the 'project' field.")
        return [{"file": "Unknown", "error": "Missing 'project' in input JSON.", "success": False}]

    # Determine the correct project path.
    if str(user_provided_repo_path).endswith(project_rel_path):
        project_abs_path = user_provided_repo_path
    else:
        project_abs_path = user_provided_repo_path / project_rel_path
    
    logger.info(f"ℹ️  Starting patch processing for {patch_hash}. Using project path: {project_abs_path}")
    
    if not project_abs_path.is_dir():
        logger.error(f"Project path is not a valid directory: {project_abs_path}")
        return [{
            "file": f.get("failed_file", "Unknown"),
            "error": f"Invalid project path: {project_abs_path}",
            "success": False
        } for f in patch.get("rejected_files", [{"failed_file": "Unknown"}])]

    all_file_results = []
    for file_conflict in patch.get("rejected_files", []):
        try:
            failed_file_path_str = file_conflict.get("failed_file")
            if not failed_file_path_str:
                logger.warning(f"  -> ⚠️  file_conflict entry is missing 'failed_file' key for patch {patch_hash}.")
                continue
            failed_file_path = Path(failed_file_path_str)
            logger.info(f"\n▶️  Processing file: {failed_file_path.name} from patch {patch_hash}")

            rej_content = file_conflict.get("rej_content")
            if not rej_content:
                logger.warning(f"  -> 'rej_content' is missing or empty for {failed_file_path.name}. This file cannot be fixed.")
                all_file_results.append({
                    "file": str(failed_file_path),
                    "error": "Missing or empty rej_content.",
                    "success": False
                })
                continue
            
            # This is the state of the source file after the failed patch attempt.
            original_source = Path(failed_file_path).read_text(encoding="utf-8")

            if 'upstream_patch_content' not in patch:
                logger.error(f"  -> Missing 'upstream_patch_content' in input JSON for {failed_file_path.name}.")
                all_file_results.append({
                    "file": str(failed_file_path),
                    "error": "Missing 'upstream_patch_content' in input JSON.",
                    "success": False
                })
                continue
            
            # Prepare the inputs for the correction agent.
            output_dir = BASE_DIR / PATHS_CONFIG.get("llm_generated_patches_dir")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = (output_dir / 
                           f"{patch_hash}_{failed_file_path.name}_"
                           f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.diff")

            original_inputs = {
                "rej_content": rej_content,
                "original_source_content": original_source,
                "upstream_patch_content": patch["upstream_patch_content"],
                "target_filename": str(failed_file_path.relative_to(project_abs_path)),
                "repo_path": str(project_abs_path),
                "commit_hash": patch_hash,
                "output_path": str(output_path)
            }
            
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
-   **Incorrect Hunk Calculations**: Double-check your line counts for the `@@ ... @@` header. An off-by-one error will cause the patch to fail.
-   **Mixing up added/removed lines**: Be careful with `+` and `-`.
-   **Extra Text**: Do not add any text outside the ```diff block.

Now, await the task."""

            gemini_agent = GeminiAgent(
                model_name=LLM_CONFIG.get("model_name"),
                temperature=LLM_CONFIG.get("temperature", 0.0),
                system_prompt=system_prompt,
                key_rotator=key_rotator
            )
            
            agent = PatchCorrectionAgent(
                gemini_agent=gemini_agent,
                original_inputs=original_inputs,
                max_retries=LLM_CONFIG.get("max_retries", 3),
                repo_lock=repo_lock
            )

            # Run the agent and get the result.
            logger.info(f"  -> Calling LLM Agent for {failed_file_path.name}...")
            result = await agent.run()

            if result.get("success"):
                # Add context to the result for downstream processing
                result['project'] = project_rel_path
                result['original_file'] = str(failed_file_path)
                
                output_path.write_text(result["downstream_llm_diff_output"],
                                       encoding="utf-8")
                result['saved_patch_file'] = output_path.name
                logger.info(f"  -> ✅ Successfully generated and saved patch to "
                            f"{output_path.name}")
            else:
                result['project'] = project_rel_path
                result['original_file'] = str(failed_file_path)
                logger.error(f"  -> ❌ Failed to generate patch for {failed_file_path.name}")

            all_file_results.append(result)

        except (IOError, TypeError, KeyError, ValueError) as e:
            logger.critical(f"  -> ❌ CRITICAL ERROR while preparing to process {file_conflict.get('failed_file', 'Unknown')}: {e}")
            all_file_results.append({
                "file": file_conflict.get('failed_file', 'Unknown'),
                "error": f"Critical setup error: {e}",
                "success": False
            })
            continue

    return all_file_results

async def run_llm_correction_step():
    """
    Runs the LLM patch correction process.
    This function is a generator that yields progress updates.
    """
    logger.info("--- Starting LLM Patch Runner ---")
    
    failed_patch_json_path = BASE_DIR / PATHS_CONFIG.get("llm_input_report")

    try:
        with open(failed_patch_json_path, "r", encoding="utf-8") as f:
            failed_patches_data = json.load(f)
    except FileNotFoundError:
        message = f"Input file not found at '{failed_patch_json_path}'."
        logger.error(message)
        yield {"type": "error", "message": message}
        return
    except json.JSONDecodeError:
        message = f"Could not decode JSON from '{failed_patch_json_path}'. The file might be empty or corrupt."
        logger.error(message)
        yield {"type": "error", "message": message}
        return

    patches_to_process = failed_patches_data.get("patches", [])
    total_patches = len(patches_to_process)
    logger.info(f"Found {total_patches} failed patches to correct.")

    yield {"type": "progress", "completed": 0, "total": total_patches}
    if not patches_to_process:
        return

    repo_lock = asyncio.Lock()
    sem = asyncio.Semaphore(LLM_CONFIG.get("concurrency", 5))
    all_results = []
    processed_count = 0

    async def process_one_patch(patch: dict):
        """Helper to process one patch entry under the semaphore."""
        async with sem:
            return await process_patch_entry(patch, repo_lock)

    tasks = [process_one_patch(patch) for patch in patches_to_process]

    for future in asyncio.as_completed(tasks):
        results_for_patch = await future
        all_results.extend(results_for_patch)
        processed_count += 1
        yield {"type": "progress", "completed": processed_count, "total": total_patches}

    # Aggregate token counts and create reports
    total_input_tokens = sum(r.get("token_counts", {}).get("input", 0) for r in all_results)
    total_output_tokens = sum(r.get("token_counts", {}).get("output", 0) for r in all_results)
    total_tokens = sum(r.get("token_counts", {}).get("total", 0) for r in all_results)

    successful_patches = [
        {"patch_file": r.get("saved_patch_file"), "patch_url": None, "source": "LLM", "project": r.get("project")}
        for r in all_results if r.get("success") and r.get("saved_patch_file")
    ]
    
    successful_patches_file = BASE_DIR / PATHS_CONFIG.get("llm_successful_patches_report")
    successful_patches_file.parent.mkdir(parents=True, exist_ok=True)
    with open(successful_patches_file, "w", encoding="utf-8") as f:
        json.dump({"patches": successful_patches}, f, indent=4)
    logger.info(f"Saved {len(successful_patches)} successfully generated patches to {successful_patches_file}")

    debug_report_path = BASE_DIR / PATHS_CONFIG.get("llm_detailed_output_report")
    failed_patches_list = [
        {"patch_file": r.get("file"), "project": r.get("project"), "error": r.get("validation_results", [{}])[-1].get("apply_error", "Unknown LLM failure")}
        for r in all_results if not r.get("success")
    ]
    debug_report_data = {
        "summary": {
            "total_patches_processed": len(all_results),
            "successful_patches": len(successful_patches),
            "failed_patches": len(failed_patches_list),
            "failed_patches_list": failed_patches_list,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
        },
        "results": [r for r in all_results if 'downstream_llm_diff_output' in r and r.pop('downstream_llm_diff_output', None)],
    }

    with open(debug_report_path, "w", encoding="utf-8") as f:
        json.dump(debug_report_data, f, indent=2)
    logger.info(f"Full LLM agent execution report saved to {debug_report_path}")

    logger.info("--- LLM Patch Runner Finished ---")
    yield {"type": "summary", "data": debug_report_data['summary']}


async def main():
    """Main asynchronous entry point for the LLM patch runner."""
    # When run as a script, just iterate through the async generator to execute it.
    async for _ in run_llm_correction_step():
        pass

if __name__ == "__main__":
    asyncio.run(main())

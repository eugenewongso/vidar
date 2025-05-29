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
from google.cloud import aiplatform_v1beta1
from google.cloud.aiplatform_v1beta1.types import CountTokensRequest, Content, Part
from android_patch_manager import AndroidPatchManager
from pathlib import Path


client = aiplatform_v1beta1.PredictionServiceClient()

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
if not api_keys:
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



def count_tokens_gemini(text, project: str, location: str = "us-central1", model: str = "gemini-2.5-pro-preview-05-06"):
    """
    Count tokens using the Gemini model from Google Cloud AI Platform.

    Args:
        text (str): Input text to count tokens for.
        project (str): Google Cloud project ID.
        location (str): Location of the model.
        model (str): Model name.

    Returns:
        int: Total token count.
    """
    publisher_model = f"projects/{project}/locations/{location}/publishers/google/models/{model}"
    request = CountTokensRequest(
        endpoint=publisher_model,
        contents=[Content(role="user", parts=[Part(text=text)])]
    )
    response = client.count_tokens(request=request)
    return response.total_tokens

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
                if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                    print(f"⚠️ API quota/rate limit hit: {e}")
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

    base_task_prompt = f"""
        Your system instructions describe your role as an advanced security patching assistant.

        You are provided with the following inputs:

        1. Original Source File (Content of the file to be patched):
        ```
        {dependencies.original_source_file_content}
        ```

        2. .rej File Content (Contains rejected hunks from a previous patch attempt):
        ```
        {dependencies.rej_file_content}
        ```

        Target Filename (use this for `--- a/` and `+++ b/` lines in your diff output): `{target_filename_for_diff}`

        Your Task:
        Following your system instructions, analyze the '.rej File Content' in conjunction with the 'Original Source File'.
        Your goal is to generate a new, corrected unified diff. This diff should incorporate the intended changes from the rejected hunks, modified as necessary to apply cleanly to the 'Original Source File'.

        Output Requirements (Strictly Enforced):
        - Your entire response must be ONLY the unified diff.
        - Do not include any introductory text, explanations, comments, or summaries.
        - The diff must be in the standard unified diff format, suitable for application with `patch -p1`.
        - Start the diff with `--- a/{target_filename_for_diff}` and `+++ b/{target_filename_for_diff}`.
    """

    print(f"Processing for diff generation: {vulnerability_id} - {target_filename_for_diff}")
    
    total_start_time = time.monotonic()
    validation_results = []
    
    for attempt in range(max_retries):
        print(f"🔄 Attempt {attempt + 1} of {max_retries} for {target_filename_for_diff}")
        
        # Add retry-specific guidance to prompt
        retry_guidance = ""
        if attempt > 0:
            prev_errors = []
            for r in validation_results:
                if not r["valid"]:
                    if not r["format_valid"]:
                        prev_errors.append("[Format Error]")
                    elif not r["apply_valid"]:
                        prev_errors.append("[Apply Error]")


            retry_guidance = f"""

        IMPORTANT - Previous attempts failed validation:
        {chr(10).join(f"- Attempt {i+1}: {err}" for i, err in enumerate(prev_errors))}

        Please fix these issues and regenerate your diff.

        Guidelines:
        1. Start the diff with `--- a/{target_filename_for_diff}` and `+++ b/{target_filename_for_diff}`
        2. Use valid hunk headers like `@@ -line,count +line,count @@`
        3. The diff must apply cleanly using `patch -p1` to the provided original file
        4. Do NOT include explanations or extra text — only the unified diff
        """
            if 'generated_diff' in locals():
                retry_guidance += f"\nHere is a snippet of your last invalid diff:\n{generated_diff[:500]}"

            print("Previous errors:", prev_errors)
        task_prompt = base_task_prompt + retry_guidance
        
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
    
    args = parser.parse_args()

    system_prompt="""You are an advanced security patching assistant. Your primary task is to generate a correctly formatted unified diff (.diff) file that successfully applies security patches.
    You will be given:
    1. An 'Original Source File' (the vulnerable code).
    2. A '.rej File Content' (containing rejected hunks from a previously failed patch application).

    Your goal is to:
    - Analyze the rejected hunk(s) in the '.rej File Content' to understand why the original patch application failed.
    - Modify the hunk(s) so that they apply cleanly and correctly to the provided 'Original Source File'.
    - Ensure your output is a valid unified diff that can be applied using a standard utility like `patch -p1`.

    Constraints:
    - Your output MUST be ONLY the unified diff. Do not include any explanations, comments, or any other text.
    - The diff must be in the correct unified diff format (starting with `--- a/...` and `+++ b/...`).
    - Do not alter unrelated code in the 'Original Source File'.
    - Only modify what is absolutely necessary within the hunk(s) to make the patch apply correctly and achieve the intended security remediation.
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
                        file_conflict.update(generated_diff)
                        
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
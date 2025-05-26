import asyncio
import os
import datetime
import sys
import json
import argparse
import copy # Added for deepcopy
import difflib # Added for generating diffs
import time # Added for timing LLM calls
import logfire
from dataclasses import dataclass
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Optional, List, Dict, Any
import tiktoken
import re


# Load environment variables from .env file
load_dotenv()

# Configure Logfire for logging (if token is available)
logfire.configure(send_to_logfire='if-token-present')

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

api_keys = os.getenv("GOOGLE_API_KEYS", "").split(",")
if not api_keys or api_keys == [""]:
    raise ValueError("Missing API keys in GOOGLE_API_KEYS")

key_rotator = APIKeyRotator(api_keys)


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

def get_all_token_counts(text: str, project: str = "", skip_gemini: bool = False):
    result = {
        "openai": count_tokens_tiktoken(text),
        "general": count_tokens_general(text),
    }
    return result

@dataclass
class SupportDependencies: 
    diff_content: str
    vulnerable_code_content: str

@dataclass
class LLMResult:
    data: str
    token_count: Optional[int]

class GeminiAgent:
    def __init__(self, model_name: str, system_prompt: str, key_rotator: APIKeyRotator):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.key_rotator = key_rotator
        self._configure_genai()

    def _configure_genai(self):
        current_key = self.key_rotator.get_current_key()
        genai.configure(api_key=current_key)
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.system_prompt
        )

    async def run(self, prompt: str, deps: Optional[SupportDependencies] = None):

        for attempt in range(len(self.key_rotator.api_keys)):
            try:
                response = self.model.generate_content(prompt)
                token_count = None
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    token_count = getattr(response.usage_metadata, "total_token_count", None)
                return LLMResult(data=response.text, token_count=token_count)

            except Exception as e:
                if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                    print(f"⚠️ API quota/rate limit hit: {e}")
                    self.key_rotator.rotate_key()
                    self._configure_genai()
                else:
                    raise e
        raise RuntimeError("All API keys exhausted or failed.")


def save_partial_output(path: str, data: Any):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"💾 Partial output saved to: {path}")
    except Exception as e:
        print(f"⚠️ Failed to save partial output to {path}: {e}")

# Define the agent using Gemini
# LatestGeminiModelNames = Literal[
#     "gemini-1.5-flash",
#     "gemini-1.5-flash-8b",
#     "gemini-1.5-pro",
#     "gemini-1.0-pro",
#     "gemini-2.0-flash-exp",
#     "gemini-2.0-flash-thinking-exp-01-21",
#     "gemini-exp-1206",
#     "gemini-2.0-flash",
#     "gemini-2.0-flash-lite-preview-02-05",
#     "gemini-2.5-flash-preview-04-17",
#     "gemini-2.5-pro-preview-05-06",
# ]
patch_porter_agent = GeminiAgent(
    model_name='gemini-2.5-pro-preview-05-06',
    system_prompt="""
    You are a patch porting agent specializing in resolving merge conflicts and applying diff-like patch content to remediate security vulnerabilities in codebases.
    
    IMPORTANT: Your response must contain ONLY the patched code with no additional comments, explanations, or formatting changes.
    DO NOT include any explanations about what you did, DO NOT include any headers or footers.
    DO NOT change indentation, whitespace, or formatting of the original file unless necessary for the patch.
    Preserve all tabs, spaces, and line endings exactly as they appear in the original file.
    Just output the final patched code file with the security fixes applied and nothing else.
    """,
    key_rotator=key_rotator
)

async def process_single_entry(
    patch_content: str,
    vuln_code_content: str,
    output_filename_base: str,
    vulnerability_id: str,
    failure_details: Dict[str, Any]
):
    """
    Processes a single vulnerability entry using the Gemini agent.
    Returns a tuple: (modified_code_string_or_None, reason_message_or_None, time_taken_or_None).
    """
    # Type checks for content fields
    if not isinstance(patch_content, str):
        reason = f"Skipping entry because patch_content (from rej_file_content) is not a string."
        print(f"{reason} for {output_filename_base} in {vulnerability_id}")
        return None, reason, None
    if not isinstance(vuln_code_content, str):
        reason = f"Skipping entry because vuln_code_content (from downstream_file_content_patched_upstream_only) is not a string."
        print(f"{reason} for {output_filename_base} in {vulnerability_id}")
        return None, reason, None

    # Existing checks for empty/placeholder content (now safe due to type checks above)
    if not vuln_code_content or vuln_code_content.strip() == "```" or vuln_code_content.strip() == "":
        reason = "Skipping entry due to empty or placeholder vulnerable code content."
        print(f"{reason} for {output_filename_base} in {vulnerability_id}")
        return None, reason, None
    if not patch_content or patch_content.strip() == "```" or patch_content.strip() == "":
        reason = "Skipping entry due to empty or placeholder patch content."
        print(f"{reason} for {output_filename_base} in {vulnerability_id}")
        return None, reason, None

    dependencies = SupportDependencies(
        diff_content=patch_content,
        vulnerable_code_content=vuln_code_content,
    )

    task_prompt = f"""
        You will be provided with the following:

        Patch Content (Diff-like):
        ```
        {dependencies.diff_content}
        ```

        Vulnerable Codebase Content:
        ```
        {dependencies.vulnerable_code_content}
        ```

        Instructions:

        1. Carefully analyze the provided Patch Content and Vulnerable Codebase Content.
        2. The Patch Content describes changes to be made to the Vulnerable Codebase Content.
        3. Apply the changes described in Patch Content to the Vulnerable Codebase Content.
        4. If there are merge conflicts, resolve them while maintaining the integrity and functionality of the Vulnerable Codebase Content.
        5. Ensure the patched code remains functional and does not introduce new issues.
        6. Provide the patched codebase as your final output. 

        STRICT REQUIREMENTS:
        - Output **ONLY** the complete patched code.
        - Do NOT include any explanations, comments, or anything else.
        - The response must be **exactly** the final patched file with the security fixes applied.
        - Preserve original formatting (indentation, whitespace, line endings) as much as possible, only changing what's necessary for the patch.
    """

    print(f"Processing: {vulnerability_id} - {output_filename_base}")
    start_time = time.time()
    try:
        result = await patch_porter_agent.run(task_prompt)
        modified_code = result.data.strip()
        gemini_token_count = result.token_count
        print(f"Gemini token count: {gemini_token_count}")
        end_time = time.time()
        time_taken = end_time - start_time
        print(f"✅ LLM processing successful for: {vulnerability_id} - {output_filename_base} (took {time_taken:.2f}s)")
        return modified_code, None, time_taken, gemini_token_count

    except Exception as e:
        end_time = time.time()
        time_taken = end_time - start_time
        reason = f"Error during LLM processing: {e}"
        print(f"Error processing entry for {output_filename_base} in {vulnerability_id}: {e} (took {time_taken:.2f}s)")
        # Log more details from failure_details if needed
        print(f"  Failure details: Patch SHA: {failure_details.get('downstream_patch', 'N/A')}, Repo: {failure_details.get('repo_path', 'N/A')}")
        return None, reason, time_taken, 0


async def main():
    parser = argparse.ArgumentParser(description="Process vulnerability JSON, apply patches using an LLM, and output an updated JSON.")
    parser.add_argument("input_json_file_path", help="Path to the input JSON file.")
    parser.add_argument(
        "--target_downstream_version", "-v",
        help="(Optional) Filter by specific downstream_version (e.g., '14'). If not provided, all versions will be processed.",
        default=None
    )
    parser.add_argument("--output_json_path", "-o", help="Path to save the output JSON file. Defaults to outputs/output_android_{version}_{timestamp}.json", default=None)
    parser.add_argument(
        "--resume_from_id",
        help="(Optional) Vulnerability ID to resume from. All earlier entries will be skipped.",
        default=None
    )

    
    args = parser.parse_args()

    # Generate timestamp once for potential use in both output and report filenames
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    output_json_path_to_use = args.output_json_path
    if output_json_path_to_use is None:
        outputs_dir = "outputs"
        os.makedirs(outputs_dir, exist_ok=True)
        # Use the generated timestamp_str
        version_label = args.target_downstream_version or "all_versions"
        output_json_path_to_use = os.path.join(
            outputs_dir,
            f"output_android_{version_label}_{timestamp_str}.json"
        )

    # Define report filename early so it's available throughout
    report_dir = os.path.join("outputs", "report")
    os.makedirs(report_dir, exist_ok=True)
    version_label = args.target_downstream_version or "all_versions"
    report_filename = os.path.join(report_dir, f"{version_label}_{timestamp_str}.json")



    # Initialize report data structure
    report_data = {
        "run_timestamp": timestamp_str,
        "target_downstream_version": args.target_downstream_version,
        "input_json_file": args.input_json_file_path,
        "main_output_json_file": output_json_path_to_use,
        "summary": {
            "total_file_conflicts_matching_version": 0,
            "files_attempted_for_llm": 0,
            "files_successfully_processed_by_llm": 0,
            "files_skipped_pre_llm_call": 0,
            "files_skipped_or_errored_in_processing_function": 0,
        },
        "successfully_processed_files_log": [],
        "skipped_or_errored_files_log": []
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


    # Create a deep copy of the input data to modify
    output_data = copy.deepcopy(input_data)

    resume_from_id = args.resume_from_id
    resume_mode = resume_from_id is not None
    resuming = False  # Flag to track when to start

    # Prepare the subset of data to process based on resume_from_id
    if resume_mode:
        resume_index = next((i for i, item in enumerate(output_data) if item.get("id") == resume_from_id), None)
        if resume_index is None:
            print(f"❌ Error: resume_from_id '{resume_from_id}' not found.")
            return
        data_to_process = output_data[resume_index:]
    else:
        data_to_process = output_data

    total_to_process = len(data_to_process)


    for idx, vulnerability_item in enumerate(data_to_process, start=1):
        vulnerability_id = vulnerability_item.get("id", "unknown_vuln_id")
        print(f"\n📍 Processing {idx} of {total_to_process}: {vulnerability_id}")

        if not isinstance(vulnerability_item, dict):
            # This check might be redundant if input_data was already validated,
            # but good for safety if structure isn't guaranteed.
            print(f"Skipping non-dictionary item: {vulnerability_item}")
            continue

        vulnerability_id = vulnerability_item.get("id", "unknown_vuln_id")
        original_failures = vulnerability_item.get("failures", []) 

        if not isinstance(original_failures, list):
            print(f"Skipping vulnerability {vulnerability_id} due to invalid 'failures' format (expected a list).")
            # vulnerability_item["failures"] = [] 
            continue
        
        filtered_failures_for_item = [] 

        for failure in original_failures: 
            if not isinstance(failure, dict):
                print(f"Skipping non-dictionary failure in {vulnerability_id}: {failure}")
                continue

            if args.target_downstream_version is None or failure.get("downstream_version") == args.target_downstream_version:
                # This failure matches the target version, so we process it.
                file_conflicts = failure.get("file_conflicts", []) 
                if not isinstance(file_conflicts, list):
                    print(f"Skipping failure in {vulnerability_id} (patch: {failure.get('downstream_patch','N/A')}) due to invalid 'file_conflicts' format (expected a list).")
                    # failure["file_conflicts"] = [] 
                    # filtered_failures_for_item.append(failure) # Add failure even if file_conflicts is broken, as it matches version
                    continue

                # Process file_conflicts for this failure
                for file_conflict in file_conflicts: 
                    if not isinstance(file_conflict, dict):
                        print(f"Skipping non-dictionary file_conflict in {vulnerability_id} for failure {failure.get('downstream_patch','N/A')}: {file_conflict}")
                        continue
                    
                    patch_content = file_conflict.get("rej_file_content")
                    vuln_code_content = file_conflict.get("downstream_file_content_patched_upstream_only")
                    original_vuln_code_content = file_conflict.get("downstream_file_content")
                    original_file_name = file_conflict.get("file_name")

                    report_data["summary"]["total_file_conflicts_matching_version"] += 1
                    
                    skip_reason = None
                    if not original_file_name:
                        skip_reason = "Missing 'file_name'"
                    elif not patch_content:
                        skip_reason = "Missing 'rej_file_content'"
                    elif not vuln_code_content:
                        skip_reason = "Missing 'downstream_file_content_patched_upstream_only'"

                    if skip_reason:
                        print(f"Skipping file_conflict in {vulnerability_id} (file: {original_file_name or 'N/A'}) due to: {skip_reason}.")
                        report_data["summary"]["files_skipped_pre_llm_call"] += 1
                        report_data["skipped_or_errored_files_log"].append({
                            "vulnerability_id": vulnerability_id,
                            "file_name": original_file_name or "Unknown",
                            "patch_sha": failure.get('downstream_patch', 'N/A'),
                            "reason": skip_reason
                        })
                        file_conflict["downstream_patched_file_llm_output"] = f"skipped, {skip_reason}"
                        continue
                    
                    report_data["summary"]["files_attempted_for_llm"] += 1
                    modified_code, error_reason_from_processing, time_taken_for_llm, gemini_token_count = await process_single_entry(
                        patch_content=patch_content,
                        vuln_code_content=vuln_code_content,
                        output_filename_base=original_file_name,
                        vulnerability_id=vulnerability_id,
                        failure_details=failure
                    )

                    if modified_code is not None:
                        token_counts = get_all_token_counts(modified_code, project="", skip_gemini=True)
                        if gemini_token_count is not None:
                            token_counts["gemini"] = gemini_token_count
                        file_conflict["llm_output_token_counts"] = token_counts
                        file_conflict["downstream_patched_file_llm_output"] = modified_code
                        if time_taken_for_llm is not None:
                            file_conflict["llm_time_taken_seconds"] = round(time_taken_for_llm, 2)
                        
                        # Calculate and add LLM_diff_content
                        if vuln_code_content and modified_code:
                            diff = difflib.unified_diff(
                                original_vuln_code_content.splitlines(keepends=True),
                                modified_code.splitlines(keepends=True),
                                fromfile='original',
                                tofile='patched'
                            )
                            file_conflict["LLM_diff_content"] = "".join(diff)
                        else:
                            file_conflict["LLM_diff_content"] = "Could not generate diff (original or patched content missing)."

                        report_data["summary"]["files_successfully_processed_by_llm"] += 1
                        log_entry = {
                            "vulnerability_id": vulnerability_id,
                            "file_name": original_file_name,
                            "patch_sha": failure.get('downstream_patch', 'N/A')
                        }
                        if time_taken_for_llm is not None:
                            log_entry["llm_time_taken_seconds"] = round(time_taken_for_llm, 2)
                        report_data["successfully_processed_files_log"].append(log_entry)
                        save_partial_output(output_json_path_to_use, output_data)
                        save_partial_output(report_filename, report_data)

                        
                    else:
                        # If modified_code is None, it means process_single_entry determined a skip/error.
                        # error_reason_from_processing will contain the reason.
                        reason_for_skip_in_output = error_reason_from_processing or "unknown processing error"
                        file_conflict["downstream_patched_file_llm_output"] = f"skipped, {reason_for_skip_in_output}"
                        if time_taken_for_llm is not None: # Store time even for errors if available
                             file_conflict["llm_time_taken_seconds"] = round(time_taken_for_llm, 2)
                        report_data["summary"]["files_skipped_or_errored_in_processing_function"] += 1
                        # Use the specific reason from process_single_entry
                        log_entry = {
                            "vulnerability_id": vulnerability_id,
                            "file_name": original_file_name,
                            "patch_sha": failure.get('downstream_patch', 'N/A'),
                            "reason": error_reason_from_processing or "Skipped or errored in processing function (unknown reason)"
                        }
                        if time_taken_for_llm is not None:
                            log_entry["llm_time_taken_seconds"] = round(time_taken_for_llm, 2)
                        report_data["skipped_or_errored_files_log"].append(log_entry)
                        save_partial_output(output_json_path_to_use, output_data)
                        save_partial_output(report_filename, report_data)

                
                # After processing all file_conflicts for this failure, add the (modified) failure to our filtered list
                filtered_failures_for_item.append(failure)
            else:
                # This failure's downstream_version does not match args.target_downstream_version.
                # So, it's not added to filtered_failures_for_item, effectively filtering it out.
                pass 
        
        # Replace the original 'failures' list in the vulnerability_item with our new filtered list
        vulnerability_item["failures"] = filtered_failures_for_item
    
    # Write the main modified data to its output JSON file
    try:
        with open(output_json_path_to_use, "w", encoding="utf-8") as f_out:
            json.dump(output_data, f_out, indent=4)
        print(f"✅ Main processed JSON successfully saved to '{output_json_path_to_use}'")
    except IOError as e:
        print(f"Error writing main output JSON to '{output_json_path_to_use}': {e}")

    # Write the summary report to its JSON file
    report_dir = os.path.join("outputs", "report")
    os.makedirs(report_dir, exist_ok=True)
    version_label = args.target_downstream_version or "all_versions"
    report_filename = os.path.join(report_dir, f"{version_label}_{timestamp_str}.json")

    try:
        with open(report_filename, "w", encoding="utf-8") as f_report:
            json.dump(report_data, f_report, indent=4)
        print(f"✅ Summary report successfully saved to '{report_filename}'")
    except IOError as e:
        print(f"Error writing summary report to '{report_filename}': {e}")

if __name__ == '__main__':
    asyncio.run(main())

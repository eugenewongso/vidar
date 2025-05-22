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
import tiktoken
from google.cloud import aiplatform_v1beta1
from google.cloud.aiplatform_v1beta1.types import CountTokensRequest, Content, Part

client = aiplatform_v1beta1.PredictionServiceClient()


# Load environment variables from .env file
load_dotenv()

# Configure Logfire for logging (if token is available)
logfire.configure(send_to_logfire='if-token-present')

# Fetch API key from environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Ensure API key is set
if not GOOGLE_API_KEY:
    raise ValueError("Missing API key. Please add GOOGLE_API_KEY to your .env file.")

# Configure Google Generative AI with API key
genai.configure(api_key=GOOGLE_API_KEY)

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

def get_all_token_counts(text: str, project: str, skip_gemini: bool = False):
    """
    Get token counts from multiple methods (OpenAI, general, Gemini).

    Args:
        text (str): Input text.
        project (str): Google Cloud project ID.
        skip_gemini (bool): Whether to skip Gemini token counting.

    Returns:
        dict: Token counts from different methods.
    """
    result = {
        "openai": count_tokens_tiktoken(text),
        "general": count_tokens_general(text),
    }
    if not skip_gemini:
        result["gemini"] = count_tokens_gemini(text, project)
    return result

@dataclass
class SupportDependencies: 
    rej_file_content: str # Renamed from diff_content for clarity
    original_source_file_content: str # Renamed from vulnerable_code_content

class GeminiAgent:
    def __init__(self, model_name: str, system_prompt: str):
        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt
        )
        
    async def run(self, prompt: str, deps: Optional[SupportDependencies] = None):
        response = self.model.generate_content(prompt)
        
        class Result:
            def __init__(self, text):
                self.data = text
                
        return Result(response.text)

patch_porter_agent = GeminiAgent(
    model_name='gemini-2.5-pro-preview-05-06', # Or your preferred model
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
)

async def process_single_entry(
    rej_content: str,          # Content from rej_file_content
    original_source_content: str, # Content from downstream_file_content
    target_filename_for_diff: str, # Original filename for context in diff headers
    vulnerability_id: str,
    failure_details: Dict[str, Any]
):
    """
    Processes a single vulnerability entry using the Gemini agent to generate a corrected diff.
    Returns the diff string or None.
    """
    # Type checks for content fields
    if not isinstance(rej_content, str):
        print(f"Skipping entry for {target_filename_for_diff} in {vulnerability_id} because .rej File Content is not a string.")
        return None
    if not isinstance(original_source_content, str):
        print(f"Skipping entry for {target_filename_for_diff} in {vulnerability_id} because Original Source File content is not a string.")
        return None

    # Existing checks for empty/placeholder content
    if not original_source_content or original_source_content.strip() == "```" or original_source_content.strip() == "":
        print(f"Skipping entry for {target_filename_for_diff} in {vulnerability_id} due to empty or placeholder Original Source File content.")
        return None
    if not rej_content or rej_content.strip() == "```" or rej_content.strip() == "": # Check if .rej is empty; it might be valid if no hunks were rejected but still indicates an issue.
        print(f"Skipping entry for {target_filename_for_diff} in {vulnerability_id} due to empty or placeholder .rej File Content. A .rej file is expected.")
        return None

    dependencies = SupportDependencies(
        rej_file_content=rej_content, 
        original_source_file_content=original_source_content,
    )

    task_prompt = f"""
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
    try:
        import time
        start_time = time.monotonic()
        result = await patch_porter_agent.run(task_prompt)
        end_time = time.monotonic()
        generated_diff = result.data.strip()

        # A simple validation for the diff format
        if not generated_diff.startswith("--- a/"):
            # If it doesn't start with "--- a/", it might be an error message or incorrect output from LLM.
            # It's also possible the LLM used a different placeholder if target_filename_for_diff was complex.
            print(f"⚠️ LLM output for {target_filename_for_diff} in {vulnerability_id} might not be a valid diff. Preview:\n{generated_diff[:250]}...")
            # Depending on strictness, you might return None here or let it pass.
            # For now, let it pass but log the warning.
        else:
            print(f"✅ LLM diff generation successful for: {vulnerability_id} - {target_filename_for_diff}")
        return {
            "downstream_llm_diff_output": generated_diff,
            "runtime_seconds": round(end_time - start_time, 2),
            "token_counts": get_all_token_counts(generated_diff, project=os.getenv("GCP_PROJECT", "neat-resolver-406722"))
        }

    except Exception as e:
        print(f"Error generating diff for {target_filename_for_diff} in {vulnerability_id}: {e}")
        print(f"  Failure details: Patch SHA: {failure_details.get('downstream_patch', 'N/A')}, Repo: {failure_details.get('repo_path', 'N/A')}")
        return None


async def main():
    parser = argparse.ArgumentParser(description="Process vulnerability JSON, generate corrected diffs using an LLM, and output an updated JSON.")
    parser.add_argument("input_json_file_path", help="Path to the input JSON file.")
    parser.add_argument("target_downstream_version", help="The downstream_version to filter by (e.g., '14').")
    parser.add_argument("--output_json_path", "-o", help="Path to save the output JSON file (with generated diffs). Defaults to outputs/approach2_output_diff_android_{version}_{timestamp}.json", default=None)
    
    args = parser.parse_args()

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    output_json_path_to_use = args.output_json_path
    if output_json_path_to_use is None:
        outputs_dir = "outputs/approach2_results" 
        os.makedirs(outputs_dir, exist_ok=True)
        output_json_path_to_use = os.path.join(outputs_dir, f"approach2_output_diff_android_{args.target_downstream_version}_{timestamp_str}.json")

    report_data = {
        "run_timestamp": timestamp_str,
        "target_downstream_version": args.target_downstream_version,
        "input_json_file": args.input_json_file_path,
        "main_output_json_file_with_diffs": output_json_path_to_use,
        "summary": {
            "total_file_conflicts_matching_version": 0,
            "files_attempted_for_llm_diff_generation": 0,
            "files_with_llm_diff_successfully_generated": 0, 
            "files_skipped_pre_llm_call": 0,
            "files_with_llm_diff_generation_errors_or_skipped_in_func": 0,
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

    for vulnerability_item in output_data:
        vulnerability_id = vulnerability_item.get("id", "unknown_vuln_id")
        failures = vulnerability_item.get("failures", [])

        for failure in failures:
            if failure.get("downstream_version") == args.target_downstream_version:
                file_conflicts = failure.get("file_conflicts", [])
                for file_conflict in file_conflicts:
                    report_data["summary"]["total_file_conflicts_matching_version"] += 1
                    
                    rej_content_for_llm = file_conflict.get("rej_file_content")
                    original_source_for_llm = file_conflict.get("downstream_file_content") # This is the "original source" for this task
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
                    generated_diff = await process_single_entry(
                        rej_content=rej_content_for_llm,
                        original_source_content=original_source_for_llm,
                        target_filename_for_diff=target_filename,
                        vulnerability_id=vulnerability_id,
                        failure_details=failure 
                    )

                    if generated_diff is not None: # Even if it's a warning, we store what LLM gave
                        file_conflict.update(generated_diff)
                        report_data["summary"]["files_with_llm_diff_successfully_generated"] += 1
                        report_data["successfully_generated_diffs_log"].append({
                            "vulnerability_id": vulnerability_id, "file_name": target_filename,
                            "patch_sha": failure.get('downstream_patch', 'N/A'),
                            "diff_preview": generated_diff["downstream_llm_diff_output"][:100] + "..." if generated_diff else "None"
                        })
                    else: # This means process_single_entry returned None (e.g. type error, empty content before LLM, or exception during LLM call)
                        report_data["summary"]["files_with_llm_diff_generation_errors_or_skipped_in_func"] += 1
                        report_data["skipped_or_errored_diff_generation_log"].append({
                            "vulnerability_id": vulnerability_id, "file_name": target_filename,
                            "patch_sha": failure.get('downstream_patch', 'N/A'),
                            "reason": "Skipped or errored in diff generation function (see console for details)"
                        })
    
    try:
        with open(output_json_path_to_use, "w", encoding="utf-8") as f_out:
            json.dump(output_data, f_out, indent=4)
        print(f"✅ Main processed JSON (with diffs) successfully saved to '{output_json_path_to_use}'")
    except IOError as e:
        print(f"Error writing main output JSON to '{output_json_path_to_use}': {e}")

    report_dir = os.path.join("outputs", "report")
    os.makedirs(report_dir, exist_ok=True)
    report_filename = os.path.join(report_dir, f"report_diff_{args.target_downstream_version}_{timestamp_str}.json")
    try:
        with open(report_filename, "w", encoding="utf-8") as f_report:
            json.dump(report_data, f_report, indent=4)
        print(f"✅ Summary diff generation report successfully saved to '{report_filename}'")
    except IOError as e:
        print(f"Error writing summary report to '{report_filename}': {e}")

if __name__ == '__main__':
    asyncio.run(main())

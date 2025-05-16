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

@dataclass
class SupportDependencies: 
    diff_content: str
    vulnerable_code_content: str

class GeminiAgent:
    def __init__(self, model_name: str, system_prompt: str):
        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt
        )
        
    async def run(self, prompt: str, deps: Optional[SupportDependencies] = None):
        # The 'deps' parameter is kept for structural similarity but might not be directly used
        # if all content is embedded in the prompt.
        # For this specific use case, 'prompt' will contain all necessary data.
        response = self.model.generate_content(prompt)
        
        class Result:
            def __init__(self, text):
                self.data = text
                
        return Result(response.text)

patch_porter_agent = GeminiAgent(
    model_name='gemini-2.5-pro-preview-05-06', # Or your preferred model
    system_prompt="""
    You are a patch porting agent specializing in resolving merge conflicts and applying diff-like patch content to remediate security vulnerabilities in codebases.
    
    IMPORTANT: Your response must contain ONLY the patched code with no additional comments, explanations, or formatting changes.
    DO NOT include any explanations about what you did, DO NOT include any headers or footers.
    DO NOT change indentation, whitespace, or formatting of the original file unless necessary for the patch.
    Preserve all tabs, spaces, and line endings exactly as they appear in the original file.
    Just output the final patched code file with the security fixes applied and nothing else.
    """
)

async def process_single_entry(
    patch_content: str,
    vuln_code_content: str,
    output_filename_base: str, # output_dir removed
    vulnerability_id: str,
    failure_details: Dict[str, Any]
):
    """
    Processes a single vulnerability entry using the Gemini agent.
    Returns the modified code string or None.
    """
    # Type checks for content fields
    if not isinstance(patch_content, str):
        print(f"Skipping entry for {output_filename_base} in {vulnerability_id} because patch_content (from rej_file_content) is not a string.")
        return None
    if not isinstance(vuln_code_content, str):
        print(f"Skipping entry for {output_filename_base} in {vulnerability_id} because vuln_code_content (from downstream_file_content) is not a string.")
        return None

    # Existing checks for empty/placeholder content (now safe due to type checks above)
    if not vuln_code_content or vuln_code_content.strip() == "```" or vuln_code_content.strip() == "":
        print(f"Skipping entry for {output_filename_base} in {vulnerability_id} due to empty or placeholder vulnerable code content.")
        return None
    if not patch_content or patch_content.strip() == "```" or patch_content.strip() == "":
        print(f"Skipping entry for {output_filename_base} in {vulnerability_id} due to empty or placeholder patch content.")
        return None

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
    try:
        result = await patch_porter_agent.run(task_prompt)
        modified_code = result.data.strip()
        print(f"✅ LLM processing successful for: {vulnerability_id} - {output_filename_base}")
        return modified_code
    except Exception as e:
        print(f"Error processing entry for {output_filename_base} in {vulnerability_id}: {e}")
        # Log more details from failure_details if needed
        print(f"  Failure details: Patch SHA: {failure_details.get('downstream_patch', 'N/A')}, Repo: {failure_details.get('repo_path', 'N/A')}")
        return None


async def main():
    parser = argparse.ArgumentParser(description="Process vulnerability JSON, apply patches using an LLM, and output an updated JSON.")
    parser.add_argument("input_json_file_path", help="Path to the input JSON file.")
    parser.add_argument("target_downstream_version", help="The downstream_version to filter by (e.g., '14').")
    parser.add_argument("--output_json_path", "-o", help="Path to save the output JSON file. Defaults to outputs/output_android_{version}_{timestamp}.json", default=None)
    
    args = parser.parse_args()

    # Generate timestamp once for potential use in both output and report filenames
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    output_json_path_to_use = args.output_json_path
    if output_json_path_to_use is None:
        outputs_dir = "outputs"
        os.makedirs(outputs_dir, exist_ok=True)
        # Use the generated timestamp_str
        output_json_path_to_use = os.path.join(outputs_dir, f"output_android_{args.target_downstream_version}_{timestamp_str}.json")

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

    for vulnerability_item in output_data: # Iterate over the copy
        if not isinstance(vulnerability_item, dict):
            # This check might be redundant if input_data was already validated,
            # but good for safety if structure isn't guaranteed.
            print(f"Skipping non-dictionary item: {vulnerability_item}")
            continue

        vulnerability_id = vulnerability_item.get("id", "unknown_vuln_id")
        failures = vulnerability_item.get("failures", []) # Get failures from the item in the copy

        if not isinstance(failures, list):
            print(f"Skipping vulnerability {vulnerability_id} due to invalid 'failures' format.")
            continue
            
        for failure in failures: # Iterate over failures in the copy
            if not isinstance(failure, dict):
                print(f"Skipping non-dictionary failure in {vulnerability_id}: {failure}")
                continue

            if failure.get("downstream_version") == args.target_downstream_version:
                file_conflicts = failure.get("file_conflicts", []) # Get file_conflicts from failure in copy
                if not isinstance(file_conflicts, list):
                    print(f"Skipping failure in {vulnerability_id} (patch: {failure.get('downstream_patch','N/A')}) due to invalid 'file_conflicts' format.")
                    continue

                for file_conflict in file_conflicts: # Iterate over file_conflicts in copy
                    if not isinstance(file_conflict, dict):
                        print(f"Skipping non-dictionary file_conflict in {vulnerability_id}: {file_conflict}")
                        continue
                    
                    patch_content = file_conflict.get("rej_file_content")
                    vuln_code_content = file_conflict.get("downstream_file_content")
                    original_file_name = file_conflict.get("file_name")

                    report_data["summary"]["total_file_conflicts_matching_version"] += 1
                    
                    skip_reason = None
                    if not original_file_name:
                        skip_reason = "Missing 'file_name'"
                    elif not patch_content:
                        skip_reason = "Missing 'rej_file_content'"
                    elif not vuln_code_content:
                        skip_reason = "Missing 'downstream_file_content'"

                    if skip_reason:
                        print(f"Skipping file_conflict in {vulnerability_id} (file: {original_file_name or 'N/A'}) due to: {skip_reason}.")
                        report_data["summary"]["files_skipped_pre_llm_call"] += 1
                        report_data["skipped_or_errored_files_log"].append({
                            "vulnerability_id": vulnerability_id,
                            "file_name": original_file_name or "Unknown",
                            "patch_sha": failure.get('downstream_patch', 'N/A'),
                            "reason": skip_reason
                        })
                        continue
                    
                    report_data["summary"]["files_attempted_for_llm"] += 1
                    modified_code = await process_single_entry(
                        patch_content=patch_content,
                        vuln_code_content=vuln_code_content,
                        output_filename_base=original_file_name,
                        vulnerability_id=vulnerability_id,
                        failure_details=failure 
                    )

                    if modified_code is not None:
                        file_conflict["downstream_llm_output"] = modified_code
                        report_data["summary"]["files_successfully_processed_by_llm"] += 1
                        report_data["successfully_processed_files_log"].append({
                            "vulnerability_id": vulnerability_id,
                            "file_name": original_file_name,
                            "patch_sha": failure.get('downstream_patch', 'N/A')
                        })
                    else:
                        report_data["summary"]["files_skipped_or_errored_in_processing_function"] += 1
                        # The reason is printed by process_single_entry (type error, empty content, or LLM error)
                        report_data["skipped_or_errored_files_log"].append({
                            "vulnerability_id": vulnerability_id,
                            "file_name": original_file_name,
                            "patch_sha": failure.get('downstream_patch', 'N/A'),
                            "reason": "Skipped or errored in processing function (see console for details)"
                        })
    
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
    report_filename = os.path.join(report_dir, f"{args.target_downstream_version}_{timestamp_str}.json")
    try:
        with open(report_filename, "w", encoding="utf-8") as f_report:
            json.dump(report_data, f_report, indent=4)
        print(f"✅ Summary report successfully saved to '{report_filename}'")
    except IOError as e:
        print(f"Error writing summary report to '{report_filename}': {e}")

if __name__ == '__main__':
    asyncio.run(main())

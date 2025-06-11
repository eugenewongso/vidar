import pandas as pd
import numpy as np
import os
import json # For formatting data for the LLM if needed
import asyncio # Added for async agent

# --- Components from approach2_blind_retry.py ---
from dotenv import load_dotenv
import google.generativeai as genai
from typing import List, Optional, Any # Added List, Optional, Any

# Load environment variables from .env file
load_dotenv()

class APIKeyRotator:
    def __init__(self, api_keys: List[str]):
        if not api_keys or api_keys == [""]:
            raise ValueError("API keys list cannot be empty.")
        self.api_keys = api_keys
        self.index = 0

    def get_current_key(self):
        return self.api_keys[self.index]

    def rotate_key(self):
        self.index = (self.index + 1) % len(self.api_keys)
        print(f"🔄 Rotating to new API key index {self.index}")
        return self.get_current_key()

class GeminiAgent: # Copied from approach2_blind_retry.py
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

    async def run(self, prompt: str, deps: Optional[Any] = None): # Matched signature, deps won\'t be used here
        for attempt_key_rotation in range(len(self.key_rotator.api_keys) + 1): # +1 to try all keys once
            try:
                response = self.model.generate_content(prompt)
                token_count = None
                if hasattr(response, "usage_metadata"):
                    token_count = getattr(response.usage_metadata, "total_token_count", None)
                # Ensure response.text is accessed correctly
                response_text = ""
                if response.parts:
                    response_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
                elif hasattr(response, 'text'):
                     response_text = response.text
                else: # Fallback for unexpected structures, or if prompt feedback is present
                    if response.prompt_feedback and response.prompt_feedback.block_reason:
                        raise RuntimeError(f"LLM call blocked: {response.prompt_feedback.block_reason_message}")


                return type("Result", (), {"data": response_text, "token_count": token_count})
            except Exception as e:
                error_message = str(e).lower()
                # Added "internal error" as per user's file context
                if "quota" in error_message or "rate limit" in error_message or "api key not valid" in error_message or "internal error" in error_message:
                    print(f"⚠️ API error encountered: {e}. Rotating key.")
                    self.key_rotator.rotate_key()
                    self._configure_genai()
                    if attempt_key_rotation == len(self.key_rotator.api_keys): # Last attempt after rotating through all keys
                        raise RuntimeError(f"All API keys failed. Last error: {e}")
                else: # Non-retryable error
                    raise e
        raise RuntimeError("All API keys exhausted or failed after multiple retries.")


# --- Configuration ---
DEFAULT_CSV_PATH = "meta_prompting_data_3.csv"

# --- Helper Functions for Data Loading and Summarization ---

def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Loads the CSV data and performs initial preparation."""
    if not os.path.exists(csv_path):
        print(f"❌ Error: CSV file not found at {csv_path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Successfully loaded {csv_path} with {len(df)} rows and {len(df.columns)} columns.")
        
        bool_cols = ['fc_llm_output_valid_overall', 'attempt_format_valid', 
                     'attempt_apply_valid', 'attempt_valid_overall']
        for col in bool_cols:
            if col in df.columns:
                 # Handle potential NaN before mapping and ensure it defaults to False
                df[col] = df[col].fillna(False).astype(str).str.lower().map({'true': True, 'false': False}).astype(bool)
        
        numeric_cols = ['original_patch_total_hunks_for_file', 'rej_file_reported_hunk_count',
                        'rej_file_actual_hunk_count', 'rej_file_total_lines', 
                        'rej_file_added_lines', 'rej_file_removed_lines',
                        'ground_truth_hunk_count', 'ground_truth_total_lines',
                        'ground_truth_added_lines', 'ground_truth_removed_lines',
                        'fc_attempts_made_overall', 'fc_runtime_total_sec',
                        'attempt_number', 'attempt_runtime_sec']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Ensure prompt columns are strings, handling potential float if all were NaN
        if 'system_prompt' in df.columns:
            df['system_prompt'] = df['system_prompt'].astype(str).fillna('MISSING_PROMPT')
        if 'base_task_prompt' in df.columns:
            df['base_task_prompt'] = df['base_task_prompt'].astype(str).fillna('MISSING_PROMPT')

        return df
    except Exception as e:
        print(f"❌ Error loading or preparing data from {csv_path}: {e}")
        return pd.DataFrame()

def generate_data_summary_for_llm(df: pd.DataFrame, max_prompts_to_detail=3, max_errors_to_detail=3) -> str:
    """Generates a textual summary of the data to be fed to the analyzer LLM."""
    if df.empty:
        return "No data available for analysis."

    summary_parts = []

    # Overall Performance
    total_file_conflicts_processed = df[['vulnerability_id', 'downstream_version', 'target_file_name']].drop_duplicates().shape[0]
    successful_file_conflicts = df[df['fc_llm_output_valid_overall'] == True][
        ['vulnerability_id', 'downstream_version', 'target_file_name']
    ].drop_duplicates().shape[0]
    overall_fc_success_rate = 0
    if total_file_conflicts_processed > 0:
        overall_fc_success_rate = (successful_file_conflicts / total_file_conflicts_processed) * 100
    
    summary_parts.append(f"Overall Performance:\n"
                         f"- Total unique file conflicts processed: {total_file_conflicts_processed}\n"
                         f"- Successfully resolved file conflicts: {successful_file_conflicts}\n"
                         f"- Overall success rate per file_conflict: {overall_fc_success_rate:.2f}%")

    # Prompt Performance (Worst N)
    df['file_conflict_id'] = df['vulnerability_id'].astype(str) + "_" + \
                              df['downstream_version'].astype(str) + "_" + \
                              df['target_file_name'].astype(str)
    prompt_groups = df.groupby(['system_prompt', 'base_task_prompt'])
    prompt_performance = []
    for (system_p, base_p), group in prompt_groups:
        num_unique_fc = group['file_conflict_id'].nunique()
        if num_unique_fc == 0: continue
        successful_fc_in_group = group[group['fc_llm_output_valid_overall'] == True]['file_conflict_id'].nunique()
        success_rate = (successful_fc_in_group / num_unique_fc) * 100
        prompt_performance.append({
            "system_prompt_short": system_p[:100] + "...",
            "base_task_prompt_short": base_p[:150] + "...",
            "success_rate": success_rate,
            "unique_fc_count": num_unique_fc
        })
    
    sorted_prompts = sorted(prompt_performance, key=lambda x: x["success_rate"])
    summary_parts.append("\nWorst Performing Prompt Combinations (by success rate):")
    for i, p_data in enumerate(sorted_prompts[:max_prompts_to_detail]):
        summary_parts.append(f"  {i+1}. System: \'{p_data['system_prompt_short']}\', Base: \'{p_data['base_task_prompt_short']}\'\n"
                             f"     Success Rate: {p_data['success_rate']:.2f}% over {p_data['unique_fc_count']} unique file conflicts.")

    # Common Errors
    failed_attempts_df = df[df['attempt_valid_overall'] == False]
    if not failed_attempts_df.empty:
        summary_parts.append("\nMost Common Errors on Failed Attempts:")
        
        top_format_errors = failed_attempts_df['attempt_format_error'].value_counts().nlargest(max_errors_to_detail)
        summary_parts.append("  Format Errors:")
        for error, count in top_format_errors.items():
            if pd.notna(error) and error.strip() and "skip" not in error.lower() and "no llm" not in error.lower():
                summary_parts.append(f"    - \"{error.strip()[:100]}...\": {count} times")
        
        top_apply_errors = failed_attempts_df['attempt_apply_error'].value_counts().nlargest(max_errors_to_detail)
        summary_parts.append("  Apply Errors:")
        for error, count in top_apply_errors.items():
            if pd.notna(error) and error.strip() and "skip" not in error.lower() and "no llm" not in error.lower():
                summary_parts.append(f"    - \"{error.strip()[:100]}...\": {count} times")
    
    # Note about data structure
    summary_parts.append("\nNote: The full CSV contains more details per attempt, including input characteristics like rej_file_hunk_count, ground_truth_hunk_count, etc.")

    return "\n".join(summary_parts)


async def run_llm_powered_meta_analysis(csv_path: str, llm_agent: GeminiAgent): # Changed type hint
    """
    Main function to load data, generate summary, and get suggestions from LLM.
    """
    df = load_and_prepare_data(csv_path)
    if df.empty:
        print(f"❌ No data loaded from {csv_path}. Cannot perform LLM meta-analysis.")
        return

    data_summary_for_llm = generate_data_summary_for_llm(df)

    meta_meta_prompt = f"""
You are an expert Prompt Engineering Analyst. Your task is to analyze a summary of experimental results 
from an LLM tasked with generating software patches (unified diffs) to fix rejected patch hunks (.rej files).
The goal is to improve the `system_prompt` and `base_task_prompt` used to instruct that patch-generating LLM.

Here is a summary of the data from multiple runs:

{data_summary_for_llm}

Based *only* on the summary provided above, please provide:
1.  Your top 2-3 most critical observations about what might be going wrong or right.
2.  Actionable suggestions for modifying the `system_prompt` or `base_task_prompt` to improve performance. 
    For each suggestion, briefly explain your reasoning based on the data summary.
    Be specific if possible (e.g., "Consider adding X to the prompt if Y error is common").
3.  If there are specific prompt combinations mentioned as low-performing, suggest a hypothesis for why,
    and how they might be improved.

Focus on clarity and conciseness in your analysis and suggestions. 
The full dataset contains more granular details like `rej_file_actual_hunk_count`, `ground_truth_hunk_count`,
`attempt_format_error`, `attempt_apply_error` for each attempt, which you can generally refer to
as factors that might be influencing the summarized results you see.
"""
    print("\n--- Sending Data to LLM Analyzer (Actual Call) ---")
    print(f"User Prompt for Analyzer LLM (first 500 chars):\n{meta_meta_prompt[:500]}...")
    print("----------------------------------------------------")

    try:
        llm_response_obj = await llm_agent.run(meta_meta_prompt) # Call the actual agent's run method
        llm_suggestions = llm_response_obj.data # Access the .data attribute for text
    except Exception as e:
        print(f"❌❌❌ LLM call for meta-analysis FAILED: {e}")
        llm_suggestions = "LLM analysis failed. Check logs."


    print("\n\n--- LLM Analyzer Agent Suggestions ---")
    print(llm_suggestions)
    print("------------------------------------")
    print("\nReview these LLM-generated suggestions to manually refine your prompts in approach2_blind_retry.py.")


if __name__ == '__main__':
    # --- Configuration ---
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv_file_path = os.path.join(current_script_dir, DEFAULT_CSV_PATH)

    # --- Initialize your LLM Agent ---
    analyzer_llm_system_prompt = "You are an expert data analyst and prompt engineering consultant. Your goal is to help improve LLM prompts based on performance data."
    
    google_api_keys_str = os.getenv("GOOGLE_API_KEYS")
    if not google_api_keys_str:
        print("❌ Error: GOOGLE_API_KEYS environment variable not set.")
        print("Please set it in your .env file or environment.")
        # exit(1) # Or handle more gracefully depending on desired behavior
        # For now, we'll let it proceed and GeminiAgent init will raise error if keys are truly empty.
    
    api_keys_list = [key.strip() for key in google_api_keys_str.split(',') if key.strip()] if google_api_keys_str else []

    if not api_keys_list:
        print("❌ Error: No valid API keys found in GOOGLE_API_KEYS.")
        # exit(1) # Or handle as above. GeminiAgent constructor will raise error.
        # For the script to be runnable even if keys are missing for a dry run (though it won't work):
        key_rotator = APIKeyRotator(["DUMMY_KEY_IF_NONE_FOUND"]) # Prevent crash if empty, actual call will fail
    else:
        key_rotator = APIKeyRotator(api_keys_list)
    
    # Use a model suitable for analysis, e.g., "gemini-1.5-pro-latest"
    # The user's approach2_blind_retry.py used "gemini-1.5-pro-preview-05-06"
    # Let's use a generally available strong model like "gemini-1.5-pro-latest" or "gemini-pro"
    analyzer_agent = GeminiAgent(
        model_name="gemini-2.5-pro-preview-05-06", # Or "gemini-pro" or your preferred analysis model
        system_prompt=analyzer_llm_system_prompt,
        key_rotator=key_rotator
    )
    
    # --- Run the LLM-powered meta-analysis ---
    # Make sure to run this with `python llm_meta_analyzer.py`
    # And ensure your GOOGLE_API_KEYS are set in your .env file or environment.
    
    async def actual_run():
        # Check if API keys are truly available before attempting the run
        if not api_keys_list:
            print("⏩ Skipping LLM analysis run as no valid API keys were configured.")
            print("Please set GOOGLE_API_KEYS in your .env file.")
            return
        await run_llm_powered_meta_analysis(default_csv_file_path, analyzer_agent)

    asyncio.run(actual_run()) # Uncomment to run
    
    print("LLM Meta-Prompt Analyzer script defined.")
    # print(f"To run, uncomment the asyncio.run(actual_run()) call in the __main__ block.")
    # print(f"Ensure your GOOGLE_API_KEYS are set in your .env file or environment.")
    # print(f"Example command: python {os.path.basename(__file__)}") 
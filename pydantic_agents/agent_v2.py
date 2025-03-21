# Input vuln file, diff file
# Output: whole (modified) 
import asyncio
import os
import datetime
import sys
import logfire
from dataclasses import dataclass
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Load environment variables from .env file
load_dotenv()

# Configure Logfire for logging (if token is available)
logfire.configure(send_to_logfire='if-token-present')

# Fetch API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# print(OPENAI_API_KEY)

# Ensure API key is set
if not OPENAI_API_KEY:
    raise ValueError("Missing API key. Please add OPENAI_API_KEY to your .env file.")

@dataclass
class SupportDependencies:
    diff_file: str
    vulnerable_codebase: str

# Configure OpenAI model with API key
model = OpenAIModel('o1', provider=OpenAIProvider(api_key=OPENAI_API_KEY))

# Define the agent using the updated method
patch_porter_agent = Agent(
    model,
    system_prompt="""
    You are a patch porting agent specializing in resolving merge conflicts and applying diff files to remediate security vulnerabilities in codebases.
    
    IMPORTANT: Your response must contain ONLY the patched code with no additional comments, explanations, or formatting changes.
    DO NOT include any explanations about what you did, DO NOT include any headers or footers.
    DO NOT change indentation, whitespace, or formatting of the original file unless necessary for the patch.
    Preserve all tabs, spaces, and line endings exactly as they appear in the original file.
    Just output the final patched code file with the security fixes applied and nothing else.
    """,
    deps_type=SupportDependencies
)

async def run_patch_porter_agent(patch_file_path=None, vuln_file_path=None, output_suffix=None):
    """
    Runs the patch porter agent with provided dependencies and saves the output to a file.
    
    Args:
        patch_file_path (str, optional): Path to the patch file. Defaults to "patch.diff".
        vuln_file_path (str, optional): Path to the vulnerable file. Defaults to "vulnerable_file.c".
        output_suffix (str, optional): Suffix to add to the output filename. Defaults to None.
    """
    # Define paths to the input files - use defaults if not provided
    diff_file_path = patch_file_path or "patch.diff"
    vulnerable_codebase_path = vuln_file_path or "vulnerable_file.c"
    
    # Get the filename for the output
    output_basename = os.path.basename(vulnerable_codebase_path) if vuln_file_path else "output_gpt.c"

    # Read file contents safely
    try:
        with open(diff_file_path, "r", encoding="utf-8") as file:
            diff_file_content = file.read()
    except FileNotFoundError:
        print(f"Error: The file '{diff_file_path}' was not found.")
        return

    try:
        with open(vulnerable_codebase_path, "r", encoding="utf-8") as file:
            vulnerable_codebase_content = file.read()
    except FileNotFoundError:
        print(f"Error: The file '{vulnerable_codebase_path}' was not found.")
        return

    # Construct dependencies with actual content
    dependencies = SupportDependencies(
        diff_file=diff_file_content,
        vulnerable_codebase=vulnerable_codebase_content,
    )

    # Task-specific instructions (formatted with runtime values)
    task_prompt = f"""
        You will be provided with the following:

        Diff File:
        {dependencies.diff_file}

        Vulnerable Codebase:
        {dependencies.vulnerable_codebase}

        Instructions:

        1. Carefully analyze the provided diff_file and vulnerable_codebase.
        2. Identify any merge conflicts that prevent the diff_file from being cleanly applied to the vulnerable_codebase.
        3. Resolve the merge conflicts while maintaining the integrity and functionality of the vulnerable_codebase.
        4. Apply the resolved diff_file to the vulnerable_codebase.
        5. Ensure the patched code remains functional and does not introduce new issues.
        6. Provide the patched codebase as your final output. 

        STRICT REQUIREMENTS:
        - Output **ONLY** the complete patched code with no formatting changes.
        - Do NOT include any explanations, comments, or anything else.
        - The response must be **exactly** the final patched file with the security fixes applied.
    """

    # Run the agent with the task-specific prompt
    result = await patch_porter_agent.run(task_prompt, deps=dependencies)

    # Ensure output is clean and strictly contains the modified code
    modified_code = result.data.strip()

    # Create the output directory if it doesn't exist
    output_dir = "outputs/output_gpt_no_desc"
    os.makedirs(output_dir, exist_ok=True)

    # Generate a filename with a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Add suffix to output filename if provided
    if output_suffix:
        output_filename = os.path.join(output_dir, f"{timestamp}_{output_suffix}")
    else:
        output_filename = os.path.join(output_dir, f"{timestamp}_output_gpt.c")

    # Write the patched file to the new timestamped filename
    with open(output_filename, "w", encoding="utf-8") as patched_file:
        patched_file.write(modified_code)

    print(f"✅ Patched file successfully saved to '{output_filename}'")
    return output_filename

# Entry point to execute the async function
if __name__ == '__main__':
    # Check if command line arguments are provided
    if len(sys.argv) >= 3:
        patch_file = sys.argv[1]
        vuln_file = sys.argv[2]
        output_suffix = sys.argv[3] if len(sys.argv) > 3 else None
        asyncio.run(run_patch_porter_agent(patch_file, vuln_file, output_suffix))
    else:
        asyncio.run(run_patch_porter_agent())
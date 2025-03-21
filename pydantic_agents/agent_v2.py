import os
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_vertex import GoogleVertexProvider
from dotenv import load_dotenv
from unidiff import PatchSet

# Load environment variables (including service account JSON path if used)
load_dotenv()

# Set up the Google Vertex AI provider
GCP_REGION = "us-central1"
GCP_PROJECT = "vidar-448622"  # Replace with your actual GCP project

# Option 1: Use application default credentials (Recommended for GCP-based deployments)
try:
    provider = GoogleVertexProvider(region=GCP_REGION)
except Exception as e:
    print(f"⚠️ Warning: Could not use application default credentials. {e}")
    provider = None

# Option 2: Use service account JSON if default credentials fail
service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if service_account_path and provider is None:
    try:
        provider = GoogleVertexProvider(service_account_file=service_account_path, region=GCP_REGION)
    except Exception as e:
        print(f"❌ Error: Could not load service account file. {e}")
        exit(1)

# Ensure provider is set correctly
if provider is None:
    raise ValueError("❌ No valid Google Vertex AI credentials found.")

@dataclass
class MergeDeps:
    """Dependencies for the merge conflict resolution agent."""
    api_key: str  # Not used for Vertex AI, but required for Pydantic AI

# Define Gemini model using Vertex AI
model = GeminiModel(
    "gemini-2.0-pro-exp-02-05",  # Use Gemini 2.0 Pro (Experimental version)
    provider=provider
)

# Initialize AI Agent
merge_agent = Agent(model, system_prompt=(
    "You are a skilled AI assistant specializing in resolving inline merge conflicts in code.\n"
    "Analyze conflicting changes, compare them with the surrounding code, and produce a clean, logically sound patch in unified diff format.\n"
    "Ensure the patch applies cleanly, includes all intended changes, and uses Unix-style line endings.\n"
    "Use the `check_diff_format` tool to verify the patch. If the patch is invalid, retry until it is correct."
))

@merge_agent.tool
async def check_diff_format(ctx: RunContext[MergeDeps], diff_content: str) -> bool:
    """Validates the generated patch using unidiff.PatchSet instead of `patch --check`."""
    try:
        PatchSet.from_string(diff_content)
        print("✅ Patch format is valid.")
        return True
    except Exception as e:
        print(f"⚠️ Patch format is invalid. Retrying...\nError: {e}")
        return False


@merge_agent.tool
async def resolve_conflict(
    ctx: RunContext[MergeDeps], inline_merge_conflict: str, code_context: str, rejected_patch: str
) -> str:
    """AI-powered merge conflict resolution tool. Returns a valid patch."""
    
    prompt = f"""
    **Inline Merge Conflict:**
    ```c
    {inline_merge_conflict}
    ```

    **Code Context:**
    ```c
    {code_context}
    ```

    **Rejected Patch (.rej file from GNU patch):**
    ```diff
    {rejected_patch}
    ```

    Your task:
    - Resolve the merge conflict by comparing the rejected patch with the inline conflict and surrounding code.
    - Produce a final patch in **standard unified diff format**.
    - Ensure the patch applies **cleanly without errors**.
    - **Remove conflict markers** (`<<<<<<<`, `=======`, `>>>>>>>`).
    - Use **Unix-style line endings** (`\n`).
    - Do **NOT** add extra explanations, only output the corrected patch.
    """

    print("🟡 Sending request to AI agent...")
    response = await merge_agent.run(prompt)

    if not response or not response.data.strip():
        print("❌ AI response was empty. Skipping.")
        return None

    return response.data.strip() + "\n"  # Ensure Unix-style newline

async def read_file(file_path: str) -> str:
    """Reads and returns the contents of a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"❌ Error: File not found - {file_path}")
        return ""
    except Exception as e:
        print(f"❌ Error reading file {file_path}: {e}")
        return ""

class PatchAgentV2:

    def __init__(self):
        self.agent = merge_agent

    async def generate_fixed_patch(self, raw_merge_conflict_path: str, code_context_path: str, rejected_patch_path: str, output_file: str):
        """Runs the AI agent to resolve the merge conflict and ensure the final patch is valid."""
        # Read contents from the provided file paths
        raw_merge_conflict = await read_file(raw_merge_conflict_path)
        code_context = await read_file(code_context_path)  
        rejected_patch = await read_file(rejected_patch_path)

        if not raw_merge_conflict or not code_context or not rejected_patch:
            print("❌ Error: One or more input files are empty or could not be read.")
            return

        
        # ADDED CARO
        max_retries = 5
        delay = 5  # Start with a 5-second delay


        for attempt in range(max_retries):
            print(f"🔄 Attempt {attempt + 1} of {max_retries}...")

            # AI resolves the conflict using explicit code context
            fixed_patch = await resolve_conflict(None, raw_merge_conflict, code_context, rejected_patch)

            if not fixed_patch:
                print("❌ AI failed to generate a patch. Retrying...")
                # ADDED CARO
                await asyncio.sleep(delay)  # Delay before retrying
                delay *= 2  # Exponential backoff
                continue

            # Debug log before calling check_diff_format
            print(f"🛠️ Checking patch format using check_diff_format (Attempt {attempt + 1})...")

            is_valid = await check_diff_format(None, fixed_patch)

            if is_valid:
                print(f"✅ Patch validated successfully on attempt {attempt + 1}. Writing to file.")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(fixed_patch)
                print(f"✅ Successfully generated a clean patch: {output_file}")
                return
            else:
                print(f"❌ Patch validation failed on attempt {attempt + 1}. Retrying...")
                # ADDED CARO
                await asyncio.sleep(delay)
                delay *= 2  # Increase delay for next retry

        print("❌ Failed to generate a valid patch after multiple attempts.")


# Example usage
async def main():
    raw_merge_conflict_path = "merge_conflict.txt"  # Path to inline merge conflict file
    code_context_path = "code_context.txt"  # Separate file for code context
    rejected_patch_path = "rejected_patch.rej"  # Path to rejected patch file
    output_file = "resolved-fix.diff"

    await generate_fixed_patch(raw_merge_conflict_path, code_context_path, rejected_patch_path, output_file)

if __name__ == "__main__":
    asyncio.run(main())
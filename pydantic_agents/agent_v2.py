import os
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
from dotenv import load_dotenv
from unidiff import PatchSet

# Load environment variables (including service account JSON path if used)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

@dataclass
class MergeDeps:
    """Dependencies for the merge conflict resolution agent."""
    api_key: str  # Required for Pydantic AI

class PatchAgentV1:
    """AI-powered merge conflict resolution agent for generating valid diff patches."""

    def __init__(self):
        """Initialize the AI agent with the Gemini model."""
        self.model = GeminiModel(
            "gemini-2.0-pro-exp-02-05",  # Use Gemini 2.0 Pro (Experimental version)
            api_key=GEMINI_API_KEY  # Pass API Key directly
        )

        # Initialize AI Agent
        self.agent = Agent(self.model, system_prompt=(
            "You are a skilled AI assistant specializing in resolving inline merge conflicts in code.\n"
            "Analyze conflicting changes, compare them with the surrounding code, and produce a clean, logically sound patch in unified diff format.\n"
            "Ensure the patch applies cleanly, includes all intended changes, and uses Unix-style line endings.\n"
            "Use the `check_diff_format` tool to verify the patch. If the patch is invalid, retry until it is correct."
        ))

        # Register tools
        self.agent.tool(self.check_diff_format)
        self.agent.tool(self.resolve_conflict)

    async def check_diff_format(self, ctx: RunContext[MergeDeps], diff_content: str) -> bool:
        """Validates the generated patch using unidiff.PatchSet instead of `patch --check`."""
        try:
            PatchSet.from_string(diff_content)
            print("✅ Patch format is valid.")
            return True
        except Exception as e:
            print(f"⚠️ Patch format is invalid. Retrying...\nError: {e}")
            return False

    async def resolve_conflict(self, ctx: RunContext[MergeDeps], inline_merge_conflict: str, code_context: str, rejected_patch: str) -> str:
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
        response = await self.agent.run(prompt)

        if not response or not response.data.strip():
            print("❌ AI response was empty. Skipping.")
            return None

        return response.data.strip() + "\n"  # Ensure Unix-style newline

    async def read_file(self, file_path: str) -> str:
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

    async def generate_fixed_patch(self, raw_merge_conflict_path: str, code_context_path: str, rejected_patch_path: str, output_file: str):
        """Runs the AI agent to resolve the merge conflict and ensure the final patch is valid."""
        # Read contents from the provided file paths
        raw_merge_conflict = await self.read_file(raw_merge_conflict_path)
        code_context = await self.read_file(code_context_path)  
        rejected_patch = await self.read_file(rejected_patch_path)

        if not raw_merge_conflict or not code_context or not rejected_patch:
            print("❌ Error: One or more input files are empty or could not be read.")
            return None

        max_retries = 3  # Limit retries to avoid infinite loops
        for attempt in range(max_retries):
            print(f"🔄 Attempt {attempt + 1} of {max_retries}...")

            # AI resolves the conflict using explicit code context
            fixed_patch = await self.resolve_conflict(None, raw_merge_conflict, code_context, rejected_patch)

            if not fixed_patch:
                print("❌ AI failed to generate a patch. Retrying...")
                continue

            # Debug log before calling check_diff_format
            print(f"🛠️ Checking patch format using check_diff_format (Attempt {attempt + 1})...")

            is_valid = await self.check_diff_format(None, fixed_patch)

            if is_valid:
                print(f"✅ Patch validated successfully on attempt {attempt + 1}. Writing to file.")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(fixed_patch)
                print(f"✅ Successfully generated a clean patch: {output_file}")
                return output_file
            else:
                print(f"❌ Patch validation failed on attempt {attempt + 1}. Retrying...")

        print("❌ Failed to generate a valid patch after multiple attempts.")
        return None

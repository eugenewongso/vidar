import os
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_vertex import GoogleVertexProvider
from dotenv import load_dotenv
from unidiff import PatchSet
from pathlib import Path
import re

# Load environment variables from a .env file
load_dotenv()

# Define Google Cloud Platform (GCP) region and project
GCP_REGION = "us-central1"
GCP_PROJECT = "neat-resolver-406722"

# Attempt to initialize the Google Vertex AI provider using default credentials
try:
    provider = GoogleVertexProvider(region=GCP_REGION)
except Exception as e:
    print(f"⚠️ Warning: Could not use application default credentials. {e}")
    provider = None

# If default credentials fail, try using a service account file
service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if service_account_path and provider is None:
    try:
        provider = GoogleVertexProvider(service_account_file=service_account_path, region=GCP_REGION)
    except Exception as e:
        print(f"❌ Error: Could not load service account file. {e}")
        exit(1)

# Raise an error if no valid credentials are found
if provider is None:
    raise ValueError("❌ No valid Google Vertex AI credentials found.")


# Function to fix hunk headers in a unified diff patch
def fix_hunk_headers(patch_text: str) -> str:
    """
    Fix the headers of hunks in a unified diff patch to ensure they match the actual content.

    Args:
        patch_text (str): The patch content as a string.

    Returns:
        str: The patch content with corrected hunk headers.
    """
    lines = patch_text.splitlines()
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('@@'):  # Identify hunk headers
            header_match = re.match(r'@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@', line)
            if not header_match:
                fixed_lines.append(line)
                i += 1
                continue

            # Extract old and new line numbers and counts from the header
            old_start, old_count, new_start, new_count = header_match.groups()
            old_start = int(old_start)
            new_start = int(new_start)
            old_count = int(old_count) if old_count else 1
            new_count = int(new_count) if new_count else 1

            # Collect the hunk body
            hunk_body = []
            i += 1
            while i < len(lines) and not lines[i].startswith('@@') and not lines[i].startswith('---') and not lines[i].startswith('+++'):
                hunk_body.append(lines[i])
                i += 1

            # Count actual lines in the hunk body
            actual_old = sum(1 for l in hunk_body if l.startswith('-') or l.startswith(' '))
            actual_new = sum(1 for l in hunk_body if l.startswith('+') or l.startswith(' '))

            # Create a corrected hunk header
            fixed_header = f"@@ -{old_start},{actual_old} +{new_start},{actual_new} @@"
            fixed_lines.append(fixed_header)
            fixed_lines.extend(hunk_body)
        else:
            fixed_lines.append(line)
            i += 1

    return '\n'.join(fixed_lines) + '\n'


@dataclass
class MergeDeps:
    """
    Data class to hold dependencies for the merge process.
    """
    api_key: str

model = GeminiModel("gemini-2.5-pro-preview-03-25", provider=provider)


merge_agent = Agent(model, system_prompt=(
    "You are a skilled AI assistant specializing in resolving inline merge conflicts in code.\n"
    "Analyze conflicting changes, compare them with the surrounding code, and produce a clean, logically sound patch in unified diff format.\n"
    "Ensure the patch applies cleanly, includes all intended changes, and uses Unix-style line endings.\n"
    "Use the `check_diff_format` tool to verify the patch. If the patch is invalid, retry until it is correct."
))

@merge_agent.tool
async def check_diff_format(ctx: RunContext[MergeDeps], diff_content: str) -> bool:
    """
    Validate the format of a unified diff patch.

    Args:
        ctx (RunContext[MergeDeps]): The runtime context for the merge process.
        diff_content (str): The patch content to validate.

    Returns:
        bool: True if the patch format is valid, False otherwise.
    """
    try:
        PatchSet.from_string(diff_content)
        print("✅ Patch format is valid.")
        return True
    except Exception as e:
        print(f"⚠️ Patch format is invalid. Retrying...\nError: {e}")
        return False
    
async def resolve_conflict(
    ctx: RunContext[MergeDeps],
    context_block: str = "",
    rej_content: str = "",
    inline_content: str = "",
    ast_content: str = ""
) -> str:
    """
    Resolve a merge conflict using the AI agent.

    Args:
        ctx (RunContext[MergeDeps]): The runtime context for the merge process.
        context_block (str): Surrounding code context for the conflict.
        rej_content (str): Rejected patch content.
        inline_content (str): Inline merge conflict content.
        ast_content (str): Abstract Syntax Tree (AST) content.

    Returns:
        str: A valid unified diff patch, or None if resolution fails.
    """
    inferred_context_lines = len(context_block.strip().splitlines()) // 2 if context_block.strip() else 0

    if ast_content.strip():
        context_description = f"""\
Here is the complete function context and dependencies extracted from the AST:
{ast_content}"""
    else:
        context_description = f"""\
Here are {inferred_context_lines} lines of surrounding code above and below the merge conflict:
{context_block if context_block.strip() else '[Not Provided]'}"""

    prompt = f"""
You need to resolve a merge conflict in a code file.

{context_description}

---

Here is the rejected patch hunk (.rej):
{rej_content if rej_content.strip() else '[Not Provided]'}

---

Here is the inline merge conflict (with <<<<<<< markers):
{inline_content if inline_content.strip() else '[Not Provided]'}

---

Requirements:
1. Resolve the conflict logically using the inputs above.
2. Output ONLY a valid unified diff format patch — no markdown formatting or explanations.
3. Ensure the patch applies cleanly.
"""

    print("🟡 Sending request to AI agent...")
    response = await merge_agent.run(prompt)

    if not response or not response.data.strip():
        print("❌ AI response was empty. Skipping.")
        return None

    patch = response.data.strip()
    patch = patch.replace("```diff", "").replace("```", "").strip() + "\n"
    return fix_hunk_headers(patch)

async def resolve_conflict_per_reject(
    ctx: RunContext[MergeDeps],
    context_block_with_inline: str = "",
    rej_content: str = ""
) -> str:
    """
    Resolve a merge conflict using a combined context block that includes inline merge markers.

    Args:
        ctx (RunContext[MergeDeps]): The runtime context for the merge process.
        context_block_with_inline (str): Code block including context and inline merge markers.
        rej_content (str): Rejected patch content (.rej).

    Returns:
        str: A valid unified diff patch or None if resolution fails.
    """
    prompt = f"""
You are an AI engineer tasked with resolving merge conflicts inside source code files.

You are given one block of code that includes both the surrounding context and an inline merge conflict marked with <<<<<<<, =======, and >>>>>>>.

---

Here is the rejected patch hunk (.rej):
{rej_content if rej_content.strip() else '[Not Provided]'}

---

Here is the full context block with inline merge conflict markers:
{context_block_with_inline if context_block_with_inline.strip() else '[Not Provided]'}

---

Instructions:
1. Analyze the conflict and surrounding context.
2. Use the rejected patch to better understand the intended change.
3. Decide which lines to keep, merge, or modify.
4. Output ONLY a valid **unified diff patch** that resolves the conflict cleanly.
5. DO NOT return anything except the patch (no prose, no markdown).

Be concise, correct, and ensure the patch applies.
"""

    print("🟡 Sending per-reject request to AI agent...")
    response = await merge_agent.run(prompt)

    if not response or not response.data.strip():
        print("❌ AI response was empty. Skipping.")
        return None

    patch = response.data.strip()
    patch = patch.replace("```diff", "").replace("```", "").strip() + "\n"
    return fix_hunk_headers(patch)


async def process_merge_conflicts(kernel_root: str, conflict_dir: str, output_dir: str):
    """
    Process merge conflicts in a directory and generate patches.

    Args:
        kernel_root (str): Root directory of the kernel source.
        conflict_dir (str): Directory containing merge conflict files.
        output_dir (str): Directory to save generated patches.
    """
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(conflict_dir):
        if not file.endswith("_inline.txt"):
            continue

        inline_file = os.path.join(conflict_dir, file)
        base_name = file.replace("_conflict.txt_inline.txt", "")
        
        # Find the original file's directory
        original_file_path = None
        for root, _, files in os.walk(kernel_root):
            if base_name + ".rej" in files and base_name + "_ast.txt" in files:
                original_file_path = root
                break

        if not original_file_path:
            print(f"❌ Could not find original file location for {base_name}. Skipping.")
            continue
        
        rej_file = os.path.join(original_file_path, base_name + ".rej")
        ast_file = os.path.join(original_file_path, base_name + "_ast.txt")
        output_file = os.path.join(output_dir, base_name.replace("/", "_") + ".diff")

        if not os.path.exists(ast_file) or not os.path.exists(rej_file):
            print(f"❌ Missing required files for {base_name}. Skipping.")
            continue

        print(f"\n🔍 Resolving patch for {base_name}...")

        max_retries = 3
        for attempt in range(max_retries):
            print(f"🔄 Attempt {attempt + 1} of {max_retries}...")
            with open(rej_file) as rf, open(ast_file) as af, open(inline_file) as inf:
                rej_content = rf.read()
                ast_content = af.read()
                inline_content = inf.read()

            # If you want to support context blocks from another source (e.g., JSON or .txt), you can read it here:
            context_block = ""  # Optional — only if you have fixed context window instead of AST

            patch = await resolve_conflict(
                ctx=None,
                ast_content=ast_content,
                context_block=context_block,
                rej_content=rej_content,
                inline_content=inline_content
            )

            if not patch:
                continue
            is_valid = await check_diff_format(None, patch)
            if is_valid:
                with open(output_file, 'w') as f:
                    f.write(patch)
                print(f"✅ Patch written to {output_file}")
                break
        else:
            print(f"❌ Failed to produce valid patch for {base_name}")

async def main():
    """
    Main function to process merge conflicts and generate patches.
    """
    kernel_root = "/Volumes/GitRepo/school/capstone/android/Xiaomi_Kernel_OpenSource"
    conflict_dir = os.path.join(kernel_root, "merge_conflicts")
    output_dir = os.path.join(kernel_root, "generated_patches")

    print(f"📂 Merge conflicts: {conflict_dir}")
    print(f"📂 Output patches: {output_dir}")
    await process_merge_conflicts(kernel_root, conflict_dir, output_dir)

if __name__ == "__main__":
    asyncio.run(main())
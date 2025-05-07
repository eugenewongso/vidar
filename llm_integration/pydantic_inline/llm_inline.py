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

# Load environment variables
load_dotenv()

GCP_REGION = "us-central1"
GCP_PROJECT = "dev-smoke-452808-t2"

try:
    provider = GoogleVertexProvider(region=GCP_REGION)
except Exception as e:
    print(f"⚠️ Warning: Could not use application default credentials. {e}")
    provider = None

service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if service_account_path and provider is None:
    try:
        provider = GoogleVertexProvider(service_account_file=service_account_path, region=GCP_REGION)
    except Exception as e:
        print(f"❌ Error: Could not load service account file. {e}")
        exit(1)

if provider is None:
    raise ValueError("❌ No valid Google Vertex AI credentials found.")


def fix_hunk_headers(patch_text: str) -> str:
    lines = patch_text.splitlines()
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('@@'):
            header_match = re.match(r'@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@', line)
            if not header_match:
                fixed_lines.append(line)
                i += 1
                continue

            old_start, old_count, new_start, new_count = header_match.groups()
            old_start = int(old_start)
            new_start = int(new_start)
            old_count = int(old_count) if old_count else 1
            new_count = int(new_count) if new_count else 1

            # Collect hunk body
            hunk_body = []
            i += 1
            while i < len(lines) and not lines[i].startswith('@@') and not lines[i].startswith('---') and not lines[i].startswith('+++'):
                hunk_body.append(lines[i])
                i += 1

            # Count actual lines
            actual_old = sum(1 for l in hunk_body if l.startswith('-') or l.startswith(' '))
            actual_new = sum(1 for l in hunk_body if l.startswith('+') or l.startswith(' '))

            fixed_header = f"@@ -{old_start},{actual_old} +{new_start},{actual_new} @@"
            fixed_lines.append(fixed_header)
            fixed_lines.extend(hunk_body)
        else:
            fixed_lines.append(line)
            i += 1

    return '\n'.join(fixed_lines) + '\n'


@dataclass
class MergeDeps:
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
    try:
        PatchSet.from_string(diff_content)
        print("✅ Patch format is valid.")
        return True
    except Exception as e:
        print(f"⚠️ Patch format is invalid. Retrying...\nError: {e}")
        return False

@merge_agent.tool
async def resolve_conflict(
    ctx: RunContext[MergeDeps],
    rej_file: str,
    ast_file: str,
    inline_file: str
) -> str:
    with open(rej_file, 'r') as f:
        rej_content = f.read()

    with open(ast_file, 'r') as f:
        ast_content = f.read()

    with open(inline_file, 'r') as f:
        inline_content = f.read()

    prompt = f"""
    You need to resolve a merge conflict in a kernel code file.

    First, here is the complete function context and dependencies:
    {ast_content}

    Here's the rejected patch hunk:
    {rej_content}

    And here is the isolated inline conflict from a failed 3-way merge:
    {inline_content}

    Requirements:
    1. Use the inline conflict and rejected hunk to understand what failed.
    2. Resolve the issue in a way that fits into the function context.
    3. Output ONLY a valid unified diff format patch with no formatting markers.
    4. Ensure clean application and preserve logic.
    """


    print("🟡 Sending request to AI agent...")
    response = await merge_agent.run(prompt)

    if not response or not response.data.strip():
        print("❌ AI response was empty. Skipping.")
        return None

    # DEBUG: Print raw LLM output before formatting
    print("\n🧾 Raw AI response:")
    print("=" * 60)
    print(response.data)
    print("=" * 60)


    # 🧼 Clean markdown formatting if present
    patch = response.data.strip()
    patch = patch.replace("```diff", "").replace("```", "").strip() + "\n"
    patch = fix_hunk_headers(patch)  # 🧩 Fix malformed hunk headers


    return patch


async def process_merge_conflicts(kernel_root: str, conflict_dir: str, output_dir: str):
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
            patch = await resolve_conflict(None, rej_file, ast_file, inline_file)
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
    kernel_root = "/Volumes/GitRepo/school/capstone/android/Xiaomi_Kernel_OpenSource"
    conflict_dir = os.path.join(kernel_root, "merge_conflicts")
    output_dir = os.path.join(kernel_root, "generated_patches")

    print(f"📂 Merge conflicts: {conflict_dir}")
    print(f"📂 Output patches: {output_dir}")
    await process_merge_conflicts(kernel_root, conflict_dir, output_dir)

if __name__ == "__main__":
    asyncio.run(main())
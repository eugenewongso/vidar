import json
import asyncio
import time
import argparse
from llm_inline import resolve_conflict_per_reject, check_diff_format

def remove_duplicate_diff_headers(patch: str) -> str:
    """
    Remove diff/git/file headers from the LLM-generated patch.
    This ensures that only the actual patch content is retained.
    
    Args:
        patch (str): The patch content as a string.
    
    Returns:
        str: The filtered patch content without diff/git/file headers.
    """
    lines = patch.splitlines()
    filtered = []
    skip_prefixes = ("diff --git", "---", "+++")
    for line in lines:
        if not any(line.startswith(prefix) for prefix in skip_prefixes):
            filtered.append(line)
    return "\n".join(filtered)


async def process_context_blocks(json_path: str, output_path: str, context_levels: list[int], combine: bool):
    print(f"📂 Loading input JSON from: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    for entry_index, entry in enumerate(data.get("failures", [])):
        if not isinstance(entry, dict):
            print(f"⚠️ Skipping entry {entry_index + 1} because it's not a dictionary. Found: {type(entry).__name__}")
            continue

        for failure in entry.get("failures", []):
            for conflict in failure.get("file_conflicts", []):
                for ctx in context_levels:
                    key = f"downstream_file_content_context_{ctx}"
                    if key not in conflict:
                        continue

                    context_blocks = conflict[key]
                    total_duration = 0.0

                    if combine:
                        combined_context = "\n\n---\n\n".join(context_blocks)
                        print(f"🔍 Running LLM for COMBINED context {ctx} with {len(context_blocks)} blocks...")
                        start_time = time.time()
                        try:
                            patch = await resolve_conflict_per_reject(
                                ctx=None,
                                context_block_with_inline=combined_context,
                                rej_content=conflict.get("rej_file_content", "")
                            )
                            if patch:
                                valid = await check_diff_format(None, patch)
                                if valid:
                                    conflict[f"llm_patch_context_{ctx}"] = remove_duplicate_diff_headers(patch)
                                    conflict[f"llm_patch_context_{ctx}_duration_seconds"] = round(time.time() - start_time, 2)
                                    print(f"✅ Valid COMBINED patch for context {ctx}")
                                else:
                                    print(f"❌ Invalid COMBINED patch format for context {ctx}")
                        except Exception as e:
                            print(f"❌ Error during COMBINED LLM call for context {ctx}: {e}")
                    else:
                        patches = []
                        for i, context in enumerate(context_blocks):
                            print(f"🔍 Running LLM for context {ctx}, block {i}...")
                            start_time = time.time()
                            try:
                                patch = await resolve_conflict_per_reject(
                                    ctx=None,
                                    context_block_with_inline=context,
                                    rej_content=conflict.get("rej_file_content", "")
                                )
                                if patch:
                                    valid = await check_diff_format(None, patch)
                                    if valid:
                                        patches.append(remove_duplicate_diff_headers(patch))
                                        print(f"✅ Valid patch block {i} for context {ctx}")
                                    else:
                                        print(f"❌ Invalid patch format (context {ctx}, block {i})")
                            except Exception as e:
                                print(f"❌ Error (context {ctx}, block {i}): {e}")
                            finally:
                                total_duration += time.time() - start_time

                        if patches:
                            combined_patch = "\n".join(patches).strip() + "\n"
                            conflict[f"llm_patch_context_{ctx}"] = combined_patch
                            conflict[f"llm_patch_context_{ctx}_duration_seconds"] = round(total_duration, 2)
                            print(f"📝 Combined patch saved for context {ctx} (took {round(total_duration, 2)}s)")
                        else:
                            print(f"⚠️ No valid patches for context {ctx}")

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n💾 Saved final output to: {output_path}")



def parse_args():
    """
    Parse command-line arguments for the script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process context blocks with LLM")
    parser.add_argument("--input", type=str, required=True, help="Input JSON path")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--context-levels", type=int, nargs="+", default=[3, 5, 10, 20],
                        help="List of context window sizes to use (default: 3 5 10 20)")
    parser.add_argument(
    "--combine", action="store_true",
    help="If set, combine all context blocks per context window into one LLM call"
)

    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments and run the main processing function
    args = parse_args()
    asyncio.run(process_context_blocks(args.input, args.output, args.context_levels, args.combine))


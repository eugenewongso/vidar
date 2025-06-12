import os
import json
import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from llm_agent_module import GeminiAgent, APIKeyRotator
from android_patch_manager import AndroidPatchManager
from validation_utils import validate_patch_format, validate_patch_applicability_in_repo
from support_utils import get_repo_url_from_osv, get_all_token_counts, SupportDependencies

# Load environment variables
load_dotenv()
api_keys = os.getenv("GOOGLE_API_KEYS", "").split(",")
if not api_keys or api_keys == [""]:
    raise ValueError("Missing API keys in GOOGLE_API_KEYS")
key_rotator = APIKeyRotator(api_keys)

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
FAILED_PATCH_JSON = BASE_DIR / "failed_patch.json"
PATCH_OUTPUT_DIR = BASE_DIR / "patch_adoption" / "generated_patches"
REPORT_OUTPUT_PATH = BASE_DIR / "reports" / "1_llm_output.json"
PATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# GeminiAgent setup
system_prompt = """You are an advanced security patching assistant. You will be given a source file and a .rej file. Generate a corrected unified diff that applies cleanly using `patch -p1`. Do not include explanations. Strictly return a valid unified diff."""
patch_porter_agent = GeminiAgent(
    model_name="gemini-2.5-pro-preview-05-06",
    system_prompt=system_prompt,
    key_rotator=key_rotator
)

async def process_patch_entry(patch: dict, kernel_path: Path):
    patch_hash = Path(patch["patch_file"]).stem
    patch_url = patch["patch_url"]
    downstream_repo_url = patch_url.split("+/", 1)[0]
    repo_path = AndroidPatchManager.clone_repo(downstream_repo_url, kernel_path.parent / "android_repos")

    AndroidPatchManager.clean_repo(repo_path)
    AndroidPatchManager.checkout_commit(repo_path, f"{patch_hash}^")

    results = []
    for file in patch["rejected_files"]:
        reject_file_path = Path(file["reject_file"])
        failed_file_path = Path(file["failed_file"])

        try:
            rej_content = reject_file_path.read_text(encoding="utf-8")
            original_source = failed_file_path.read_text(encoding="utf-8")
        except Exception as e:
            results.append({"file": str(failed_file_path), "error": str(e)})
            continue

        dependencies = SupportDependencies(
            rej_file_content=rej_content, original_source_file_content=original_source
        )

        prompt = f"""
        Original Source File:
        ```
        {original_source}
        ```

        Rejected Hunks:
        ```
        {rej_content}
        ```

        Please generate a corrected unified diff starting with `--- a/...` and `+++ b/...`. Strictly return only the diff.
        """

        validation_results = []
        for attempt in range(3):
            try:
                start_time = datetime.now()
                result = await patch_porter_agent.run(prompt, deps=dependencies)
                generated_diff = result.data.strip()
                format_valid, format_error = validate_patch_format(generated_diff)
                apply_valid, apply_error = (False, "Skipped due to format error")
                if format_valid:
                    apply_valid, apply_error = validate_patch_applicability_in_repo(generated_diff, str(repo_path))

                validation = {
                    "attempt": attempt + 1,
                    "runtime_seconds": (datetime.now() - start_time).total_seconds(),
                    "format_valid": format_valid,
                    "apply_valid": apply_valid,
                    "format_error": format_error,
                    "apply_error": apply_error,
                    "valid": format_valid and apply_valid,
                    "token_counts": get_all_token_counts(generated_diff, result.token_count)
                }
                validation_results.append(validation)

                if validation["valid"]:
                    output_path = PATCH_OUTPUT_DIR / f"{patch_hash}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.diff"
                    output_path.write_text(generated_diff, encoding="utf-8")
                    return {
                        "patch_hash": patch_hash,
                        "patch_url": patch_url,
                        "file": str(failed_file_path),
                        "output_path": str(output_path),
                        "validation_results": validation_results,
                        "success": True
                    }
            except Exception as e:
                validation_results.append({"attempt": attempt + 1, "error": str(e), "valid": False})

        return {
            "patch_hash": patch_hash,
            "patch_url": patch_url,
            "file": str(failed_file_path),
            "validation_results": validation_results,
            "success": False
        }

async def main():
    kernel_path = Path(os.getenv("KERNEL_PATH", "/data/androidOS14"))
    with open(FAILED_PATCH_JSON, "r", encoding="utf-8") as f:
        failed_patches = json.load(f)

    all_results = []
    for patch in failed_patches["patches"]:
        result = await process_patch_entry(patch, kernel_path)
        all_results.append(result)

    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"✅ Final report saved to {REPORT_OUTPUT_PATH}")

if __name__ == "__main__":
    asyncio.run(main())

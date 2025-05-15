import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
import re

from pathlib import Path

# ---------- CONFIG ----------
SCRIPT_DIR = Path.cwd() / "kernel_cve_evaluation"
REPO_PATH = SCRIPT_DIR.parent / "linux"
PATCH_TOOL = "gpatch"  # or "patch"
STRIP_LEVEL = "1"
INPUT_JSON = SCRIPT_DIR.parent / "reports" / "2025_full_cve_report_20250320_142906.json"
OUTPUT_JSON = SCRIPT_DIR.parent / "report_with_inline_conflict5.json"
# ----------------------------


def git_checkout(commit):
    subprocess.run(["git", "reset", "--hard", commit], cwd=REPO_PATH, check=True)

def generate_patch(upstream_commit, output_file):
    with open(output_file, "w") as f:
        subprocess.run(["git", "format-patch", "-1", upstream_commit, "--stdout"],
                       cwd=REPO_PATH, stdout=f, check=True)

def apply_patch(patch_path):
    result = subprocess.run(
        [PATCH_TOOL, "-p", STRIP_LEVEL, "-i", str(patch_path), "--ignore-whitespace"],
        cwd=REPO_PATH,
        capture_output=True,
        text=True
    )
    return result.returncode == 0, result.stdout + result.stderr

def find_rej_files():
    return list(Path(REPO_PATH).rglob("*.rej"))

def combine_rej_files(rej_paths, output_path):
    with open(output_path, "w") as outfile:
        for file in rej_paths:
            with open(file, "r") as f:
                outfile.write(f.read().strip() + "\n\n")

def format_conflict_blocks(conflict_text: str, file_name: str = "unknown_file") -> str:
    """
    Formats a raw conflict block string into a Markdown code block with clearer labels.

    :param conflict_text: Raw text containing Git-style conflict markers.
    :param file_name: Optional name of the file for context.
    :return: A formatted, Markdown-friendly diff-style string.
    """
    blocks = re.findall(r"(<<<<<<<.*?=======.*?>>>>>>>)", conflict_text, re.DOTALL)
    formatted_blocks = []

    for block in blocks:
        lines = block.strip().splitlines()
        section1 = []
        section2 = []
        mode = None

        for line in lines:
            if line.startswith("<<<<<<<"):
                mode = "head"
                continue
            elif line.startswith("======="):
                mode = "incoming"
                continue
            elif line.startswith(">>>>>>>"):
                mode = None
                continue

            if mode == "head":
                section1.append(line)
            elif mode == "incoming":
                section2.append(line)

        formatted_block = (
            f"```diff\n"
            f"// Conflict in: {file_name}\n"
            f"<<<<<<< CURRENT VERSION\n" + "\n".join(f"- {l}" for l in section1) + "\n"
            f"=======\n" + "\n".join(f"+ {l}" for l in section2) + "\n"
            f">>>>>>> INCOMING PATCH\n"
            f"```"
        )
        formatted_blocks.append(formatted_block)

    return "\n\n".join(formatted_blocks) if formatted_blocks else "No conflict markers found."


def extract_inline_conflict_from_patch(patch_file):
    result = subprocess.run(
        [PATCH_TOOL, "--merge", "-p", STRIP_LEVEL, "--output=-", "-i", str(patch_file)],
        cwd=REPO_PATH,
        capture_output=True,
        text=True
    )
    # This captures the output with conflict markers (e.g. <<<<<<< HEAD)
    pattern = re.compile(r"<<<<<<<.*?=======.*?>>>>>>>", re.DOTALL)
    conflicts = pattern.findall(result.stdout)
    return "\n\n".join(conflicts).strip() if conflicts else None

def format_rej_file_content(raw_rej_text: str) -> str:
    """
    Wraps entire .rej content into a single Markdown diff block.
    Preserves original Git-style headers and formatting.
    """
    return f"```diff\n{raw_rej_text.strip()}\n```" if raw_rej_text.strip() else "No rejected diff content found."


def enrich_failures_with_conflicts(report_path):
    original_cwd = os.getcwd()
    os.chdir(REPO_PATH)

    try:
        with open(report_path, "r") as f:
            data = json.load(f)

        for group in ["cves_with_all_failures", "cves_with_partial_failures"]:
            for cve in data.get(group, []):
                for attempt in cve.get("patch_attempts", []):
                    upstream_commit = attempt["upstream_commit"]
                    patch_file = Path(original_cwd) / "tmp_patch.diff"

                    try:
                        git_checkout(upstream_commit)
                        generate_patch(attempt["upstream_patch"], patch_file)
                    except Exception as e:
                        print(f"⚠️ Error generating patch for {upstream_commit}: {e}")
                        continue

                    # Filter only failed patch results
                    failed_results = [r for r in attempt["patch_results"] if r.get("result") == "failure"]
                    if not failed_results:
                        continue  # Skip if no failed results

                    for result in failed_results:
                        downstream_commit = result["downstream_commit"]
                        try:
                            git_checkout(downstream_commit)
                        except Exception as e:
                            print(f"⚠️ Failed to checkout {downstream_commit}: {e}")
                            result["inline_merge_conflict"] = ""
                            result["rej_file_content"] = ""
                            continue

                        for old in find_rej_files():
                            try:
                                old.unlink()
                            except Exception:
                                pass

                        success, _ = apply_patch(patch_file)
                        if success:
                            result["inline_merge_conflict"] = ""
                            result["rej_file_content"] = ""
                            continue

                        with TemporaryDirectory() as tempdir:
                            combined_rej = Path(tempdir) / "combined.rej"
                            rej_files = find_rej_files()
                            combine_rej_files(rej_files, combined_rej)

                            raw_rej_content = ""
                            for rfile in rej_files:
                                try:
                                    raw_rej_content += f"--- {rfile.relative_to(REPO_PATH)} ---\n"
                                    raw_rej_content += Path(rfile).read_text().strip() + "\n\n"
                                except Exception as e:
                                    raw_rej_content += f"⚠️ Failed to read {rfile.name}: {e}\n\n"

                            # conflict_text = extract_inline_conflict(combined_rej)
                            conflict_text = extract_inline_conflict_from_patch(patch_file)


                            result["rej_file_content"] = format_rej_file_content(raw_rej_content)
                            result["inline_merge_conflict"] = (
                                format_conflict_blocks(conflict_text, file_name="unknown") if conflict_text else "No conflict markers found."
                            )

                with open(Path(original_cwd) / OUTPUT_JSON, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"💾 Progress saved after {cve['cve_url']}")

    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    enrich_failures_with_conflicts(INPUT_JSON)

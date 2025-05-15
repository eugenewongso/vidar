import os
import subprocess
import tempfile
import re
from pathlib import Path
from google.cloud import aiplatform_v1beta1
from google.cloud.aiplatform_v1beta1.types import CountTokensRequest, Content, Part
import tiktoken

client = aiplatform_v1beta1.PredictionServiceClient()

def parse_version(v):
    """Turn '12L' or '14' into comparable tuples."""
    if v.endswith('L'):
        return (int(v[:-1]), 1)
    return (int(v), 0)

def is_newer_version(source, target):
    """Return True if `source` is newer than `target`."""
    return parse_version(source) > parse_version(target)


class AndroidPatchManager:
    PATCH_TOOL = "gpatch"
    STRIP_LEVEL = 1

    @staticmethod
    def clone_repo(repo_url, repo_base):
        repo_name = repo_url.split('/')[-1]
        repo_path = os.path.join(repo_base, repo_name)

        if not os.path.exists(repo_path):
            print(f"Cloning {repo_url} into {repo_path}")
            subprocess.run(["git", "clone", "--no-single-branch", repo_url, repo_path], check=True)
        else:
            print(f"Using cached repo at {repo_path}")
            subprocess.run(["git", "fetch", "--all"], cwd=repo_path, check=True)

        return repo_path
    
    @staticmethod
    def count_tokens_gemini(text, project: str, location: str = "us-central1", model: str = "gemini-2.5-pro-preview-03-25"):
        publisher_model = f"projects/{project}/locations/{location}/publishers/google/models/{model}"
        request = CountTokensRequest(
            endpoint=publisher_model,
            contents=[Content(role="user", parts=[Part(text=text)])]
        )
        response = client.count_tokens(request=request)
        return response.total_tokens
    
    @staticmethod
    def count_tokens_general(text: str):
        # Rough estimate: ~1 token = 0.75 words or ~4 chars/token
        word_count = len(re.findall(r'\w+', text))
        char_estimate = len(text) // 4
        return {
            "word_based": word_count,
            "char_based": char_estimate
        }


    @staticmethod
    def count_tokens_tiktoken(text: str, model: str = "gpt-3.5-turbo"):
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

    @staticmethod
    def get_all_token_counts(text: str, project: str, skip_gemini: bool = False):
        result = {
            "openai": AndroidPatchManager.count_tokens_tiktoken(text),
            "general": AndroidPatchManager.count_tokens_general(text),
        }
        if not skip_gemini:
            result["gemini"] = AndroidPatchManager.count_tokens_gemini(text, project)
        return result


    @staticmethod
    def filter_patch_file(patch_file_path, relevant_files):
        with open(patch_file_path, 'r') as f:
            patch_lines = f.readlines()

        filtered_patch_lines = []
        include_block = False
        current_file = None

        for line in patch_lines:
            if line.startswith('diff --git'):
                match = re.match(r'diff --git a/(.*?) b/(.*?)$', line)
                current_file = match.group(1) if match else None
                include_block = current_file in relevant_files
            if include_block:
                filtered_patch_lines.append(line)

        with open(patch_file_path, 'w') as f:
            f.writelines(filtered_patch_lines)

        print(f"🗂️ Filtered patch to include only: {relevant_files}")

    
    @staticmethod
    def clean_repo(repo_path):
        # Reset uncommitted changes and clean untracked files
        subprocess.run(["git", "reset", "--hard"], cwd=repo_path, check=True)
        subprocess.run(["git", "clean", "-fd"], cwd=repo_path, check=True)


    @staticmethod
    def checkout_downstream_branch(repo_path, downstream_version):
        # Clean repo before checkout
        AndroidPatchManager.clean_repo(repo_path)

        normalized_version = downstream_version.strip()
        branch_name = f"android{normalized_version}-release"

        print(f"📦 Trying to checkout: {branch_name}")

        try:
            subprocess.run(["git", "checkout", branch_name], cwd=repo_path, check=True)
        except subprocess.CalledProcessError as e:
            raise ValueError(f"❌ Failed to checkout branch {branch_name}: {e}")

        return branch_name


    @staticmethod
    def checkout_commit(repo_path, commit_hash):
        subprocess.run(["git", "reset", "--hard", commit_hash], cwd=repo_path, check=True)

    @staticmethod
    def generate_combined_patch(repo_path, commit_hashes):
        patch_file = tempfile.NamedTemporaryFile(delete=False, suffix=".diff")
        combined_patch_content = ""
        with open(patch_file.name, "w") as f:
            for commit_hash in commit_hashes:
                result = subprocess.run(
                    ["git", "format-patch", "-1", commit_hash, "--stdout"],
                    cwd=repo_path, capture_output=True, text=True, check=True
                )
                f.write(result.stdout)
                combined_patch_content += result.stdout  # Collect content
        return patch_file.name, combined_patch_content


    @staticmethod
    def apply_patch(repo_path, patch_file, use_merge=False):
        patch_cmd = [AndroidPatchManager.PATCH_TOOL, "-p", str(AndroidPatchManager.STRIP_LEVEL), "-i", patch_file, "--ignore-whitespace"]
        if use_merge:
            patch_cmd.insert(1, "--merge")
        result = subprocess.run(patch_cmd, cwd=repo_path, capture_output=True, text=True)

        success = result.returncode == 0
        output = (result.stdout + result.stderr).strip()

        total_hunks_match = re.search(r"(\d+) out of (\d+) hunks FAILED", output)
        failed_hunks = re.findall(r"Hunk #(\d+) FAILED", output)
        if total_hunks_match:
            total_hunks = int(total_hunks_match.group(2))
        else:
            # Set total_hunks at least as high as the highest failed hunk if total_hunks wasn't parsed
            total_hunks = max([int(h) for h in failed_hunks], default=0)
        failed_hunks_list = [int(h) for h in failed_hunks] if failed_hunks else []

        return success, output, total_hunks, failed_hunks_list


    @staticmethod
    def extract_conflicts(repo_path, patch_file, downstream_version, upstream_commit, patch_error_output, total_hunks, failed_hunks_list):
        file_conflicts = []
        rej_files = list(Path(repo_path).rglob("*.rej"))

        # If .rej files exist, extract as usual
        for rej in rej_files:
            file_name = str(rej.relative_to(repo_path)).replace(".rej", "")
            content = rej.read_text()

            file_path = os.path.join(repo_path, file_name)
            downstream_content = Path(file_path).read_text() if os.path.exists(file_path) else ""

            AndroidPatchManager.checkout_commit(repo_path, upstream_commit)
            upstream_file_path = os.path.join(repo_path, file_name)
            upstream_content = Path(upstream_file_path).read_text() if os.path.exists(upstream_file_path) else ""

            AndroidPatchManager.checkout_downstream_branch(repo_path, downstream_version)

            filtered_patch_path = tempfile.NamedTemporaryFile(delete=False, suffix=".diff").name
            with open(patch_file, "r") as orig, open(filtered_patch_path, "w") as filtered:
                include = False
                for line in orig:
                    if line.startswith("diff --git"):
                        include = file_name in line
                    if include:
                        filtered.write(line)

            AndroidPatchManager.clean_repo(repo_path)
            _, merge_output, _, _ = AndroidPatchManager.apply_patch(repo_path, filtered_patch_path, use_merge=True)

            inline_conflicts = AndroidPatchManager.parse_inline_conflicts(repo_path, file_name, downstream_version, upstream_commit)

            # Calculate inline conflict token summary
            total_inline_tokens = {
                "gemini": 0,
                "openai": 0,
                "general_word": 0,
                "general_char": 0
            }
            for conflict in inline_conflicts:
                tokens = conflict.get("merge_conflict_tokens", {})
                total_inline_tokens["gemini"] += tokens.get("gemini", 0)
                total_inline_tokens["openai"] += tokens.get("openai", 0)
                general = tokens.get("general", {})
                total_inline_tokens["general_word"] += general.get("word_based", 0)
                total_inline_tokens["general_char"] += general.get("char_based", 0)

            file_conflicts.append({
                "file_name": file_name,
                "total_hunks": total_hunks,
                "failed_hunks": failed_hunks_list,
                "inline_merge_conflicts": inline_conflicts,
                "inline_merge_token_summary": total_inline_tokens,
                "rej_file_content": f"```diff\n{content.strip()}\n```",
                "rej_file_tokens": AndroidPatchManager.get_all_token_counts(content, project="neat-resolver-406722", skip_gemini=False),
                "patch_apply_output": patch_error_output,
                "inline_merge_output": merge_output,
                "upstream_file_content": f"```{file_name.split('.')[-1]}\n{upstream_content.strip()}\n```" if upstream_content else "",
                "upstream_file_tokens": AndroidPatchManager.get_all_token_counts(upstream_content, project="neat-resolver-406722", skip_gemini=False),
                "downstream_file_content": f"```{file_name.split('.')[-1]}\n{downstream_content.strip()}\n```" if downstream_content else "",
                "downstream_file_tokens": AndroidPatchManager.get_all_token_counts(downstream_content, project="neat-resolver-406722", skip_gemini=False)
            })


        # If no .rej files were found AND "can't find file to patch" was in the patch output, handle it
        if not rej_files and "can't find file to patch at input line" in patch_error_output:
            matches = re.findall(r'diff --git a/(.*?) b/', patch_error_output)
            for missing_file in matches:
                file_conflicts.append({
                    "file_name": missing_file,
                    "total_hunks": 0,
                    "failed_hunks": [],
                    "inline_merge_conflicts": [],
                    "rej_file_content": "",
                    "patch_apply_output": patch_error_output,
                    "inline_merge_output": "",
                    "reason": f"File '{missing_file}' is missing in downstream repo",
                    "upstream_file_content": "",
                    "downstream_file_content": ""
                })


        return file_conflicts





    @staticmethod
    def parse_inline_conflicts(repo_path, file_name, downstream_version, upstream_commit):
        file_path = os.path.join(repo_path, file_name)
        if not os.path.exists(file_path):
            return []
        text = Path(file_path).read_text()
        blocks = re.findall(r"(<<<<<<<.*?=======.*?>>>>>>>)", text, re.DOTALL)
        conflicts = []
        for i, block in enumerate(blocks, start=1):
            lines = block.strip().splitlines()
            current_lines, incoming_lines, mode = [], [], None
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
                    current_lines.append(line)
                elif mode == "incoming":
                    incoming_lines.append(line)

            # Custom conflict marker
            formatted_block = (
                f"<<<<<<< DOWNSTREAM (version {downstream_version})\n"
                f"{chr(10).join(current_lines)}\n"
                f"=======\n"
                f"{chr(10).join(incoming_lines)}\n"
                f">>>>>>> UPSTREAM PATCH (commit {upstream_commit})"
            )

            # Count tokens for the formatted block
            token_counts = AndroidPatchManager.get_all_token_counts(
                formatted_block, project="neat-resolver-406722", skip_gemini=True
            )

            conflicts.append({
                "hunk_number": i,
                "merge_conflict": formatted_block,
                "merge_conflict_tokens": token_counts  # 🧠 added this field
            })
        return conflicts



    @staticmethod
    def process_vulnerability(vuln_data, repo_base):
        result_entry = {
            "id": vuln_data['id'],
            "aliases": vuln_data.get('aliases', []),
            "vulnerability_url": f"https://api.osv.dev/v1/vulns/{vuln_data['id']}",
            "severity": vuln_data.get('affected', [{}])[0].get('ecosystem_specific', {}).get('severity', 'Unknown'),
            "patch_attempts": []
        }

        upstream_fixes = []
        downstream_versions = []

        for affected in vuln_data.get("affected", []):
            version = affected.get("versions", [])[0]
            if "next" in version:
                upstream_fixes.extend(affected["ecosystem_specific"].get("fixes", []))
            elif version != "15":
                downstream_versions.append({
                    "version": version,
                    "fixes": affected["ecosystem_specific"].get("fixes", [])
                })

        if not upstream_fixes:
            return {
                "vulnerability_url": result_entry["vulnerability_url"],
                "skipped": True,
                "error": "No upstream fixes found"
            }

        upstream_commits = AndroidPatchManager.extract_upstream_commits(upstream_fixes)
        repo_url = AndroidPatchManager.get_repo_url(vuln_data.get("affected", [])[0])

        mismatch_versions = []
        matched_versions = []

        for dv in downstream_versions:
            downstream_commits = AndroidPatchManager.extract_upstream_commits(dv["fixes"])
            if len(upstream_commits) != len(downstream_commits):
                mismatch_versions.append({
                    "downstream_version": dv["version"],
                    "upstream_commit_count": len(upstream_commits),
                    "downstream_commit_count": len(downstream_commits)
                })
            else:
                matched_versions.append(dv["version"])

        if mismatch_versions:
            print(f"⚠️ Skipping {vuln_data['id']} due to mismatched commit counts in downstream versions.")
            return {
                "vulnerability_url": result_entry["vulnerability_url"],
                "skipped": True,
                "commit_mismatch": True,
                "id": vuln_data['id'],
                "severity": result_entry["severity"],
                "mismatch_versions": mismatch_versions,
                "matched_versions": matched_versions
            }

        if not repo_url:
            return {
                "vulnerability_url": result_entry["vulnerability_url"],
                "skipped": True,
                "error": "No valid repo URL found"
            }

        try:
            repo_path = AndroidPatchManager.clone_repo(repo_url, repo_base)

            try:
                AndroidPatchManager.clean_repo(repo_path)
                subprocess.run(["git", "checkout", "main"], cwd=repo_path, check=True)
                print(f"🔀 Checked out to upstream branch: main")
            except subprocess.CalledProcessError:
                print(f"⚠️ Failed to checkout 'main' branch in repo: {repo_url}")
                return {
                    "vulnerability_url": result_entry["vulnerability_url"],
                    "skipped": True,
                    "error": "Main branch not found"
                }

            patch_file, upstream_patch_content = AndroidPatchManager.generate_combined_patch(repo_path, upstream_commits)

            patch_summary = {
                "upstream_commits": upstream_commits,
                "upstream_branch_used": "main",
                "upstream_patch_content": upstream_patch_content,
                "upstream_patch_tokens": AndroidPatchManager.get_all_token_counts(upstream_patch_content, project="neat-resolver-406722"),
                "total_downstream_versions_tested": len(downstream_versions),
                "successful_patches": 0,
                "failed_patches": 0,
                "patch_results": []
            }

            relevant_files = set()
            for affected in vuln_data.get("affected", []):
                for sig in affected.get("ecosystem_specific", {}).get("vanir_signatures", []):
                    target_file = sig.get("target", {}).get("file")
                    if target_file:
                        relevant_files.add(target_file)

            if relevant_files:
                AndroidPatchManager.filter_patch_file(patch_file, relevant_files)

            if os.stat(patch_file).st_size == 0:
                print(f"⚠️ Filtered patch is empty, skipping vulnerability.")
                return {
                    "vulnerability_url": result_entry["vulnerability_url"],
                    "skipped": True,
                    "error": "Filtered patch is empty"
                }

            for dv in downstream_versions:
                version = dv["version"]
                ground_truth_fixes = dv["fixes"]
                ground_truth_commits = AndroidPatchManager.extract_upstream_commits(ground_truth_fixes)
                ground_truth_commit = ground_truth_commits[0] if ground_truth_commits else None

                if not ground_truth_commit:
                    print(f"⚠️ No ground truth commit for downstream version {version}, skipping.")
                    patch_summary["patch_results"].append({
                        "downstream_version": version,
                        "branch_used": None,
                        "result": "skipped",
                        "reason": "No ground truth commit",
                        "downstream_patch": None
                    })
                    continue

                try:
                    used_branch = AndroidPatchManager.checkout_downstream_branch(repo_path, downstream_version=version)
                except ValueError as e:
                    print(f"⚠️ {e}, skipping version {version}")
                    patch_summary["patch_results"].append({
                        "downstream_version": version,
                        "result": "skipped",
                        "reason": str(e),
                        "downstream_patch": ground_truth_commit
                    })
                    continue

                try:
                    subprocess.run(["git", "cat-file", "-e", f"{ground_truth_commit}^{{commit}}"], cwd=repo_path, check=True)
                except subprocess.CalledProcessError:
                    print(f"⚠️ Ground truth commit {ground_truth_commit} not found in repo, skipping version {version}.")
                    patch_summary["patch_results"].append({
                        "downstream_version": version,
                        "result": "skipped",
                        "reason": "Commit not found in repo",
                        "downstream_patch": ground_truth_commit
                    })
                    continue

                subprocess.run(["git", "reset", "--hard", f"{ground_truth_commit}^"], cwd=repo_path, check=True)
                AndroidPatchManager.clean_repo(repo_path)

                success, patch_output, total_hunks, failed_hunks_list = AndroidPatchManager.apply_patch(repo_path, patch_file)

                downstream_patch_content = ""
                try:
                    result = subprocess.run(
                        ["git", "show", ground_truth_commit],
                        cwd=repo_path, capture_output=True, text=True, check=True
                    )
                    downstream_patch_content = result.stdout
                except subprocess.CalledProcessError:
                    print(f"⚠️ Could not retrieve downstream patch content for {ground_truth_commit}")

                patch_result = {
                    "downstream_version": version,
                    "branch_used": used_branch,
                    "downstream_patch": ground_truth_commit,
                    "repo_path": repo_path,
                    "result": "success" if success else "failure",
                    "downstream_patch_content": downstream_patch_content,
                    "downstream_patch_tokens": AndroidPatchManager.get_all_token_counts(
                        downstream_patch_content, project="neat-resolver-406722", skip_gemini=True
                    ),
                }

                if not success:
                    file_conflicts = AndroidPatchManager.extract_conflicts(
                        repo_path, patch_file, version, upstream_commits[0], patch_output, total_hunks, failed_hunks_list
                    )
                    patch_result["file_conflicts"] = file_conflicts
                    patch_summary["failed_patches"] += 1
                else:
                    patch_summary["successful_patches"] += 1

                patch_summary["patch_results"].append(patch_result)

            result_entry["patch_attempts"].append(patch_summary)
            result_entry["cross_patch_attempts"] = []

            
            matched_versions = sorted(matched_versions, key=parse_version, reverse=True)

            for source in matched_versions:
                source_commit = None
                for dv in downstream_versions:
                    if dv["version"] == source:
                        fixes = dv["fixes"]
                        commits = AndroidPatchManager.extract_upstream_commits(fixes)
                        source_commit = commits[0] if commits else None
                        break

                if not source_commit:
                    continue

                try:
                    AndroidPatchManager.checkout_downstream_branch(repo_path, source)
                    patch_file, _ = AndroidPatchManager.generate_combined_patch(repo_path, [source_commit])
                except Exception as e:
                    print(f"⚠️ Failed to generate patch from version {source}: {e}")
                    continue

                for target in matched_versions:
                    if target == source:
                        continue
                    if not is_newer_version(source, target):
                        print(f"⏩ Skipping forward-port: {source} → {target}")
                        continue

                    try:
                        # 1. Clean repo
                        AndroidPatchManager.clean_repo(repo_path)

                        # 2. Checkout target branch
                        AndroidPatchManager.checkout_downstream_branch(repo_path, target)

                        # 3. Reset to the parent of the target version's ground-truth patch
                        target_commit = None
                        for dv in downstream_versions:
                            if dv["version"] == target:
                                commits = AndroidPatchManager.extract_upstream_commits(dv["fixes"])
                                target_commit = commits[0] if commits else None
                                break

                        if not target_commit:
                            raise Exception(f"❌ No ground-truth patch commit for target version {target}")

                        subprocess.run(["git", "reset", "--hard", f"{target_commit}^"], cwd=repo_path, check=True)


                        # 4. Apply patch
                        success, patch_output, _, _ = AndroidPatchManager.apply_patch(repo_path, patch_file)

                        result_entry["cross_patch_attempts"].append({
                            "from": source,
                            "to": target,
                            "result": "success" if success else "failure",
                            "patch_output": patch_output
                        })

                    except Exception as e:
                        print(f"⚠️ Cross-patch {source} → {target} failed: {e}")
                        result_entry["cross_patch_attempts"].append({
                            "from": source,
                            "to": target,
                            "result": "error",
                            "reason": str(e)
                        })

                        result_entry["cross_patch_attempts"].append({
                            "from": source,
                            "to": target,
                            "result": "success" if success else "failure",
                            "patch_output": patch_output
                        })

                    except Exception as e:
                        print(f"⚠️ Cross-patch {source} → {target} failed: {e}")
                        result_entry["cross_patch_attempts"].append({
                            "from": source,
                            "to": target,
                            "result": "error",
                            "reason": str(e)
                        })

        except Exception as e:
            print(f"❌ Unexpected error while processing {vuln_data['id']}: {e}")
            return {
                "vulnerability_url": result_entry["vulnerability_url"],
                "skipped": True,
                "error": str(e)
            }

        return result_entry


    @staticmethod
    def get_repo_url(affected):
        package_name = affected.get("package", {}).get("name", "")
        if package_name.startswith("platform/"):
            return f"https://android.googlesource.com/{package_name}"
        return None


    @staticmethod
    def extract_upstream_commits(fix_urls):
        print(f"Fix URLs: {fix_urls}")
        commits = []
        for url in fix_urls:
            if "+" in url:
                commit = url.split("+")[-1].lstrip("/")
                commits.append(commit)
        return commits


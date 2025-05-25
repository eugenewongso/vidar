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
    """Convert version strings like '12L' or '14' into tuples for comparison."""
    if v.endswith('L'):
        return (int(v[:-1]), 1)
    return (int(v), 0)

def is_newer_version(source, target):
    """Check if the source version is newer than the target version."""
    return parse_version(source) > parse_version(target)


class AndroidPatchManager:
    PATCH_TOOL = "patch"  # Default patch tool
    STRIP_LEVEL = 1  # Default strip level for patch application

    @staticmethod
    def clone_repo(repo_url, repo_base):
        """
        Clone a Git repository or fetch updates if it already exists locally.

        Args:
            repo_url (str): URL of the repository to clone.
            repo_base (str): Base directory to store the cloned repository.

        Returns:
            str: Path to the cloned repository.
        """
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
    def count_tokens_gemini(text, project: str, location: str = "us-central1", model: str = "gemini-2.5-pro-preview-05-06"):
        """
        Count tokens using the Gemini model from Google Cloud AI Platform.

        Args:
            text (str): Input text to count tokens for.
            project (str): Google Cloud project ID.
            location (str): Location of the model.
            model (str): Model name.

        Returns:
            int: Total token count.
        """
        publisher_model = f"projects/{project}/locations/{location}/publishers/google/models/{model}"
        request = CountTokensRequest(
            endpoint=publisher_model,
            contents=[Content(role="user", parts=[Part(text=text)])]
        )
        response = client.count_tokens(request=request)
        return response.total_tokens
    
    @staticmethod
    def count_tokens_general(text: str):
        """
        Estimate token count based on word and character counts.

        Args:
            text (str): Input text.

        Returns:
            dict: Estimated token counts based on words and characters.
        """
        # Rough estimate: ~1 token = 0.75 words or ~4 chars/token
        word_count = len(re.findall(r'\w+', text))
        char_estimate = len(text) // 4
        return {
            "word_based": word_count,
            "char_based": char_estimate
        }


    @staticmethod
    def count_tokens_tiktoken(text: str, model: str = "gpt-3.5-turbo"):
        """
        Count tokens using the Tiktoken library.

        Args:
            text (str): Input text.
            model (str): Model name.

        Returns:
            int: Total token count.
        """
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

    @staticmethod
    def get_all_token_counts(text: str, project: str, skip_gemini: bool = False):
        """
        Get token counts from multiple methods (OpenAI, general, Gemini).

        Args:
            text (str): Input text.
            project (str): Google Cloud project ID.
            skip_gemini (bool): Whether to skip Gemini token counting.

        Returns:
            dict: Token counts from different methods.
        """
        result = {
            "openai": AndroidPatchManager.count_tokens_tiktoken(text),
            "general": AndroidPatchManager.count_tokens_general(text),
        }
        if not skip_gemini:
            result["gemini"] = AndroidPatchManager.count_tokens_gemini(text, project)
        return result


    @staticmethod
    def filter_patch_file(patch_file_path, relevant_files):
        """
        Filter a patch file to include only changes for relevant files.

        Args:
            patch_file_path (str): Path to the patch file.
            relevant_files (list): List of relevant file paths to include.
        """
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
        """
        Reset uncommitted changes and clean untracked files in a Git repository.

        Args:
            repo_path (str): Path to the repository.
        """
        # Reset uncommitted changes and clean untracked files
        subprocess.run(["git", "reset", "--hard"], cwd=repo_path, check=True)
        subprocess.run(["git", "clean", "-fd"], cwd=repo_path, check=True)


    @staticmethod
    def checkout_downstream_branch(repo_path, downstream_version):
        """
        Checkout a downstream branch in the repository.

        Args:
            repo_path (str): Path to the repository.
            downstream_version (str): Version of the downstream branch.

        Returns:
            str: Name of the checked-out branch.
        """
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
        """
        Reset the repository to a specific commit.

        Args:
            repo_path (str): Path to the repository.
            commit_hash (str): Commit hash to reset to.
        """
        subprocess.run(["git", "reset", "--hard", commit_hash], cwd=repo_path, check=True)

    @staticmethod
    def generate_combined_patch(repo_path, commit_hashes):
        """
        Generate a combined patch file from multiple commits.

        Args:
            repo_path (str): Path to the repository.
            commit_hashes (list): List of commit hashes.

        Returns:
            tuple: Path to the patch file and its content.
        """
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
        """
        Apply a patch to the repository.

        Args:
            repo_path (str): Path to the repository.
            patch_file (str): Path to the patch file.
            use_merge (bool): Whether to use merge mode.

        Returns:
            tuple: Success status, output, total hunks, and failed hunks list.
        """
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
        """
        Extract conflicts from a failed patch application.

        Args:
            repo_path (str): Path to the repository.
            patch_file (str): Path to the patch file.
            downstream_version (str): Downstream version.
            upstream_commit (str): Upstream commit hash.
            patch_error_output (str): Output from the patch application.
            total_hunks (int): Total hunks in the patch.
            failed_hunks_list (list): List of failed hunks.

        Returns:
            list: Details of file conflicts.
        """
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

            # AndroidPatchManager.clean_repo(repo_path)
            _, merge_output, _, _ = AndroidPatchManager.apply_patch(repo_path, filtered_patch_path, use_merge=True)
            downstream_with_conflict_markers = Path(file_path).read_text() if os.path.exists(file_path) else ""

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
                "downstream_file_tokens": AndroidPatchManager.get_all_token_counts(downstream_content, project="neat-resolver-406722", skip_gemini=False),
                "downstream_file_content_with_markers": f"```{file_name.split('.')[-1]}\n{downstream_with_conflict_markers.strip()}\n```" if downstream_with_conflict_markers else "",

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
        """
        Parse inline merge conflicts in a file.

        Args:
            repo_path (str): Path to the repository.
            file_name (str): Name of the file with conflicts.
            downstream_version (str): Downstream version.
            upstream_commit (str): Upstream commit hash.

        Returns:
            list: Details of inline conflicts.
        """
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
                "merge_conflict_tokens": token_counts
            })
        return conflicts

    @staticmethod
    def _prepare_patch_summary(repo_path, upstream_commits):
        """
        Prepare a summary of the upstream patch.

        Args:
            repo_path (str): Path to the repository.
            upstream_commits (list): List of upstream commit hashes.

        Returns:
            tuple: Patch file path, patch content, and summary.
        """
        AndroidPatchManager.clean_repo(repo_path)
        try:
            subprocess.run(["git", "checkout", "main"], cwd=repo_path, check=True)
            print(f"🔀 Checked out to upstream branch: main")
        except subprocess.CalledProcessError:
            raise RuntimeError("Main branch not found")

        patch_file, upstream_patch_content = AndroidPatchManager.generate_combined_patch(repo_path, upstream_commits)
        summary = {
            "upstream_commits": upstream_commits,
            "upstream_branch_used": "main",
            "upstream_patch_content": upstream_patch_content,
            "upstream_patch_tokens": AndroidPatchManager.get_all_token_counts(upstream_patch_content, project="neat-resolver-406722"),
            "total_downstream_versions_tested": 0,
            "successful_patches": 0,
            "failed_patches": 0,
            "patch_results": []
        }
        return patch_file, upstream_patch_content, summary


    @staticmethod
    def _apply_patch_to_downstream_versions(repo_path, patch_file, downstream_versions, upstream_commits):
        """
        Apply a patch to multiple downstream versions.

        Args:
            repo_path (str): Path to the repository.
            patch_file (str): Path to the patch file.
            downstream_versions (list): List of downstream versions.
            upstream_commits (list): List of upstream commit hashes.

        Returns:
            dict: Summary of patch application results.
        """
        summary = {
            "total_downstream_versions_tested": len(downstream_versions),
            "successful_patches": 0,
            "failed_patches": 0,
            "patch_results": []
        }

        for dv in downstream_versions:
            version = dv["version"]
            ground_truth_commits = AndroidPatchManager.extract_commit_hashes(dv["fixes"])
            ground_truth_commit = ground_truth_commits[0] if ground_truth_commits else None

            if not ground_truth_commit:
                summary["patch_results"].append({
                    "downstream_version": version,
                    "branch_used": None,
                    "result": "skipped",
                    "reason": "No ground truth commit",
                    "downstream_patch": None
                })
                continue

            try:
                branch = AndroidPatchManager.checkout_downstream_branch(repo_path, version)
            except Exception as e:
                summary["patch_results"].append({
                    "downstream_version": version,
                    "result": "skipped",
                    "reason": str(e),
                    "downstream_patch": ground_truth_commit
                })
                continue

            subprocess.run(["git", "cat-file", "-e", f"{ground_truth_commit}^{{commit}}"], cwd=repo_path, check=True)
            subprocess.run(["git", "reset", "--hard", f"{ground_truth_commit}^"], cwd=repo_path, check=True)
            AndroidPatchManager.clean_repo(repo_path)

            success, output, total_hunks, failed_hunks = AndroidPatchManager.apply_patch(repo_path, patch_file)

            downstream_patch_content = ""
            try:
                result = subprocess.run(["git", "show", ground_truth_commit], cwd=repo_path, capture_output=True, text=True, check=True)
                downstream_patch_content = result.stdout
            except subprocess.CalledProcessError:
                pass

            result_entry = {
                "downstream_version": version,
                "branch_used": branch,
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
                    repo_path, patch_file, version, upstream_commits[0], output, total_hunks, failed_hunks
                )
                result_entry["file_conflicts"] = file_conflicts
                summary["failed_patches"] += 1
            else:
                summary["successful_patches"] += 1

            summary["patch_results"].append(result_entry)

        return summary
    
    @staticmethod
    def _attempt_cross_patch_forwarding(repo_path, matched_versions, downstream_versions):
        """
        Attempt to forward patches across downstream versions.

        Args:
            repo_path (str): Path to the repository.
            matched_versions (list): List of matched downstream versions.
            downstream_versions (list): List of all downstream versions.

        Returns:
            list: Results of cross-patch forwarding attempts.
        """
        results = []
        matched_versions_sorted = sorted(matched_versions, key=parse_version, reverse=True)

        for source in matched_versions_sorted:
            source_commit = None
            for dv in downstream_versions:
                if (dv["version"] == source):
                    commits = AndroidPatchManager.extract_commit_hashes(dv["fixes"])
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

            for target in matched_versions_sorted:
                if target == source or not is_newer_version(source, target):
                    continue

                try:
                    AndroidPatchManager.clean_repo(repo_path)
                    AndroidPatchManager.checkout_downstream_branch(repo_path, target)

                    target_commit = None
                    for dv in downstream_versions:
                        if dv["version"] == target:
                            commits = AndroidPatchManager.extract_commit_hashes(dv["fixes"])
                            target_commit = commits[0] if commits else None
                            break

                    if not target_commit:
                        raise Exception(f"❌ No ground-truth patch commit for target version {target}")

                    subprocess.run(["git", "reset", "--hard", f"{target_commit}^"], cwd=repo_path, check=True)
                    success, output, *_ = AndroidPatchManager.apply_patch(repo_path, patch_file)

                    results.append({
                        "from": source,
                        "to": target,
                        "result": "success" if success else "failure",
                        "patch_output": output
                    })
                except Exception as e:
                    print(f"⚠️ Cross-patch {source} → {target} failed: {e}")
                    results.append({
                        "from": source,
                        "to": target,
                        "result": "error",
                        "reason": str(e)
                    })

        return results



    @staticmethod
    def process_vulnerability(vuln_data, repo_base, allowed_downstream_versions=None):
        """
        Process a vulnerability by applying patches and analyzing results.

        Args:
            vuln_data (dict): Vulnerability data.
            repo_base (str): Base directory for repositories.

        Returns:
            dict: Results of the vulnerability processing.
        """
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
                if allowed_downstream_versions and version not in allowed_downstream_versions:
                    continue
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

        upstream_commits = AndroidPatchManager.extract_commit_hashes(upstream_fixes)
        repo_url = AndroidPatchManager.get_repo_url(vuln_data.get("affected", [])[0])

        mismatch_versions = []
        matched_versions = []

        for dv in downstream_versions:
            downstream_commits = AndroidPatchManager.extract_commit_hashes(dv["fixes"])
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
                patch_file, upstream_patch_content, patch_summary = AndroidPatchManager._prepare_patch_summary(repo_path, upstream_commits)
            except RuntimeError as e:
                print(f"⚠️ {e}")
                return {
                    "vulnerability_url": result_entry["vulnerability_url"],
                    "skipped": True,
                    "error": str(e)
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

            downstream_patch_results = AndroidPatchManager._apply_patch_to_downstream_versions(
                repo_path, patch_file, downstream_versions, upstream_commits
            )
            patch_summary.update(downstream_patch_results)
            result_entry["patch_attempts"].append(patch_summary)

            cross_patch_attempts = AndroidPatchManager._attempt_cross_patch_forwarding(
                repo_path, matched_versions, downstream_versions
            )
            result_entry["cross_patch_attempts"] = cross_patch_attempts

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
        """
        Get the repository URL for a given affected package.

        Args:
            affected (dict): Affected package data.

        Returns:
            str: Repository URL.
        """
        package_name = affected.get("package", {}).get("name", "")
        if package_name.startswith("platform/"):
            return f"https://android.googlesource.com/{package_name}"
        return None


    @staticmethod
    def extract_commit_hashes(fix_urls):
        """
        Extract commit hashes from fix URLs.

        Args:
            fix_urls (list): List of fix URLs.

        Returns:
            list: List of commit hashes.
        """
        print(f"Fix URLs: {fix_urls}")
        commits = []
        for url in fix_urls:
            if "+" in url:
                commit = url.split("+")[-1].lstrip("/")
                commits.append(commit)
        return commits


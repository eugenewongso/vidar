import json
import os
import subprocess
import requests
from datetime import datetime
import re
from pathlib import Path

class KernelCVEPatchManager:
    def __init__(self, json_url, repo_path=None):
        self.json_url = json_url
        self.repo_path = repo_path
        self.patch_command = "gpatch"
        self.strip_level = 1
        self.patch_folder = os.path.join(os.getcwd(), "patches")
        os.makedirs(self.patch_folder, exist_ok=True)
        self.upstream_files = []


    def fetch_json(self):
        """Fetch CVE JSON file (supports local path or URL)."""
        if os.path.exists(self.json_url):
            try:
                with open(self.json_url, "r") as f:
                    return json.load(f), None
            except Exception as e:
                return None, f"Error reading local JSON: {e}"
        else:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(self.json_url, headers=headers)
            if response.status_code != 200:
                return None, f"Error fetching JSON: {response.status_code}"

            try:
                return response.json(), None
            except Exception as e:
                return None, f"Error decoding JSON: {e}"



    def extract_rej_conflicts(repo_path):
        conflict_data = []
        for rej_file in Path(repo_path).rglob("*.rej"):
            try:
                content = rej_file.read_text()
                relative_path = str(rej_file.relative_to(repo_path))
                conflict_data.append({
                    "file": relative_path.replace(".rej", ""),
                    "rej_content": content
                })
            except Exception:
                pass
        return conflict_data



    def clone_repository(self, repo_url):
        """ Clone the Linux kernel repository if not already cloned """
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        clone_path = os.path.join(os.getcwd(), repo_name)

        if not os.path.exists(clone_path):
            subprocess.run(["git", "clone", repo_url, clone_path], check=True)
        self.repo_path = clone_path

    def checkout_version(self, commit_hash, index, total_commits):
        """ Reset to the previous commit before the specified commit (downstream_patch) and print structured logs """
        try:
            result = subprocess.run(
                ["git", "rev-parse", f"{commit_hash}^"], 
                cwd=self.repo_path, capture_output=True, text=True, check=True
            )
            downstream_commit = result.stdout.strip()

            if not downstream_commit:
                print(f"\n⚠️ No parent commit found for {commit_hash}, using {commit_hash} directly.")
                downstream_commit = commit_hash

            commit_message_result = subprocess.run(
                ["git", "log", "-1", "--pretty=format:%s", downstream_commit], 
                cwd=self.repo_path, capture_output=True, text=True
            )
            commit_message = commit_message_result.stdout.strip() if commit_message_result.returncode == 0 else "Unknown Commit Message"

            print(f"\n🔄 Checking out downstream commit ({index}/{total_commits})")
            print(f"   ├─ Downstream Patch: {commit_hash}")
            print(f"   ├─ Downstream Commit: {downstream_commit}")
            print(f"   ├─ Commit Message: \"{commit_message}\"")

            subprocess.run(["git", "reset", "--hard", downstream_commit], cwd=self.repo_path, check=True)
            subprocess.run(["git", "clean", "-fd"], cwd=self.repo_path, check=True)

            return downstream_commit

        except subprocess.CalledProcessError:
            print(f"\n⚠️ Failed to find downstream commit for {commit_hash}, using {commit_hash} directly.")
            subprocess.run(["git", "reset", "--hard", commit_hash], cwd=self.repo_path, check=True)
            subprocess.run(["git", "clean", "-fd"], cwd=self.repo_path, check=True)
            return commit_hash


    def parse_inline_conflicts(repo_path, file_name, downstream_version, upstream_commit):
        file_path = os.path.join(repo_path, file_name)
        
        if not Path(file_path).is_file():
            print(f"⚠️ Skipping inline conflict parsing for: {file_path} is not a file.")
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

            formatted_block = (
                f"<<<<<<< DOWNSTREAM (version {downstream_version})\n"
                f"{chr(10).join(current_lines)}\n"
                f"=======\n"
                f"{chr(10).join(incoming_lines)}\n"
                f">>>>>>> UPSTREAM PATCH (commit {upstream_commit})"
            )

            conflicts.append({
                "hunk_number": i,
                "merge_conflict": formatted_block
            })
        return conflicts


    def generate_patch(self, fixed_commit):
        patch_cmd = ["git", "format-patch", "-1", fixed_commit, "--stdout"]
        patch_file = os.path.join(self.patch_folder, f"{fixed_commit}.diff")
        
        with open(patch_file, "w") as file:
            subprocess.run(patch_cmd, cwd=self.repo_path, stdout=file, text=True, check=True)

        
        # Get list of files affected
        file_list_cmd = ["git", "show", "--name-only", "--pretty=format:", fixed_commit]
        result = subprocess.run(file_list_cmd, cwd=self.repo_path, capture_output=True, text=True)
        files_touched = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
        
        self.upstream_files = files_touched  # Save for later
        
        return patch_file


    def apply_patch(self, patch_file, use_merge=False):
        """ Apply the patch and extract hunk failure details """
        patch_cmd = [self.patch_command, "-p", str(self.strip_level), "-i", patch_file, "--ignore-whitespace"]
        if use_merge:
            patch_cmd.append("--merge")

        result = subprocess.run(patch_cmd, cwd=self.repo_path, capture_output=True, text=True)

        success = result.returncode == 0
        output = (result.stdout + result.stderr).strip()
        
        # Extract total hunks and failed hunk numbers
        total_hunks_match = re.search(r"(\d+) out of (\d+) hunks FAILED", output)
        failed_hunks = re.findall(r"Hunk #(\d+) FAILED", output)
        total_hunks = int(total_hunks_match.group(2)) if total_hunks_match else 0
        failed_hunks_list = [int(h) for h in failed_hunks] if failed_hunks else []

        return success, output, total_hunks, failed_hunks_list


    def get_commit_date(self, commit_hash):
        """Retrieve commit date from Git history."""
        try:
            result = subprocess.run(
                ["git", "show", "-s", "--format=%ci", commit_hash],
                cwd=self.repo_path, capture_output=True, text=True, check=True
            )
            return result.stdout.strip()  # Returns YYYY-MM-DD HH:MM:SS format
        except subprocess.CalledProcessError:
            print(f"⚠️ Failed to retrieve commit date for {commit_hash}.")
            return "Unknown"

    def process_cve(self):
        """ Process a single CVE JSON file and return structured results """
        data, error = self.fetch_json()
    
        if data is None:
            return {
                "cve_url": self.json_url,
                "skipped": True,
                "error": error
            }

        affected_entries = data["containers"]["cna"]["affected"]

        processed_repos = set()
        report_entry = {
            "cve_id": data.get("cveMetadata", {}).get("cveID", "UNKNOWN"),
            "cve_url": self.json_url,
            "patch_attempts": []
        }


        for entry in affected_entries:
            repo_url = entry["repo"]
            if repo_url in processed_repos:
                continue

            self.clone_repository(repo_url)
            processed_repos.add(repo_url)

            versions = entry.get("versions", [])

            # Extract versions and their commit dates from the CVE JSON
            git_versions = []
            for v in versions:
                if "lessThan" in v:
                    commit_hash = v["lessThan"]
                    commit_date = self.get_commit_date(commit_hash)
                    git_versions.append((commit_hash, commit_date))

            # Sort commits based on date (oldest first)
            git_versions.sort(key=lambda x: x[1])

            if not git_versions:
                continue

            # Use the first commit in the list as the upstream patch (lessThan)
            upstream_patch, upstream_commit_date = git_versions[0]  

            # Get the commit before the upstream patch (lessThan^) as the upstream commit
            try:
                upstream_commit = subprocess.run(
                    ["git", "rev-parse", f"{upstream_patch}^"],
                    cwd=self.repo_path, capture_output=True, text=True, check=True
                ).stdout.strip()
            except subprocess.CalledProcessError:
                print(f"\n⚠️ Failed to find parent commit for {upstream_patch}, using {upstream_patch} directly.")
                upstream_commit = upstream_patch


            patch_summary = {
                "upstream_commit": upstream_commit,
                "upstream_commit_date": upstream_commit_date,
                "upstream_patch": upstream_patch,
                "total_versions_tested": len(git_versions) - 1,
                "successful_patches": 0,
                "failed_patches": 0,
                "patch_results": []
            }


            for index, (commit_hash, commit_date) in enumerate(git_versions[1:], start=1):
                downstream_commit = self.checkout_version(commit_hash, index, len(git_versions) - 1)

                # Add downstream file content
                downstream_file_content = {}
                for file in self.upstream_files:
                    file_path = os.path.join(self.repo_path, file)
                    if os.path.exists(file_path):
                        downstream_file_content[file] = Path(file_path).read_text()


                try:
                    result = subprocess.run(
                        ["git", "show", commit_hash],
                        cwd=self.repo_path, capture_output=True, text=True, check=True
                    )
                    downstream_patch_content = result.stdout
                except subprocess.CalledProcessError:
                    downstream_patch_content = ""


                # Generate patch file for upstream commit if it hasn't been generated
                if index == 1:
                    patch_file = self.generate_patch(upstream_patch)
                    with open(patch_file, "r") as pf:
                        patch_summary["upstream_patch_content"] = pf.read()

                    # Add upstream file content
                    upstream_file_content = {}
                    for file in self.upstream_files:
                        file_path = os.path.join(self.repo_path, file)
                        if os.path.exists(file_path):
                            upstream_file_content[file] = Path(file_path).read_text()
                    patch_summary["upstream_file_content"] = upstream_file_content


                # Apply the patch and extract hunk failure details
                success, error_msg, total_hunks, failed_hunks = self.apply_patch(patch_file)

                result_entry = {
                    "downstream_patch": commit_hash,
                    "downstream_commit": downstream_commit,
                    "commit_date": commit_date,
                    "result": "success" if success else "failure",
                    "patch_apply_output": error_msg,
                    "downstream_patch_content": downstream_patch_content,
                    "downstream_file_content": downstream_file_content,
                }


                if not success:
                    result_entry["error"] = error_msg
                    result_entry["total_hunks"] = total_hunks
                    result_entry["total_failed_hunks"] = len(failed_hunks)
                    result_entry["failed_hunks"] = failed_hunks

                    # Get rej and inline conflicts
                    rej_conflicts = KernelCVEPatchManager.extract_rej_conflicts(self.repo_path)
                    file_conflicts = []
                    for rej in rej_conflicts:
                        file_name = rej["file"]
                        # Re-apply patch using --merge to get inline markers
                        self.checkout_version(commit_hash, index, len(git_versions) - 1)  # reset state
                        _, _, _, _ = self.apply_patch(patch_file, use_merge=True)

                        # Then parse inline conflicts
                        inline_conflicts = KernelCVEPatchManager.parse_inline_conflicts(
                            self.repo_path,
                            file_name=file_name,
                            downstream_version=commit_hash,
                            upstream_commit=upstream_commit
                        )

                        # 🧠 Sync hunk number if safe (only 1 failed hunk and 1 conflict)
                        if len(failed_hunks) == 1 and len(inline_conflicts) == 1:
                            inline_conflicts[0]["hunk_number"] = failed_hunks[0]

                        
                        file_conflicts.append({
                            "file_name": file_name,
                            "rej_file_content": f"```diff\n{rej['rej_content'].strip()}\n```",
                            "inline_merge_conflicts": inline_conflicts,
                            "patch_apply_output": error_msg
                        })


                    result_entry["file_conflicts"] = file_conflicts


                patch_summary["patch_results"].append(result_entry)

                if success:
                    patch_summary["successful_patches"] += 1
                else:
                    patch_summary["failed_patches"] += 1



            report_entry["patch_attempts"].append(patch_summary)

        return report_entry

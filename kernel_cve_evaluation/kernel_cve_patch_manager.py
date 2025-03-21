import json
import os
import subprocess
import requests
from datetime import datetime
import re

class KernelCVEPatchManager:
    def __init__(self, json_url, repo_path=None):
        self.json_url = json_url
        self.repo_path = repo_path
        self.patch_command = "gpatch"
        self.strip_level = 1
        self.patch_folder = os.path.join(os.getcwd(), "patches")
        os.makedirs(self.patch_folder, exist_ok=True)

    def fetch_json(self):
        """ Fetch CVE JSON file and return data or an error message """
        response = requests.get(self.json_url)
        if response.status_code != 200:
            return None, f"Error fetching JSON: {response.status_code}"  # Return None and an error message
        return response.json(), None  # Return data and no error



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

            return downstream_commit

        except subprocess.CalledProcessError:
            print(f"\n⚠️ Failed to find downstream commit for {commit_hash}, using {commit_hash} directly.")
            subprocess.run(["git", "reset", "--hard", commit_hash], cwd=self.repo_path, check=True)
            return commit_hash



    def generate_patch(self, fixed_commit):
        """ Generate a patch from the fixed commit """
        patch_cmd = ["git", "format-patch", "-1", fixed_commit, "--stdout"]
        patch_file = os.path.join(self.patch_folder, f"{fixed_commit}.diff")
        with open(patch_file, "w") as file:
            subprocess.run(patch_cmd, cwd=self.repo_path, stdout=file, check=True)
        return patch_file

    def apply_patch(self, patch_file):
        """ Apply the patch to the current branch and extract hunk failure details """
        patch_cmd = [self.patch_command, "-p", str(self.strip_level), "-i", patch_file, "--ignore-whitespace"]
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
        report_entry = {"cve_url": self.json_url, "patch_attempts": []}

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

                # Generate patch file for upstream commit if it hasn't been generated
                if index == 1:
                    patch_file = self.generate_patch(upstream_patch)

                # Apply the patch and extract hunk failure details
                success, error_msg, total_hunks, failed_hunks = self.apply_patch(patch_file)

                result_entry = {
                    "downstream_patch": commit_hash,  # This is where the patch is applied
                    "downstream_commit": downstream_commit,  # This is the parent commit before patching
                    "commit_date": commit_date,  # Include commit date
                    "result": "success" if success else "failure"
                }

                if not success:
                    result_entry["error"] = error_msg
                    result_entry["total_hunks"] = total_hunks
                    result_entry["total_failed_hunks"] = len(failed_hunks)
                    result_entry["failed_hunks"] = failed_hunks

                patch_summary["patch_results"].append(result_entry)

                if success:
                    patch_summary["successful_patches"] += 1
                else:
                    patch_summary["failed_patches"] += 1



            report_entry["patch_attempts"].append(patch_summary)

        return report_entry

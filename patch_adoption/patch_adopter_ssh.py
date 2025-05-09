import subprocess
import os
import re
import json
import time
import tempfile

class RemotePatchAdopter:
    """Applies patches to a remote Android source directory using SSH."""

    def __init__(self, ssh_host, remote_kernel_path, local_patch_dir, local_report_output_path):
        """
        Initialize the remote patch adopter.

        :param ssh_host: SSH host (e.g., 'tsetia@is-vanir-capstone')
        :param remote_kernel_path: Path to the kernel source on the remote host
        :param local_patch_dir: Local directory where patch files are stored
        :param local_report_output_path: Local path to save the patch application report
        """
        self.ssh_host = ssh_host
        self.remote_kernel_path = remote_kernel_path
        self.local_patch_dir = local_patch_dir
        self.local_report_output_path = local_report_output_path
        self.strip_level = 1
        self.patch_results = {"patches": []}

    def apply_patch(self, local_patch_file, patch_url):
        """
        Copies a patch to the remote host and applies it using SSH.

        :param local_patch_file: Path to the patch file on the local machine
        :param patch_url: URL of the patch
        :return: Patch application details including status and output message
        """
        if not os.path.exists(local_patch_file):
            print(f"❌ Patch file not found: {local_patch_file}")
            return {
                "patch_file": os.path.basename(local_patch_file),
                "patch_url": patch_url,
                "status": "Rejected",
                "detailed_status": "Rejected: Missing Patch File",
                "rejected_files": [],
                "message_output": "Patch file not found locally."
            }

        # Create a temporary file name on the remote server
        remote_patch_file = f"/tmp/{os.path.basename(local_patch_file)}"

        try:
            # Copy the patch file to the remote host
            copy_cmd = ["scp", local_patch_file, f"{self.ssh_host}:{remote_patch_file}"]
            subprocess.run(copy_cmd, check=True, capture_output=True, text=True)
            
            # Run the patch command on the remote host
            ssh_cmd = [
                "ssh", self.ssh_host,
                f"cd {self.remote_kernel_path} && patch -p{self.strip_level} -i {remote_patch_file} -f"
            ]
            
            result = subprocess.run(ssh_cmd, capture_output=True, text=True)
            console_output = result.stdout + result.stderr
            print(console_output)
            
            # Determine detailed status from the output
            detailed_status = self.determine_detailed_status(console_output)
            
            # Extract failed files
            rejected_files = self.extract_failed_files(console_output)
            
            # Determine rejected file paths (no need to collect the .rej files from remote)
            formatted_rejected_files = [{"failed_file": f, "reject_file": f"{self.remote_kernel_path}/{f}.rej"} 
                                       for f in rejected_files]
            
            # Overall status
            overall_status = "Applied Successfully" if not formatted_rejected_files else "Rejected"
            
            return {
                "patch_file": os.path.basename(local_patch_file),
                "patch_url": patch_url,
                "status": overall_status,
                "detailed_status": detailed_status,
                "rejected_files": formatted_rejected_files,
                "message_output": console_output
            }
            
        except subprocess.CalledProcessError as e:
            console_output = (e.stdout or "") + (e.stderr or "")
            print(console_output)
            return {
                "patch_file": os.path.basename(local_patch_file),
                "patch_url": patch_url,
                "status": "Rejected",
                "detailed_status": "Rejected: Error Running Remote Command",
                "rejected_files": [],
                "message_output": console_output
            }
            
        finally:
            # Clean up the remote patch file
            try:
                subprocess.run(["ssh", self.ssh_host, f"rm -f {remote_patch_file}"], 
                              capture_output=True, check=False)
            except:
                pass

    def determine_detailed_status(self, console_output):
        """
        Determines the detailed status of the patch application.

        :param console_output: The output of the patch command.
        :return: Detailed status string.
        """
        # Check if patch was already applied
        if "Reversed (or previously applied) patch detected" in console_output:
            return "Applied Successfully: Already Applied"
        
        # Check if files weren't found
        if "can't find file to patch" in console_output:
            return "Skipped: Files Not Found"
        
        # Check if any hunks failed
        if "FAILED" in console_output and "hunk" in console_output:
            return "Rejected: Failed Hunks"
        
        # Check if there were offsets but all hunks were applied
        if "offset" in console_output and "FAILED" not in console_output:
            return "Applied Successfully: With Offsets"
        
        # Default to clean application if none of the above
        return "Applied Successfully: Clean"

    def extract_failed_files(self, console_output):
        """
        Extracts failed file paths from the patch output.

        :param console_output: The output of the patch command.
        :return: List of failed file paths.
        """
        failed_files = []
        pattern = re.compile(r"patching file (\S+)\nHunk #\d+ FAILED")

        for match in pattern.findall(console_output):
            failed_files.append(match.strip())

        return failed_files

    def save_report(self):
        """
        Saves the patch application report as a JSON file.
        """
        with open(self.local_report_output_path, "w") as report:
            json.dump(self.patch_results, report, indent=4)

        print(f"📄 Patch report saved to: {self.local_report_output_path}")

    def generate_summary(self):
        """
        Generates a summary of patch application results by detailed status.
        """
        status_counts = {}
        
        for patch in self.patch_results["patches"]:
            detailed_status = patch.get("detailed_status", "Unknown")
            status_counts[detailed_status] = status_counts.get(detailed_status, 0) + 1
        
        summary = {
            "total_patches": len(self.patch_results["patches"]),
            "status_counts": status_counts
        }
        
        print("\n===== Patch Application Summary =====")
        print(f"Total patches processed: {summary['total_patches']}")
        print("Status breakdown:")
        for status, count in status_counts.items():
            print(f"  - {status}: {count}")
        
        return summary


# === Main Patch Application Logic ===

if __name__ == "__main__":
    # Configuration (update these to match your setup)
    ssh_host = "tsetia@is-vanir-capstone"
    remote_kernel_path = "/data/androidOS14"
    local_patch_dir = "/Users/theophilasetiawan/Desktop/files/capstone/vidar/fetch_patch_output/diff_output"
    parsed_report_path = "/Users/theophilasetiawan/Desktop/files/capstone/vidar/reports/parsed_report.json"
    report_output_path = "/Users/theophilasetiawan/Desktop/files/capstone/vidar/reports/patch_application_report.json"

    # Ensure local patch directory exists
    if not os.path.isdir(local_patch_dir):
        print(f"❌ Error: Local patch directory not found at {local_patch_dir}")
        exit(1)

    # Ensure parsed report exists
    if not os.path.isfile(parsed_report_path):
        print(f"❌ Error: Parsed report not found at {parsed_report_path}")
        exit(1)

    # Verify SSH connection
    try:
        test_cmd = ["ssh", ssh_host, "echo 'SSH connection successful'"]
        result = subprocess.run(test_cmd, capture_output=True, text=True, check=True)
        print(result.stdout.strip())
    except subprocess.CalledProcessError:
        print("❌ Error: Failed to connect to the remote host via SSH")
        print("Make sure you can connect without password (using SSH keys)")
        exit(1)

    # Verify remote kernel path
    try:
        test_cmd = ["ssh", ssh_host, f"test -d {remote_kernel_path} && echo 'Directory exists'"]
        result = subprocess.run(test_cmd, capture_output=True, text=True, check=True)
        if "Directory exists" not in result.stdout:
            print(f"❌ Error: Remote directory {remote_kernel_path} not found")
            exit(1)
    except subprocess.CalledProcessError:
        print(f"❌ Error: Remote directory {remote_kernel_path} not found")
        exit(1)

    # Load parsed report
    try:
        with open(parsed_report_path, "r") as f:
            parsed_report = json.load(f)
        print(f"📄 Loaded parsed report with {len(parsed_report['patches'])} patches")
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"❌ Error loading parsed report: {e}")
        exit(1)

    # Initialize remote patch adopter
    patcher = RemotePatchAdopter(ssh_host, remote_kernel_path, local_patch_dir, report_output_path)

    # Iterate through patches
    for patch in parsed_report["patches"]:
        patch_file = os.path.join(local_patch_dir, patch["patch_file"])

        print(f"\n🔍 Attempting to apply patch: {patch_file}")
        patch_result = patcher.apply_patch(patch_file, patch["patch_url"])

        patcher.patch_results["patches"].append(patch_result)

    # Save the final report
    patcher.save_report()
    
    # Generate and display summary
    patcher.generate_summary() 
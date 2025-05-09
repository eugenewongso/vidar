import subprocess
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class InlineMergeGenerator:
    """Handles in-file merge conflict extraction from `.rej` files using `patch --merge`."""

    def __init__(self, kernel_path, report_path):
        self.kernel_path = Path(kernel_path).resolve()
        self.report_path = Path(report_path).resolve()
        self.merge_conflicts_dir = self.kernel_path / "merge_conflicts"
        self.merge_conflicts_dir.mkdir(parents=True, exist_ok=True)
        self.patch_command = "gpatch"  # Use GNU patch on macOS

    def get_rejected_files_from_report(self):
        """Extract rejected files and `.rej` paths from `patch_application_report.json`."""
        if not self.report_path.exists():
            logger.error(f"❌ Patch application report not found: {self.report_path}")
            return []

        with open(self.report_path, "r") as f:
            report_data = json.load(f)

        rejected_files = []
        for patch in report_data.get("patches", []):
            if patch["status"] == "Rejected":
                for rejected_file in patch["rejected_files"]:
                    rejected_files.append(rejected_file["reject_file"])

        return rejected_files

    def generate_infile_merge_conflict(self, reject_file: str):
        """
        Runs `gpatch --merge -p0 --output=<file>` on a `.rej` file,
        saves the inline merge conflict to `merge_conflicts/`, and extracts conflict lines.
        """
        if not os.path.exists(reject_file):
            logger.warning(f"❌ No `.rej` file found at {reject_file}. Skipping.")
            return None

        kernel_root = str(self.kernel_path)
        conflict_file = self.merge_conflicts_dir / f"{Path(reject_file).stem}_conflict.txt"

        try:
            logger.info(f"🔄 Running `gpatch --merge -p0 --output={conflict_file}` inside `{kernel_root}`")

            with open(reject_file, "r") as rej_file:
                subprocess.run(
                    [self.patch_command, "--merge", "-p0", f"--output={conflict_file}"],
                    cwd=kernel_root,
                    text=True,
                    stdin=rej_file,
                    check=False
                )

            if conflict_file.exists() and conflict_file.stat().st_size > 0:
                rejected_lines = self.extract_rejected_lines(conflict_file)
                logger.info(f"✅ Merge conflict saved at: {conflict_file}")
                return {
                    "conflict_file": str(conflict_file),
                    "rejected_lines": rejected_lines
                }
            else:
                logger.warning(f"⚠️ `gpatch` didn't generate conflicts for `{reject_file}`.")
                return None

        except subprocess.TimeoutExpired:
            logger.error(f"❌ `gpatch` timed out while processing {reject_file}")
            return None

    def extract_rejected_lines(self, conflict_file: Path):
        """Reads the merge conflict file and returns line numbers of conflict markers."""
        rejected_lines = []
        try:
            with open(conflict_file, "r") as f:
                for line_num, line in enumerate(f, start=1):
                    if line.strip().startswith(("<<<<<<<", "=======", ">>>>>>>")):
                        rejected_lines.append(line_num)  # Already an int
            logger.info(f"📌 Extracted rejected line numbers from `{conflict_file}`: {rejected_lines}")
        except Exception as e:
            logger.error(f"❌ Error while extracting rejected lines from `{conflict_file}`: {e}")
        return rejected_lines


    def process_all_rejects(self):
        """Processes all `.rej` files and returns list of dicts with conflict file and line info."""
        logger.info("🔍 Processing rejected patches from report...")

        rejected_files = self.get_rejected_files_from_report()
        rejected_patch_info = []

        for reject_file in rejected_files:
            logger.info(f"📂 Processing: {reject_file}")
            result = self.generate_infile_merge_conflict(reject_file)

            if result:
                conflict_file = result["conflict_file"]
                rejected_lines = result["rejected_lines"]


                # Reconstruct the original failed source file path (e.g., drivers/staging/android/ion/ion.c)
                failed_file_rel_path = os.path.relpath(reject_file, self.kernel_path).replace(".rej", "")

                rejected_patch_info.append({
                    "conflict_file": str(conflict_file),
                    "rejected_lines": rejected_lines,
                    "failed_file": failed_file_rel_path
                })
            else:
                logger.warning(f"⚠️ Failed to process: {reject_file}")

        logger.info(f"🔁 Total rejected patches: {len(rejected_patch_info)}")
        return rejected_patch_info




if __name__ == "__main__":
    KERNEL_PATH = "/Volumes/GitRepo/school/capstone/android/base"
    REPORT_PATH = "reports/patch_application_report.json"

    generator = InlineMergeGenerator(KERNEL_PATH, REPORT_PATH)
    generator.process_all_rejects()

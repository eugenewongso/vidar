import os
import json
import argparse
from patch_adoption.ast_processor import ASTProcessor
from patch_adoption.inline_merge_generator import InlineMergeGenerator

class RejectedPatchProcessor:
    def __init__(self, kernel_path, report_output_path, process_ast=True, process_merge=True, process_conflict=False):
        self.kernel_path = kernel_path
        self.report_output_path = report_output_path
        self.process_ast = process_ast
        self.process_merge = process_merge
        self.process_conflict = process_conflict
        self.ast_processor = ASTProcessor()
        self.merge_generator = InlineMergeGenerator(self.kernel_path, report_output_path)

    def process_rejected_patches(self):
        """Process rejected patches based on the selected options."""

        rejected_patches = []

        # Always generate rejected_lines if any processing is requested
        if self.process_merge or self.process_ast or self.process_conflict:
            print("🔍 Generating rejected lines from merge conflicts...")
            rejected_patches = self.merge_generator.process_all_rejects()
            print(f"🔁 Total rejected patches: {len(rejected_patches)}")
            for rp in rejected_patches:
                print(f"📄 Conflict file: {rp['conflict_file']}, lines: {rp['rejected_lines']}")

        if self.process_conflict:
            self.extract_inline_conflicts(rejected_patches)

        if self.process_merge and not self.process_ast:
            return

        if self.process_ast:
            for rejected_patch in rejected_patches:
                self.process_patch(rejected_patch)

    def extract_inline_conflicts(self, rejected_patches):
        """Extract inline merge conflict blocks from conflict files and save to separate files."""
        for patch in rejected_patches:
            conflict_path = os.path.join(self.kernel_path, patch['conflict_file'])
            output_path = conflict_path + "_inline.txt"
            try:
                with open(conflict_path, 'r') as f:
                    lines = f.readlines()

                in_conflict = False
                conflict_block = []
                extracted = []

                for i, line in enumerate(lines):
                    if line.startswith("<<<<<<<"):
                        in_conflict = True
                        conflict_block = [f"# Conflict starting at line {i + 1}\n", line]
                    elif in_conflict:
                        conflict_block.append(line)
                        if line.startswith("=======") or line.startswith(">>>>>>>"):
                            if line.startswith(">>>>>>>"):
                                extracted.extend(conflict_block)
                                extracted.append("\n")  # separate blocks
                                in_conflict = False

                if extracted:
                    with open(output_path, 'w') as out:
                        out.writelines(extracted)
                    print(f"📝 Extracted inline conflicts to {output_path}")
                else:
                    print(f"⚠️ No inline conflicts found in {conflict_path}")

            except FileNotFoundError:
                print(f"❌ Conflict file not found: {conflict_path}")


    def process_patch(self, rejected_patch):
        """Process a single rejected patch by generating AST with extracted rejected lines."""
        conflict_file = rejected_patch["conflict_file"]
        rejected_lines = rejected_patch["rejected_lines"]

        failed_file = rejected_patch["failed_file"]
        full_path = os.path.join(self.kernel_path, failed_file)

        print(f"\n🔍 Debug: Processing {failed_file}")
        print(f"📌 Rejected lines: {rejected_lines}")
        print(f"📂 Full path: {full_path}")

        rejected_lines = [int(l) for l in rejected_lines]
        ast_context = self.ast_processor.get_function_context(full_path, rejected_lines)
        if ast_context:
            ast_context_path = full_path + "_ast.txt"
            self.ast_processor.save_context(ast_context, ast_context_path)
            print(f"✅ AST saved at: {ast_context_path}")
        else:
            print(f"⚠️ AST extraction failed for {full_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process rejected patches for AST and inline merge conflicts.")
    parser.add_argument("--ast", action="store_true", help="Only process AST")
    parser.add_argument("--merge", action="store_true", help="Only process rejected_lines for merge analysis")
    parser.add_argument("--conflict", action="store_true", help="Extract inline merge conflict blocks")

    args = parser.parse_args()

    process_ast = args.ast or not (args.ast or args.merge or args.conflict)
    process_merge = args.merge or not (args.ast or args.merge or args.conflict)
    process_conflict = args.conflict or not (args.ast or args.merge or args.conflict)

    KERNEL_PATH = "/Volumes/GitRepo/school/capstone/android/Xiaomi_Kernel_OpenSource"
    REPORT_PATH = "reports/patch_application_report.json"

    processor = RejectedPatchProcessor(KERNEL_PATH, REPORT_PATH, process_ast, process_merge, process_conflict)
    processor.process_rejected_patches()

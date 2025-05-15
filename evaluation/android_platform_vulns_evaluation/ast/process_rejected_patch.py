import json
import argparse
import os
import re

class RejectedPatchProcessorFromJSON:
    def __init__(self, json_path, use_ast=True, context_lines=[3]):
        self.json_path = json_path
        self.use_ast = use_ast
        self.context_lines = context_lines

        if use_ast:
            from ast_processor import ASTProcessor
            self.ast_processor = ASTProcessor()

    def run(self):
        with open(self.json_path, "r") as f:
            data = json.load(f)

        for entry in data:
            for failure in entry.get("failures", []):
                for conflict in failure.get("file_conflicts", []):
                    file_path = conflict["file_name"]
                    source_code = conflict.get("downstream_file_content", "")
                    if not source_code:
                        print(f"⚠️ No source content for {file_path}")
                        continue

                    print(f"\n📂 Processing: {file_path}")

                    if self.use_ast:
                        self.process_with_ast(file_path, source_code, conflict)
                    else:
                        self.extract_context_block(file_path, source_code, conflict)

        suffix = "_with_ast.json" if self.use_ast else f"_with_context_{'_'.join(map(str, self.context_lines))}.json"
        output_path = self.json_path.replace(".json", suffix)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n💾 Saved output to: {output_path}")

    def process_with_ast(self, file_path, source_code, conflict):
        rej_content = conflict.get("rej_file_content", "")
        rejected_lines = self.extract_rejected_line_numbers(rej_content)

        try:
            context = self.ast_processor.get_function_context_from_source(
                source_code, rejected_lines, file_name=file_path
            )
            if context:
                conflict["ast_context"] = {
                    "functions": context.functions,
                    "dependent_types": context.dependent_types,
                    "includes": context.includes
                }
                conflict["ast_dump"] = (
                    "// Required includes\n" +
                    '\n'.join(f"#include {inc}" for inc in context.includes) + "\n\n" +
                    "// Required types and dependencies\n" +
                    '\n'.join(context.dependent_types) + "\n\n" +
                    "// Extracted Functions\n\n" +
                    '\n\n'.join(context.functions)
                )
                print(f"✅ AST embedded for: {file_path}")
            else:
                print(f"⚠️ AST extraction failed: {file_path}")
        except Exception as e:
            print(f"❌ Error processing AST for {file_path}: {e}")

    def extract_context_from_inline_output(self, conflict, source_lines, ctx):
        blocks = []
        merge_output = conflict.get("inline_merge_output", "")
        pattern = r"Hunk #\d+ NOT MERGED at (\d+)-(\d+)"
        matches = re.findall(pattern, merge_output)

        for start, end in matches:
            start = int(start)
            end = int(end)
            context_start = max(0, start - 1 - ctx)  # line numbers are 1-based
            context_end = min(len(source_lines), end + ctx)
            snippet = "\n".join(source_lines[context_start:context_end])
            blocks.append(snippet)
        return blocks


    def extract_context_block(self, file_path, source_code, conflict):
        source_lines = source_code.splitlines()

        # First: Try to match downstream blocks normally (if any)
        conflict_blocks = conflict.get("inline_merge_conflicts", [])
        fallback_done = False

        for ctx in self.context_lines:
            context_blocks = []

            for block in conflict_blocks:
                text = block.get("merge_conflict", "")
                if not text:
                    continue

                try:
                    match = re.search(r"<<<<<<< DOWNSTREAM.*?\n(.*?)=======\n", text, re.DOTALL)
                    if match:
                        downstream_raw = match.group(1)
                        downstream = downstream_raw.encode().decode("unicode_escape").strip()
                    else:
                        downstream = ""
                except Exception as e:
                    print(f"⚠️ Error parsing merge conflict: {e}")
                    downstream = ""

                if downstream:
                    match_start_char = source_code.find(downstream)
                    if match_start_char != -1:
                        match_start_line = source_code[:match_start_char].count("\n")
                        start = max(0, match_start_line - ctx)
                        end = min(len(source_lines), match_start_line + downstream.count("\n") + ctx)
                        snippet = "\n".join(source_lines[start:end])
                        context_blocks.append(snippet)
                        continue

                # Fall back to line numbers if downstream snippet not found
                fallback_done = True
                context_blocks += self.extract_context_from_inline_output(conflict, source_lines, ctx)

            key = f"downstream_file_content_context_{ctx}"
            if context_blocks:
                conflict[key] = context_blocks
                print(f"📌 Context ({ctx} lines) extracted for: {file_path}")
            else:
                print(f"⚠️ No context blocks extracted for {ctx} lines: {file_path}")

        if fallback_done:
            print(f"🔁 Fallback used for: {file_path}")


    def extract_rejected_line_numbers(self, rej_diff_text):
        lines = rej_diff_text.split("\n")
        rejected_line_numbers = []
        for line in lines:
            if line.startswith("@@"):
                try:
                    parts = line.split("@@")[1].strip()
                    target = parts.split(" ")[-1] if " " in parts else parts
                    if target.startswith('+'):
                        base = int(target[1:].split(',')[0])
                        rejected_line_numbers.append(base)
                except Exception as e:
                    print(f"⚠️ Failed to parse hunk line: {line} ({e})")
        return rejected_line_numbers

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process rejected patches from JSON")
    parser.add_argument("--json", required=True, help="Path to the filtered_failures JSON file")
    parser.add_argument("--no-ast", action="store_true", help="Use context blocks instead of AST")
    parser.add_argument("--context", type=int, nargs="+", default=[3], help="List of context line counts to extract")

    args = parser.parse_args()
    processor = RejectedPatchProcessorFromJSON(
        json_path=args.json,
        use_ast=not args.no_ast,
        context_lines=args.context
    )
    processor.run()

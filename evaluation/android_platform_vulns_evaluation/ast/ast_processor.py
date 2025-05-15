from clang import cindex
try:
    cindex.Config.set_library_file("/opt/homebrew/Cellar/llvm/20.1.4/lib/libclang.dylib")
    print("✅ Clang initialized successfully.")
except Exception as e:
    print(f"❌ Failed to initialize Clang: {e}")
from clang.cindex import TranslationUnit
from dataclasses import dataclass
from typing import List, Optional
import os

@dataclass
class ASTContext:
    functions: List[str]
    dependent_types: List[str]
    includes: List[str]

class ASTProcessor:
    def __init__(self):
        self.index = cindex.Index.create()

    def get_function_context_from_source(self, source_code: str, rejected_lines: List[int], file_name="tmp.c") -> Optional[ASTContext]:
        if not source_code.strip():
            print(f"❌ Error: Empty source code provided.")
            return None

        print(f"🔍 Debug: AST parsing {file_name}, rejected lines: {rejected_lines}")


        tu = self.index.parse(
            path=file_name,
            args=['-x', 'c', '-std=c11', '-I/usr/include', '-I/usr/local/include'],
            unsaved_files=[(file_name, source_code)],
            options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
        )
        if not tu.cursor:
            print(f"❌ AST Parsing Failed: {file_name}")

            return None

        functions = {}
        call_graph = {}
        helper_functions = set()
        dependent_types = set()
        includes = []

        for cursor in tu.cursor.walk_preorder():
            if cursor.kind == cindex.CursorKind.FUNCTION_DECL:
                start, end = cursor.extent.start.line, cursor.extent.end.line
                name = cursor.spelling
                call_graph[name] = set()

                file_lines = source_code.splitlines(keepends=True)

                function_body = file_lines[start - 1:end]

                for child in cursor.get_children():
                    if child.kind == cindex.CursorKind.CALL_EXPR and child.displayname:
                        call_graph[name].add(child.displayname)

                functions[name] = {
                    "start": start,
                    "end": end,
                    "calls": call_graph[name],
                    "code": self._trim_function(function_body, start, rejected_lines)
                }

            if cursor.kind in [
                cindex.CursorKind.STRUCT_DECL,
                cindex.CursorKind.TYPEDEF_DECL,
                cindex.CursorKind.ENUM_DECL,
                cindex.CursorKind.TYPE_REF
            ]:
                dependent_types.add(cursor.type.spelling)

            if cursor.kind == cindex.CursorKind.INCLUSION_DIRECTIVE:
                includes.append(cursor.spelling)

        selected_functions = set()
        for name, info in functions.items():
            if any(info["start"] <= line <= info["end"] for line in rejected_lines):
                selected_functions.add(name)
                helper_functions.update(call_graph[name])

        selected_functions.update(fn for fn in helper_functions if fn in functions)
        extracted_funcs = [functions[name]["code"] for name in selected_functions]
        filtered_dependencies = [t for t in dependent_types if any(t in f for f in extracted_funcs)]

        print(f"✅ Extracted functions: {list(selected_functions)}")
        print(f"✅ Extracted dependencies: {filtered_dependencies}")
        print(f"✅ Extracted includes: {includes}")

        return ASTContext(
            functions=extracted_funcs,
            dependent_types=filtered_dependencies,
            includes=includes
        )

    def _trim_function(self, body: List[str], start_line: int, rejected_lines: List[int]) -> str:
        if len(body) <= 50:
            return ''.join(body)

        local_rejected = [l - start_line for l in rejected_lines if start_line <= l < start_line + len(body)]
        snippet = []

        snippet += body[:15]
        snippet += ["\n..."]

        if local_rejected:
            min_line = max(min(local_rejected) - 3, 15)
            max_line = min(max(local_rejected) + 4, len(body) - 10)
            snippet += body[min_line:max_line]
            snippet += ["\n..."]

        snippet += body[-10:]
        return ''.join(snippet)

    def save_context(self, context: ASTContext, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        content = "// Required includes\n"
        content += '\n'.join(f"#include {inc}" for inc in context.includes)
        content += "\n\n// Required types and dependencies\n"
        content += '\n'.join(context.dependent_types)
        content += "\n\n// Extracted Functions\n\n"
        content += '\n\n'.join(context.functions)
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"💾 Debug: AST context saved at {output_path}")
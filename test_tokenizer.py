from tree_sitter import Language, Parser

LIB_PATH = "build/my-languages.so"

LANGUAGE_MAP = {
    "python": Language(LIB_PATH, "python"),
    "c": Language(LIB_PATH, "c"),
    "cpp": Language(LIB_PATH, "cpp"),
    "java": Language(LIB_PATH, "java"),
    "javascript": Language(LIB_PATH, "javascript"),
}

EXT_TO_LANG = {
    "py": "python",
    "c": "c", "h": "c",
    "cpp": "cpp", "cc": "cpp", "cxx": "cpp", "hpp": "cpp",
    "java": "java",
    "js": "javascript"
}

def get_language_from_filename(filename: str) -> str:
    ext = filename.rsplit('.', 1)[-1].lower()
    return EXT_TO_LANG.get(ext, None)

def tokenize_code_tree_sitter(code: str, lang_name: str) -> list[str]:
    parser = Parser()
    parser.set_language(LANGUAGE_MAP[lang_name])
    tree = parser.parse(code.encode("utf8"))
    root = tree.root_node

    def walk(node):
        if node.child_count == 0:
            return [code[node.start_byte:node.end_byte]]
        tokens = []
        for child in node.children:
            tokens.extend(walk(child))
        return tokens

    return walk(root)

# uncomment for testing
if __name__ == "__main__":
    java_code = """
    public class HelloWorld {
        public static void main(String[] args) {
            System.out.println("Hello, World!");
        }
    }
    """
    tokens = tokenize_code_tree_sitter(java_code, "java")
    print("Extracted tokens:")
    print(tokens)

from tree_sitter import Language

Language.build_library(
    # Output path for the shared library
    'build/my-languages.so',
    # Paths to the grammar repositories
    [
        'vendor/tree-sitter-c',
        'vendor/tree-sitter-cpp',
        'vendor/tree-sitter-python',
        'vendor/tree-sitter-java',
        'vendor/tree-sitter-javascript',
    ]
)

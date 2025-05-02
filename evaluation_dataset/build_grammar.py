from tree_sitter import Language

Language.build_library(
    # Output path for the compiled shared object
    'build/my-languages.so',
    
    # List of language grammar repos to include (you cloned this earlier)
    [
        'vendor/tree-sitter-c'
    ]
)

print("✅ Built grammar at build/my-languages.so")

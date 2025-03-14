from pathlib import Path


"""
📍 File: paths.py
This file defines all shared paths used in the project.

✅ You **do NOT need to hardcode absolute paths** like "/Users/yourname/..."

The `PROJECT_ROOT` is automatically detected as the root of the project, assuming this file lives inside the `vidar/` folder.

🔁 If your project folder is located somewhere else (e.g. ~/Documents/google-capstone), you do **not** need to change anything unless the folder structure itself is different.

💡 Example Project Structure:
google-capstone/
├── android/
│   └── Xiaomi_Kernel_OpenSource/
└── vidar/
    ├── patch_adoption/
    │   └── generated_patches/
    ├── reports/
    │   └── parsed_report.json
    ├── llm_integration/
    │   └── failed_patch.json
    └── paths.py  ← You are here

📦 If you cloned this repo and the structure is the same, you're all set!
Otherwise, make sure your local folders follow this structure or update paths accordingly.

"""

# Project root is the current folder
PROJECT_ROOT = Path(__file__).resolve().parent

# Kernel repo
KERNEL_PATH = PROJECT_ROOT.parent / "android" / "Xiaomi_Kernel_OpenSource"

# Output files
PARSED_REPORT_PATH = PROJECT_ROOT / "reports" / "parsed_report.json"
GENERATED_PATCHES_DIR = PROJECT_ROOT / "patch_adoption" / "generated_patches"
FAILED_PATCH_JSON = PROJECT_ROOT / "llm_integration" / "failed_patch.json"

# Output folder for raw diffs (fetched from upstream)
FETCHED_DIFF_DIR = PROJECT_ROOT / "fetch_patch_output" / "diff_output"

# Final report of patch application status
PATCH_APPLICATION_REPORT = PROJECT_ROOT / "reports" / "patch_application_report.json"

# Failed ones from gnu patch
FAILED_PATCH_JSON = PROJECT_ROOT / "llm_integration" / "failed_patch.json"

# Temporary location to test individual patches (can be set dynamically) 
# I think this it not necessary for now but will update -fila
SINGLE_PATCH_FILE = PROJECT_ROOT / "patch_adoption" / "generated_patches" / "f913f0123e6cff4dbc7c1e17d13b7a59a54475d2.diff_fixed.diff"

FAILED_PATCH_JSON = PROJECT_ROOT / "llm_integration" / "failed_patch.json"


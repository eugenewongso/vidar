from vertexai.preview.generative_models import (
    GenerativeModel,
    GenerationConfig,
    Content,
    Part,
    SafetySetting
)
import vertexai
import os
import json

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from paths import KERNEL_PATH
from paths import FAILED_PATCH_JSON


class LLMPatchGenerator:
    """Handles AI-based patch porting from upstream to downstream versions."""

    def __init__(self, kernel_path, project_id="vidar-452910", location="us-central1"):
        """Initializes the Vertex AI client and sets the kernel path."""
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel("gemini-1.0-pro")  # Use Gemini model name
        self.kernel_path = kernel_path

    def load_file(self, file_path, base_path=None):
        """Reads a file and returns its content as a string."""
        if base_path:
            file_path = os.path.join(base_path, file_path)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def generate_patch(self, upstream_diff_path, downstream_version_path, output_path=None):
        """Generates a security patch porting suggestion based on AI analysis."""
        try:
            upstream_diff = self.load_file(upstream_diff_path, base_path=self.kernel_path)
            downstream_version = self.load_file(downstream_version_path, base_path=self.kernel_path)

        except FileNotFoundError as e:
            print(e)
            return None

        user_prompt = f"""Task:
Your task is to port the security patch from the upstream_diff_file to the downstream_version.
The output should be a clean diff file that can be applied to the downstream_version without introducing any merge conflicts.

Upstream Diff:
{upstream_diff}

Downstream Version Code:
{downstream_version}

Instructions:
1. Analyze the upstream_diff_file to understand the changes introduced by the security patch.
2. Carefully examine the downstream_version codebase to identify any differences that might lead to merge conflicts or incompatibility issues.
3. Intelligently adapt the patch to the downstream_version, resolving any potential conflicts.
4. The output should be a single, clean diff file ready to be applied to the downstream_version.
5. Ensure syntactic correctness, logical consistency, and functional integrity after applying the patch.
6. Preserve indentation, structure, and accuracy in hunk headers (e.g., @@ -1,5 +1,5 @@)."""

        print("🚀 Generating ported patch...")

        contents = [
            Content(role="user", parts=[Part.from_text(user_prompt)])
        ]

        config = GenerationConfig(
            temperature=1,
            top_p=1,
            seed=0,
            max_output_tokens=2048,
            response_mime_type="text/plain"
        )

        safety_settings = [
            SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
            SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
            SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
        ]

        response_text = ""
        for chunk in self.model.generate_content(
            contents=contents,
            generation_config=config,
            safety_settings=safety_settings,
            stream=True
        ):
            response_text += chunk.text

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(response_text)
            print(f"✅ Ported patch saved to {output_path}")

        return response_text


# === Main Script ===
if __name__ == "__main__":
    kernel_path = str(KERNEL_PATH)
    old_volume_prefix = str(KERNEL_PATH)
    failed_patch_path = str(FAILED_PATCH_JSON)

    with open(failed_patch_path, "r") as f:
        failed_patches = json.load(f)

    generator = LLMPatchGenerator(kernel_path)

    for patch in failed_patches["patch"]:
        patch_file = patch["patch_file"]
        patch_url = patch["patch_url"]
        rejected_files = patch["rejected_files"]

        for file in rejected_files:
            failed_file = file["failed_file"]
            reject_file = file["reject_file"]

            if reject_file.startswith(old_volume_prefix):
                relative_reject_path = os.path.relpath(reject_file, old_volume_prefix)
                reject_file = relative_reject_path

            print(f"\n🔍 Processing failed patch: {patch_file}")
            print(f" - Failed File: {failed_file}")
            print(f" - Reject File: {reject_file}")

            output_patch_file = f"{patch_file}_fixed.diff"
            generated_patch = generator.generate_patch(reject_file, failed_file, output_patch_file)

            if generated_patch:
                print(f"🎉 Successfully generated patch for {patch_file}")
            else:
                print(f"⚠️ Failed to generate patch for {patch_file}")

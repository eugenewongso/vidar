from google import genai
from google.genai import types
import os
import json

class LLMPatchGenerator:
    """Handles AI-based patch porting from upstream to downstream versions."""

    def __init__(self, kernel_path, project_id="dev-smoke-452808-t2", location="us-central1"):
        """Initializes the Google Vertex AI client and sets the kernel path."""
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location
        )
        self.model = "gemini-2.0-pro-exp-02-05"
        self.kernel_path = kernel_path  # Path to the kernel source directory

    def load_file(self, file_path, base_path=None):
        """Reads a file and returns its content as a string. Uses base_path if provided."""
        if base_path:
            file_path = os.path.join(base_path, file_path)  # Ensure absolute path

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def generate_patch(self, upstream_diff_path, downstream_version_path, output_path=None):
        """Generates a security patch porting suggestion based on AI analysis."""
        try:
            upstream_diff = self.load_file(upstream_diff_path)  # No base path needed, already absolute
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

        system_instruction = """You are a specialized AI assistant designed for patch porting in software development. 
Your expertise lies in adapting security fixes from newer (upstream) codebases to older (downstream) versions, 
ensuring compatibility and resolving any merge conflicts."""

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_prompt)]
            )
        ]

        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=1,
            seed=0,
            max_output_tokens=2048,
            response_modalities=["TEXT"],
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
            ],
            system_instruction=[types.Part.from_text(text=system_instruction)],
        )

        # Call the AI model
        print("🚀 Generating ported patch...")
        response_text = ""

        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=generate_content_config,
        ):
            response_text += chunk.text

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(response_text)
            print(f"✅ Ported patch saved to {output_path}")

        return response_text


# === Main Script ===
if __name__ == "__main__":
    kernel_path = "/Volumes/GitRepo/school/capstone/android/Xiaomi_Kernel_OpenSource" # Hardcode
    failed_patch_path = os.path.join(os.path.dirname(__file__), "failed_patch.json")

    # Load failed patches JSON
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

            print(f"\n🔍 Processing failed patch: {patch_file}")
            print(f" - Failed File: {failed_file}")
            print(f" - Reject File: {reject_file}")

            # Generate new patched diff
            output_patch_file = f"{patch_file}_fixed.diff"
            generated_patch = generator.generate_patch(reject_file, failed_file, output_patch_file)

            if generated_patch:
                print(f"🎉 Successfully generated patch for {patch_file}")
            else:
                print(f"⚠️ Failed to generate patch for {patch_file}")

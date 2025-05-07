from google import genai
from google.genai import types
import os
import json
from datetime import datetime

class LLMPatchGenerator:
    """Handles AI-based patch porting from upstream to downstream versions."""

    def __init__(self, kernel_path, project_id="dev-smoke-452808-t2", location="us-central1"):
        """Initializes the Google Vertex AI client and sets the kernel path."""
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location
        )
        self.model = "gemini-2.5-pro-preview-03-25"
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
            return None, str(e)

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

        success = True
        error_message = None
        try:
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(response_text)
                print(f"✅ Ported patch saved to {output_path}")
        except Exception as e:
            success = False
            error_message = str(e)
            print(f"⚠️ Error saving patch: {error_message}")

        return response_text, error_message


# === Main Script ===
if __name__ == "__main__":
    kernel_path = "/Volumes/GitRepo/school/capstone/android/Xiaomi_Kernel_OpenSource"
    failed_patch_path = os.path.join(os.path.dirname(__file__), "failed_patch.json")

    # Create output directories
    base_dir = os.path.dirname(os.path.dirname(__file__))
    generated_patches_dir = os.path.join(base_dir, "patch_adoption", "generated_patches")
    reports_dir = os.path.join(base_dir, "reports")
    
    os.makedirs(generated_patches_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # Load failed patches JSON
    with open(failed_patch_path, "r") as f:
        failed_patches = json.load(f)

    generator = LLMPatchGenerator(kernel_path)
    
    # Initialize list to store individual patch results
    patch_results = []

    for patch in failed_patches["patches"]:
        patch_hash = os.path.splitext(patch["patch_file"])[0]  # Remove .diff extension
        patch_url = patch["patch_url"]
        message_output = patch["message_output"]

        for file in patch["rejected_files"]:
            failed_file = file["failed_file"]
            reject_file = file["reject_file"]

            print(f"\n🔍 Processing failed patch: {patch_hash}")
            print(f" - Failed File: {failed_file}")
            print(f" - Reject File: {reject_file}")

            # Generate output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{patch_hash}_{timestamp}.diff"
            output_path = os.path.join(generated_patches_dir, output_filename)

            # Generate new patched diff
            generated_patch, error = generator.generate_patch(reject_file, failed_file, output_path)

            # Create result entry for this specific patch attempt
            result = {
                "patch_hash": patch_hash,
                "patch_url": patch_url,
                "related_file": failed_file,
                "message": message_output,
                "generated_patch_path": output_path if not error else None,
                "success": bool(generated_patch and not error),
                "error": error,
                "timestamp": datetime.now().isoformat()
            }
            
            patch_results.append(result)

            if generated_patch and not error:
                print(f"🎉 Successfully generated patch for {patch_hash}")
            else:
                print(f"⚠️ Failed to generate patch for {patch_hash}")

    # Save results to report file
    report_path = os.path.join(reports_dir, "1_llm_output.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(patch_results, f, indent=2)

    print(f"\n✅ Results saved to {report_path}")

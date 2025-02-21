import os
import time
import google.generativeai as genai

class LLMPatchGenerator:
    """Handles file uploads and patch generation using the Gemini API."""
    
    def __init__(self, api_key=None, model_name="gemini-2.0-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "your-api-key")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name=model_name, generation_config={
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        })
    
    def upload_to_gemini(self, path, mime_type="text/plain"):
        """Uploads a file to Gemini."""
        file = genai.upload_file(path, mime_type=mime_type)
        print(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file
    
    def wait_for_files_active(self, files):
        """Waits until all uploaded files are in ACTIVE state."""
        print("Waiting for file processing...")
        for name in (file.name for file in files):
            file = genai.get_file(name)
            while file.state.name == "PROCESSING":
                print(".", end="", flush=True)
                time.sleep(10)
                file = genai.get_file(name)
            if file.state.name != "ACTIVE":
                raise Exception(f"File {file.name} failed to process")
        print("\nAll files are ready.")
    
    def generate_patch(self, file_paths):
        """Handles file uploads, waits for processing, and initiates patch generation."""
        files = [self.upload_to_gemini(path) for path in file_paths]
        self.wait_for_files_active(files)
        chat_session = self.model.start_chat(history=[
            {"role": "user", "parts": [
                files[0], files[1],
                "You will generate a diff file to apply security patches to the affected classes/functions..."
            ]},
        ])
        
        response = chat_session.send_message("Step 1: Analyze the reference patch and affected files. Summarize key differences.")
        time.sleep(3)
        response = chat_session.send_message("Step 2: Generate the diff file based on your analysis.")
        
        diff_start = response.text.find("<diff_output>")
        diff_end = response.text.find("</diff_output>")
        diff_content = response.text[diff_start + len("<diff_output>"):diff_end].strip() if diff_start != -1 and diff_end != -1 else response.text
        
        patch_file = "generated_patch.diff"
        with open(patch_file, "w") as f:
            f.write(diff_content)
        print(f"✅ Patch saved as {patch_file}")
        return patch_file

    def refine_patch(self, failed_patch, error_log):
        """Refines a previously generated patch using error logs and feedback."""
        chat_session = self.model.start_chat(history=[
            {
                "role": "user",
                "parts": [
                    failed_patch,
                    error_log,
                    "The previous patch did not apply cleanly. Analyze the error logs and adjust the patch accordingly."
                ],
            },
        ])
        
        response = chat_session.send_message("Refine the patch based on the errors and provide an updated diff file.")
        
        # Extract diff content
        diff_start = response.text.find("<diff_output>")
        diff_end = response.text.find("</diff_output>")
        refined_diff = response.text[diff_start + len("<diff_output>"):diff_end].strip() if diff_start != -1 and diff_end != -1 else response.text

        # Save refined patch
        refined_patch_file = "refined_patch.diff"
        with open(refined_patch_file, "w") as f:
            f.write(refined_diff)
        
        print(f"✅ Refined Patch saved as {refined_patch_file}")
        return refined_patch_file


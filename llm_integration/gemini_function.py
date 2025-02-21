import os
import time
import json
import google.generativeai as genai

class GeminiPatchGenerator:
    """Handles file uploads and patch generation using the Gemini API."""
    
    def __init__(self, api_key=None, model_name="gemini-2.0-flash", kernel_source_dir="Xiaomi_Kernel_OpenSource/"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "your-api-key")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name=model_name, generation_config={
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        })
        
        # Define the kernel source directory
        self.KERNEL_SOURCE_DIR = os.path.abspath(kernel_source_dir)
    
    def get_kernel_file_path(self, filename):
        """Constructs the absolute path to a kernel source file."""
        return os.path.join(self.KERNEL_SOURCE_DIR, filename)
    
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
    
    def generate_patch(self, json_file):
        """Reads JSON file, uploads relevant files, waits for processing, and generates patch."""
        with open(json_file, "r") as f:
            patch_data = json.load(f)
        
        file_paths = []
        for patch in patch_data["patches"]:
            for file_path in patch["files"].keys():
                # Resolve kernel file paths before using them
                resolved_path = self.get_kernel_file_path(file_path)
                file_paths.append(resolved_path)  # Store the absolute file path

        # Print resolved paths for debugging
        print("Resolved kernel file paths:")
        for path in file_paths:
            print(f" - {path}")

        files = [self.upload_to_gemini(path) for path in file_paths]
        self.wait_for_files_active(files)
        
        chat_session = self.model.start_chat(history=[
            {"role": "user", "parts": [
                files[0],
                    """You are an advanced security patching assistant. Your task is to generate a diff file that applies security patches to specific classes/functions within the provided affected files.

                    ### Identify Affected Code in the Source Files
                    - Review the provided affected files.
                    - Locate classes/functions that require security patching.
                    - Pay *close attention to line numbers* to ensure precise modifications.
                    - Understand the purpose of the existing code to ensure seamless integration of the patch.

                    ### Analyze the Security Patch Requirements
                    - Determine what security vulnerabilities exist in the affected code.
                    - Identify whether the fix requires:
                    - Input validation (e.g., sanitization, escaping user input)
                    - Secure memory management (e.g., avoiding buffer overflows, null pointer dereferencing)
                    - Permission handling (e.g., enforcing authentication/authorization)
                    - Cryptographic security (e.g., replacing weak algorithms)
                    - Fixing race conditions or concurrency issues

                    ### Apply the Security Patch
                    - Modify the code accordingly:
                    - Insert, update, or replace *only the necessary lines*.
                    - Ensure *correct syntax and indentation*.
                    - Follow *secure coding best practices* to prevent vulnerabilities.
                    - If the security fix introduces *logic conflicts*, suggest the most suitable resolution.

                    ### Generate and Format the Diff File
                    - Construct a *valid diff file* incorporating the security patch.
                    - Ensure that:
                    - The diff format is *correct and readable*.
                    - The patch integrates *seamlessly* into the existing codebase.
                    """
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

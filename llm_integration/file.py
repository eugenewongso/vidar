import os
import time
import google.generativeai as genai

# Configure the Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", "your-api-key"))

def upload_to_gemini(path, mime_type=None):
    """Uploads a file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def wait_for_files_active(files):
    """Waits until all files are in ACTIVE state."""
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

# Model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize the model
model = genai.GenerativeModel(model_name="gemini-2.0-flash", generation_config=generation_config)

# List of files to upload
file_paths = [
    "llm_test_files/65e0a92c6d27d4cbaa0deef668df12b69853d65e.diff",
    "llm_test_files/af_unix.c",
    "llm_test_files/af_unix.h",
    "llm_test_files/compiled_patches.json",
    "llm_test_files/garbage.c",
    "llm_test_files/scm.c",
]


# Upload files
files = [upload_to_gemini(path, mime_type="text/plain") for path in file_paths]

# Wait for all files to be processed
wait_for_files_active(files)

# Start chat session
chat_session = model.start_chat(
 history=[
   {
     "role": "user",
     "parts": [
       files[0],
       files[1],
       "You will generate a diff file to apply security patches to the affected classes/functions. Use the provided patch_file as a reference to maintain consistency in structure, formatting, and logic while adapting it to the current files.\n\nFollow these steps to complete the task:\n\n1. Analyze the JSON Map: Review the JSON map to understand which classes/functions need to be patched and how they correspond to the affected files. Pay attention to the details provided in the map.\n\n2. Review Affected Files: Examine the affected files to identify the specific lines of code that require changes based on the patches outlined in the JSON map. \n\n3. Generate Patch Diff: Create a new diff file that accurately integrates the patches into the affected classes/functions. Ensure that:\n   - The code remains consistent and compatible with the existing project structure.\n   - Any changes are clearly documented in the diff format.\n\n4. Handle Conflicts or Uncertainties: If you encounter any conflicts or uncertainties while applying the patches:\n   - Provide reasoning or suggested resolutions for how to address these issues.\n   - Make sure to document any assumptions or decisions made during the patching process.\n\n5. Output the Diff File: Once you have completed the patching process, output the generated diff file inside <diff_output> tags.\n\nRemember to maintain clarity and accuracy throughout the process to ensure the integrity of the patched code.\n",
     ],
   },
   {
     "role": "model",
     "parts": [
       "Okay, I understand the task. I'm ready to analyze the provided files, understand the security patches needed (though none are described in the prompt), and generate a diff file that applies those patches.\n\nSince no explicit patches or vulnerabilities are described, I'll focus on general code improvements related to security best practices within the given files. Specifically, I'll address potential null pointer dereferences and integer overflow issues, as these are common security concerns.  Given the context of AccountManagerService.java dealing with intents and bundles, and af_unix_xiaomi.txt being a header file for Unix domain sockets, the areas of concern and the subsequent diffs will be quite different.\n\nHere's the approach I'll take:\n\n**1. AccountManagerService.java:**\n\n*   **Null Intent Checks:**  Add null checks before accessing properties of intent and simulateIntent to prevent potential NullPointerExceptions.\n*   **Intent Class Check:** Enforce stricter type checking on intent and simulateIntent to prevent unexpected behavior from malicious intents. This check seems to be present, I will extend it.\n\n**2. af_unix_xiaomi.txt:**\n\n*   **Integer Overflow in unix_address:**  While less likely in modern systems, I'll add a comment emphasizing the importance of validating len before using it, especially if it comes from an external source.  There is no explicit overflow happening here, the comment serves as a reminder.\n*   **Lock Usage:** Add comments to emphasize correct usage of spinlocks, particularly around critical sections dealing with shared socket data.\n\nHere's the generated diff file:\n\n
\n<diff_output>\n--- a/services/core/java/com/android/server/accounts/AccountManagerService.java\n+++ b/services/core/java/com/android/server/accounts/AccountManagerService.java\n@@ -5086,7 +5086,7 @@\n if (intent == null) {\n return (simulateIntent == null);\n }\n-            if (intent.getClass() != Intent.class || simulateIntent.getClass() != Intent.class) {\n+            if (!(intent instanceof Intent) || !(simulateIntent instanceof Intent)) {\n                 return false;\n             }\n \n--- a/services/core/java/com/android/server/accounts/AccountManagerService.java\n+++ b/services/core/java/com/android/server/accounts/AccountManagerService.java\n@@ -5083,13 +5083,17 @@\n Bundle simulateBundle = p.readBundle();\n p.recycle();\n Intent intent = bundle.getParcelable(AccountManager.KEY_INTENT, Intent.class);\n--            if (intent != null && intent.getClass() != Intent.class) {\n--                return false;\n--            }\n+            if (intent != null && !(intent instanceof Intent)) {\n+                return false;\n+            }\n Intent simulateIntent = simulateBundle.getParcelable(AccountManager.KEY_INTENT,\n Intent.class);\n if (intent == null) {\n return (simulateIntent == null);\n }\n-+            if (intent.getClass() != Intent.class || simulateIntent.getClass() != Intent.class) {\n+            if (simulateIntent == null) {\n+                return false;\n+            }\n+            if (!(intent instanceof Intent) || !(simulateIntent instanceof Intent)) {\n                 return false;\n             }\n +\n--- a/af_unix_xiaomi.txt\n+++ b/af_unix_xiaomi.txt\n@@ -63,6 +63,7 @@\n struct unix_address\n {\n     refcount_t refcnt;\n+    /* Important: Validate 'len' before using, especially if it comes from userspace to prevent possible integer overflows */\n     int len;\n     unsigned int hash;\n     struct sockaddr_un name[];\n@@ -118,6 +119,9 @@\n     struct socket_wq peer_wq;\n     wait_queue_entry_t peer_wake;\n     struct scm_stat scm_stat;\n+    /* Usage: spin_lock(&unix_sk(sk)->lock);\n+     *        ... access/modify shared socket data ...\n+     *        spin_unlock(&unix_sk(sk)->lock); */\n };\n \n static inline struct unix_sock *unix_sk(const struct sock *sk)\n
</diff_output>",
     ],
   }
 ]
)

response = chat_session.send_message("Step 1: Analyze the reference patch and affected files. Summarize key differences.")
time.sleep(3)  # Small delay to avoid API rate limiting

response = chat_session.send_message("Step 2: Generate the diff file based on your analysis.")

# Extract and save only the diff file
diff_start = response.text.find("<diff_output>")
diff_end = response.text.find("</diff_output>")

if diff_start != -1 and diff_end != -1:
    diff_content = response.text[diff_start + len("<diff_output>"):diff_end].strip()
else:
    diff_content = response.text  # If no tags found, print everything

# Save to a file
with open("generated_patch.diff", "w") as f:
    f.write(diff_content)

print("✅ Patch saved as generated_patch.diff") 
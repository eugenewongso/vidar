import asyncio
import os
import datetime
import sys
import json
import argparse
import copy # Added for deepcopy
import logfire
from dataclasses import dataclass
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Optional, List, Dict, Any
import re
import time
import tiktoken
import subprocess
import tempfile
from unidiff import PatchSet
from android_patch_manager import AndroidPatchManager
from pathlib import Path

class APIKeyRotator:
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.index = 0

    def get_current_key(self):
        return self.api_keys[self.index]

    def rotate_key(self):
        self.index = (self.index + 1) % len(self.api_keys)
        print(f"🔄 Rotating to new API key index {self.index}")
        return self.get_current_key()



# Load environment variables from .env file
load_dotenv()

# Configure Logfire for logging (if token is available)
logfire.configure(send_to_logfire='if-token-present')

api_keys = os.getenv("GOOGLE_API_KEYS", "").split(",")
if not api_keys or api_keys == [""]:
    raise ValueError("Missing API keys in GOOGLE_API_KEYS")

key_rotator = APIKeyRotator(api_keys)

def get_repo_url_from_osv(vuln_id: str, osv_dir: str = "osv_data_android") -> Optional[str]:
    """
    Load OSV metadata JSON and construct repo URL from package name.
    """
    try:
        osv_path = os.path.join(osv_dir, f"{vuln_id}.json")
        with open(osv_path, "r", encoding="utf-8") as f:
            osv_data = json.load(f)
        affected = osv_data.get("affected", [])
        if affected and "package" in affected[0]:
            package_name = affected[0]["package"]["name"]  # e.g., "platform/frameworks/base"
            return f"https://android.googlesource.com/{package_name}"
    except Exception as e:
        print(f"❌ Failed to load repo_url from OSV for {vuln_id}: {e}")
    return None


def save_partial_output(path: str, data: Any):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"💾 Partial output saved to: {path}")
    except Exception as e:
        print(f"⚠️ Failed to save partial output to {path}: {e}")

def count_tokens_general(text: str):
    """
    Estimate token count based on word and character counts.

    Args:
        text (str): Input text.

    Returns:
        dict: Estimated token counts based on words and characters.
    """
    # Rough estimate: ~1 token = 0.75 words or ~4 chars/token
    word_count = len(re.findall(r'\w+', text))
    char_estimate = len(text) // 4
    return {
        "word_based": word_count,
        "char_based": char_estimate
    }


def count_tokens_tiktoken(text: str, model: str = "gpt-3.5-turbo"):
    """
    Count tokens using the Tiktoken library.

    Args:
        text (str): Input text.
        model (str): Model name.

    Returns:
        int: Total token count.
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def get_all_token_counts(text: str, gemini_token_count: Optional[int] = None):
    result = {
        "openai": count_tokens_tiktoken(text),
        "general": count_tokens_general(text),
    }
    if gemini_token_count is not None:
        result["gemini"] = gemini_token_count
    return result

@dataclass
class SupportDependencies: 
    rej_file_content: str # Renamed from diff_content for clarity
    original_source_file_content: str # Renamed from vulnerable_code_content

class GeminiAgent:
    def __init__(self, model_name: str, system_prompt: str, key_rotator: APIKeyRotator):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.key_rotator = key_rotator
        self._configure_genai()

    def _configure_genai(self):
        key = self.key_rotator.get_current_key()
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.system_prompt
        )

    async def run(self, prompt: str, deps: Optional[SupportDependencies] = None):
        for attempt in range(len(self.key_rotator.api_keys)):
            try:
                response = self.model.generate_content(prompt)
                token_count = None
                if hasattr(response, "usage_metadata"):
                    token_count = getattr(response.usage_metadata, "total_token_count", None)
                return type("Result", (), {"data": response.text, "token_count": token_count})
            except Exception as e:
                error_message = str(e).lower()
                if "quota" in error_message or "rate limit" in error_message or "internal error" in error_message:
                    print(f"⚠️ API error encountered: {e}")
                    self.key_rotator.rotate_key()
                    self._configure_genai()
                else:
                    raise e
        raise RuntimeError("All API keys exhausted or failed.")

def validate_patch_format(patch_content: str) -> tuple[bool, str]:
    """
    Validate patch format using unidiff library.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        PatchSet.from_string(patch_content)
        return True, "Valid patch format"
    except Exception as e:
        return False, f"Invalid patch format: {str(e)}"

def validate_patch_applicability(patch_content: str, original_file_content: str, target_filename: str) -> tuple[bool, str]:
    """
    Test if patch can be applied using GNU patch dry run.
    
    Returns:
        tuple: (can_apply, error_message)
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create original file
            original_file_path = os.path.join(temp_dir, target_filename)
            with open(original_file_path, 'w', encoding='utf-8') as f:
                f.write(original_file_content)
            
            # Create patch file
            patch_file_path = os.path.join(temp_dir, "test.patch")
            with open(patch_file_path, 'w', encoding='utf-8') as f:
                f.write(patch_content)
            
            # Test patch application with dry run
            result = subprocess.run(
                ['patch', '--dry-run', '-p1', '-i', patch_file_path],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return True, "Patch can be applied successfully"
            else:
                return False, f"Patch cannot be applied: {result.stderr}"
                
    except subprocess.TimeoutExpired:
        return False, "Patch validation timed out"
    except Exception as e:
        return False, f"Error during patch validation: {str(e)}"


def validate_patch_applicability_in_repo(patch_content: str, repo_path: str) -> tuple[bool, str]:
    """
    Test if patch can be applied using GNU patch dry run in actual repo.
    
    Returns:
        tuple: (can_apply, error_message)
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".diff", mode="w") as f:
            f.write(patch_content)
            patch_file_path = f.name

        result = subprocess.run(
            ['patch', '--dry-run', '-p1', '-i', patch_file_path],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            return True, "Patch applies cleanly in repo"
        else:
            return False, f"Patch failed in repo: {(result.stdout + result.stderr).strip()}"
    except subprocess.TimeoutExpired:
        return False, "Patch validation timed out"
    except Exception as e:
        return False, f"Error during patch validation: {str(e)}"


async def process_single_entry_with_retry(
    rej_content: str,
    original_source_content: str,
    target_filename_for_diff: str,
    vulnerability_id: str,
    failure_details: Dict[str, Any],
    upstream_patch_content: str,
    patch_porter_agent: GeminiAgent,
    max_retries: int = 3
):
    """
    Processes a single vulnerability entry with retry logic for validation.
    """
    # Type checks for content fields
    if not isinstance(rej_content, str):
        print(f"Skipping entry for {target_filename_for_diff} in {vulnerability_id} because .rej File Content is not a string.")
        return {
            "llm_output_valid": False,
            "runtime_seconds": 0,
            "attempts_made": 0,
            "validation_results": [],
            "error": ".rej File Content is not a string",
            "last_format_error": None,
            "last_apply_error": None
        }

    if not isinstance(original_source_content, str):
        print(f"Skipping entry for {target_filename_for_diff} in {vulnerability_id} because Original Source File content is not a string.")
        return {
            "llm_output_valid": False,
            "runtime_seconds": 0,
            "attempts_made": 0,
            "validation_results": [],
            "error": "Original Source File content is not a string",
            "last_format_error": None,
            "last_apply_error": None
        }


    # Existing checks for empty/placeholder content
    if not original_source_content or original_source_content.strip() == "```" or original_source_content.strip() == "":
        print(f"Skipping entry for {target_filename_for_diff} in {vulnerability_id} due to empty or placeholder Original Source File content.")
        return {
            "llm_output_valid": False,
            "runtime_seconds": 0,
            "attempts_made": 0,
            "validation_results": [],
            "error": "Empty or placeholder Original Source File content",
            "last_format_error": None,
            "last_apply_error": None
        }

    if not rej_content or rej_content.strip() == "```" or rej_content.strip() == "":
        print(f"Skipping entry for {target_filename_for_diff} in {vulnerability_id} due to empty or placeholder .rej File Content. A .rej file is expected.")
        return {
            "llm_output_valid": False,
            "runtime_seconds": 0,
            "attempts_made": 0,
            "validation_results": [],
            "error": "Empty or placeholder .rej File Content",
            "last_format_error": None,
            "last_apply_error": None
        }


    dependencies = SupportDependencies(
        rej_file_content=rej_content, 
        original_source_file_content=original_source_content,
    )

    base_task_prompt = f"""You are an expert software patch generation assistant. Your task is to resolve the conflicts in the provided `.rej` file by generating a corrected unified diff that applies cleanly to the 'Original Source File'.
Adhere strictly to your system instructions regarding format and content, especially the requirement for a single ```diff markdown block output.

Inputs:
1.  **Original Source File**: The content of the file that needs to be patched.
    ```text
    {dependencies.original_source_file_content}
    ```

2.  **.rej File Content**: The rejected hunks from a previous patch attempt. These are the changes you MUST adapt and correct. Pay close attention to the `@@ ... @@` headers in this content to determine the correct number of lines for your output hunk.
    ```text
    {dependencies.rej_file_content}
    ```

3.  **Target Filename**: Use this exact name for the `--- a/` and `+++ b/` lines in your diff.
    `{target_filename_for_diff}`

Your Specific Task:
Generate a corrected unified diff.
*   The diff must take the changes described in the '.rej File Content' and make them apply cleanly to the 'Original Source File'.
*   Address line number offsets or context mismatches.
*   Preserve the original intent of the security patch.
*   Strictly adhere to line counts and structure indicated by the `.rej` file's hunk headers.
*   Ensure all lines within your generated hunks start with ' ' (space), '+', or '-'.

Output Requirements (Strictly Enforced as per System Instructions):
1.  **ONLY THE DIFF in a ```diff block**: Your entire response must be *exclusively* a single markdown ```diff code block, starting with ```diff\\n and ending with ```.
2.  **Correct Headers**: Start the diff content (inside the block) with `--- a/{target_filename_for_diff}` and `+++ b/{target_filename_for_diff}`.
3.  **Valid Hunks**: Ensure all hunk headers (`@@ -old,count +new,count @@`) are correct and accurately reflect the changes and context, matching the structure derived from the input `.rej` file.

Example of Expected Output Format (Illustrative - actual content will depend on inputs):
```diff
--- a/{target_filename_for_diff}
+++ b/{target_filename_for_diff}
@@ -12,5 +12,5 @@ // Example: this header must match the logic of the resolved .rej hunk
 context line 1 (unchanged)
 context line 2 (unchanged)
-old line to be removed
+new line to be added
 context line 3 (unchanged)
```
"""

    print(f"Processing for diff generation: {vulnerability_id} - {target_filename_for_diff}")
    
    total_start_time = time.monotonic()
    validation_results = []
    
    for attempt in range(max_retries):
        print(f"🔄 Attempt {attempt + 1} of {max_retries} for {target_filename_for_diff}")
        
        # For a truly blind retry with consistent guidelines, the task_prompt is always the base_task_prompt.
        task_prompt = base_task_prompt
        
        if attempt > 0:
            # Log previous errors to console for debugging, but they are not added to the LLM prompt.
            prev_errors = []
            for r in validation_results:
                if not r["valid"]:
                    if not r["format_valid"]:
                        prev_errors.append(f"[Format Error] {r['format_error']}")
                    elif not r["apply_valid"]:
                        prev_errors.append(f"[Apply Error] {r['apply_error']}")
            
            if prev_errors:
                print("Previous errors (for console, not added to LLM prompt for blind retry):", prev_errors)
            else:
                print("Previous attempt failed, retrying with the same prompt (blind retry).")

        try:
            start_time = time.monotonic()
            result = await patch_porter_agent.run(task_prompt)
            end_time = time.monotonic()
            generated_diff = result.data.strip()

            # # Print LLM output for debugging
            # print("🔍 LLM-Generated Unified Diff:")
            # print("-" * 60)
            # print(generated_diff[:2000])  # Limit to 2000 chars to avoid console flooding
            # print("-" * 60)


            # Validation 1: Check patch format
            format_valid, format_error = validate_patch_format(generated_diff)
            
            # Validation 2: Check if patch can be applied (only if format is valid)
            apply_valid, apply_error = False, "Skipped due to format error"
            if format_valid:
                repo_path_str = failure_details.get("repo_path")
                downstream_version = failure_details.get("downstream_version")
                downstream_patch_sha = failure_details.get("downstream_patch")

                if not repo_path_str or not downstream_version or not downstream_patch_sha or not upstream_patch_content:
                    print(f"⚠️ Missing required repo metadata for validation: {vulnerability_id} - {target_filename_for_diff}")
                    return None

                repo_path = Path(repo_path_str)

                # Clean repo
                AndroidPatchManager.clean_repo(repo_path)

                # Checkout branch
                AndroidPatchManager.checkout_downstream_branch(repo_path, downstream_version)

                # Reset to the commit before the downstream patch
                AndroidPatchManager.checkout_commit(repo_path, f"{downstream_patch_sha}^")

                # Save upstream patch to a temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".patch", mode="w") as temp_patch_file:
                    temp_patch_file.write(upstream_patch_content)
                    temp_patch_path = temp_patch_file.name

                # Actually apply the upstream patch (allow partial success)
                success, message, stdout, stderr = AndroidPatchManager.apply_patch(repo_path, temp_patch_path)

                if not success:
                    print(f"⚠️ Partial application of upstream patch (some hunks rejected) for {vulnerability_id} - {target_filename_for_diff}")
                    # Continue — this is expected!
                else:
                    print(f"✅ Upstream patch applied (fully or partially) for {vulnerability_id} - {target_filename_for_diff}")


                apply_valid, apply_error = validate_patch_applicability_in_repo(
                    generated_diff,
                    str(repo_path) 
                )

            
            validation_result = {
                "attempt": attempt + 1,
                "format_valid": format_valid,
                "format_error": format_error,
                "apply_valid": apply_valid,
                "apply_error": apply_error,
                "valid": format_valid and apply_valid,
                "runtime_seconds": round(end_time - start_time, 2)
            }
            validation_results.append(validation_result)
            
            print(f"📊 Validation results for attempt {attempt + 1}:")
            print(f"  - Format valid: {format_valid}")
            print(f"  - Can apply: {apply_valid}")
            
            if validation_result["valid"]:
                total_end_time = time.monotonic()
                print(f"✅ LLM diff generation and validation successful for: {vulnerability_id} - {target_filename_for_diff}")
                return {
                    "downstream_llm_diff_output": generated_diff,
                    "llm_output_valid": True,
                    "runtime_seconds": round(total_end_time - total_start_time, 2),
                    "attempts_made": attempt + 1,
                    "validation_results": validation_results,
                    "token_counts": get_all_token_counts(generated_diff, gemini_token_count=result.token_count)
                }
            else:
                print(f"⚠️ Validation failed for attempt {attempt + 1}, retrying...")
                if attempt == max_retries - 1:
                    print(f"❌ All {max_retries} attempts failed validation for {target_filename_for_diff}")

        except Exception as e:
            validation_results.append({
                "attempt": attempt + 1,
                "error": f"Exception during generation: {str(e)}",
                "valid": False,
                "format_valid": False,
                "format_error": "Skipped due to exception",
                "apply_valid": False,
                "apply_error": "Skipped due to exception"
            })

            print(f"❌ Error on attempt {attempt + 1} for {target_filename_for_diff}: {e}")
            if attempt == max_retries - 1:
                print(f"❌ All {max_retries} attempts failed for {target_filename_for_diff}")

    # If we get here, all attempts failed
    total_end_time = time.monotonic()
    return {
        "downstream_llm_diff_output": None,
        "runtime_seconds": round(total_end_time - total_start_time, 2),
        "llm_output_valid": False,
        "attempts_made": max_retries,
        "validation_results": validation_results,
        "error": "All validation attempts failed" if validation_results else "Exception occurred before any validation",
        "last_format_error": validation_results[-1]["format_error"] if validation_results else None,
        "last_apply_error": validation_results[-1]["apply_error"] if validation_results else None
    }

async def main():
    parser = argparse.ArgumentParser(description="Process vulnerability JSON, generate corrected diffs using an LLM, and output an updated JSON.")
    parser.add_argument("input_json_file_path", help="Path to the input JSON file.")
    parser.add_argument(
        "--target_downstream_version", "-v",
        help="(Optional) Filter by specific downstream_version (e.g., '14'). If not provided, all versions will be processed.",
        default=None
    )
    parser.add_argument("--output_json_path", "-o", help="Path to save the output JSON file (with generated diffs). Defaults to outputs/approach2_output_diff_android_{version}_{timestamp}.json", default=None)
    parser.add_argument(
        "--resume_from_id",
        help="(Optional) Vulnerability ID to resume from. All earlier entries will be skipped.",
        default=None
    )

    
    args = parser.parse_args()

    system_prompt = '''You are an expert software patch generation assistant.
Your primary task is to generate *perfectly formatted* unified diffs, enclosed in a ```diff markdown block, to resolve software conflicts presented in `.rej` files.
Adherence to the unified diff format and the specified output structure is paramount.

Key Objectives:
1.  **Analyze Rejects**: Carefully examine the provided '.rej File Content' and 'Original Source File'.
2.  **Generate Corrected Diff**: Create a new unified diff that incorporates the intended changes from the .rej file, adjusted to apply successfully to the 'Original Source File'.
3.  **Minimal Changes**: Modify *only* what is absolutely necessary within the rejected hunks to ensure they apply correctly and preserve the original intent of the patch.
4.  **Security Focus**: While ensuring correct formatting, the goal is to apply a security patch. Ensure your corrections maintain or achieve the intended security remediation.

STRICT FORMATTING AND OUTPUT REQUIREMENTS:
*   **Markdown Diff Block**: Your entire response MUST be a single markdown code block of type 'diff'. Start with ```diff (with a newline after it) and end with ```.
*   **No Extra Text**: Absolutely NO explanations, NO comments, NO apologies, NO summaries, or any other text outside this single ```diff ... ``` block.
*   **Diff Content - Headers**: Inside the block, the diff MUST start with `--- a/FILENAME` and `+++ b/FILENAME` lines, where FILENAME is the target filename provided to you in the task.
*   **Diff Content - Hunk Structure and Line Counts**:
    *   You MUST generate hunks that precisely match the line counts implied by the input `.rej` file's hunk headers (e.g., `@@ -old_start,old_lines +new_start,new_lines @@`).
    *   The output hunk should contain `old_lines` from the original (prefixed with ' ' for context, '-' for removed) and `new_lines` for the patched version (prefixed with ' ' for context, '+' for added).
    *   Do NOT add extraneous context lines beyond what is necessary to make the patch apply cleanly and match the implied line counts.
*   **Diff Content - Line Prefixes**: Every content line *within* a diff hunk (after the `@@ ... @@` header) MUST start with a ' ' (space for context), '-' (for removed lines), or '+' (for added lines).
*   **No Nested Code Blocks**: Do NOT use further markdown code fences (like ```) *inside* the main diff content.

Here is an example of a successful transformation. Pay close attention to how the `.rej` file is resolved against the original source snippets to produce the "Expected Output Diff".

**Few-Shot Example:**

**Target Filename:**
`services/usb/java/com/android/server/usb/UsbDeviceManager.java`

**Original Source (services/usb/java/com/android/server/usb/UsbDeviceManager.java):**
```java
/*
 * Copyright (C) 2011 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions an
 * limitations under the License.
 */

package com.android.server.usb;

import static android.hardware.usb.UsbPortStatus.DATA_ROLE_DEVICE;
import static android.hardware.usb.UsbPortStatus.DATA_ROLE_HOST;
import static android.hardware.usb.UsbPortStatus.MODE_AUDIO_ACCESSORY;
import static android.hardware.usb.UsbPortStatus.POWER_ROLE_SINK;
import static android.hardware.usb.UsbPortStatus.POWER_ROLE_SOURCE;

import static com.android.internal.usb.DumpUtils.writeAccessory;
import static com.android.internal.util.dump.DumpUtils.writeStringIfNotNull;

import android.app.ActivityManager;
import android.app.KeyguardManager;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.BroadcastReceiver;
import android.content.ComponentName;
import android.content.ContentResolver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.content.res.Resources;
import android.debug.AdbManagerInternal;
import android.debug.AdbNotifications;
import android.debug.AdbTransportType;
import android.debug.IAdbTransport;
import android.hardware.usb.ParcelableUsbPort;
import android.hardware.usb.UsbAccessory;
import android.hardware.usb.UsbConfiguration;
import android.hardware.usb.UsbConstants;
import android.hardware.usb.UsbDevice;
import android.hardware.usb.UsbInterface;
import android.hardware.usb.UsbManager;
import android.hardware.usb.UsbPort;
import android.hardware.usb.UsbPortStatus;
import android.hardware.usb.gadget.V1_0.GadgetFunction;
import android.hardware.usb.gadget.V1_0.IUsbGadget;
import android.hardware.usb.gadget.V1_0.IUsbGadgetCallback;
import android.hardware.usb.gadget.V1_0.Status;
import android.hidl.manager.V1_0.IServiceManager;
import android.hidl.manager.V1_0.IServiceNotification;
import android.os.BatteryManager;
import android.os.Environment;
import android.os.FileUtils;
import android.os.Handler;
import android.os.HwBinder;
import android.os.Looper;
import android.os.Message;
import android.os.ParcelFileDescriptor;
import android.os.RemoteException;
import android.os.SystemClock;
import android.os.SystemProperties;
import android.os.UEventObserver;
import android.os.UserHandle;
import android.os.UserManager;
import android.os.storage.StorageManager;
import android.os.storage.StorageVolume;
import android.provider.Settings;
import android.service.usb.UsbDeviceManagerProto;
import android.service.usb.UsbHandlerProto;
import android.util.Pair;
import android.util.Slog;

import com.android.internal.annotations.GuardedBy;
import com.android.internal.logging.MetricsLogger;
import com.android.internal.logging.nano.MetricsProto.MetricsEvent;
import com.android.internal.messages.nano.SystemMessageProto.SystemMessage;
import com.android.internal.notification.SystemNotificationChannels;
import com.android.internal.os.SomeArgs;
import com.android.internal.util.dump.DualDumpOutputStream;
import com.android.server.FgThread;
import com.android.server.LocalServices;
import com.android.server.wm.ActivityTaskManagerInternal;

import java.io.File;
import java.io.FileDescriptor;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Locale;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Scanner;
import java.util.Set;

/**
 * UsbDeviceManager manages USB state in device mode.
 */
public class UsbDeviceManager implements ActivityTaskManagerInternal.ScreenObserver {

    private static final String TAG = UsbDeviceManager.class.getSimpleName();
    private static final boolean DEBUG = false;

    /**
     * The name of the xml file in which screen unlocked functions are stored.
     */
    private static final String USB_PREFS_XML = "UsbDeviceManagerPrefs.xml";

    /**
     * The SharedPreference setting per user that stores the screen unlocked functions between
     * sessions.
     */
    static final String UNLOCKED_CONFIG_PREF = "usb-screen-unlocked-config-%d";

    /**
     * ro.bootmode value when phone boots into usual Android.
     */
    private static final String NORMAL_BOOT = "normal";

    private static final String USB_STATE_MATCH =
            "DEVPATH=/devices/virtual/android_usb/android0";
    private static final String ACCESSORY_START_MATCH =
            "DEVPATH=/devices/virtual/misc/usb_accessory";
    private static final String FUNCTIONS_PATH =
            "/sys/class/android_usb/android0/functions";
    private static final String STATE_PATH =
            "/sys/class/android_usb/android0/state";
    private static final String RNDIS_ETH_ADDR_PATH =
            "/sys/class/android_usb/android0/f_rndis/ethaddr";
    private static final String AUDIO_SOURCE_PCM_PATH =
            "/sys/class/android_usb/android0/f_audio_source/pcm";
    private static final String MIDI_ALSA_PATH =
            "/sys/class/android_usb/android0/f_midi/alsa";

    private static final int MSG_UPDATE_STATE = 0;
    private static final int MSG_ENABLE_ADB = 1;
    private static final int MSG_SET_CURRENT_FUNCTIONS = 2;
    private static final int MSG_SYSTEM_READY = 3;
    private static final int MSG_BOOT_COMPLETED = 4;
    private static final int MSG_USER_SWITCHED = 5;
    private static final int MSG_UPDATE_USER_RESTRICTIONS = 6;
    private static final int MSG_UPDATE_PORT_STATE = 7;
    private static final int MSG_ACCESSORY_MODE_ENTER_TIMEOUT = 8;
    private static final int MSG_UPDATE_CHARGING_STATE = 9;
    private static final int MSG_UPDATE_HOST_STATE = 10;
    private static final int MSG_LOCALE_CHANGED = 11;
    private static final int MSG_SET_SCREEN_UNLOCKED_FUNCTIONS = 12;
    private static final int MSG_UPDATE_SCREEN_LOCK = 13;
    private static final int MSG_SET_CHARGING_FUNCTIONS = 14;
    private static final int MSG_SET_FUNCTIONS_TIMEOUT = 15;
    private static final int MSG_GET_CURRENT_USB_FUNCTIONS = 16;
    private static final int MSG_FUNCTION_SWITCH_TIMEOUT = 17;
    private static final int MSG_GADGET_HAL_REGISTERED = 18;
    private static final int MSG_RESET_USB_GADGET = 19;

    private static final int AUDIO_MODE_SOURCE = 1;

    // Delay for debouncing USB disconnects.
    // We often get rapid connect/disconnect events when enabling USB functions,
    // which need debouncing.
    private static final int UPDATE_DELAY = 1000;

    // Timeout for entering USB request mode.
    // Request is cancelled if host does not configure device within 10 seconds.
    private static final int ACCESSORY_REQUEST_TIMEOUT = 10 * 1000;

    private static final String BOOT_MODE_PROPERTY = "ro.bootmode";

    private static final String ADB_NOTIFICATION_CHANNEL_ID_TV = "usbdevicemanager.adb.tv";
    private UsbHandler mHandler;

    private final Object mLock = new Object();

    private final Context mContext;
    private final ContentResolver mContentResolver;
    @GuardedBy("mLock")
    private UsbProfileGroupSettingsManager mCurrentSettings;
    private final boolean mHasUsbAccessory;
    @GuardedBy("mLock")
    private String[] mAccessoryStrings;
    private final UEventObserver mUEventObserver;

    private static Set<Integer> sBlackListedInterfaces;
    private HashMap<Long, FileDescriptor> mControlFds;

    static {
        sBlackListedInterfaces = new HashSet<>();
        sBlackListedInterfaces.add(UsbConstants.USB_CLASS_AUDIO);
        sBlackListedInterfaces.add(UsbConstants.USB_CLASS_COMM);
        sBlackListedInterfaces.add(UsbConstants.USB_CLASS_HID);
        sBlackListedInterfaces.add(UsbConstants.USB_CLASS_PRINTER);
        sBlackListedInterfaces.add(UsbConstants.USB_CLASS_MASS_STORAGE);
        sBlackListedInterfaces.add(UsbConstants.USB_CLASS_HUB);
        sBlackListedInterfaces.add(UsbConstants.USB_CLASS_CDC_DATA);
        sBlackListedInterfaces.add(UsbConstants.USB_CLASS_CSCID);
        sBlackListedInterfaces.add(UsbConstants.USB_CLASS_CONTENT_SEC);
        sBlackListedInterfaces.add(UsbConstants.USB_CLASS_VIDEO);
        sBlackListedInterfaces.add(UsbConstants.USB_CLASS_WIRELESS_CONTROLLER);
    }

    /*
     * Listens for uevent messages from the kernel to monitor the USB state
     */
    private final class UsbUEventObserver extends UEventObserver {
        @Override
        public void onUEvent(UEventObserver.UEvent event) {
            if (DEBUG) Slog.v(TAG, "USB UEVENT: " + event.toString());

            String state = event.get("USB_STATE");
            String accessory = event.get("ACCESSORY");
            if (state != null) {
                mHandler.updateState(state);
            } else if ("START".equals(accessory)) {
                if (DEBUG) Slog.d(TAG, "got accessory start");
                startAccessoryMode();
            }
        }
    }

    @Override
    public void onKeyguardStateChanged(boolean isShowing) {
        int userHandle = ActivityManager.getCurrentUser();
        boolean secure = mContext.getSystemService(KeyguardManager.class)
                .isDeviceSecure(userHandle);
        if (DEBUG) {
            Slog.v(TAG, "onKeyguardStateChanged: isShowing:" + isShowing + " secure:" + secure
                    + " user:" + userHandle);
        }
        // We are unlocked when the keyguard is down or non-secure.
        mHandler.sendMessage(MSG_UPDATE_SCREEN_LOCK, (isShowing && secure));
    }

    @Override
    public void onAwakeStateChanged(boolean isAwake) {
        // ignore
    }

    /** Called when a user is unlocked. */
    public void onUnlockUser(int userHandle) {
        onKeyguardStateChanged(false);
    }

    public UsbDeviceManager(Context context, UsbAlsaManager alsaManager,
            UsbSettingsManager settingsManager, UsbPermissionManager permissionManager) {
        mContext = context;
        mContentResolver = context.getContentResolver();
        PackageManager pm = mContext.getPackageManager();
        mHasUsbAccessory = pm.hasSystemFeature(PackageManager.FEATURE_USB_ACCESSORY);
        initRndisAddress();

        boolean halNotPresent = false;
        try {
            IUsbGadget.getService(true);
        } catch (RemoteException e) {
            Slog.e(TAG, "USB GADGET HAL present but exception thrown", e);
        } catch (NoSuchElementException e) {
            halNotPresent = true;
            Slog.i(TAG, "USB GADGET HAL not present in the device", e);
        }

        mControlFds = new HashMap<>();
        FileDescriptor mtpFd = nativeOpenControl(UsbManager.USB_FUNCTION_MTP);
        if (mtpFd == null) {
            Slog.e(TAG, "Failed to open control for mtp");
        }
        mControlFds.put(UsbManager.FUNCTION_MTP, mtpFd);
        FileDescriptor ptpFd = nativeOpenControl(UsbManager.USB_FUNCTION_PTP);
        if (ptpFd == null) {
            Slog.e(TAG, "Failed to open control for ptp");
        }
        mControlFds.put(UsbManager.FUNCTION_PTP, ptpFd);

        if (halNotPresent) {
            /**
             * Initialze the legacy UsbHandler
             */
            mHandler = new UsbHandlerLegacy(FgThread.get().getLooper(), mContext, this,
                    alsaManager, permissionManager);
        } else {
            /**
             * Initialize HAL based UsbHandler
             */
            mHandler = new UsbHandlerHal(FgThread.get().getLooper(), mContext, this,
                    alsaManager, permissionManager);
        }

        if (nativeIsStartRequested()) {
            if (DEBUG) Slog.d(TAG, "accessory attached at boot");
            startAccessoryMode();
        }

        BroadcastReceiver portReceiver = new BroadcastReceiver() {
            @Override
            public void onReceive(Context context, Intent intent) {
                ParcelableUsbPort port = intent.getParcelableExtra(UsbManager.EXTRA_PORT);
                UsbPortStatus status = intent.getParcelableExtra(UsbManager.EXTRA_PORT_STATUS);
                mHandler.updateHostState(
                        port.getUsbPort(context.getSystemService(UsbManager.class)), status);
            }
        };

        BroadcastReceiver chargingReceiver = new BroadcastReceiver() {
            @Override
            public void onReceive(Context context, Intent intent) {
                int chargePlug = intent.getIntExtra(BatteryManager.EXTRA_PLUGGED, -1);
                boolean usbCharging = chargePlug == BatteryManager.BATTERY_PLUGGED_USB;
                mHandler.sendMessage(MSG_UPDATE_CHARGING_STATE, usbCharging);
            }
        };

        BroadcastReceiver hostReceiver = new BroadcastReceiver() {
            @Override
            public void onReceive(Context context, Intent intent) {
                Iterator devices = ((UsbManager) context.getSystemService(Context.USB_SERVICE))
                        .getDeviceList().entrySet().iterator();
                if (intent.getAction().equals(UsbManager.ACTION_USB_DEVICE_ATTACHED)) {
                    mHandler.sendMessage(MSG_UPDATE_HOST_STATE, devices, true);
                } else {
                    mHandler.sendMessage(MSG_UPDATE_HOST_STATE, devices, false);
                }
            }
        };

        BroadcastReceiver languageChangedReceiver = new BroadcastReceiver() {
            @Override
            public void onReceive(Context context, Intent intent) {
                mHandler.sendEmptyMessage(MSG_LOCALE_CHANGED);
            }
        };

        mContext.registerReceiver(portReceiver,
                new IntentFilter(UsbManager.ACTION_USB_PORT_CHANGED));
        mContext.registerReceiver(chargingReceiver,
                new IntentFilter(Intent.ACTION_BATTERY_CHANGED));

        IntentFilter filter =
                new IntentFilter(UsbManager.ACTION_USB_DEVICE_ATTACHED);
        filter.addAction(UsbManager.ACTION_USB_DEVICE_DETACHED);
        mContext.registerReceiver(hostReceiver, filter);

        mContext.registerReceiver(languageChangedReceiver,
                new IntentFilter(Intent.ACTION_LOCALE_CHANGED));

        // Watch for USB configuration changes
        mUEventObserver = new UsbUEventObserver();
        mUEventObserver.startObserving(USB_STATE_MATCH);
        mUEventObserver.startObserving(ACCESSORY_START_MATCH);
    }

    UsbProfileGroupSettingsManager getCurrentSettings() {
        synchronized (mLock) {
            return mCurrentSettings;
        }
    }

    String[] getAccessoryStrings() {
        synchronized (mLock) {
            return mAccessoryStrings;
        }
    }

    public void systemReady() {
        if (DEBUG) Slog.d(TAG, "systemReady");

        LocalServices.getService(ActivityTaskManagerInternal.class).registerScreenObserver(this);

        mHandler.sendEmptyMessage(MSG_SYSTEM_READY);
    }

    public void bootCompleted() {
        if (DEBUG) Slog.d(TAG, "boot completed");
        mHandler.sendEmptyMessage(MSG_BOOT_COMPLETED);
    }

    public void setCurrentUser(int newCurrentUserId, UsbProfileGroupSettingsManager settings) {
        synchronized (mLock) {
            mCurrentSettings = settings;
            mHandler.obtainMessage(MSG_USER_SWITCHED, newCurrentUserId, 0).sendToTarget();
        }
    }

    public void updateUserRestrictions() {
        mHandler.sendEmptyMessage(MSG_UPDATE_USER_RESTRICTIONS);
    }

    private void startAccessoryMode() {
        if (!mHasUsbAccessory) return;

        mAccessoryStrings = nativeGetAccessoryStrings();
        boolean enableAudio = (nativeGetAudioMode() == AUDIO_MODE_SOURCE);
        // don't start accessory mode if our mandatory strings have not been set
        boolean enableAccessory = (mAccessoryStrings != null &&
                mAccessoryStrings[UsbAccessory.MANUFACTURER_STRING] != null &&
                mAccessoryStrings[UsbAccessory.MODEL_STRING] != null);

        long functions = UsbManager.FUNCTION_NONE;
        if (enableAccessory) {
            functions |= UsbManager.FUNCTION_ACCESSORY;
        }
        if (enableAudio) {
            functions |= UsbManager.FUNCTION_AUDIO_SOURCE;
        }

        if (functions != UsbManager.FUNCTION_NONE) {
            mHandler.sendMessageDelayed(mHandler.obtainMessage(MSG_ACCESSORY_MODE_ENTER_TIMEOUT),
                    ACCESSORY_REQUEST_TIMEOUT);
            setCurrentFunctions(functions);
        }
    }

    private static void initRndisAddress() {
        // configure RNDIS ethernet address based on our serial number using the same algorithm
        // we had been previously using in kernel board files
        final int ETH_ALEN = 6;
        int address[] = new int[ETH_ALEN];
        // first byte is 0x02 to signify a locally administered address
        address[0] = 0x02;

        String serial = SystemProperties.get("ro.serialno", "1234567890ABCDEF");
        int serialLength = serial.length();
        // XOR the USB serial across the remaining 5 bytes
        for (int i = 0; i < serialLength; i++) {
            address[i % (ETH_ALEN - 1) + 1] ^= (int) serial.charAt(i);
        }
        String addrString = String.format(Locale.US, "%02X:%02X:%02X:%02X:%02X:%02X",
                address[0], address[1], address[2], address[3], address[4], address[5]);
        try {
            FileUtils.stringToFile(RNDIS_ETH_ADDR_PATH, addrString);
        } catch (IOException e) {
            Slog.e(TAG, "failed to write to " + RNDIS_ETH_ADDR_PATH);
        }
    }

    abstract static class UsbHandler extends Handler {

        // current USB state
        private boolean mHostConnected;
        private boolean mSourcePower;
        private boolean mSinkPower;
        private boolean mConfigured;
        private boolean mAudioAccessoryConnected;
        private boolean mAudioAccessorySupported;

        private UsbAccessory mCurrentAccessory;
        private int mUsbNotificationId;
        private boolean mAdbNotificationShown;
        private boolean mUsbCharging;
        private boolean mHideUsbNotification;
        private boolean mSupportsAllCombinations;
        private boolean mScreenLocked;
        private boolean mSystemReady;
        private Intent mBroadcastedIntent;
        private boolean mPendingBootBroadcast;
        private boolean mAudioSourceEnabled;
        private boolean mMidiEnabled;
        private int mMidiCard;
        private int mMidiDevice;

        private final Context mContext;
        private final UsbAlsaManager mUsbAlsaManager;
        private final UsbPermissionManager mPermissionManager;
        private NotificationManager mNotificationManager;

        protected boolean mConnected;
        protected long mScreenUnlockedFunctions;
        protected boolean mBootCompleted;
        protected boolean mCurrentFunctionsApplied;
        protected boolean mUseUsbNotification;
        protected long mCurrentFunctions;
        protected final UsbDeviceManager mUsbDeviceManager;
        protected final ContentResolver mContentResolver;
        protected SharedPreferences mSettings;
        protected int mCurrentUser;
        protected boolean mCurrentUsbFunctionsReceived;

        /**
         * The persistent property which stores whether adb is enabled or not.
         * May also contain vendor-specific default functions for testing purposes.
         */
        protected static final String USB_PERSISTENT_CONFIG_PROPERTY = "persist.sys.usb.config";

        UsbHandler(Looper looper, Context context, UsbDeviceManager deviceManager,
                UsbAlsaManager alsaManager, UsbPermissionManager permissionManager) {
            super(looper);
            mContext = context;
            mUsbDeviceManager = deviceManager;
            mUsbAlsaManager = alsaManager;
            mPermissionManager = permissionManager;
            mContentResolver = context.getContentResolver();

            mCurrentUser = ActivityManager.getCurrentUser();
            mScreenLocked = true;

            mSettings = getPinnedSharedPrefs(mContext);
            if (mSettings == null) {
                Slog.e(TAG, "Couldn't load shared preferences");
            } else {
                mScreenUnlockedFunctions = UsbManager.usbFunctionsFromString(
                        mSettings.getString(
                                String.format(Locale.ENGLISH, UNLOCKED_CONFIG_PREF, mCurrentUser),
                                ""));
            }

            // We do not show the USB notification if the primary volume supports mass storage.
            // The legacy mass storage UI will be used instead.
            final StorageManager storageManager = StorageManager.from(mContext);
            final StorageVolume primary =
                    storageManager != null ? storageManager.getPrimaryVolume() : null;

            boolean massStorageSupported = primary != null && primary.allowMassStorage();
            mUseUsbNotification = !massStorageSupported && mContext.getResources().getBoolean(
                    com.android.internal.R.bool.config_usbChargingMessage);
        }

        public void sendMessage(int what, boolean arg) {
            removeMessages(what);
            Message m = Message.obtain(this, what);
            m.arg1 = (arg ? 1 : 0);
            sendMessage(m);
        }

        public void sendMessage(int what, Object arg) {
            removeMessages(what);
            Message m = Message.obtain(this, what);
            m.obj = arg;
            sendMessage(m);
        }

        public void sendMessage(int what, Object arg, boolean arg1) {
            removeMessages(what);
            Message m = Message.obtain(this, what);
            m.obj = arg;
            m.arg1 = (arg1 ? 1 : 0);
            sendMessage(m);
        }

        public void sendMessage(int what, boolean arg1, boolean arg2) {
            removeMessages(what);
            Message m = Message.obtain(this, what);
            m.arg1 = (arg1 ? 1 : 0);
            m.arg2 = (arg2 ? 1 : 0);
            sendMessage(m);
        }

        public void sendMessageDelayed(int what, boolean arg, long delayMillis) {
            removeMessages(what);
            Message m = Message.obtain(this, what);
            m.arg1 = (arg ? 1 : 0);
            sendMessageDelayed(m, delayMillis);
        }

        public void updateState(String state) {
            int connected, configured;

            if ("DISCONNECTED".equals(state)) {
                connected = 0;
                configured = 0;
            } else if ("CONNECTED".equals(state)) {
                connected = 1;
                configured = 0;
            } else if ("CONFIGURED".equals(state)) {
                connected = 1;
                configured = 1;
            } else {
                Slog.e(TAG, "unknown state " + state);
                return;
            }
            removeMessages(MSG_UPDATE_STATE);
            if (connected == 1) removeMessages(MSG_FUNCTION_SWITCH_TIMEOUT);
            Message msg = Message.obtain(this, MSG_UPDATE_STATE);
            msg.arg1 = connected;
            msg.arg2 = configured;
            // debounce disconnects to avoid problems bringing up USB tethering
            sendMessageDelayed(msg, (connected == 0) ? UPDATE_DELAY : 0);
        }

        public void updateHostState(UsbPort port, UsbPortStatus status) {
            if (DEBUG) {
                Slog.i(TAG, "updateHostState " + port + " status=" + status);
            }

            SomeArgs args = SomeArgs.obtain();
            args.arg1 = port;
            args.arg2 = status;

            removeMessages(MSG_UPDATE_PORT_STATE);
            Message msg = obtainMessage(MSG_UPDATE_PORT_STATE, args);
            // debounce rapid transitions of connect/disconnect on type-c ports
            sendMessageDelayed(msg, UPDATE_DELAY);
        }

        private void setAdbEnabled(boolean enable) {
            if (DEBUG) Slog.d(TAG, "setAdbEnabled: " + enable);

            if (enable) {
                setSystemProperty(USB_PERSISTENT_CONFIG_PROPERTY, UsbManager.USB_FUNCTION_ADB);
            } else {
                setSystemProperty(USB_PERSISTENT_CONFIG_PROPERTY, "");
            }

            setEnabledFunctions(mCurrentFunctions, true);
            updateAdbNotification(false);
        }

        protected boolean isUsbTransferAllowed() {
            UserManager userManager = (UserManager) mContext.getSystemService(Context.USER_SERVICE);
            return !userManager.hasUserRestriction(UserManager.DISALLOW_USB_FILE_TRANSFER);
        }

        private void updateCurrentAccessory() {
            // We are entering accessory mode if we have received a request from the host
            // and the request has not timed out yet.
            boolean enteringAccessoryMode = hasMessages(MSG_ACCESSORY_MODE_ENTER_TIMEOUT);

            if (mConfigured && enteringAccessoryMode) {
                // successfully entered accessory mode
                String[] accessoryStrings = mUsbDeviceManager.getAccessoryStrings();
                if (accessoryStrings != null) {
                    UsbSerialReader serialReader = new UsbSerialReader(mContext, mPermissionManager,
                            accessoryStrings[UsbAccessory.SERIAL_STRING]);

                    mCurrentAccessory = new UsbAccessory(
                            accessoryStrings[UsbAccessory.MANUFACTURER_STRING],
                            accessoryStrings[UsbAccessory.MODEL_STRING],
                            accessoryStrings[UsbAccessory.DESCRIPTION_STRING],
                            accessoryStrings[UsbAccessory.VERSION_STRING],
                            accessoryStrings[UsbAccessory.URI_STRING],
                            serialReader);

                    serialReader.setDevice(mCurrentAccessory);

                    Slog.d(TAG, "entering USB accessory mode: " + mCurrentAccessory);
                    // defer accessoryAttached if system is not ready
                    if (mBootCompleted) {
                        mUsbDeviceManager.getCurrentSettings().accessoryAttached(mCurrentAccessory);
                    } // else handle in boot completed
                } else {
                    Slog.e(TAG, "nativeGetAccessoryStrings failed");
                }
            } else {
                if (!enteringAccessoryMode) {
                    notifyAccessoryModeExit();
                } else if (DEBUG) {
                    Slog.v(TAG, "Debouncing accessory mode exit");
                }
            }
        }

        private void notifyAccessoryModeExit() {
            // make sure accessory mode is off
            // and restore default functions
            Slog.d(TAG, "exited USB accessory mode");
            setEnabledFunctions(UsbManager.FUNCTION_NONE, false);

            if (mCurrentAccessory != null) {
                if (mBootCompleted) {
                    mPermissionManager.usbAccessoryRemoved(mCurrentAccessory);
                }
                mCurrentAccessory = null;
            }
        }

        protected SharedPreferences getPinnedSharedPrefs(Context context) {
            final File prefsFile = new File(
                    Environment.getDataSystemDeDirectory(UserHandle.USER_SYSTEM), USB_PREFS_XML);
            return context.createDeviceProtectedStorageContext()
                    .getSharedPreferences(prefsFile, Context.MODE_PRIVATE);
        }

        private boolean isUsbStateChanged(Intent intent) {
            final Set<String> keySet = intent.getExtras().keySet();
            if (mBroadcastedIntent == null) {
                for (String key : keySet) {
                    if (intent.getBooleanExtra(key, false)) {
                        return true;
                    }
                }
            } else {
                if (!keySet.equals(mBroadcastedIntent.getExtras().keySet())) {
                    return true;
                }
                for (String key : keySet) {
                    if (intent.getBooleanExtra(key, false) !=
                            mBroadcastedIntent.getBooleanExtra(key, false)) {
                        return true;
                    }
                }
            }
            return false;
        }

        protected void updateUsbStateBroadcastIfNeeded(long functions) {
            // send a sticky broadcast containing current USB state
            Intent intent = new Intent(UsbManager.ACTION_USB_STATE);
            intent.addFlags(Intent.FLAG_RECEIVER_REPLACE_PENDING
                    | Intent.FLAG_RECEIVER_INCLUDE_BACKGROUND
                    | Intent.FLAG_RECEIVER_FOREGROUND);
            intent.putExtra(UsbManager.USB_CONNECTED, mConnected);
            intent.putExtra(UsbManager.USB_HOST_CONNECTED, mHostConnected);
            intent.putExtra(UsbManager.USB_CONFIGURED, mConfigured);
            intent.putExtra(UsbManager.USB_DATA_UNLOCKED,
                    isUsbTransferAllowed() && isUsbDataTransferActive(mCurrentFunctions));

            long remainingFunctions = functions;
            while (remainingFunctions != 0) {
                intent.putExtra(UsbManager.usbFunctionsToString(
                        Long.highestOneBit(remainingFunctions)), true);
                remainingFunctions -= Long.highestOneBit(remainingFunctions);
            }

            // send broadcast intent only if the USB state has changed
            if (!isUsbStateChanged(intent)) {
                if (DEBUG) {
                    Slog.d(TAG, "skip broadcasting " + intent + " extras: " + intent.getExtras());
                }
                return;
            }

            if (DEBUG) Slog.d(TAG, "broadcasting " + intent + " extras: " + intent.getExtras());
            sendStickyBroadcast(intent);
            mBroadcastedIntent = intent;
        }

        protected void sendStickyBroadcast(Intent intent) {
            mContext.sendStickyBroadcastAsUser(intent, UserHandle.ALL);
        }

        private void updateUsbFunctions() {
            updateMidiFunction();
        }

        private void updateMidiFunction() {
            boolean enabled = (mCurrentFunctions & UsbManager.FUNCTION_MIDI) != 0;
            if (enabled != mMidiEnabled) {
                if (enabled) {
                    Scanner scanner = null;
                    try {
                        scanner = new Scanner(new File(MIDI_ALSA_PATH));
                        mMidiCard = scanner.nextInt();
                        mMidiDevice = scanner.nextInt();
                    } catch (FileNotFoundException e) {
                        Slog.e(TAG, "could not open MIDI file", e);
                        enabled = false;
                    } finally {
                        if (scanner != null) {
                            scanner.close();
                        }
                    }
                }
                mMidiEnabled = enabled;
            }
            mUsbAlsaManager.setPeripheralMidiState(
                    mMidiEnabled && mConfigured, mMidiCard, mMidiDevice);
        }

        private void setScreenUnlockedFunctions() {
            setEnabledFunctions(mScreenUnlockedFunctions, false);
        }

        private static class AdbTransport extends IAdbTransport.Stub {
            private final UsbHandler mHandler;

            AdbTransport(UsbHandler handler) {
                mHandler = handler;
            }

            @Override
            public void onAdbEnabled(boolean enabled, byte transportType) {
                if (transportType == AdbTransportType.USB) {
                    mHandler.sendMessage(MSG_ENABLE_ADB, enabled);
                }
            }
        }

        /**
         * Returns the functions that are passed down to the low level driver once adb and
         * charging are accounted for.
         */
        long getAppliedFunctions(long functions) {
            if (functions == UsbManager.FUNCTION_NONE) {
                return getChargingFunctions();
            }
            if (isAdbEnabled()) {
                return functions | UsbManager.FUNCTION_ADB;
            }
            return functions;
        }

        @Override
        public void handleMessage(Message msg) {
            switch (msg.what) {
                case MSG_UPDATE_STATE:
                    mConnected = (msg.arg1 == 1);
                    mConfigured = (msg.arg2 == 1);

                    updateUsbNotification(false);
                    updateAdbNotification(false);
                    if (mBootCompleted) {
                        updateUsbStateBroadcastIfNeeded(getAppliedFunctions(mCurrentFunctions));
                    }
                    if ((mCurrentFunctions & UsbManager.FUNCTION_ACCESSORY) != 0) {
                        updateCurrentAccessory();
                    }
                    if (mBootCompleted) {
                        if (!mConnected && !hasMessages(MSG_ACCESSORY_MODE_ENTER_TIMEOUT)
                                && !hasMessages(MSG_FUNCTION_SWITCH_TIMEOUT)) {
                            // restore defaults when USB is disconnected
                            if (!mScreenLocked
                                    && mScreenUnlockedFunctions != UsbManager.FUNCTION_NONE) {
                                setScreenUnlockedFunctions();
                            } else {
                                setEnabledFunctions(UsbManager.FUNCTION_NONE, false);
                            }
                        }
                        updateUsbFunctions();
                    } else {
                        mPendingBootBroadcast = true;
                    }
                    break;
                case MSG_UPDATE_PORT_STATE:
                    SomeArgs args = (SomeArgs) msg.obj;
                    boolean prevHostConnected = mHostConnected;
                    UsbPort port = (UsbPort) args.arg1;
                    UsbPortStatus status = (UsbPortStatus) args.arg2;
                    mHostConnected = status.getCurrentDataRole() == DATA_ROLE_HOST;
                    mSourcePower = status.getCurrentPowerRole() == POWER_ROLE_SOURCE;
                    mSinkPower = status.getCurrentPowerRole() == POWER_ROLE_SINK;
                    mAudioAccessoryConnected = (status.getCurrentMode() == MODE_AUDIO_ACCESSORY);
                    mAudioAccessorySupported = port.isModeSupported(MODE_AUDIO_ACCESSORY);
                    // Ideally we want to see if PR_SWAP and DR_SWAP is supported.
                    // But, this should be suffice, since, all four combinations are only supported
                    // when PR_SWAP and DR_SWAP are supported.
                    mSupportsAllCombinations = status.isRoleCombinationSupported(
                            POWER_ROLE_SOURCE, DATA_ROLE_HOST)
                            && status.isRoleCombinationSupported(POWER_ROLE_SINK, DATA_ROLE_HOST)
                            && status.isRoleCombinationSupported(POWER_ROLE_SOURCE,
                            DATA_ROLE_DEVICE)
                            && status.isRoleCombinationSupported(POWER_ROLE_SINK, DATA_ROLE_DEVICE);

                    args.recycle();
                    updateUsbNotification(false);
                    if (mBootCompleted) {
                        if (mHostConnected || prevHostConnected) {
                            updateUsbStateBroadcastIfNeeded(getAppliedFunctions(mCurrentFunctions));
                        }
                    } else {
                        mPendingBootBroadcast = true;
                    }
                    break;
                case MSG_UPDATE_CHARGING_STATE:
                    mUsbCharging = (msg.arg1 == 1);
                    updateUsbNotification(false);
                    break;
                case MSG_UPDATE_HOST_STATE:
                    Iterator devices = (Iterator) msg.obj;
                    boolean connected = (msg.arg1 == 1);

                    if (DEBUG) {
                        Slog.i(TAG, "HOST_STATE connected:" + connected);
                    }

                    mHideUsbNotification = false;
                    while (devices.hasNext()) {
                        Map.Entry pair = (Map.Entry) devices.next();
                        if (DEBUG) {
                            Slog.i(TAG, pair.getKey() + " = " + pair.getValue());
                        }
                        UsbDevice device = (UsbDevice) pair.getValue();
                        int configurationCount = device.getConfigurationCount() - 1;
                        while (configurationCount >= 0) {
                            UsbConfiguration config = device.getConfiguration(configurationCount);
                            configurationCount--;
                            int interfaceCount = config.getInterfaceCount() - 1;
                            while (interfaceCount >= 0) {
                                UsbInterface intrface = config.getInterface(interfaceCount);
                                interfaceCount--;
                                if (sBlackListedInterfaces.contains(intrface.getInterfaceClass())) {
                                    mHideUsbNotification = true;
                                    break;
                                }
                            }
                        }
                    }
                    updateUsbNotification(false);
                    break;
                case MSG_ENABLE_ADB:
                    setAdbEnabled(msg.arg1 == 1);
                    break;
                case MSG_SET_CURRENT_FUNCTIONS:
                    long functions = (Long) msg.obj;
                    setEnabledFunctions(functions, false);
                    break;
                case MSG_SET_SCREEN_UNLOCKED_FUNCTIONS:
                    mScreenUnlockedFunctions = (Long) msg.obj;
                    if (mSettings != null) {
                        SharedPreferences.Editor editor = mSettings.edit();
                        editor.putString(String.format(Locale.ENGLISH, UNLOCKED_CONFIG_PREF,
                                mCurrentUser),
                                UsbManager.usbFunctionsToString(mScreenUnlockedFunctions));
                        editor.commit();
                    }
                    if (!mScreenLocked && mScreenUnlockedFunctions != UsbManager.FUNCTION_NONE) {
                        // If the screen is unlocked, also set current functions.
                        setScreenUnlockedFunctions();
                    } else {
                        setEnabledFunctions(UsbManager.FUNCTION_NONE, false);
                    }
                    break;
                case MSG_UPDATE_SCREEN_LOCK:
                    if (msg.arg1 == 1 == mScreenLocked) {
                        break;
                    }
                    mScreenLocked = msg.arg1 == 1;
                    if (!mBootCompleted) {
                        break;
                    }
                    if (mScreenLocked) {
                        if (!mConnected) {
                            setEnabledFunctions(UsbManager.FUNCTION_NONE, false);
                        }
                    } else {
                        if (mScreenUnlockedFunctions != UsbManager.FUNCTION_NONE
                                && mCurrentFunctions == UsbManager.FUNCTION_NONE) {
                            // Set the screen unlocked functions if current function is charging.
                            setScreenUnlockedFunctions();
                        }
                    }
                    break;
                case MSG_UPDATE_USER_RESTRICTIONS:
                    // Restart the USB stack if USB transfer is enabled but no longer allowed.
                    if (isUsbDataTransferActive(mCurrentFunctions) && !isUsbTransferAllowed()) {
                        setEnabledFunctions(UsbManager.FUNCTION_NONE, true);
                    }
                    break;
                case MSG_SYSTEM_READY:
                    mNotificationManager = (NotificationManager)
                            mContext.getSystemService(Context.NOTIFICATION_SERVICE);

                    LocalServices.getService(
                            AdbManagerInternal.class).registerTransport(new AdbTransport(this));

                    // Ensure that the notification channels are set up
                    if (isTv()) {
                        // TV-specific notification channel
                        mNotificationManager.createNotificationChannel(
                                new NotificationChannel(ADB_NOTIFICATION_CHANNEL_ID_TV,
                                        mContext.getString(
                                                com.android.internal.R.string
                                                        .adb_debugging_notification_channel_tv),
                                        NotificationManager.IMPORTANCE_HIGH));
                    }
                    mSystemReady = true;
                    finishBoot();
                    break;
                case MSG_LOCALE_CHANGED:
                    updateAdbNotification(true);
                    updateUsbNotification(true);
                    break;
                case MSG_BOOT_COMPLETED:
                    mBootCompleted = true;
                    finishBoot();
                    break;
                case MSG_USER_SWITCHED: {
                    if (mCurrentUser != msg.arg1) {
                        if (DEBUG) {
                            Slog.v(TAG, "Current user switched to " + msg.arg1);
                        }
                        mCurrentUser = msg.arg1;
                        mScreenLocked = true;
                        mScreenUnlockedFunctions = UsbManager.FUNCTION_NONE;
                        if (mSettings != null) {
                            mScreenUnlockedFunctions = UsbManager.usbFunctionsFromString(
                                    mSettings.getString(String.format(Locale.ENGLISH,
                                            UNLOCKED_CONFIG_PREF, mCurrentUser), ""));
                        }
                        setEnabledFunctions(UsbManager.FUNCTION_NONE, false);
                    }
                    break;
                }
                case MSG_ACCESSORY_MODE_ENTER_TIMEOUT: {
                    if (DEBUG) {
                        Slog.v(TAG, "Accessory mode enter timeout: " + mConnected);
                    }
                    if (!mConnected || (mCurrentFunctions & UsbManager.FUNCTION_ACCESSORY) == 0) {
                        notifyAccessoryModeExit();
                    }
                    break;
                }
            }
        }

        protected void finishBoot() {
            if (mBootCompleted && mCurrentUsbFunctionsReceived && mSystemReady) {
                if (mPendingBootBroadcast) {
                    updateUsbStateBroadcastIfNeeded(getAppliedFunctions(mCurrentFunctions));
                    mPendingBootBroadcast = false;
                }
                if (!mScreenLocked
                        && mScreenUnlockedFunctions != UsbManager.FUNCTION_NONE) {
                    setScreenUnlockedFunctions();
                } else {
                    setEnabledFunctions(UsbManager.FUNCTION_NONE, false);
                }
                if (mCurrentAccessory != null) {
                    mUsbDeviceManager.getCurrentSettings().accessoryAttached(mCurrentAccessory);
                }

                updateUsbNotification(false);
                updateAdbNotification(false);
                updateUsbFunctions();
            }
        }

        protected boolean isUsbDataTransferActive(long functions) {
            return (functions & UsbManager.FUNCTION_MTP) != 0
                    || (functions & UsbManager.FUNCTION_PTP) != 0;
        }

        public UsbAccessory getCurrentAccessory() {
            return mCurrentAccessory;
        }

        protected void updateUsbNotification(boolean force) {
            if (mNotificationManager == null || !mUseUsbNotification
                    || ("0".equals(getSystemProperty("persist.charging.notify", "")))) {
                return;
            }

            // Dont show the notification when connected to a USB peripheral
            // and the link does not support PR_SWAP and DR_SWAP
            if (mHideUsbNotification && !mSupportsAllCombinations) {
                if (mUsbNotificationId != 0) {
                    mNotificationManager.cancelAsUser(null, mUsbNotificationId,
                            UserHandle.ALL);
                    mUsbNotificationId = 0;
                    Slog.d(TAG, "Clear notification");
                }
                return;
            }

            int id = 0;
            int titleRes = 0;
            Resources r = mContext.getResources();
            CharSequence message = r.getText(
                    com.android.internal.R.string.usb_notification_message);
            if (mAudioAccessoryConnected && !mAudioAccessorySupported) {
                titleRes = com.android.internal.R.string.usb_unsupported_audio_accessory_title;
                id = SystemMessage.NOTE_USB_AUDIO_ACCESSORY_NOT_SUPPORTED;
            } else if (mConnected) {
                if (mCurrentFunctions == UsbManager.FUNCTION_MTP) {
                    titleRes = com.android.internal.R.string.usb_mtp_notification_title;
                    id = SystemMessage.NOTE_USB_MTP;
                } else if (mCurrentFunctions == UsbManager.FUNCTION_PTP) {
                    titleRes = com.android.internal.R.string.usb_ptp_notification_title;
                    id = SystemMessage.NOTE_USB_PTP;
                } else if (mCurrentFunctions == UsbManager.FUNCTION_MIDI) {
                    titleRes = com.android.internal.R.string.usb_midi_notification_title;
                    id = SystemMessage.NOTE_USB_MIDI;
                } else if (mCurrentFunctions == UsbManager.FUNCTION_RNDIS) {
                    titleRes = com.android.internal.R.string.usb_tether_notification_title;
                    id = SystemMessage.NOTE_USB_TETHER;
                } else if (mCurrentFunctions == UsbManager.FUNCTION_ACCESSORY) {
                    titleRes = com.android.internal.R.string.usb_accessory_notification_title;
                    id = SystemMessage.NOTE_USB_ACCESSORY;
                }
                if (mSourcePower) {
                    if (titleRes != 0) {
                        message = r.getText(
                                com.android.internal.R.string.usb_power_notification_message);
                    } else {
                        titleRes = com.android.internal.R.string.usb_supplying_notification_title;
                        id = SystemMessage.NOTE_USB_SUPPLYING;
                    }
                } else if (titleRes == 0) {
                    titleRes = com.android.internal.R.string.usb_charging_notification_title;
                    id = SystemMessage.NOTE_USB_CHARGING;
                }
            } else if (mSourcePower) {
                titleRes = com.android.internal.R.string.usb_supplying_notification_title;
                id = SystemMessage.NOTE_USB_SUPPLYING;
            } else if (mHostConnected && mSinkPower && mUsbCharging) {
                titleRes = com.android.internal.R.string.usb_charging_notification_title;
                id = SystemMessage.NOTE_USB_CHARGING;
            }
            if (id != mUsbNotificationId || force) {
                // clear notification if title needs changing
                if (mUsbNotificationId != 0) {
                    mNotificationManager.cancelAsUser(null, mUsbNotificationId,
                            UserHandle.ALL);
                    Slog.d(TAG, "Clear notification");
                    mUsbNotificationId = 0;
                }
                // Not relevant for automotive.
                if (mContext.getPackageManager().hasSystemFeature(
                        PackageManager.FEATURE_AUTOMOTIVE)
                        && id == SystemMessage.NOTE_USB_CHARGING) {
                    mUsbNotificationId = 0;
                    return;
                }

                if (id != 0) {
                    CharSequence title = r.getText(titleRes);
                    PendingIntent pi;
                    String channel;

                    if (titleRes
                            != com.android.internal.R.string
                            .usb_unsupported_audio_accessory_title) {
                        Intent intent = Intent.makeRestartActivityTask(
                                new ComponentName("com.android.settings",
                                        "com.android.settings.Settings$UsbDetailsActivity"));
                        pi = PendingIntent.getActivityAsUser(mContext, 0,
                                intent, 0, null, UserHandle.CURRENT);
                        channel = SystemNotificationChannels.USB;
                    } else {
                        final Intent intent = new Intent();
                        intent.setClassName("com.android.settings",
                                "com.android.settings.HelpTrampoline");
                        intent.putExtra(Intent.EXTRA_TEXT,
                                "help_url_audio_accessory_not_supported");

                        if (mContext.getPackageManager().resolveActivity(intent, 0) != null) {
                            pi = PendingIntent.getActivity(mContext, 0, intent, 0);
                        } else {
                            pi = null;
                        }

                        channel = SystemNotificationChannels.ALERTS;
                        message = r.getText(
                                com.android.internal.R.string
                                        .usb_unsupported_audio_accessory_message);
                    }

                    Notification.Builder builder = new Notification.Builder(mContext, channel)
                            .setSmallIcon(com.android.internal.R.drawable.stat_sys_adb)
                            .setWhen(0)
                            .setOngoing(true)
                            .setTicker(title)
                            .setDefaults(0)  // please be quiet
                            .setColor(mContext.getColor(
                                    com.android.internal.R.color
                                            .system_notification_accent_color))
                            .setContentTitle(title)
                            .setContentText(message)
                            .setContentIntent(pi)
                            .setVisibility(Notification.VISIBILITY_PUBLIC);

                    if (titleRes
                            == com.android.internal.R.string
                            .usb_unsupported_audio_accessory_title) {
                        builder.setStyle(new Notification.BigTextStyle()
                                .bigText(message));
                    }
                    Notification notification = builder.build();

                    mNotificationManager.notifyAsUser(null, id, notification,
                            UserHandle.ALL);
                    Slog.d(TAG, "push notification:" + title);
                    mUsbNotificationId = id;
                }
            }
        }

        protected boolean isAdbEnabled() {
            return LocalServices.getService(AdbManagerInternal.class)
                    .isAdbEnabled(AdbTransportType.USB);
        }

        protected void updateAdbNotification(boolean force) {
            if (mNotificationManager == null) return;
            final int id = SystemMessage.NOTE_ADB_ACTIVE;

            if (isAdbEnabled() && mConnected) {
                if ("0".equals(getSystemProperty("persist.adb.notify", ""))) return;

                if (force && mAdbNotificationShown) {
                    mAdbNotificationShown = false;
                    mNotificationManager.cancelAsUser(null, id, UserHandle.ALL);
                }

                if (!mAdbNotificationShown) {
                    Notification notification = AdbNotifications.createNotification(mContext,
                            AdbTransportType.USB);
                    mAdbNotificationShown = true;
                    mNotificationManager.notifyAsUser(null, id, notification, UserHandle.ALL);
                }
            } else if (mAdbNotificationShown) {
                mAdbNotificationShown = false;
                mNotificationManager.cancelAsUser(null, id, UserHandle.ALL);
            }
        }

        private boolean isTv() {
            return mContext.getPackageManager().hasSystemFeature(PackageManager.FEATURE_LEANBACK);
        }

        protected long getChargingFunctions() {
            // if ADB is enabled, reset functions to ADB
            // else enable MTP as usual.
            if (isAdbEnabled()) {
                return UsbManager.FUNCTION_ADB;
            } else {
                return UsbManager.FUNCTION_MTP;
            }
        }

        protected void setSystemProperty(String prop, String val) {
            SystemProperties.set(prop, val);
        }

        protected String getSystemProperty(String prop, String def) {
            return SystemProperties.get(prop, def);
        }

        protected void putGlobalSettings(ContentResolver contentResolver, String setting, int val) {
            Settings.Global.putInt(contentResolver, setting, val);
        }

        public long getEnabledFunctions() {
            return mCurrentFunctions;
        }

        public long getScreenUnlockedFunctions() {
            return mScreenUnlockedFunctions;
        }

        /**
         * Dump a functions mask either as proto-enums (if dumping to proto) or a string (if dumping
         * to a print writer)
         */
        private void dumpFunctions(DualDumpOutputStream dump, String idName, long id,
                long functions) {
            // UsbHandlerProto.UsbFunction matches GadgetFunction
            for (int i = 0; i < 63; i++) {
                if ((functions & (1L << i)) != 0) {
                    if (dump.isProto()) {
                        dump.write(idName, id, 1L << i);
                    } else {
                        dump.write(idName, id, GadgetFunction.toString(1L << i));
                    }
                }
            }
        }

        public void dump(DualDumpOutputStream dump, String idName, long id) {
            long token = dump.start(idName, id);

            dumpFunctions(dump, "current_functions", UsbHandlerProto.CURRENT_FUNCTIONS,
                    mCurrentFunctions);
            dump.write("current_functions_applied", UsbHandlerProto.CURRENT_FUNCTIONS_APPLIED,
                    mCurrentFunctionsApplied);
            dumpFunctions(dump, "screen_unlocked_functions",
                    UsbHandlerProto.SCREEN_UNLOCKED_FUNCTIONS, mScreenUnlockedFunctions);
            dump.write("screen_locked", UsbHandlerProto.SCREEN_LOCKED, mScreenLocked);
            dump.write("connected", UsbHandlerProto.CONNECTED, mConnected);
            dump.write("configured", UsbHandlerProto.CONFIGURED, mConfigured);
            if (mCurrentAccessory != null) {
                writeAccessory(dump, "current_accessory", UsbHandlerProto.CURRENT_ACCESSORY,
                        mCurrentAccessory);
            }
            dump.write("host_connected", UsbHandlerProto.HOST_CONNECTED, mHostConnected);
            dump.write("source_power", UsbHandlerProto.SOURCE_POWER, mSourcePower);
            dump.write("sink_power", UsbHandlerProto.SINK_POWER, mSinkPower);
            dump.write("usb_charging", UsbHandlerProto.USB_CHARGING, mUsbCharging);
            dump.write("hide_usb_notification", UsbHandlerProto.HIDE_USB_NOTIFICATION,
                    mHideUsbNotification);
            dump.write("audio_accessory_connected", UsbHandlerProto.AUDIO_ACCESSORY_CONNECTED,
                    mAudioAccessoryConnected);

            try {
                writeStringIfNotNull(dump, "kernel_state", UsbHandlerProto.KERNEL_STATE,
                        FileUtils.readTextFile(new File(STATE_PATH), 0, null).trim());
            } catch (Exception e) {
                Slog.e(TAG, "Could not read kernel state", e);
            }

            try {
                writeStringIfNotNull(dump, "kernel_function_list",
                        UsbHandlerProto.KERNEL_FUNCTION_LIST,
                        FileUtils.readTextFile(new File(FUNCTIONS_PATH), 0, null).trim());
            } catch (Exception e) {
                Slog.e(TAG, "Could not read kernel function list", e);
            }

            dump.end(token);
        }

        /**
         * Evaluates USB function policies and applies the change accordingly.
         */
        protected abstract void setEnabledFunctions(long functions, boolean forceRestart);
    }

    private static final class UsbHandlerLegacy extends UsbHandler {
        /**
         * The non-persistent property which stores the current USB settings.
         */
        private static final String USB_CONFIG_PROPERTY = "sys.usb.config";

        /**
         * The non-persistent property which stores the current USB actual state.
         */
        private static final String USB_STATE_PROPERTY = "sys.usb.state";

        private HashMap<String, HashMap<String, Pair<String, String>>> mOemModeMap;
        private String mCurrentOemFunctions;
        private String mCurrentFunctionsStr;
        private boolean mUsbDataUnlocked;

        UsbHandlerLegacy(Looper looper, Context context, UsbDeviceManager deviceManager,
                UsbAlsaManager alsaManager, UsbPermissionManager permissionManager) {
            super(looper, context, deviceManager, alsaManager, permissionManager);
            try {
                readOemUsbOverrideConfig(context);
                // Restore default functions.
                mCurrentOemFunctions = getSystemProperty(getPersistProp(false),
                        UsbManager.USB_FUNCTION_NONE);
                if (isNormalBoot()) {
                    mCurrentFunctionsStr = getSystemProperty(USB_CONFIG_PROPERTY,
                            UsbManager.USB_FUNCTION_NONE);
                    mCurrentFunctionsApplied = mCurrentFunctionsStr.equals(
                            getSystemProperty(USB_STATE_PROPERTY, UsbManager.USB_FUNCTION_NONE));
                } else {
                    mCurrentFunctionsStr = getSystemProperty(getPersistProp(true),
                            UsbManager.USB_FUNCTION_NONE);
                    mCurrentFunctionsApplied = getSystemProperty(USB_CONFIG_PROPERTY,
                            UsbManager.USB_FUNCTION_NONE).equals(
                            getSystemProperty(USB_STATE_PROPERTY, UsbManager.USB_FUNCTION_NONE));
                }
                mCurrentFunctions = UsbManager.FUNCTION_NONE;
                mCurrentUsbFunctionsReceived = true;

                String state = FileUtils.readTextFile(new File(STATE_PATH), 0, null).trim();
                updateState(state);
            } catch (Exception e) {
                Slog.e(TAG, "Error initializing UsbHandler", e);
            }
        }

        private void readOemUsbOverrideConfig(Context context) {
            String[] configList = context.getResources().getStringArray(
                    com.android.internal.R.array.config_oemUsbModeOverride);

            if (configList != null) {
                for (String config : configList) {
                    String[] items = config.split(":");
                    if (items.length == 3 || items.length == 4) {
                        if (mOemModeMap == null) {
                            mOemModeMap = new HashMap<>();
                        }
                        HashMap<String, Pair<String, String>> overrideMap =
                                mOemModeMap.get(items[0]);
                        if (overrideMap == null) {
                            overrideMap = new HashMap<>();
                            mOemModeMap.put(items[0], overrideMap);
                        }

                        // Favoring the first combination if duplicate exists
                        if (!overrideMap.containsKey(items[1])) {
                            if (items.length == 3) {
                                overrideMap.put(items[1], new Pair<>(items[2], ""));
                            } else {
                                overrideMap.put(items[1], new Pair<>(items[2], items[3]));
                            }
                        }
                    }
                }
            }
        }

        private String applyOemOverrideFunction(String usbFunctions) {
            if ((usbFunctions == null) || (mOemModeMap == null)) {
                return usbFunctions;
            }

            String bootMode = getSystemProperty(BOOT_MODE_PROPERTY, "unknown");
            Slog.d(TAG, "applyOemOverride usbfunctions=" + usbFunctions + " bootmode=" + bootMode);

            Map<String, Pair<String, String>> overridesMap =
                    mOemModeMap.get(bootMode);
            // Check to ensure that the oem is not overriding in the normal
            // boot mode
            if (overridesMap != null && !(bootMode.equals(NORMAL_BOOT)
                    || bootMode.equals("unknown"))) {
                Pair<String, String> overrideFunctions =
                        overridesMap.get(usbFunctions);
                if (overrideFunctions != null) {
                    Slog.d(TAG, "OEM USB override: " + usbFunctions
                            + " ==> " + overrideFunctions.first
                            + " persist across reboot "
                            + overrideFunctions.second);
                    if (!overrideFunctions.second.equals("")) {
                        String newFunction;
                        if (isAdbEnabled()) {
                            newFunction = addFunction(overrideFunctions.second,
                                    UsbManager.USB_FUNCTION_ADB);
                        } else {
                            newFunction = overrideFunctions.second;
                        }
                        Slog.d(TAG, "OEM USB override persisting: " + newFunction + "in prop: "
                                + getPersistProp(false));
                        setSystemProperty(getPersistProp(false), newFunction);
                    }
                    return overrideFunctions.first;
                } else if (isAdbEnabled()) {
                    String newFunction = addFunction(UsbManager.USB_FUNCTION_NONE,
                            UsbManager.USB_FUNCTION_ADB);
                    setSystemProperty(getPersistProp(false), newFunction);
                } else {
                    setSystemProperty(getPersistProp(false), UsbManager.USB_FUNCTION_NONE);
                }
            }
            // return passed in functions as is.
            return usbFunctions;
        }

        private boolean waitForState(String state) {
            // wait for the transition to complete.
            // give up after 1 second.
            String value = null;
            for (int i = 0; i < 20; i++) {
                // State transition is done when sys.usb.state is set to the new configuration
                value = getSystemProperty(USB_STATE_PROPERTY, "");
                if (state.equals(value)) return true;
                SystemClock.sleep(50);
            }
            Slog.e(TAG, "waitForState(" + state + ") FAILED: got " + value);
            return false;
        }

        private void setUsbConfig(String config) {
            if (DEBUG) Slog.d(TAG, "setUsbConfig(" + config + ")");
            /**
             * set the new configuration
             * we always set it due to b/23631400, where adbd was getting killed
             * and not restarted due to property timeouts on some devices
             */
            setSystemProperty(USB_CONFIG_PROPERTY, config);
        }

        @Override
        protected void setEnabledFunctions(long usbFunctions, boolean forceRestart) {
            boolean usbDataUnlocked = isUsbDataTransferActive(usbFunctions);
            if (DEBUG) {
                Slog.d(TAG, "setEnabledFunctions functions=" + usbFunctions + ", "
                        + "forceRestart=" + forceRestart + ", usbDataUnlocked=" + usbDataUnlocked);
            }

            if (usbDataUnlocked != mUsbDataUnlocked) {
                mUsbDataUnlocked = usbDataUnlocked;
                updateUsbNotification(false);
                forceRestart = true;
            }

            /**
             * Try to set the enabled functions.
             */
            final long oldFunctions = mCurrentFunctions;
            final boolean oldFunctionsApplied = mCurrentFunctionsApplied;
            if (trySetEnabledFunctions(usbFunctions, forceRestart)) {
                return;
            }

            /**
             * Didn't work.  Try to revert changes.
             * We always reapply the policy in case certain constraints changed such as
             * user restrictions independently of any other new functions we were
             * trying to activate.
             */
            if (oldFunctionsApplied && oldFunctions != usbFunctions) {
                Slog.e(TAG, "Failsafe 1: Restoring previous USB functions.");
                if (trySetEnabledFunctions(oldFunctions, false)) {
                    return;
                }
            }

            /**
             * Still didn't work.  Try to restore the default functions.
             */
            Slog.e(TAG, "Failsafe 2: Restoring default USB functions.");
            if (trySetEnabledFunctions(UsbManager.FUNCTION_NONE, false)) {
                return;
            }

            /**
             * Now we're desperate.  Ignore the default functions.
             * Try to get ADB working if enabled.
             */
            Slog.e(TAG, "Failsafe 3: Restoring empty function list (with ADB if enabled).");
            if (trySetEnabledFunctions(UsbManager.FUNCTION_NONE, false)) {
                return;
            }

            /**
             * Ouch.
             */
            Slog.e(TAG, "Unable to set any USB functions!");
        }

        private boolean isNormalBoot() {
            String bootMode = getSystemProperty(BOOT_MODE_PROPERTY, "unknown");
            return bootMode.equals(NORMAL_BOOT) || bootMode.equals("unknown");
        }

        protected String applyAdbFunction(String functions) {
            // Do not pass null pointer to the UsbManager.
            // There isn't a check there.
            if (functions == null) {
                functions = "";
            }
            if (isAdbEnabled()) {
                functions = addFunction(functions, UsbManager.USB_FUNCTION_ADB);
            } else {
                functions = removeFunction(functions, UsbManager.USB_FUNCTION_ADB);
            }
            return functions;
        }

        private boolean trySetEnabledFunctions(long usbFunctions, boolean forceRestart) {
            String functions = null;
            if (usbFunctions != UsbManager.FUNCTION_NONE) {
                functions = UsbManager.usbFunctionsToString(usbFunctions);
            }
            mCurrentFunctions = usbFunctions;
            if (functions == null || applyAdbFunction(functions)
                    .equals(UsbManager.USB_FUNCTION_NONE)) {
                functions = UsbManager.usbFunctionsToString(getChargingFunctions());
            }
            functions = applyAdbFunction(functions);

            String oemFunctions = applyOemOverrideFunction(functions);

            if (!isNormalBoot() && !mCurrentFunctionsStr.equals(functions)) {
                setSystemProperty(getPersistProp(true), functions);
            }

            if ((!functions.equals(oemFunctions)
                    && !mCurrentOemFunctions.equals(oemFunctions))
                    || !mCurrentFunctionsStr.equals(functions)
                    || !mCurrentFunctionsApplied
                    || forceRestart) {
                Slog.i(TAG, "Setting USB config to " + functions);
                mCurrentFunctionsStr = functions;
                mCurrentOemFunctions = oemFunctions;
                mCurrentFunctionsApplied = false;

                /**
                 * Kick the USB stack to close existing connections.
                 */
                setUsbConfig(UsbManager.USB_FUNCTION_NONE);

                if (!waitForState(UsbManager.USB_FUNCTION_NONE)) {
                    Slog.e(TAG, "Failed to kick USB config");
                    return false;
                }

                /**
                 * Set the new USB configuration.
                 */
                setUsbConfig(oemFunctions);

                if (mBootCompleted
                        && (containsFunction(functions, UsbManager.USB_FUNCTION_MTP)
                        || containsFunction(functions, UsbManager.USB_FUNCTION_PTP))) {
                    /**
                     * Start up dependent services.
                     */
                    updateUsbStateBroadcastIfNeeded(getAppliedFunctions(mCurrentFunctions));
                }

                if (!waitForState(oemFunctions)) {
                    Slog.e(TAG, "Failed to switch USB config to " + functions);
                    return false;
                }

                mCurrentFunctionsApplied = true;
            }
            return true;
        }

        private String getPersistProp(boolean functions) {
            String bootMode = getSystemProperty(BOOT_MODE_PROPERTY, "unknown");
            String persistProp = USB_PERSISTENT_CONFIG_PROPERTY;
            if (!(bootMode.equals(NORMAL_BOOT) || bootMode.equals("unknown"))) {
                if (functions) {
                    persistProp = "persist.sys.usb." + bootMode + ".func";
                } else {
                    persistProp = "persist.sys.usb." + bootMode + ".config";
                }
            }
            return persistProp;
        }

        private static String addFunction(String functions, String function) {
            if (UsbManager.USB_FUNCTION_NONE.equals(functions)) {
                return function;
            }
            if (!containsFunction(functions, function)) {
                if (functions.length() > 0) {
                    functions += ",";
                }
                functions += function;
            }
            return functions;
        }

        private static String removeFunction(String functions, String function) {
            String[] split = functions.split(",");
            for (int i = 0; i < split.length; i++) {
                if (function.equals(split[i])) {
                    split[i] = null;
                }
            }
            if (split.length == 1 && split[0] == null) {
                return UsbManager.USB_FUNCTION_NONE;
            }
            StringBuilder builder = new StringBuilder();
            for (int i = 0; i < split.length; i++) {
                String s = split[i];
                if (s != null) {
                    if (builder.length() > 0) {
                        builder.append(",");
                    }
                    builder.append(s);
                }
            }
            return builder.toString();
        }

        static boolean containsFunction(String functions, String function) {
            int index = functions.indexOf(function);
            if (index < 0) return false;
            if (index > 0 && functions.charAt(index - 1) != ',') return false;
            int charAfter = index + function.length();
            if (charAfter < functions.length() && functions.charAt(charAfter) != ',') return false;
            return true;
        }
    }

    private static final class UsbHandlerHal extends UsbHandler {

        /**
         * Proxy object for the usb gadget hal daemon.
         */
        @GuardedBy("mGadgetProxyLock")
        private IUsbGadget mGadgetProxy;

        private final Object mGadgetProxyLock = new Object();

        /**
         * Cookie sent for usb gadget hal death notification.
         */
        private static final int USB_GADGET_HAL_DEATH_COOKIE = 2000;

        /**
         * Keeps track of the latest setCurrentUsbFunctions request number.
         */
        private int mCurrentRequest = 0;

        /**
         * The maximum time for which the UsbDeviceManager would wait once
         * setCurrentUsbFunctions is called.
         */
        private static final int SET_FUNCTIONS_TIMEOUT_MS = 3000;

        /**
         * Conseration leeway to make sure that the hal callback arrives before
         * SET_FUNCTIONS_TIMEOUT_MS expires. If the callback does not arrive
         * within SET_FUNCTIONS_TIMEOUT_MS, UsbDeviceManager retries enabling
         * default functions.
         */
        private static final int SET_FUNCTIONS_LEEWAY_MS = 500;

        /**
         * While switching functions, a disconnect is excpect as the usb gadget
         * us torn down and brought back up. Wait for SET_FUNCTIONS_TIMEOUT_MS +
         * ENUMERATION_TIME_OUT_MS before switching back to default fumctions when
         * switching functions.
         */
        private static final int ENUMERATION_TIME_OUT_MS = 2000;

        /**
         * Gadget HAL fully qualified instance name for registering for ServiceNotification.
         */
        protected static final String GADGET_HAL_FQ_NAME =
                "android.hardware.usb.gadget@1.0::IUsbGadget";

        protected boolean mCurrentUsbFunctionsRequested;

        UsbHandlerHal(Looper looper, Context context, UsbDeviceManager deviceManager,
                UsbAlsaManager alsaManager, UsbPermissionManager permissionManager) {
            super(looper, context, deviceManager, alsaManager, permissionManager);
            try {
                ServiceNotification serviceNotification = new ServiceNotification();

                boolean ret = IServiceManager.getService()
                        .registerForNotifications(GADGET_HAL_FQ_NAME, "", serviceNotification);
                if (!ret) {
                    Slog.e(TAG, "Failed to register usb gadget service start notification");
                    return;
                }

                synchronized (mGadgetProxyLock) {
                    mGadgetProxy = IUsbGadget.getService(true);
                    mGadgetProxy.linkToDeath(new UsbGadgetDeathRecipient(),
                            USB_GADGET_HAL_DEATH_COOKIE);
                    mCurrentFunctions = UsbManager.FUNCTION_NONE;
                    mCurrentUsbFunctionsRequested = true;
                    mGadgetProxy.getCurrentUsbFunctions(new UsbGadgetCallback());
                }
                String state = FileUtils.readTextFile(new File(STATE_PATH), 0, null).trim();
                updateState(state);
            } catch (NoSuchElementException e) {
                Slog.e(TAG, "Usb gadget hal not found", e);
            } catch (RemoteException e) {
                Slog.e(TAG, "Usb Gadget hal not responding", e);
            } catch (Exception e) {
                Slog.e(TAG, "Error initializing UsbHandler", e);
            }
        }


        final class UsbGadgetDeathRecipient implements HwBinder.DeathRecipient {
            @Override
            public void serviceDied(long cookie) {
                if (cookie == USB_GADGET_HAL_DEATH_COOKIE) {
                    Slog.e(TAG, "Usb Gadget hal service died cookie: " + cookie);
                    synchronized (mGadgetProxyLock) {
                        mGadgetProxy = null;
                    }
                }
            }
        }

        final class ServiceNotification extends IServiceNotification.Stub {
            @Override
            public void onRegistration(String fqName, String name, boolean preexisting) {
                Slog.i(TAG, "Usb gadget hal service started " + fqName + " " + name);
                if (!fqName.equals(GADGET_HAL_FQ_NAME)) {
                    Slog.e(TAG, "fqName does not match");
                    return;
                }

                sendMessage(MSG_GADGET_HAL_REGISTERED, preexisting);
            }
        }

        @Override
        public void handleMessage(Message msg) {
            switch (msg.what) {
                case MSG_SET_CHARGING_FUNCTIONS:
                    setEnabledFunctions(UsbManager.FUNCTION_NONE, false);
                    break;
                case MSG_SET_FUNCTIONS_TIMEOUT:
                    Slog.e(TAG, "Set functions timed out! no reply from usb hal");
                    if (msg.arg1 != 1) {
                        // Set this since default function may be selected from Developer options
                        setEnabledFunctions(mScreenUnlockedFunctions, false);
                    }
                    break;
                case MSG_GET_CURRENT_USB_FUNCTIONS:
                    Slog.e(TAG, "prcessing MSG_GET_CURRENT_USB_FUNCTIONS");
                    mCurrentUsbFunctionsReceived = true;

                    if (mCurrentUsbFunctionsRequested) {
                        Slog.e(TAG, "updating mCurrentFunctions");
                        // Mask out adb, since it is stored in mAdbEnabled
                        mCurrentFunctions = ((Long) msg.obj) & ~UsbManager.FUNCTION_ADB;
                        Slog.e(TAG,
                                "mCurrentFunctions:" + mCurrentFunctions + "applied:" + msg.arg1);
                        mCurrentFunctionsApplied = msg.arg1 == 1;
                    }
                    finishBoot();
                    break;
                case MSG_FUNCTION_SWITCH_TIMEOUT:
                    /**
                     * Dont force to default when the configuration is already set to default.
                     */
                    if (msg.arg1 != 1) {
                        // Set this since default function may be selected from Developer options
                        setEnabledFunctions(mScreenUnlockedFunctions, false);
                    }
                    break;
                case MSG_GADGET_HAL_REGISTERED:
                    boolean preexisting = msg.arg1 == 1;
                    synchronized (mGadgetProxyLock) {
                        try {
                            mGadgetProxy = IUsbGadget.getService();
                            mGadgetProxy.linkToDeath(new UsbGadgetDeathRecipient(),
                                    USB_GADGET_HAL_DEATH_COOKIE);
                            if (!mCurrentFunctionsApplied && !preexisting) {
                                setEnabledFunctions(mCurrentFunctions, false);
                            }
                        } catch (NoSuchElementException e) {
                            Slog.e(TAG, "Usb gadget hal not found", e);
                        } catch (RemoteException e) {
                            Slog.e(TAG, "Usb Gadget hal not responding", e);
                        }
                    }
                    break;
                case MSG_RESET_USB_GADGET:
                    synchronized (mGadgetProxyLock) {
                        if (mGadgetProxy == null) {
                            Slog.e(TAG, "reset Usb Gadget mGadgetProxy is null");
                            break;
                        }

                        try {
                            android.hardware.usb.gadget.V1_1.IUsbGadget gadgetProxy =
                                    android.hardware.usb.gadget.V1_1.IUsbGadget
                                            .castFrom(mGadgetProxy);
                            gadgetProxy.reset();
                        } catch (RemoteException e) {
                            Slog.e(TAG, "reset Usb Gadget failed", e);
                        }
                    }
                    break;
                default:
                    super.handleMessage(msg);
            }
        }

        private class UsbGadgetCallback extends IUsbGadgetCallback.Stub {
            int mRequest;
            long mFunctions;
            boolean mChargingFunctions;

            UsbGadgetCallback() {
            }

            UsbGadgetCallback(int request, long functions,
                    boolean chargingFunctions) {
                mRequest = request;
                mFunctions = functions;
                mChargingFunctions = chargingFunctions;
            }

            @Override
            public void setCurrentUsbFunctionsCb(long functions,
                    int status) {
                /**
                 * Callback called for a previous setCurrenUsbFunction
                 */
                if ((mCurrentRequest != mRequest) || !hasMessages(MSG_SET_FUNCTIONS_TIMEOUT)
                        || (mFunctions != functions)) {
                    return;
                }

                removeMessages(MSG_SET_FUNCTIONS_TIMEOUT);
                Slog.e(TAG, "notifyCurrentFunction request:" + mRequest + " status:" + status);
                if (status == Status.SUCCESS) {
                    mCurrentFunctionsApplied = true;
                } else if (!mChargingFunctions) {
                    Slog.e(TAG, "Setting default fuctions");
                    sendEmptyMessage(MSG_SET_CHARGING_FUNCTIONS);
                }
            }

            @Override
            public void getCurrentUsbFunctionsCb(long functions,
                    int status) {
                sendMessage(MSG_GET_CURRENT_USB_FUNCTIONS, functions,
                        status == Status.FUNCTIONS_APPLIED);
            }
        }

        private void setUsbConfig(long config, boolean chargingFunctions) {
            if (true) Slog.d(TAG, "setUsbConfig(" + config + ") request:" + ++mCurrentRequest);
            /**
             * Cancel any ongoing requests, if present.
             */
            removeMessages(MSG_FUNCTION_SWITCH_TIMEOUT);
            removeMessages(MSG_SET_FUNCTIONS_TIMEOUT);
            removeMessages(MSG_SET_CHARGING_FUNCTIONS);

            synchronized (mGadgetProxyLock) {
                if (mGadgetProxy == null) {
                    Slog.e(TAG, "setUsbConfig mGadgetProxy is null");
                    return;
                }
                try {
                    if ((config & UsbManager.FUNCTION_ADB) != 0) {
                        /**
                         * Start adbd if ADB function is included in the configuration.
                         */
                        LocalServices.getService(AdbManagerInternal.class)
                                .startAdbdForTransport(AdbTransportType.USB);
                    } else {
                        /**
                         * Stop adbd otherwise
                         */
                        LocalServices.getService(AdbManagerInternal.class)
                                .stopAdbdForTransport(AdbTransportType.USB);
                    }
                    UsbGadgetCallback usbGadgetCallback = new UsbGadgetCallback(mCurrentRequest,
                            config, chargingFunctions);
                    mGadgetProxy.setCurrentUsbFunctions(config, usbGadgetCallback,
                            SET_FUNCTIONS_TIMEOUT_MS - SET_FUNCTIONS_LEEWAY_MS);
                    sendMessageDelayed(MSG_SET_FUNCTIONS_TIMEOUT, chargingFunctions,
                            SET_FUNCTIONS_TIMEOUT_MS);
                    if (mConnected) {
                        // Only queue timeout of enumeration when the USB is connected
                        sendMessageDelayed(MSG_FUNCTION_SWITCH_TIMEOUT, chargingFunctions,
                                SET_FUNCTIONS_TIMEOUT_MS + ENUMERATION_TIME_OUT_MS);
                    }
                    if (DEBUG) Slog.d(TAG, "timeout message queued");
                } catch (RemoteException e) {
                    Slog.e(TAG, "Remoteexception while calling setCurrentUsbFunctions", e);
                }
            }
        }

        @Override
        protected void setEnabledFunctions(long functions, boolean forceRestart) {
            if (DEBUG) {
                Slog.d(TAG, "setEnabledFunctions functions=" + functions + ", "
                        + "forceRestart=" + forceRestart);
            }
            if (mCurrentFunctions != functions
                    || !mCurrentFunctionsApplied
                    || forceRestart) {
                Slog.i(TAG, "Setting USB config to " + UsbManager.usbFunctionsToString(functions));
                mCurrentFunctions = functions;
                mCurrentFunctionsApplied = false;
                // set the flag to false as that would be stale value
                mCurrentUsbFunctionsRequested = false;

                boolean chargingFunctions = functions == UsbManager.FUNCTION_NONE;
                functions = getAppliedFunctions(functions);

                // Set the new USB configuration.
                setUsbConfig(functions, chargingFunctions);

                if (mBootCompleted && isUsbDataTransferActive(functions)) {
                    // Start up dependent services.
                    updateUsbStateBroadcastIfNeeded(functions);
                }
            }
        }
    }

    /* returns the currently attached USB accessory */
    public UsbAccessory getCurrentAccessory() {
        return mHandler.getCurrentAccessory();
    }

    /**
     * opens the currently attached USB accessory.
     *
     * @param accessory accessory to be openened.
     * @param uid Uid of the caller
     */
    public ParcelFileDescriptor openAccessory(UsbAccessory accessory,
            UsbUserPermissionManager permissions, int uid) {
        UsbAccessory currentAccessory = mHandler.getCurrentAccessory();
        if (currentAccessory == null) {
            throw new IllegalArgumentException("no accessory attached");
        }
        if (!currentAccessory.equals(accessory)) {
            String error = accessory.toString()
                    + " does not match current accessory "
                    + currentAccessory;
            throw new IllegalArgumentException(error);
        }
        permissions.checkPermission(accessory, uid);
        return nativeOpenAccessory();
    }

    public long getCurrentFunctions() {
        return mHandler.getEnabledFunctions();
    }

    /**
     * Returns a dup of the control file descriptor for the given function.
     */
    public ParcelFileDescriptor getControlFd(long usbFunction) {
        FileDescriptor fd = mControlFds.get(usbFunction);
        if (fd == null) {
            return null;
        }
        try {
            return ParcelFileDescriptor.dup(fd);
        } catch (IOException e) {
            Slog.e(TAG, "Could not dup fd for " + usbFunction);
            return null;
        }
    }

    public long getScreenUnlockedFunctions() {
        return mHandler.getScreenUnlockedFunctions();
    }

    /**
     * Adds function to the current USB configuration.
     *
     * @param functions The functions to set, or empty to set the charging function.
     */
    public void setCurrentFunctions(long functions) {
        if (DEBUG) {
            Slog.d(TAG, "setCurrentFunctions(" + UsbManager.usbFunctionsToString(functions) + ")");
        }
        if (functions == UsbManager.FUNCTION_NONE) {
            MetricsLogger.action(mContext, MetricsEvent.ACTION_USB_CONFIG_CHARGING);
        } else if (functions == UsbManager.FUNCTION_MTP) {
            MetricsLogger.action(mContext, MetricsEvent.ACTION_USB_CONFIG_MTP);
        } else if (functions == UsbManager.FUNCTION_PTP) {
            MetricsLogger.action(mContext, MetricsEvent.ACTION_USB_CONFIG_PTP);
        } else if (functions == UsbManager.FUNCTION_MIDI) {
            MetricsLogger.action(mContext, MetricsEvent.ACTION_USB_CONFIG_MIDI);
        } else if (functions == UsbManager.FUNCTION_RNDIS) {
            MetricsLogger.action(mContext, MetricsEvent.ACTION_USB_CONFIG_RNDIS);
        } else if (functions == UsbManager.FUNCTION_ACCESSORY) {
            MetricsLogger.action(mContext, MetricsEvent.ACTION_USB_CONFIG_ACCESSORY);
        }
        mHandler.sendMessage(MSG_SET_CURRENT_FUNCTIONS, functions);
    }

    /**
     * Sets the functions which are set when the screen is unlocked.
     *
     * @param functions Functions to set.
     */
    public void setScreenUnlockedFunctions(long functions) {
        if (DEBUG) {
            Slog.d(TAG, "setScreenUnlockedFunctions("
                    + UsbManager.usbFunctionsToString(functions) + ")");
        }
        mHandler.sendMessage(MSG_SET_SCREEN_UNLOCKED_FUNCTIONS, functions);
    }

    /**
     * Resets the USB Gadget.
     */
    public void resetUsbGadget() {
        if (DEBUG) {
            Slog.d(TAG, "reset Usb Gadget");
        }

        mHandler.sendMessage(MSG_RESET_USB_GADGET, null);
    }

    private void onAdbEnabled(boolean enabled) {
        mHandler.sendMessage(MSG_ENABLE_ADB, enabled);
    }

    /**
     * Write the state to a dump stream.
     */
    public void dump(DualDumpOutputStream dump, String idName, long id) {
        long token = dump.start(idName, id);

        if (mHandler != null) {
            mHandler.dump(dump, "handler", UsbDeviceManagerProto.HANDLER);
        }

        dump.end(token);
    }

    private native String[] nativeGetAccessoryStrings();

    private native ParcelFileDescriptor nativeOpenAccessory();

    private native FileDescriptor nativeOpenControl(String usbFunction);

    private native boolean nativeIsStartRequested();

    private native int nativeGetAudioMode();
}
```

**.rej File Content:**
```text
--- services/usb/java/com/android/server/usb/UsbDeviceManager.java
+++ services/usb/java/com/android/server/usb/UsbDeviceManager.java
@@ -80,9 +80,9 @@ import android.os.storage.StorageVolume;
 import android.provider.Settings;
 import android.service.usb.UsbDeviceManagerProto;
 import android.service.usb.UsbHandlerProto;
+import android.text.TextUtils;
 import android.util.Pair;
 import android.util.Slog;
-import android.text.TextUtils;
 
 import com.android.internal.R;
 import com.android.internal.annotations.GuardedBy;
@@ -880,7 +880,7 @@ public class UsbDeviceManager implements ActivityTaskManagerInternal.ScreenObser
             }
         }
 
-        private void notifyAccessoryModeExit(int operationId) {
+        protected void notifyAccessoryModeExit(int operationId) {
             // make sure accessory mode is off
             // and restore default functions
             Slog.d(TAG, "exited USB accessory mode");
@@ -2313,8 +2313,13 @@ public class UsbDeviceManager implements ActivityTaskManagerInternal.ScreenObser
                      */
                     operationId = sUsbOperationCount.incrementAndGet();
                     if (msg.arg1 != 1) {
-                        // Set this since default function may be selected from Developer options
-                        setEnabledFunctions(mScreenUnlockedFunctions, false, operationId);
+                        if (mCurrentFunctions == UsbManager.FUNCTION_ACCESSORY) {
+                            notifyAccessoryModeExit(operationId);
+                        } else {
+                            // Set this since default function may be selected from Developer
+                            // options
+                            setEnabledFunctions(mScreenUnlockedFunctions, false, operationId);
+                        }
                     }
                     break;
                 case MSG_GADGET_HAL_REGISTERED:
```

**Expected Output Diff:**
```diff
--- a/services/usb/java/com/android/server/usb/UsbDeviceManager.java
+++ b/services/usb/java/com/android/server/usb/UsbDeviceManager.java
@@ -719,7 +719,7 @@ public class UsbDeviceManager implements ActivityTaskManagerInternal.ScreenObser
             }
         }
 
-        private void notifyAccessoryModeExit() {
+        protected void notifyAccessoryModeExit() {
             // make sure accessory mode is off
             // and restore default functions
             Slog.d(TAG, "exited USB accessory mode");
@@ -1957,8 +1957,13 @@ public class UsbDeviceManager implements ActivityTaskManagerInternal.ScreenObser
                      * Dont force to default when the configuration is already set to default.
                      */
                     if (msg.arg1 != 1) {
-                        // Set this since default function may be selected from Developer options
-                        setEnabledFunctions(mScreenUnlockedFunctions, false);
+                        if (mCurrentFunctions == UsbManager.FUNCTION_ACCESSORY) {
+                            notifyAccessoryModeExit();
+                        } else {
+                            // Set this since default function may be selected from Developer
+                            // options
+                            setEnabledFunctions(mScreenUnlockedFunctions, false);
+                        }
                     }
                     break;
                 case MSG_GADGET_HAL_REGISTERED:
```
'''

    patch_porter_agent = GeminiAgent(
        model_name="gemini-2.5-pro-preview-05-06",
        system_prompt=system_prompt,
        key_rotator=key_rotator
    )


    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    version_label = args.target_downstream_version or "all_versions"


    output_json_path_to_use = args.output_json_path
    if output_json_path_to_use is None:
        outputs_dir = "outputs/approach2_results" 
        os.makedirs(outputs_dir, exist_ok=True)
        output_json_path_to_use = os.path.join(
            outputs_dir,
            f"approach2_output_diff_android_{version_label}_{timestamp_str}.json"
        )


    report_data = {
        "run_timestamp": timestamp_str,
        "target_downstream_version": args.target_downstream_version or "all_versions",
        "input_json_file": args.input_json_file_path,
        "main_output_json_file_with_diffs": output_json_path_to_use,
        "summary": {
            "total_file_conflicts_matching_version": 0,
            "files_attempted_for_llm_diff_generation": 0,
            "files_with_llm_diff_successfully_generated": 0, 
            "files_skipped_pre_llm_call": 0,
            "files_with_llm_diff_generation_errors_or_skipped_in_func": 0,
            "successful_attempts_histogram": {},
            "total_runtime_seconds_all": 0,
            "total_runtime_seconds_successful": 0
        },
        "successfully_generated_diffs_log": [],
        "skipped_or_errored_diff_generation_log": []
    }

    try:
        with open(args.input_json_file_path, "r", encoding="utf-8") as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input JSON file not found at '{args.input_json_file_path}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{args.input_json_file_path}'")
        return

    if not isinstance(input_data, list):
        print("Error: Expected a list of vulnerabilities in the input JSON file.")
        return

    output_data = copy.deepcopy(input_data)

    resume_from_id = args.resume_from_id
    resume_mode = resume_from_id is not None

    if resume_mode:
        resume_index = next((i for i, item in enumerate(output_data) if item.get("id") == resume_from_id), None)
        if resume_index is None:
            print(f"❌ Error: resume_from_id '{resume_from_id}' not found in input.")
            return
        output_data = output_data[resume_index:]
        print(f"⏩ Resuming from vulnerability ID: {resume_from_id}")


    report_dir = os.path.join("outputs", "report")
    os.makedirs(report_dir, exist_ok=True)
    report_filename = os.path.join(
        report_dir,
        f"report_diff_{version_label}_{timestamp_str}.json"
    )

    for vulnerability_item in output_data:
        vulnerability_id = vulnerability_item.get("id", "unknown_vuln_id")
        failures = vulnerability_item.get("failures", [])

        for failure in failures:
            if args.target_downstream_version is None or failure.get("downstream_version") == args.target_downstream_version:
                file_conflicts = failure.get("file_conflicts", [])
                for file_conflict in file_conflicts:
                    report_data["summary"]["total_file_conflicts_matching_version"] += 1
                    
                    rej_content_for_llm = file_conflict.get("rej_file_content")
                    original_source_for_llm = file_conflict.get("downstream_file_content_patched_upstream_only")    # This is the "original source" for this task
                    target_filename = file_conflict.get("file_name")
                    
                    skip_reason = None
                    if not target_filename: skip_reason = "Missing 'file_name' (target filename for diff)"
                    elif not rej_content_for_llm: skip_reason = "Missing 'rej_file_content' (rejected hunks)"
                    elif not original_source_for_llm: skip_reason = "Missing 'downstream_file_content' (original source file)"

                    if skip_reason:
                        print(f"Skipping file_conflict in {vulnerability_id} (file: {target_filename or 'N/A'}) due to: {skip_reason}.")
                        report_data["summary"]["files_skipped_pre_llm_call"] += 1
                        report_data["skipped_or_errored_diff_generation_log"].append({
                            "vulnerability_id": vulnerability_id, "file_name": target_filename or "Unknown",
                            "patch_sha": failure.get('downstream_patch', 'N/A'), "reason": skip_reason
                        })
                        continue
                    
                    report_data["summary"]["files_attempted_for_llm_diff_generation"] += 1

                    REPO_BASE = Path("android_repos")

                    vuln_id = vulnerability_item.get("id")
                    repo_url = get_repo_url_from_osv(vuln_id)

                    repo_path_str = failure.get("repo_path")
                    repo_path = Path(repo_path_str) if repo_path_str else None

                    if not repo_path_str:
                        if not repo_url:
                            print(f"⚠️ Skipping due to missing repo_path and repo_url in failure entry for {vulnerability_id}")
                            report_data["summary"]["files_skipped_pre_llm_call"] += 1
                            report_data["skipped_or_errored_diff_generation_log"].append({
                                "vulnerability_id": vulnerability_id,
                                "file_name": target_filename or "Unknown",
                                "patch_sha": failure.get('downstream_patch', 'N/A'),
                                "reason": "Missing repo_path and repo_url"
                            })
                            continue
                        # Clone the repo into android_repos/
                        repo_path = AndroidPatchManager.clone_repo(repo_url, REPO_BASE)
                        failure["repo_path"] = str(repo_path)  # Save for use in downstream
                    else:
                        repo_path = Path(repo_path_str)
                        if not repo_path.exists():
                            if repo_url:
                                print(f"📦 Repo not found at {repo_path}, cloning from {repo_url}")
                                repo_path = AndroidPatchManager.clone_repo(repo_url, REPO_BASE)
                                failure["repo_path"] = str(repo_path)
                            else:
                                print(f"⚠️ Repo path does not exist and repo_url missing for {vulnerability_id}")
                                continue
                        else:
                            print(f"📂 Using existing repo at {repo_path}")

                    
                    # Use the new retry function
                    generated_diff = await process_single_entry_with_retry(
                        rej_content=rej_content_for_llm,
                        original_source_content=original_source_for_llm,
                        target_filename_for_diff=target_filename,
                        vulnerability_id=vulnerability_id,
                        failure_details=failure,
                        upstream_patch_content=vulnerability_item.get("upstream_patch_content"),
                        patch_porter_agent=patch_porter_agent,
                        max_retries=3
                    )

                    runtime = generated_diff.get("runtime_seconds", 0)
                    report_data["summary"]["total_runtime_seconds_all"] = (
                        report_data["summary"].get("total_runtime_seconds_all", 0) + runtime
                    )

                    if generated_diff.get("llm_output_valid"):
                        report_data["summary"]["total_runtime_seconds_successful"] = (
                            report_data["summary"].get("total_runtime_seconds_successful", 0) + runtime
                        )

                    if generated_diff is not None:
                        file_conflict.update(generated_diff) # Update the current file_conflict in output_data
                        
                        # Incrementally save the entire output_data after this file_conflict has been processed
                        # and output_data has been updated.
                        # Ensure vulnerability_item is accessible for the print message.
                        # It should be, as file_conflict is nested within vulnerability_item's loops.
                        current_fc_name_for_save = file_conflict.get("file_name", "unknown_file")
                        vuln_id_for_save = vulnerability_item.get("id", "unknown_vuln_id")
                        print(f"💾 Incrementally saving output after processing file: {current_fc_name_for_save} for vuln: {vuln_id_for_save}")
                        save_partial_output(output_json_path_to_use, output_data)

                        # Incrementally save the report_data
                        print(f"📊 Incrementally saving report data after processing file: {current_fc_name_for_save} for vuln: {vuln_id_for_save}")
                        save_partial_output(report_filename, report_data)
                        
                        if generated_diff.get("llm_output_valid"):
                            report_data["summary"]["files_with_llm_diff_successfully_generated"] += 1
                            # Track number of attempts in histogram (e.g., "1 run", "2 runs")
                            attempts = generated_diff.get("attempts_made", 0)
                            label = f"{attempts} run" if attempts == 1 else f"{attempts} runs"
                            histogram = report_data["summary"]["successful_attempts_histogram"]
                            histogram[label] = histogram.get(label, 0) + 1

                            report_data["successfully_generated_diffs_log"].append({
                                "vulnerability_id": vulnerability_id,
                                "file_name": target_filename,
                                "patch_sha": failure.get('downstream_patch', 'N/A'),
                                "downstream_version": failure.get("downstream_version", "N/A"),
                                "diff_preview": generated_diff["downstream_llm_diff_output"][:100] + "..." if generated_diff.get("downstream_llm_diff_output") else "None"
                            })
                        else:
                            report_data["summary"]["files_with_llm_diff_generation_errors_or_skipped_in_func"] += 1
                            report_data["skipped_or_errored_diff_generation_log"].append({
                                "vulnerability_id": vulnerability_id,
                                "file_name": target_filename,
                                "patch_sha": failure.get('downstream_patch', 'N/A'),
                                "reason": generated_diff.get("error", "Validation failed"),
                                "last_format_error": generated_diff.get("last_format_error"),
                                "last_apply_error": generated_diff.get("last_apply_error")
                            })

    
    try:
        with open(output_json_path_to_use, "w", encoding="utf-8") as f_out:
            json.dump(output_data, f_out, indent=4)
        print(f"✅ Main processed JSON (with diffs) successfully saved to '{output_json_path_to_use}'")
    except IOError as e:
        print(f"Error writing main output JSON to '{output_json_path_to_use}': {e}")

    try:
        with open(report_filename, "w", encoding="utf-8") as f_report:
            json.dump(report_data, f_report, indent=4)
        print(f"✅ Summary diff generation report successfully saved to '{report_filename}'")
    except IOError as e:
        print(f"Error writing summary report to '{report_filename}': {e}")

if __name__ == '__main__':
    asyncio.run(main())

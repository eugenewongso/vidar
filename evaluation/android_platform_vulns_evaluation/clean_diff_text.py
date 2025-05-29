from unidiff import PatchSet
import io

def clean_diff_text(diff_text_str: str) -> str:
    """
    Removes standard diff headers (--- a/..., +++ b/..., --- original, +++ patched)
    and returns only the hunk content starting from the first '@@ '.
    If no '@@ ' is found, returns an empty string, as it implies no comparable hunk data.
    """

    print(diff_text_str)
    if not isinstance(diff_text_str, str):
        return ""
    
    lines = diff_text_str.splitlines() # Work with lines without keepends for easier joining
    print("lines", lines)
    
    hunk_start_index = -1
    for i, line in enumerate(lines):
        if line.startswith("@@ "):
            hunk_start_index = i
            break
            
    if hunk_start_index != -1:
        # If a hunk header is found, take all lines from there and rejoin
        return "\n".join(lines[hunk_start_index:])
    else:
        # No hunk data found (e.g., empty diff, or diff only showed file mode changes)
        return ""
    
def clean_diff_text_2(diff_text_str: str) -> str:
    """
    Uses unidiff to extract and return only the hunk content from a diff string.
    Removes redundant headers and avoids duplication.
    """
    if not isinstance(diff_text_str, str):
        return ""

    try:
        patch = PatchSet(io.StringIO(diff_text_str))
        cleaned_hunks = [str(hunk) for patched_file in patch for hunk in patched_file]
        return "\n".join(cleaned_hunks)
    except Exception as e:
        print(f"Error parsing diff: {e}")
        return ""

    

def main():
    """
    Main function to clean diff text from a given file.
    Reads the diff text from the input file, cleans it, and writes the cleaned text to the output file.
    """
    downstream_patch_content = "commit 2c5add83a18d87ea4a46bc8ab7f675e32c8d6a56\nAuthor: Billy Huang <billyhuang@google.com>\nDate:   Wed Oct 2 14:27:47 2024 -0700\n\n    RESTRICT AUTOMERGE backport \"opp: validate that content uri belongs to current user\"\n    \n    Bug: 296915500\n    Flag: EXEMPT trivial fix with complete testing coverage\n    Test: atest GoogleBluetoothInstrumentationTests:BluetoothOppSendFileInfoTest\n    Ignore-AOSP-First: fix for undisclosed vulnerability\n    Merged-In: I76b25fcd446d5e0530308e21deafa68d0b768edc\n    Change-Id: I0b6423025c95c13eeea3cbf584212913b5fbf307\n\ndiff --git a/android/app/src/com/android/bluetooth/opp/BluetoothOppSendFileInfo.java b/android/app/src/com/android/bluetooth/opp/BluetoothOppSendFileInfo.java\nindex 2adb8e5f44..7ce134341a 100644\n--- a/android/app/src/com/android/bluetooth/opp/BluetoothOppSendFileInfo.java\n+++ b/android/app/src/com/android/bluetooth/opp/BluetoothOppSendFileInfo.java\n@@ -32,6 +32,8 @@\n \n package com.android.bluetooth.opp;\n \n+import static android.os.UserHandle.myUserId;\n+\n import android.content.ContentResolver;\n import android.content.Context;\n import android.content.res.AssetFileDescriptor;\n@@ -39,6 +41,7 @@ import android.database.Cursor;\n import android.database.sqlite.SQLiteException;\n import android.net.Uri;\n import android.provider.OpenableColumns;\n+import android.text.TextUtils;\n import android.util.EventLog;\n import android.util.Log;\n \n@@ -49,6 +52,7 @@ import java.io.File;\n import java.io.FileInputStream;\n import java.io.FileNotFoundException;\n import java.io.IOException;\n+import java.util.Objects;\n \n /**\n  * This class stores information about a single sending file It will only be\n@@ -117,6 +121,11 @@ public class BluetoothOppSendFileInfo {\n                 return SEND_FILE_INFO_ERROR;\n             }\n \n+            if (isContentUriForOtherUser(uri)) {\n+                Log.e(TAG, \"Uri: \" + uri + \" is invalid for user \" + myUserId());\n+                return SEND_FILE_INFO_ERROR;\n+            }\n+\n             contentType = contentResolver.getType(uri);\n             Cursor metadataCursor;\n             try {\n@@ -253,6 +262,12 @@ public class BluetoothOppSendFileInfo {\n         return new BluetoothOppSendFileInfo(fileName, contentType, length, is, 0);\n     }\n \n+    private static boolean isContentUriForOtherUser(Uri uri) {\n+        String uriUserId = uri.getUserInfo();\n+        return !TextUtils.isEmpty(uriUserId)\n+                && !Objects.equals(uriUserId, String.valueOf(myUserId()));\n+    }\n+\n     private static long getStreamSize(FileInputStream is) throws IOException {\n         long length = 0;\n         byte[] unused = new byte[4096];\ndiff --git a/android/app/tests/unit/src/com/android/bluetooth/opp/BluetoothOppSendFileInfoTest.java b/android/app/tests/unit/src/com/android/bluetooth/opp/BluetoothOppSendFileInfoTest.java\nindex 756836afaa..acb58272fb 100644\n--- a/android/app/tests/unit/src/com/android/bluetooth/opp/BluetoothOppSendFileInfoTest.java\n+++ b/android/app/tests/unit/src/com/android/bluetooth/opp/BluetoothOppSendFileInfoTest.java\n@@ -17,6 +17,8 @@\n \n package com.android.bluetooth.opp;\n \n+import static android.os.UserHandle.myUserId;\n+\n import static com.google.common.truth.Truth.assertThat;\n \n import static org.mockito.ArgumentMatchers.any;\n@@ -31,7 +33,6 @@ import android.content.res.AssetFileDescriptor;\n import android.database.MatrixCursor;\n import android.net.Uri;\n import android.provider.OpenableColumns;\n-import android.util.Log;\n \n import androidx.test.platform.app.InstrumentationRegistry;\n import androidx.test.runner.AndroidJUnit4;\n@@ -120,6 +121,110 @@ public class BluetoothOppSendFileInfoTest {\n         assertThat(info).isEqualTo(BluetoothOppSendFileInfo.SEND_FILE_INFO_ERROR);\n     }\n \n+    @Test\n+    public void generateFileInfo_withContentUriForOtherUser_returnsSendFileInfoError()\n+            throws Exception {\n+        String type = \"image/jpeg\";\n+        Uri uri = buildContentUriWithEncodedAuthority((myUserId() + 1) + \"@media\");\n+\n+        long fileLength = 1000;\n+        String fileName = \"pic.jpg\";\n+\n+        FileInputStream fs = mock(FileInputStream.class);\n+        AssetFileDescriptor fd = mock(AssetFileDescriptor.class);\n+        doReturn(fileLength).when(fd).getLength();\n+        doReturn(fs).when(fd).createInputStream();\n+\n+        doReturn(fd).when(mCallProxy).contentResolverOpenAssetFileDescriptor(any(), eq(uri), any());\n+\n+        mCursor =\n+                new MatrixCursor(new String[] {OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE});\n+        mCursor.addRow(new Object[] {fileName, fileLength});\n+\n+        doReturn(mCursor)\n+                .when(mCallProxy)\n+                .contentResolverQuery(any(), eq(uri), any(), any(), any(), any());\n+\n+        BluetoothOppSendFileInfo info =\n+                BluetoothOppSendFileInfo.generateFileInfo(mContext, uri, type, true);\n+\n+        assertThat(info).isEqualTo(BluetoothOppSendFileInfo.SEND_FILE_INFO_ERROR);\n+    }\n+\n+    @Test\n+    public void generateFileInfo_withContentUriForImplicitUser_returnsInfoWithCorrectLength()\n+            throws Exception {\n+        String type = \"image/jpeg\";\n+        Uri uri = buildContentUriWithEncodedAuthority(\"media\");\n+\n+        long fileLength = 1000;\n+        String fileName = \"pic.jpg\";\n+\n+        FileInputStream fs = mock(FileInputStream.class);\n+        AssetFileDescriptor fd = mock(AssetFileDescriptor.class);\n+        doReturn(fileLength).when(fd).getLength();\n+        doReturn(fs).when(fd).createInputStream();\n+\n+        doReturn(fd).when(mCallProxy).contentResolverOpenAssetFileDescriptor(any(), eq(uri), any());\n+\n+        mCursor =\n+                new MatrixCursor(new String[] {OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE});\n+        mCursor.addRow(new Object[] {fileName, fileLength});\n+\n+        doReturn(mCursor)\n+                .when(mCallProxy)\n+                .contentResolverQuery(any(), eq(uri), any(), any(), any(), any());\n+\n+        BluetoothOppSendFileInfo info =\n+                BluetoothOppSendFileInfo.generateFileInfo(mContext, uri, type, true);\n+\n+        assertThat(info.mInputStream).isEqualTo(fs);\n+        assertThat(info.mFileName).isEqualTo(fileName);\n+        assertThat(info.mLength).isEqualTo(fileLength);\n+        assertThat(info.mStatus).isEqualTo(0);\n+    }\n+\n+    @Test\n+    public void generateFileInfo_withContentUriForSameUser_returnsInfoWithCorrectLength()\n+            throws Exception {\n+        String type = \"image/jpeg\";\n+        Uri uri = buildContentUriWithEncodedAuthority(myUserId() + \"@media\");\n+\n+        long fileLength = 1000;\n+        String fileName = \"pic.jpg\";\n+\n+        FileInputStream fs = mock(FileInputStream.class);\n+        AssetFileDescriptor fd = mock(AssetFileDescriptor.class);\n+        doReturn(fileLength).when(fd).getLength();\n+        doReturn(fs).when(fd).createInputStream();\n+\n+        doReturn(fd).when(mCallProxy).contentResolverOpenAssetFileDescriptor(any(), eq(uri), any());\n+\n+        mCursor =\n+                new MatrixCursor(new String[] {OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE});\n+        mCursor.addRow(new Object[] {fileName, fileLength});\n+\n+        doReturn(mCursor)\n+                .when(mCallProxy)\n+                .contentResolverQuery(any(), eq(uri), any(), any(), any(), any());\n+\n+        BluetoothOppSendFileInfo info =\n+                BluetoothOppSendFileInfo.generateFileInfo(mContext, uri, type, true);\n+\n+        assertThat(info.mInputStream).isEqualTo(fs);\n+        assertThat(info.mFileName).isEqualTo(fileName);\n+        assertThat(info.mLength).isEqualTo(fileLength);\n+        assertThat(info.mStatus).isEqualTo(0);\n+    }\n+\n+    private static Uri buildContentUriWithEncodedAuthority(String authority) {\n+        return new Uri.Builder()\n+                .scheme(\"content\")\n+                .encodedAuthority(authority)\n+                .path(\"external/images/media/1\")\n+                .build();\n+    }\n+\n     @Test\n     public void generateFileInfo_withoutPermissionForAccessingUri_returnsSendFileInfoError() {\n         String type = \"text/plain\";\n"
    LLM_diff_content = "--- original\n+++ patched\n@@ -258,11 +258,11 @@\n             PermissionControllerStatsLog.write(REVIEW_PERMISSIONS_FRAGMENT_RESULT_REPORTED,\n                     changeId, mViewModel.getPackageInfo().applicationInfo.uid,\n                     group.getPackageName(),\n-                    permission.getName(), permission.isGrantedIncludingAppOp());\n+                    permission.getName(), permission.isGranted());\n             Log.v(LOG_TAG, \"Permission grant via permission review changeId=\" + changeId + \" uid=\"\n                     + mViewModel.getPackageInfo().applicationInfo.uid + \" packageName=\"\n                     + group.getPackageName() + \" permission=\"\n-                    + permission.getName() + \" granted=\" + permission.isGrantedIncludingAppOp());\n+                    + permission.getName() + \" granted=\" + permission.isGranted());\n         }\n     }\n \n"


    # cleaned_diff_text = clean_diff_text(downstream_patch_content)
    # cleaned_diff_text = clean_diff_text(LLM_diff_content)

    cleaned_diff_text = clean_diff_text_2(LLM_diff_content)


    print("Cleaned diff text:" + cleaned_diff_text)

if __name__ == "__main__":
    main()
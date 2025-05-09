#!/bin/bash

KERNEL_DIR="/Volumes/GitRepo/school/capstone/android/Xiaomi_Kernel_OpenSource"
PATCH_DIR="$KERNEL_DIR/generated_patches"
LOG_FILE="$KERNEL_DIR/patch_application_log.txt"

cd "$KERNEL_DIR" || { echo "❌ Kernel dir not found."; exit 1; }

echo "📝 Starting patch application at $(date)" > "$LOG_FILE"

for patch in "$PATCH_DIR"/*.diff; do
  echo "🔧 Applying $patch..." | tee -a "$LOG_FILE"
  if gpatch -p1 < "$patch"; then
    echo "✅ Applied: $patch" | tee -a "$LOG_FILE"
  else
    echo "❌ Failed: $patch" | tee -a "$LOG_FILE"
  fi
done

echo "📄 Finished patch application." | tee -a "$LOG_FILE"

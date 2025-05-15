#!/bin/bash

# Define an array of years to run for
years=(2024 2023)

# Loop through each year
for year in "${years[@]}"
do
  echo "▶️ Running for $year..."
  python3 osv_patch_runner.py \
    --after "${year}-01-01" \
    --before "${year}-12-31"
done

echo "✅ All runs complete."

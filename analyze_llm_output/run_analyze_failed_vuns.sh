#!/bin/bash

# Example usage of the analyze_failed_vulns.py script

echo "🔧 Installing chart dependencies (optional for charts only)..."
pip install -r requirements_charts.txt

echo ""
echo "📊 Running analysis with both charts and tables..."

# Create output directory name with timestamp
OUTPUT_DIR="vulnerability_analysis_$(date +%Y%m%d_%H%M%S)"

# Usage with tables and charts - all outputs in one directory
python3 analyze_failed_vulns.py \
    "outputs/report/report_approach2_output_diff_android_all_versions_smart_retry.json" \
    "outputs/report/report_all_versions_20242025_approach_2_llm_output_no_guideline.json" \
    "outputs/report/report_all_versions_20242025_approach_2_llm_output_less_error_message.json" \
    "outputs/report/report_all_versions_20242025_approach_2_llm_output_no_guideline_less_error_message.json" \
    "outputs/report/report_approach2_output_diff_android_all_versions_blind_retry_rerun_2.json" \
    "outputs/report/report_approach2_output_diff_android_all_versions_blind_retry_without.json" \
    --output-dir "$OUTPUT_DIR" \
    --tables \
    --charts \
    --verbose \
    -o analysis_complete.json \
    --table-file vulnerability_analysis_report.html \
    --chart-dir charts

echo ""
echo "✅ Analysis complete!"
echo ""
echo "📁 All outputs saved to: $OUTPUT_DIR/"
echo "  📄 JSON Summary: $OUTPUT_DIR/analysis_complete.json"
echo "  📋 HTML Tables: $OUTPUT_DIR/vulnerability_analysis_report.html"
echo "  📊 Charts saved to: $OUTPUT_DIR/charts/"
echo ""
echo "Generated files in $OUTPUT_DIR/:"
echo "  📋 vulnerability_analysis_report.html - Interactive HTML tables (open in browser)"
echo "  📈 charts/top_failed_vulnerabilities.png - Top failed vulnerabilities"
echo "  🥧 charts/failure_reasons_pie.png - Distribution of failure reasons"
echo "  📊 charts/source_file_distribution.png - Source file distribution for top failures"
echo "  📁 charts/top_failed_files.png - Most frequently failing files"

echo ""
echo "🌐 To view the HTML report, open: $OUTPUT_DIR/vulnerability_analysis_report.html in your web browser"

# Example: Tables only (no chart dependencies needed)
echo ""
echo "📋 Running analysis with tables only (no chart dependencies required)..."
# SIMPLE_OUTPUT_DIR="simple_analysis_$(date +%Y%m%d_%H%M%S)"
# python3 analyze_failed_vulns.py "outputs/report/report_diff_all_versions_20250529_233253.json" \
#     --output-dir "$SIMPLE_OUTPUT_DIR" \
#     --tables \
#     -o tables_only_analysis.json \
#     --table-file tables_only_report.html

# Example with multiple report files (uncomment and modify paths as needed)
# echo ""
# echo "📊 Running analysis with multiple report files..."
# MULTI_OUTPUT_DIR="multi_analysis_$(date +%Y%m%d_%H%M%S)"
# python3 analyze_failed_vulns.py \
#     "outputs/report/report1.json" \
#     "outputs/report/report2.json" \
#     "outputs/report/report3.json" \
#     --output-dir "$MULTI_OUTPUT_DIR" \
#     --tables \
#     --charts \
#     -o multi_report_analysis.json \
#     --table-file multi_report_tables.html \
#     --chart-dir charts

echo ""
echo "🎯 Usage examples:"
echo "  Tables only:  python3 analyze_failed_vulns.py report.json --output-dir results --tables"
echo "  Charts only:  python3 analyze_failed_vulns.py report.json --output-dir results --charts"
echo "  Both:         python3 analyze_failed_vulns.py report.json --output-dir results --tables --charts"
echo "  Multiple:     python3 analyze_failed_vulns.py report1.json report2.json --output-dir results --tables" 
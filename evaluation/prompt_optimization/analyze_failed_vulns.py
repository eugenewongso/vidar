"""
Script to analyze multiple report JSON files and summarize failed vulnerabilities.
Identifies which vulnerabilities failed most frequently across different runs.
"""

import json
import argparse
import os
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np


def load_report_file(file_path: str) -> Dict[str, Any]:
    """Load and parse a report JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return {}


def extract_failures_from_report(report_data: Dict[str, Any], source_file: str) -> List[Dict[str, Any]]:
    """Extract failure information from a single report."""
    failures = []
    
    # Get failures from skipped_or_errored_diff_generation_log
    skipped_errors = report_data.get("skipped_or_errored_diff_generation_log", [])
    
    for entry in skipped_errors:
        failure_info = {
            "vulnerability_id": entry.get("vulnerability_id", "unknown"),
            "file_name": entry.get("file_name", "unknown"),
            "patch_sha": entry.get("patch_sha", "unknown"),
            "vuln_key": entry.get("vulnerability_id", "unknown"),  # Use just vulnerability_id as key
            "reason": entry.get("reason", "unknown"),
            "last_format_error": entry.get("last_format_error"),
            "last_apply_error": entry.get("last_apply_error"),
            "source_file": source_file,
            "run_timestamp": report_data.get("run_timestamp", "unknown"),
            "target_version": report_data.get("target_downstream_version", "unknown")
        }
        failures.append(failure_info)
    
    return failures


def analyze_failures(all_failures: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze all failures and create summary statistics."""
    
    # First, group failures by source file and deduplicate vulnerabilities within each file
    failures_by_source = defaultdict(list)
    for failure in all_failures:
        failures_by_source[failure["source_file"]].append(failure)
    
    # Count unique vulnerabilities per file, then sum across files
    vuln_failure_counts = Counter()
    vuln_details = defaultdict(lambda: {
        "total_files_with_failures": 0,
        "vulnerability_id": "",
        "unique_patches": set(),
        "unique_files": set(),
        "unique_reasons": set(),
        "source_files": set(),
        "failure_details": []
    })
    
    # Count failures by reason (still counting all individual failures)
    reason_counts = Counter()
    
    # Count failures by file (still counting all individual failures)
    file_failure_counts = Counter()
    
    # Track failure reasons for each file
    file_failure_reasons = defaultdict(lambda: defaultdict(int))
    
    # Track which source files each vulnerability comes from
    vuln_source_mapping = defaultdict(set)
    
    # Process each source file
    for source_file, failures_in_file in failures_by_source.items():
        # Get unique vulnerabilities in this file
        unique_vulns_in_file = set()
        
        for failure in failures_in_file:
            vuln_key = failure["vuln_key"]
            file_name = failure["file_name"]
            reason = failure["reason"]
            patch_sha = failure["patch_sha"]
            
            # Add to unique vulnerabilities for this file
            unique_vulns_in_file.add(vuln_key)
            
            # Count individual failures for reason and file statistics
            reason_counts[reason] += 1
            file_failure_counts[file_name] += 1
            
            # Track failure reasons for each file
            file_failure_reasons[file_name][reason] += 1
            
            # Update detailed info for vulnerability
            vuln_details[vuln_key]["vulnerability_id"] = failure["vulnerability_id"]
            vuln_details[vuln_key]["unique_patches"].add(patch_sha)
            vuln_details[vuln_key]["unique_files"].add(file_name)
            vuln_details[vuln_key]["unique_reasons"].add(reason)
            vuln_details[vuln_key]["source_files"].add(source_file)
            vuln_details[vuln_key]["failure_details"].append(failure)
            
            # Track source mapping
            vuln_source_mapping[vuln_key].add(source_file)
        
        # Count each unique vulnerability in this file only once
        for vuln_key in unique_vulns_in_file:
            vuln_failure_counts[vuln_key] += 1
            vuln_details[vuln_key]["total_files_with_failures"] += 1
    
    # Convert sets to lists for JSON serialization
    for vuln_key in vuln_details:
        vuln_details[vuln_key]["unique_patches"] = list(vuln_details[vuln_key]["unique_patches"])
        vuln_details[vuln_key]["unique_files"] = list(vuln_details[vuln_key]["unique_files"])
        vuln_details[vuln_key]["unique_reasons"] = list(vuln_details[vuln_key]["unique_reasons"])
        vuln_details[vuln_key]["source_files"] = list(vuln_details[vuln_key]["source_files"])
        vuln_source_mapping[vuln_key] = list(vuln_source_mapping[vuln_key])
    
    # Convert file_failure_reasons to regular dict for JSON serialization
    file_failure_reasons_dict = {}
    for file_name, reasons in file_failure_reasons.items():
        file_failure_reasons_dict[file_name] = dict(reasons)
    
    return {
        "vuln_failure_counts": dict(vuln_failure_counts),
        "vuln_details": dict(vuln_details),
        "vuln_source_mapping": dict(vuln_source_mapping),
        "reason_counts": dict(reason_counts),
        "file_failure_counts": dict(file_failure_counts),
        "file_failure_reasons": file_failure_reasons_dict,
        "total_failures": len(all_failures),
        "unique_vulnerabilities": len(vuln_failure_counts),
        "unique_files": len(file_failure_counts)
    }


def create_html_tables(analysis_results: Dict[str, Any], output_file: str, report_files: List[str] = None) -> None:
    """Create interactive HTML tables for failure analysis."""
    
    vuln_failure_counts = analysis_results["vuln_failure_counts"]
    vuln_details = analysis_results["vuln_details"]
    reason_counts = analysis_results["reason_counts"]
    file_failure_counts = analysis_results["file_failure_counts"]
    file_failure_reasons = analysis_results["file_failure_reasons"]
    
    # Sort data for tables
    top_vulns = sorted(vuln_failure_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    top_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
    top_files = sorted(file_failure_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    
    # Prepare report files information
    if report_files:
        report_files_display = ", ".join([os.path.basename(f) for f in report_files])
        num_files = len(report_files)
    else:
        report_files_display = "Unknown"
        num_files = 0
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vulnerability Failure Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1, h2 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .cross-input-card {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: white;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            position: sticky;
            top: 0;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .rank {{
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
        }}
        .vuln-id {{
            font-family: monospace;
            font-size: 0.9em;
            background-color: #e8f5e8;
            padding: 2px 6px;
            border-radius: 4px;
        }}
        .file-name {{
            font-family: monospace;
            font-size: 0.85em;
            max-width: 300px;
            word-break: break-word;
        }}
        .failure-count {{
            font-weight: bold;
            color: #d32f2f;
            text-align: center;
        }}
        .source-files {{
            font-size: 0.8em;
            color: #666;
        }}
        .reason {{
            max-width: 250px;
            word-wrap: break-word;
        }}
        .percentage {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .metadata {{
            background-color: #f0f0f0;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            font-size: 0.9em;
        }}
        .collapsible {{
            background-color: #4CAF50;
            color: white;
            cursor: pointer;
            padding: 18px;
            width: 100%;
            border: none;
            text-align: left;
            outline: none;
            font-size: 15px;
            margin-top: 10px;
        }}
        .collapsible:hover {{
            background-color: #45a049;
        }}
        .content {{
            padding: 0 18px;
            display: none;
            overflow: hidden;
            background-color: #f1f1f1;
        }}
        .content.active {{
            display: block;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Vulnerability Failure Analysis Report</h1>
        
        <div class="metadata">
            <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
            <strong>Report Files Analyzed:</strong> {num_files}<br>
            <strong>Source Files:</strong> {report_files_display}<br>
            <strong>Analysis:</strong> Vulnerabilities that failed in one or more source files
        </div>
        
        <div class="summary-stats">
            <div class="stat-card">
                <div class="stat-number">{analysis_results["total_failures"]}</div>
                <div class="stat-label">Total Failures</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{analysis_results["unique_vulnerabilities"]}</div>
                <div class="stat-label">Unique Vulnerabilities</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{analysis_results["unique_files"]}</div>
                <div class="stat-label">Unique Files Affected</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(reason_counts)}</div>
                <div class="stat-label">Different Failure Types</div>
            </div>
        </div>

        <h2>🔥 Top 20 Failed Vulnerabilities</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Vulnerability ID</th>
                    <th>Total Failures</th>
                    <th>Files Affected</th>
                    <th>Failure Types</th>
                    <th>Patches Involved</th>
                </tr>
            </thead>
            <tbody>"""
    
    for i, (vuln_key, files_with_failures) in enumerate(top_vulns, 1):
        details = vuln_details[vuln_key]
        individual_failures = len(details["failure_details"])
        
        html_content += f"""
                <tr>
                    <td class="rank">#{i}</td>
                    <td class="vuln-id">{details["vulnerability_id"]}</td>
                    <td class="failure-count">{individual_failures}</td>
                    <td>{len(details["unique_files"])}</td>
                    <td>{len(details["unique_reasons"])}</td>
                    <td>{len(details["unique_patches"])}</td>
                </tr>"""
    
    html_content += """
            </tbody>
        </table>

        <h2>❌ Failure Reasons Breakdown</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Failure Reason</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
            </thead>
            <tbody>"""
    
    total_individual_failures = sum(reason_counts.values())
    for i, (reason, count) in enumerate(top_reasons, 1):
        percentage = (count / total_individual_failures) * 100
        html_content += f"""
                <tr>
                    <td class="rank">#{i}</td>
                    <td class="reason">{reason}</td>
                    <td class="failure-count">{count}</td>
                    <td class="percentage">{percentage:.1f}%</td>
                </tr>"""
    
    html_content += """
            </tbody>
        </table>

        <h2>📁 Top 20 Most Failed Files</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>File Name</th>
                    <th>Failure Count</th>
                    <th>Percentage of Total</th>
                    <th>Failure Reasons</th>
                </tr>
            </thead>
            <tbody>"""
    
    for i, (file_name, count) in enumerate(top_files, 1):
        percentage = (count / total_individual_failures) * 100
        # Get failure reasons for this file
        reasons = file_failure_reasons.get(file_name, {})
        reasons_text = ", ".join([f"{reason} ({cnt})" for reason, cnt in sorted(reasons.items(), key=lambda x: x[1], reverse=True)])
        
        html_content += f"""
                <tr>
                    <td class="rank">#{i}</td>
                    <td class="file-name">{file_name}</td>
                    <td class="failure-count">{count}</td>
                    <td class="percentage">{percentage:.1f}%</td>
                    <td class="reason">{reasons_text}</td>
                </tr>"""
    
    html_content += """
            </tbody>
        </table>

        <h2>📋 Detailed Breakdown by Vulnerability</h2>"""
    
    for vuln_key, files_with_failures in top_vulns:
        details = vuln_details[vuln_key]
        individual_failures = len(details["failure_details"])
        
        html_content += f"""
        <button type="button" class="collapsible">{details["vulnerability_id"]} - {individual_failures} failures in {len(details["unique_files"])} files</button>
        <div class="content">
            <table style="margin-left: 20px;">
                <thead>
                    <tr>
                        <th>File Name</th>
                        <th>Failure Reason</th>
                        <th>Source Report</th>
                        <th>Target Version</th>
                    </tr>
                </thead>
                <tbody>"""
        
        for failure in details["failure_details"]:
            html_content += f"""
                    <tr>
                        <td class="file-name">{failure.get('file_name', 'unknown')}</td>
                        <td class="reason">{failure.get('reason', 'unknown')}</td>
                        <td class="source-files">{failure.get('source_file', 'unknown')}</td>
                        <td>{failure.get('target_version', 'unknown')}</td>
                    </tr>"""
            
        html_content += """
                </tbody>
            </table>
        </div>"""
    
    html_content += """
    </div>

    <script>
        // Add collapsible functionality
        var coll = document.getElementsByClassName("collapsible");
        for (var i = 0; i < coll.length; i++) {
            coll[i].addEventListener("click", function() {
                this.classList.toggle("active");
                var content = this.nextElementSibling;
                if (content.style.display === "block") {
                    content.style.display = "none";
                } else {
                    content.style.display = "block";
                }
            });
        }
    </script>
</body>
</html>"""
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"HTML tables saved to: {output_file}")


def create_charts(analysis_results: Dict[str, Any], output_dir: str = "charts") -> None:
    """Create visualizations for failure analysis."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    vuln_failure_counts = analysis_results["vuln_failure_counts"]
    vuln_details = analysis_results["vuln_details"]
    reason_counts = analysis_results["reason_counts"]
    file_failure_counts = analysis_results["file_failure_counts"]
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Top Failed Vulnerabilities (by files with failures)
    top_vulns = sorted(vuln_failure_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    if top_vulns:
        fig, ax = plt.subplots(figsize=(12, 8))
        vuln_ids = [vuln_details[vuln]["vulnerability_id"] for vuln, _ in top_vulns]
        files_with_failures = [count for _, count in top_vulns]
        
        bars = ax.barh(range(len(vuln_ids)), files_with_failures, color=sns.color_palette("viridis", len(vuln_ids)))
        ax.set_yticks(range(len(vuln_ids)))
        ax.set_yticklabels(vuln_ids, fontsize=10)
        ax.set_xlabel('Number of Files with Failures')
        ax.set_title('Top 15 Failed Vulnerabilities\n(by number of files affected)', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, files_with_failures)):
            individual_failures = len(vuln_details[top_vulns[i][0]]["failure_details"])
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                   f'{count} files\n({individual_failures} total)', 
                   va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/top_failed_vulnerabilities.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Failure Reasons Distribution
    if reason_counts:
        fig, ax = plt.subplots(figsize=(10, 8))
        reasons = list(reason_counts.keys())
        counts = list(reason_counts.values())
        
        colors = sns.color_palette("Set3", len(reasons))
        wedges, texts, autotexts = ax.pie(counts, labels=reasons, autopct='%1.1f%%', 
                                         colors=colors, startangle=90)
        
        ax.set_title('Distribution of Failure Reasons\n(by individual failure occurrences)', fontsize=14, fontweight='bold')
        
        # Improve text readability
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/failure_reasons_pie.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Source File Distribution for Top Failed Vulnerabilities
    if top_vulns and len(top_vulns) > 0:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Get top 10 for this chart
        top_10_vulns = top_vulns[:10]
        vuln_labels = [vuln_details[vuln]["vulnerability_id"] for vuln, _ in top_10_vulns]
        
        # Count source files for each vulnerability
        source_counts = []
        for vuln, _ in top_10_vulns:
            details = vuln_details[vuln]
            source_files = details["source_files"]
            source_counts.append(len(source_files))
        
        # Create stacked bar chart
        colors = sns.color_palette("viridis", len(top_10_vulns))
        bars = ax.bar(range(len(vuln_labels)), source_counts, color=colors)
        
        ax.set_xticks(range(len(vuln_labels)))
        ax.set_xticklabels(vuln_labels, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Number of Source Reports')
        ax.set_title('Source Report Distribution\nfor Top 10 Failed Vulnerabilities', fontsize=14, fontweight='bold')
        
        # Add value labels
        for bar, count in zip(bars, source_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/source_file_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Top Failed Files (by individual failure count)
    top_files = sorted(file_failure_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    if top_files:
        fig, ax = plt.subplots(figsize=(12, 8))
        file_names = [os.path.basename(file) for file, _ in top_files]
        failure_counts = [count for _, count in top_files]
        
        bars = ax.barh(range(len(file_names)), failure_counts, color=sns.color_palette("plasma", len(file_names)))
        ax.set_yticks(range(len(file_names)))
        ax.set_yticklabels(file_names, fontsize=10)
        ax.set_xlabel('Individual Failure Count')
        ax.set_title('Top 15 Files with Most Individual Failures', fontsize=14, fontweight='bold')
        
        # Add value labels
        for bar, count in zip(bars, failure_counts):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                   str(count), va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/top_failed_files.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Charts saved to {output_dir}/ directory:")
    print(f"  - top_failed_vulnerabilities.png (files affected)")
    print(f"  - failure_reasons_pie.png (individual failures)")  
    print(f"  - source_file_distribution.png")
    print(f"  - top_failed_files.png (individual failures)")


def create_summary_report(analysis: Dict[str, Any], report_files: List[str]) -> Dict[str, Any]:
    """Create a comprehensive summary report."""
    
    # Sort vulnerabilities by failure count (descending)
    sorted_vulnerabilities = sorted(
        analysis["vuln_failure_counts"].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Sort reasons by count (descending)
    sorted_reasons = sorted(
        analysis["reason_counts"].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Sort files by failure count (descending)
    sorted_files = sorted(
        analysis["file_failure_counts"].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Create top 10 lists
    top_10_vulnerabilities = sorted_vulnerabilities[:10]
    top_10_files = sorted_files[:10]
    
    # Calculate cross-input aggregation statistics
    cross_input_stats = {
        "total_unique_vulnerabilities": len(analysis["vuln_failure_counts"]),
        "vulnerabilities_with_multiple_failures": sum(1 for count in analysis["vuln_failure_counts"].values() if count > 1),
        "vulnerabilities_appearing_in_multiple_reports": 0,
        "max_failures_for_single_vulnerability": max(analysis["vuln_failure_counts"].values()) if analysis["vuln_failure_counts"] else 0,
        "average_failures_per_vulnerability": round(sum(analysis["vuln_failure_counts"].values()) / len(analysis["vuln_failure_counts"]), 2) if analysis["vuln_failure_counts"] else 0
    }
    
    # Count vulnerabilities appearing in multiple source reports
    for vuln_key in analysis["vuln_failure_counts"]:
        source_files = analysis["vuln_source_mapping"].get(vuln_key, [])
        if len(source_files) > 1:
            cross_input_stats["vulnerabilities_appearing_in_multiple_reports"] += 1
    
    summary = {
        "analysis_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "input_report_files": report_files,
        "summary_statistics": {
            "total_failures_analyzed": analysis["total_failures"],
            "unique_vulnerabilities_with_failures": analysis["unique_vulnerabilities"],
            "unique_files_with_failures": analysis["unique_files"],
            "total_report_files_analyzed": len(report_files)
        },
        "cross_input_aggregation_stats": cross_input_stats,
        "all_vuln_failure_totals": [
            {
                "vuln_key": vuln_key,
                "vulnerability_id": analysis["vuln_details"][vuln_key]["vulnerability_id"],
                "total_failures_across_all_inputs": count,
                "appears_in_num_reports": len(analysis["vuln_source_mapping"].get(vuln_key, [])),
                "source_report_files": analysis["vuln_source_mapping"].get(vuln_key, []),
                "unique_files_affected": len(analysis["vuln_details"][vuln_key]["unique_files"]),
                "unique_failure_reasons": len(analysis["vuln_details"][vuln_key]["unique_reasons"])
            }
            for vuln_key, count in sorted_vulnerabilities
        ],
        "top_10_most_failed_vulnerabilities": [
            {
                "vuln_key": vuln_key,
                "vulnerability_id": analysis["vuln_details"][vuln_key]["vulnerability_id"],
                "failure_count": count,
                "unique_files_affected": len(analysis["vuln_details"][vuln_key]["unique_files"]),
                "unique_failure_reasons": len(analysis["vuln_details"][vuln_key]["unique_reasons"]),
                "source_files": analysis["vuln_details"][vuln_key]["source_files"]
            }
            for vuln_key, count in top_10_vulnerabilities
        ],
        "top_10_most_failed_files": [
            {
                "file_name": file_name,
                "failure_count": count,
                "failure_reasons": analysis["file_failure_reasons"].get(file_name, {})
            }
            for file_name, count in top_10_files
        ],
        "failure_reasons_breakdown": [
            {
                "reason": reason,
                "count": count,
                "percentage": round((count / analysis["total_failures"]) * 100, 2)
            }
            for reason, count in sorted_reasons
        ],
        "detailed_vuln_analysis": {}
    }
    
    # Add detailed analysis for top 10 vulnerabilities
    for vuln_key, _ in top_10_vulnerabilities:
        details = analysis["vuln_details"][vuln_key]
        summary["detailed_vuln_analysis"][vuln_key] = {
            "vulnerability_id": details["vulnerability_id"],
            "total_failures": details["total_files_with_failures"],
            "files_affected": details["unique_files"],
            "failure_reasons": details["unique_reasons"],
            "source_reports": details["source_files"],
            "failure_breakdown_by_reason": {}
        }
        
        # Count failures by reason for this vulnerability
        reason_breakdown = Counter()
        for failure in details["failure_details"]:
            reason_breakdown[failure["reason"]] += 1
        
        summary["detailed_vuln_analysis"][vuln_key]["failure_breakdown_by_reason"] = dict(reason_breakdown)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze failed vulnerabilities across multiple report files")
    parser.add_argument("report_files", nargs="+", help="Paths to report JSON files")
    parser.add_argument("-o", "--output", help="Output file name for summary (default: failure_analysis_summary.json)", 
                       default="failure_analysis_summary.json")
    parser.add_argument("--output-dir", help="Directory to save all output files (default: current directory)", 
                       default=".")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed output to console")
    parser.add_argument("--charts", "-c", action="store_true", help="Generate charts and visualizations")
    parser.add_argument("--chart-dir", help="Directory name for chart output within output-dir (default: charts)", default="charts")
    parser.add_argument("--tables", "-t", action="store_true", help="Generate HTML tables report")
    parser.add_argument("--table-file", help="HTML table output file name (default: failure_analysis_tables.html)", 
                       default="failure_analysis_tables.html")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct full paths for all outputs within the output directory
    json_output_path = output_dir / args.output
    html_output_path = output_dir / args.table_file
    chart_output_dir = output_dir / args.chart_dir
    
    print(f"🔍 Analyzing {len(args.report_files)} report files...")
    print(f"📁 Output directory: {output_dir.absolute()}")
    
    # Load all report files and extract failures
    all_failures = []
    valid_files = []
    
    for file_path in args.report_files:
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue
            
        print(f"📄 Processing: {file_path}")
        report_data = load_report_file(file_path)
        
        if not report_data:
            continue
            
        failures = extract_failures_from_report(report_data, os.path.basename(file_path))
        all_failures.extend(failures)
        valid_files.append(file_path)
        
        print(f"  └─ Found {len(failures)} failures")
    
    if not all_failures:
        print("❌ No failures found in any report files")
        return
    
    print(f"\n📊 Total failures found: {len(all_failures)}")
    print(f"📁 Valid report files processed: {len(valid_files)}")
    
    # Analyze failures
    analysis = analyze_failures(all_failures)
    
    # Create summary report
    summary = create_summary_report(analysis, valid_files)
    
    # Generate HTML tables if requested
    if args.tables:
        try:
            create_html_tables(analysis, str(html_output_path), valid_files)
        except Exception as e:
            print(f"❌ Error generating HTML tables: {e}")
    
    # Generate charts if requested
    if args.charts:
        try:
            create_charts(analysis, str(chart_output_dir))
        except ImportError as e:
            print(f"⚠️ Could not generate charts. Missing dependencies: {e}")
            print("💡 Install with: pip install matplotlib seaborn pandas")
        except Exception as e:
            print(f"❌ Error generating charts: {e}")
    
    # Save summary to file
    try:
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"✅ Summary saved to: {json_output_path}")
    except Exception as e:
        print(f"❌ Error saving summary: {e}")
        return
    
    # Print console summary
    print("\n" + "="*80)
    print("FAILURE ANALYSIS SUMMARY")
    print("="*80)
    print(f"📁 All outputs saved to: {output_dir.absolute()}")
    if args.tables:
        print(f"📋 HTML Report: {html_output_path}")
    if args.charts:
        print(f"📊 Charts Directory: {chart_output_dir}")
    print(f"📄 JSON Summary: {json_output_path}")
    
    print(f"\nTotal Failures: {summary['summary_statistics']['total_failures_analyzed']}")
    print(f"Unique Vulnerabilities with Failures: {summary['summary_statistics']['unique_vulnerabilities_with_failures']}")
    print(f"Unique Files with Failures: {summary['summary_statistics']['unique_files_with_failures']}")
    
    # Add cross-input aggregation summary
    print("\n🔗 CROSS-INPUT AGGREGATION SUMMARY:")
    print("-" * 50)
    cross_stats = summary["cross_input_aggregation_stats"]
    print(f"📊 Total unique vulnerabilities: {cross_stats['total_unique_vulnerabilities']}")
    print(f"🔥 Vulnerabilities with multiple failures: {cross_stats['vulnerabilities_with_multiple_failures']}")
    print(f"📁 Vulnerabilities appearing in multiple reports: {cross_stats['vulnerabilities_appearing_in_multiple_reports']}")
    print(f"🎯 Maximum failures for single vulnerability: {cross_stats['max_failures_for_single_vulnerability']}")
    print(f"📈 Average failures per vulnerability: {cross_stats['average_failures_per_vulnerability']}")
    
    print("\n🔥 TOP 10 MOST FAILED VULNERABILITIES (ACROSS ALL INPUTS):")
    print("-" * 70)
    for i, vuln in enumerate(summary["top_10_most_failed_vulnerabilities"], 1):
        # Get additional cross-input info
        all_totals = next((item for item in summary["all_vuln_failure_totals"] 
                          if item["vuln_key"] == vuln["vuln_key"]), {})
        appears_in_reports = all_totals.get("appears_in_num_reports", 0)
        
        print(f"{i:2d}. {vuln['vuln_key']} ({vuln['failure_count']} total failures)")
        print(f"    Vulnerability: {vuln['vulnerability_id']}")
        print(f"    Appears in {appears_in_reports} report file(s)")
        print(f"    Files affected: {vuln['unique_files_affected']}")
        print(f"    Failure types: {vuln['unique_failure_reasons']}")
        if args.verbose:
            print(f"    Source reports: {', '.join(vuln['source_files'])}")
        print()
    
    print("📁 TOP 10 MOST FAILED FILES:")
    print("-" * 40)
    for i, file_info in enumerate(summary["top_10_most_failed_files"], 1):
        print(f"{i:2d}. {file_info['file_name']} ({file_info['failure_count']} failures)")
        if file_info['failure_reasons']:
            reasons_text = ", ".join([f"{reason}: {count}" for reason, count in sorted(file_info['failure_reasons'].items(), key=lambda x: x[1], reverse=True)])
            print(f"    Reasons: {reasons_text}")
    
    print("\n❌ FAILURE REASONS BREAKDOWN:")
    print("-" * 40)
    for reason_info in summary["failure_reasons_breakdown"]:
        print(f"• {reason_info['reason']}: {reason_info['count']} ({reason_info['percentage']}%)")
    
    if args.verbose:
        print("\n📋 COMPLETE VULNERABILITY FAILURE TOTALS (ACROSS ALL INPUTS):")
        print("-" * 60)
        for vuln_total in summary["all_vuln_failure_totals"]:
            print(f"\n🔸 {vuln_total['vuln_key']}")
            print(f"  Total failures across all inputs: {vuln_total['total_failures_across_all_inputs']}")
            print(f"  Appears in {vuln_total['appears_in_num_reports']} report file(s): {', '.join(vuln_total['source_report_files'])}")
            print(f"  Vulnerability: {vuln_total['vulnerability_id']}")
    
    if args.verbose:
        print("\n📋 DETAILED VULNERABILITY ANALYSIS:")
        print("-" * 50)
        for vuln_key, details in summary["detailed_vuln_analysis"].items():
            print(f"\n🔸 {vuln_key}")
            print(f"  Vulnerability: {details['vulnerability_id']}")
            print(f"  Total failures: {details['total_failures']}")
            print(f"  Files: {', '.join(details['files_affected'])}")
            print(f"  Reasons: {', '.join(details['failure_reasons'])}")
            print(f"  Source reports: {', '.join(details['source_reports'])}")


if __name__ == "__main__":
    main() 
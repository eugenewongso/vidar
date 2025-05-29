import json
import os
import argparse
from collections import defaultdict

def analyze_version_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    total_file_conflicts = 0
    files_skipped = 0
    files_attempted = 0
    successful_llm = 0
    run_counts = defaultdict(int)

    for entry in data:
        if 'failures' in entry:
            failures = entry.get('failures', [])
            num_runs = len(failures)

            for failure in failures:
                for file_conflict in failure.get('file_conflicts', []):
                    total_file_conflicts += 1

                    rej_content = file_conflict.get("rej_file_content")
                    original_source = file_conflict.get("downstream_file_content_patched_upstream_only")
                    target_filename = file_conflict.get("file_name")

                    if not all([target_filename, rej_content, original_source]):
                        files_skipped += 1
                        continue

                    files_attempted += 1
                    if file_conflict.get('llm_output_valid', False):
                        successful_llm += 1
                        run_counts[num_runs] += 1

    coverage = (successful_llm / files_attempted * 100) if files_attempted > 0 else 0

    return {
        'success_ratio': f"{successful_llm}/{files_attempted}",
        'coverage': f"{coverage:.2f}%",
        'run_1': f"{run_counts[1]}/{successful_llm} ({run_counts[1]/successful_llm*100:.1f}%)" if successful_llm > 0 else "0/0 (0%)",
        'run_2': f"{run_counts[2]}/{successful_llm} ({run_counts[2]/successful_llm*100:.1f}%)" if successful_llm > 0 else "0/0 (0%)",
        'run_3': f"{run_counts[3]}/{successful_llm} ({run_counts[3]/successful_llm*100:.1f}%)" if successful_llm > 0 else "0/0 (0%)",
        'total_conflicts': total_file_conflicts,
        'files_skipped': files_skipped,
        'files_attempted': files_attempted,
        'successful_llm': successful_llm
    }

def print_results(title, results_dict, versions, total_override=None):
    print(f"\n{title}")
    total_successful = 0
    total_attempted = 0

    for version in versions:
        if version in results_dict:
            r = results_dict[version]
            total_successful += r['successful_llm']
            total_attempted += r['files_attempted']
            print(f"Version {version}\t{r['success_ratio']}\t{r['coverage']}\t{r['run_1']}\t{r['run_2']}\t{r['run_3']}\t0 sec\t0 sec")

    if total_override:
        print(f"Total\t{total_successful}/{total_attempted}\t-\t-\t-\t-\t-\t-\t{total_override}")
    else:
        print(f"Total\t{total_successful}/{total_attempted}\t-\t-\t-\t-\t-\t-\t")

def main():
    parser = argparse.ArgumentParser(description="Analyze version-separated LLM outputs")
    parser.add_argument("--dir", default="version_separated", help="Directory containing the version files")
    parser.add_argument("--output", help="Path to save the analysis results (optional)")
    args = parser.parse_args()

    results = {
        'regular': {},              # With Guideline
        'less_error': {},           # Less Error Message + Guideline
        'no_guideline': {},         # No Guideline
        'less_error_no_guide': {}   # Less Error Message + No Guideline
    }

    for filename in sorted(os.listdir(args.dir)):
        if not filename.endswith('.json') or 'skipped' in filename:
            continue

        file_path = os.path.join(args.dir, filename)
        version = filename.split('_version_')[1].replace('.json', '')

        if 'no_guideline' in filename and 'less_error_message' in filename:
            results['less_error_no_guide'][version] = analyze_version_file(file_path)
        elif 'no_guideline' in filename:
            results['no_guideline'][version] = analyze_version_file(file_path)
        elif 'less_error_message' in filename:
            results['less_error'][version] = analyze_version_file(file_path)
        else:
            results['regular'][version] = analyze_version_file(file_path)

    print("\nAnalysis Results")
    print("Setting\tSuccess (≤3 runs)\tCoverage\t1 Run\t2 Runs\t3 Runs\tRuntime (All)\tRuntime (Success)\tTotal Success")

    all_versions = ['11', '12', '12L', '13', '14']

    print_results("With Guideline:", results['regular'], all_versions, total_override="69/106")
    print_results("Less Error Message + Guideline:", results['less_error'], all_versions, total_override="61/106")
    print_results("No Guideline:", results['no_guideline'], all_versions)
    print_results("Less Error Message + No Guideline:", results['less_error_no_guide'], all_versions)

    if args.output:
        with open(args.output, 'w') as f:
            f.write("Analysis Results\n")
            f.write("Setting\tSuccess (≤3 runs)\tCoverage\t1 Run\t2 Runs\t3 Runs\tRuntime (All)\tRuntime (Success)\tTotal Success\n")

            def write_result_section(label, group):
                f.write(f"\n{label}\n")
                for version in sorted(group.keys()):
                    r = group[version]
                    f.write(f"Version {version}\t{r['success_ratio']}\t{r['coverage']}\t{r['run_1']}\t{r['run_2']}\t{r['run_3']}\t0 sec\t0 sec\t{r['successful_llm']}/{r['files_attempted']}\n")

            write_result_section("With Guideline:", results['regular'])
            write_result_section("Less Error Message + Guideline:", results['less_error'])
            write_result_section("No Guideline:", results['no_guideline'])
            write_result_section("Less Error Message + No Guideline:", results['less_error_no_guide'])

        print(f"\n💾 Saved analysis to {args.output}")

if __name__ == "__main__":
    main()

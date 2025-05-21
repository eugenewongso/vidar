import unittest
import json
import os
import tempfile
import shutil
import sys
from unittest.mock import patch, mock_open

# Add the parent directory of 'post_eval' to sys.path to allow importing compare_llm_vs_ground_truth
# This assumes the test script is run from the 'vidar' directory or that PYTHONPATH is set up.
# For robust testing, especially in CI, managing PYTHONPATH or project structure is key.
# Here, we try a common approach for local testing.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) 
# If 'post_eval' is directly under 'vidar', PROJECT_ROOT is 'vidar'.
# If 'compare_llm_vs_ground_truth' needs 'metrics' from 'vidar/metrics',
# 'vidar' (PROJECT_ROOT) should be in sys.path.
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Now try to import the script to be tested
try:
    from post_eval.approach_1 import compare_llm_vs_ground_truth
except ImportError as e:
    print(f"Failed to import compare_llm_vs_ground_truth: {e}")
    print(f"Ensure that the script can be imported. Current sys.path: {sys.path}")
    # As a fallback for the test structure, define a dummy main if import fails
    # This allows the test structure to be written, but tests will fail if import is broken.
    class compare_llm_vs_ground_truth: # type: ignore
        @staticmethod
        def main():
            raise ImportError("compare_llm_vs_ground_truth.py could not be imported for testing.")

class TestCompareLLMvsGroundTruth(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        # Mock the output directory to be within our temp_dir
        self.mock_report_dir = os.path.join(self.test_dir, "gt_comparison_reports")
        os.makedirs(self.mock_report_dir, exist_ok=True)
        
        # Keep original os.path.join and os.makedirs to avoid issues with mocks
        self.original_os_path_join = os.path.join
        self.original_os_makedirs = os.makedirs

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def create_mock_input_file(self, data):
        fd, path = tempfile.mkstemp(suffix=".json", dir=self.test_dir)
        with os.fdopen(fd, 'w') as tmp:
            json.dump(data, tmp)
        return path

    @patch('builtins.print') # Mock print to suppress console output during tests
    def run_main_with_mock_args(self, mock_print, input_data_list, expected_json_filename_part, expected_csv_filename_part):
        mock_input_path = self.create_mock_input_file(input_data_list)
        
        # Mock os.path.join to redirect output to our temp directory
        def mock_os_path_join(base, *args):
            if base == "post_eval" and args and args[0] == "gt_comparison_reports":
                return self.original_os_path_join(self.mock_report_dir, *args[1:])
            return self.original_os_path_join(base, *args)

        # Mock os.makedirs to not fail if report_dir is self.mock_report_dir
        def mock_os_makedirs(name, mode=0o777, exist_ok=False):
            if name == self.mock_report_dir: # Allow creating our mocked report dir
                 return self.original_os_makedirs(name, mode=mode, exist_ok=True)
            return self.original_os_makedirs(name, mode=mode, exist_ok=exist_ok)

        with patch('sys.argv', ['compare_llm_vs_ground_truth.py', mock_input_path]), \
             patch('os.path.join', side_effect=mock_os_path_join), \
             patch('os.makedirs', side_effect=mock_os_makedirs):
            compare_llm_vs_ground_truth.main()

        # Check for output files
        json_output_files = [f for f in os.listdir(self.mock_report_dir) if expected_json_filename_part in f and f.endswith(".json")]
        csv_output_files = [f for f in os.listdir(self.mock_report_dir) if expected_csv_filename_part in f and f.endswith(".csv")]
        
        self.assertTrue(json_output_files, f"JSON report containing '{expected_json_filename_part}' not found.")
        self.assertTrue(csv_output_files, f"CSV report containing '{expected_csv_filename_part}' not found.")
        
        json_report_path = self.original_os_path_join(self.mock_report_dir, json_output_files[0])
        with open(json_report_path, 'r') as f:
            report_data = json.load(f)
        return report_data


    def test_valid_input_single_entry(self):
        valid_data = [{
            "id": "CVE-2023-001",
            "failures": [{
                "downstream_version": "14",
                "file_conflicts": [{
                    "file_name": "test.java",
                    "downstream_file_content_ground_truth": "public class A { void foo() { System.out.println(\"Hello\"); } }",
                    "downstream_patched_file_llm_output": "public class A { void foo() { System.out.println(\"World\"); } }"
                }]
            }]
        }]
        base_filename = os.path.splitext(os.path.basename(self.create_mock_input_file([])))[0] # get temp name part
        
        report_data = self.run_main_with_mock_args(valid_data, f"gt_comparison_metrics_{base_filename}", f"gt_comparison_metrics_{base_filename}")
        
        self.assertEqual(len(report_data), 1)
        self.assertEqual(report_data[0]["cve_id"], "CVE-2023-001")
        self.assertEqual(report_data[0]["file_name"], "test.java")
        self.assertEqual(report_data[0]["metrics_status"], "computed")
        self.assertIn("normalized_edit_similarity", report_data[0]["metrics"])

    def test_missing_ground_truth(self):
        data_missing_gt = [{
            "id": "CVE-2023-002",
            "failures": [{
                "downstream_version": "14",
                "file_conflicts": [{
                    "file_name": "test_missing_gt.java",
                    # "downstream_file_content_ground_truth": "...", # Missing
                    "downstream_patched_file_llm_output": "public class B {}"
                }]
            }]
        }]
        base_filename = os.path.splitext(os.path.basename(self.create_mock_input_file([])))[0]
        report_data = self.run_main_with_mock_args(data_missing_gt, f"gt_comparison_metrics_{base_filename}", f"gt_comparison_metrics_{base_filename}")
        self.assertEqual(len(report_data), 1)
        self.assertTrue(report_data[0]["metrics_status"].startswith("Metrics not computed:"))
        self.assertIn("'downstream_file_content_ground_truth' is missing", report_data[0]["metrics_status"])
        self.assertEqual(report_data[0]["metrics"], {})

    def test_llm_output_is_skip_message(self):
        data_llm_skipped = [{
            "id": "CVE-2023-003",
            "failures": [{
                "downstream_version": "14",
                "file_conflicts": [{
                    "file_name": "test_llm_skip.java",
                    "downstream_file_content_ground_truth": "public class C {}",
                    "downstream_patched_file_llm_output": "skipped, LLM error"
                }]
            }]
        }]
        base_filename = os.path.splitext(os.path.basename(self.create_mock_input_file([])))[0]
        report_data = self.run_main_with_mock_args(data_llm_skipped, f"gt_comparison_metrics_{base_filename}", f"gt_comparison_metrics_{base_filename}")
        self.assertEqual(len(report_data), 1)
        self.assertTrue(report_data[0]["metrics_status"].startswith("Metrics not computed:"))
        self.assertIn("LLM output was 'skipped, LLM error'", report_data[0]["metrics_status"])
        self.assertEqual(report_data[0]["llm_patched_codebase"], "skipped, LLM error") # Ensure skip message is preserved
        self.assertEqual(report_data[0]["metrics"], {})

    def test_malformed_json_file(self):
        malformed_json_path = self.create_mock_input_file("this is not json")
        os.remove(malformed_json_path) # remove the valid temp file
        with open(malformed_json_path, 'w') as f: # recreate with bad content
            f.write("this is not json {")

        with patch('sys.argv', ['compare_llm_vs_ground_truth.py', malformed_json_path]), \
             patch('builtins.print') as captured_print:
            compare_llm_vs_ground_truth.main()
            
            error_printed = False
            for call_args in captured_print.call_args_list:
                if "Error: Could not decode JSON" in call_args[0][0]:
                    error_printed = True
                    break
            self.assertTrue(error_printed, "Error message for malformed JSON not printed.")

    def test_input_file_not_found(self):
        non_existent_path = os.path.join(self.test_dir, "non_existent_file.json")
        with patch('sys.argv', ['compare_llm_vs_ground_truth.py', non_existent_path]), \
             patch('builtins.print') as captured_print:
            compare_llm_vs_ground_truth.main()

            error_printed = False
            for call_args in captured_print.call_args_list:
                if "Error: Input file not found" in call_args[0][0]:
                    error_printed = True
                    break
            self.assertTrue(error_printed, "Error message for file not found not printed.")


if __name__ == '__main__':
    # This allows running the tests directly from the command line
    # For example: python post_eval/test_compare_llm_vs_ground_truth.py
    # Ensure that 'vidar' directory is in PYTHONPATH or run from 'vidar' directory.
    unittest.main()

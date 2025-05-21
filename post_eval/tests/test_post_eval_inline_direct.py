import unittest
import os
import json
import csv
from unittest.mock import patch, mock_open, MagicMock

# Assuming post_eval_inline_direct.py is in the same directory or accessible via PYTHONPATH
from post_eval_inline_direct import (
    count_tokens,
    clean_code,
    get_language_from_filename,
    compute_metrics,
    main,
    MAX_TOKENS
)

class TestHelperFunctions(unittest.TestCase):

    def test_count_tokens(self):
        self.assertEqual(count_tokens(""), 0)
        self.assertEqual(count_tokens("hello world"), 2)
        self.assertEqual(count_tokens("hello"), 1)
        self.assertEqual(count_tokens("  hello   world  "), 2) # split handles multiple spaces
        self.assertEqual(count_tokens("word1 word2 word3"), 3)

    def test_clean_code(self):
        self.assertEqual(clean_code("```python\nprint('hello')\n```"), "python\nprint('hello')")
        self.assertEqual(clean_code("```\nprint('hello')\n```"), "print('hello')")
        self.assertEqual(clean_code("print('hello')"), "print('hello')")
        self.assertEqual(clean_code("```javascript\nconsole.log('test');"), "javascript\nconsole.log('test');") # No trailing ```
        self.assertEqual(clean_code("console.log('test');\n```"), "console.log('test');") # No leading ```
        self.assertEqual(clean_code("  leading and trailing spaces  "), "leading and trailing spaces")
        self.assertEqual(clean_code("```\n  content with spaces  \n```"), "content with spaces")
        self.assertEqual(clean_code("```markdown\n# Title\n```"), "markdown\n# Title")
        self.assertEqual(clean_code("no backticks"), "no backticks")
        self.assertEqual(clean_code("  ```\n  code\n  ```  "), "code") # with surrounding spaces

    def test_get_language_from_filename(self):
        self.assertEqual(get_language_from_filename("script.py"), "python")
        self.assertEqual(get_language_from_filename("main.java"), "java")
        self.assertEqual(get_language_from_filename("app.js"), "javascript")
        self.assertEqual(get_language_from_filename("program.c"), "c")
        self.assertEqual(get_language_from_filename("module.cpp"), "cpp")
        self.assertEqual(get_language_from_filename("module.cc"), "cpp")
        self.assertEqual(get_language_from_filename("module.cxx"), "cpp")
        self.assertEqual(get_language_from_filename("header.h"), "cpp")
        self.assertEqual(get_language_from_filename("header.hpp"), "cpp")
        self.assertEqual(get_language_from_filename("code.cs"), "csharp")
        self.assertEqual(get_language_from_filename("unknown.txt"), "txt")
        self.assertEqual(get_language_from_filename("archive.tar.gz"), "gz")
        self.assertEqual(get_language_from_filename("Makefile"), "makefile") # No dot, returns full name
        self.assertEqual(get_language_from_filename("image.JPEG"), "jpeg") # Case insensitivity
        self.assertEqual(get_language_from_filename("script.PY"), "python")


class TestComputeMetrics(unittest.TestCase):
    @patch('post_eval.post_eval_inline_direct.relative_line_count_similarity')
    @patch('post_eval.post_eval_inline_direct.normalized_edit_similarity')
    @patch('post_eval.post_eval_inline_direct.token_level_edit_distance')
    @patch('post_eval.post_eval_inline_direct.compute_codebert_score')
    @patch('post_eval.post_eval_inline_direct.compute_cosine_openai_embedding')
    @patch('post_eval.post_eval_inline_direct.count_tokens')
    def test_compute_metrics_normal_case(
        self, mock_count_tokens, mock_openai, mock_codebert,
        mock_tled, mock_nes, mock_rlcs
    ):
        mock_rlcs.return_value = 0.8
        mock_nes.return_value = 0.7
        mock_tled.return_value = 10
        mock_codebert.return_value = {"precision": 0.9, "recall": 0.85, "f1": 0.87}
        mock_openai.return_value = 0.95
        mock_count_tokens.side_effect = [50, 60] # upstream, downstream

        upstream_code = "upstream code"
        downstream_code = "downstream code"
        file_name = "test.py"

        metrics = compute_metrics(upstream_code, downstream_code, file_name)

        self.assertEqual(metrics["relative_line_count_similarity"], 0.8)
        self.assertEqual(metrics["normalized_edit_similarity"], 0.7)
        self.assertEqual(metrics["token_level_edit_distance"], 10)
        self.assertEqual(metrics["codebert_score"]["precision"], 0.9)
        self.assertEqual(metrics["cosine_similarity_openai"], 0.95)
        self.assertEqual(metrics["token_count_upstream"], 50)
        self.assertEqual(metrics["token_count_downstream"], 60)
        self.assertEqual(metrics["token_count_total"], 110)

        mock_rlcs.assert_called_once_with(downstream_code, upstream_code)
        mock_nes.assert_called_once_with(downstream_code, upstream_code)
        mock_tled.assert_called_once_with(downstream_code, upstream_code)
        mock_codebert.assert_called_once_with(downstream_code, upstream_code, "python")
        mock_openai.assert_called_once_with(upstream_code, downstream_code)
        mock_count_tokens.assert_any_call(upstream_code)
        mock_count_tokens.assert_any_call(downstream_code)

    @patch('post_eval.post_eval_inline_direct.relative_line_count_similarity')
    @patch('post_eval.post_eval_inline_direct.normalized_edit_similarity')
    @patch('post_eval.post_eval_inline_direct.token_level_edit_distance')
    @patch('post_eval.post_eval_inline_direct.compute_codebert_score')
    @patch('post_eval.post_eval_inline_direct.compute_cosine_openai_embedding')
    @patch('post_eval.post_eval_inline_direct.count_tokens')
    def test_compute_metrics_codebert_error(
        self, mock_count_tokens, mock_openai, mock_codebert,
        mock_tled, mock_nes, mock_rlcs
    ):
        mock_rlcs.return_value = 0.8
        mock_nes.return_value = 0.7
        mock_tled.return_value = 10
        mock_codebert.return_value = {"error": "CodeBERT failed"}
        mock_openai.return_value = 0.95
        mock_count_tokens.side_effect = [50, 60]

        metrics = compute_metrics("up", "down", "test.java")
        self.assertEqual(metrics["codebert_score"], {"error": "CodeBERT failed"})
        mock_codebert.assert_called_once_with("down", "up", "java")


    @patch('post_eval.post_eval_inline_direct.relative_line_count_similarity')
    @patch('post_eval.post_eval_inline_direct.normalized_edit_similarity')
    @patch('post_eval.post_eval_inline_direct.token_level_edit_distance')
    @patch('post_eval.post_eval_inline_direct.compute_codebert_score')
    @patch('post_eval.post_eval_inline_direct.compute_cosine_openai_embedding')
    @patch('post_eval.post_eval_inline_direct.count_tokens')
    def test_compute_metrics_openai_skipped_due_to_tokens(
        self, mock_count_tokens, mock_openai, mock_codebert,
        mock_tled, mock_nes, mock_rlcs
    ):
        mock_rlcs.return_value = 0.8
        mock_nes.return_value = 0.7
        mock_tled.return_value = 10
        mock_codebert.return_value = {"f1": 0.87}
        # Simulate token count exceeding MAX_TOKENS
        mock_count_tokens.side_effect = [MAX_TOKENS // 2, MAX_TOKENS // 2 + 1]


        metrics = compute_metrics("up_long", "down_long", "test.js")
        self.assertEqual(metrics["cosine_similarity_openai"], "skipped")
        self.assertEqual(metrics["token_count_total"], MAX_TOKENS + 1)
        mock_openai.assert_not_called()
        mock_codebert.assert_called_once_with("down_long", "up_long", "javascript")

    @patch('post_eval.post_eval_inline_direct.relative_line_count_similarity', MagicMock(return_value="N/A"))
    @patch('post_eval.post_eval_inline_direct.normalized_edit_similarity', MagicMock(return_value="Error"))
    @patch('post_eval.post_eval_inline_direct.token_level_edit_distance', MagicMock(return_value=None))
    @patch('post_eval.post_eval_inline_direct.compute_codebert_score', MagicMock(return_value={"f1": 0.5}))
    @patch('post_eval.post_eval_inline_direct.compute_cosine_openai_embedding', MagicMock(return_value=0.12345))
    @patch('post_eval.post_eval_inline_direct.count_tokens', MagicMock(side_effect=[0,0]))
    def test_compute_metrics_non_float_results(self):
        metrics = compute_metrics("", "", "test.c")
        self.assertEqual(metrics["relative_line_count_similarity"], "N/A")
        self.assertEqual(metrics["normalized_edit_similarity"], "Error")
        self.assertEqual(metrics["token_level_edit_distance"], None) # Not rounded
        self.assertEqual(metrics["codebert_score"]["f1"], 0.5) # Rounded
        self.assertEqual(metrics["cosine_similarity_openai"], 0.1235) # Rounded


class TestMainFunction(unittest.TestCase):

    @patch('argparse.ArgumentParser')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('json.dump')
    @patch('csv.DictWriter')
    @patch('os.makedirs')
    @patch('post_eval.post_eval_inline_direct.compute_metrics')
    @patch('post_eval.post_eval_inline_direct.datetime')
    @patch('post_eval.post_eval_inline_direct.tqdm') # Mock tqdm
    def test_main_normal_flow(
        self, mock_tqdm, mock_datetime, mock_compute_metrics, mock_makedirs,
        mock_csv_writer, mock_json_dump, mock_json_load, mock_file_open, mock_arg_parser
    ):
        # --- Setup Mocks ---
        # Argparse
        mock_args = MagicMock()
        mock_args.json_input = "dummy_input.json"
        mock_arg_parser.return_value.parse_args.return_value = mock_args

        # Datetime
        mock_dt_obj = MagicMock()
        mock_dt_obj.strftime.return_value = "20230101_120000"
        mock_datetime.now.return_value = mock_dt_obj

        # JSON Load
        mock_input_data = [
            {
                "id": "CVE-2023-0001",
                "failures": [
                    {
                        "downstream_version": "v1.0",
                        "file_conflicts": [
                            {
                                "file_name": "file1.py",
                                "upstream_file_content": "upstream content 1",
                                "downstream_file_content": "downstream content 1"
                            },
                            {
                                "file_name": "file2.java",
                                "upstream_file_content": "```java\nupstream 2\n```",
                                "downstream_file_content": "downstream 2"
                            }
                        ]
                    }
                ]
            }
        ]
        mock_json_load.return_value = mock_input_data

        # compute_metrics
        mock_metrics_result_1 = {"metric1": 0.5, "token_count_total": 20, "codebert_score": {"f1": 0.6}}
        mock_metrics_result_2 = {"metric1": 0.7, "token_count_total": 30, "codebert_score": {"f1": 0.8}}
        mock_compute_metrics.side_effect = [mock_metrics_result_1, mock_metrics_result_2]

        # tqdm
        mock_tqdm_instance = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_tqdm_instance


        # --- Execute main ---
        main()

        # --- Assertions ---
        # Argparse called
        mock_arg_parser.assert_called_once()
        mock_arg_parser.return_value.parse_args.assert_called_once()

        # os.makedirs called for results directory
        expected_results_json_dir = os.path.dirname("results/summary_20230101_120000.json")
        expected_results_csv_dir = os.path.dirname("results/summary_20230101_120000.csv")
        mock_makedirs.assert_any_call(expected_results_json_dir, exist_ok=True)
        mock_makedirs.assert_any_call(expected_results_csv_dir, exist_ok=True)


        # Input file opened
        mock_file_open.assert_any_call("dummy_input.json", "r")
        mock_json_load.assert_called_once()

        # compute_metrics calls
        self.assertEqual(mock_compute_metrics.call_count, 2)
        mock_compute_metrics.assert_any_call("upstream content 1", "downstream content 1", "file1.py")
        mock_compute_metrics.assert_any_call("java\nupstream 2", "downstream content 1", "file2.java") # clean_code applied

        # tqdm calls
        mock_tqdm.assert_called_once_with(total=2, desc="Evaluating file conflicts")
        self.assertEqual(mock_tqdm_instance.update.call_count, 2)


        # Output files opened (JSON and CSV)
        # Check that open was called for writing the JSON and CSV files
        # The exact number of calls to open can be tricky due to multiple writes,
        # so we check for specific calls.
        expected_json_path = "results/summary_20230101_120000.json"
        expected_csv_path = "results/summary_20230101_120000.csv"

        # JSON dump calls
        # json.dump is called inside the loop for each result
        self.assertEqual(mock_json_dump.call_count, 2)
        # Check the content of the first call to json.dump
        args_list_json_dump = mock_json_dump.call_args_list
        
        # First call to json.dump
        first_dump_args = args_list_json_dump[0][0]
        self.assertEqual(len(first_dump_args[0]), 1) # all_results has 1 item
        self.assertEqual(first_dump_args[0][0]["cve_id"], "CVE-2023-0001")
        self.assertEqual(first_dump_args[0][0]["file_name"], "file1.py")
        self.assertEqual(first_dump_args[0][0]["metrics"], mock_metrics_result_1)

        # Second call to json.dump
        second_dump_args = args_list_json_dump[1][0]
        self.assertEqual(len(second_dump_args[0]), 2) # all_results has 2 items
        self.assertEqual(second_dump_args[0][1]["cve_id"], "CVE-2023-0001")
        self.assertEqual(second_dump_args[0][1]["file_name"], "file2.java")
        self.assertEqual(second_dump_args[0][1]["metrics"], mock_metrics_result_2)


        # CSV DictWriter calls
        # writer.writeheader() should be called once.
        # writer.writerow() should be called for each conflict.
        mock_csv_writer_instance = mock_csv_writer.return_value
        self.assertEqual(mock_csv_writer_instance.writeheader.call_count, 1) # Assuming header_written starts False
        self.assertEqual(mock_csv_writer_instance.writerow.call_count, 2)

        # Check one of the rowrite calls
        first_row_call_args = mock_csv_writer_instance.writerow.call_args_list[0][0][0]
        self.assertEqual(first_row_call_args["cve_id"], "CVE-2023-0001")
        self.assertEqual(first_row_call_args["file_name"], "file1.py")
        self.assertEqual(first_row_call_args["metric1"], 0.5) # From mock_metrics_result_1
        self.assertEqual(first_row_call_args["codebert_f1"], 0.6)


    @patch('argparse.ArgumentParser')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('os.makedirs')
    @patch('post_eval.post_eval_inline_direct.datetime')
    @patch('post_eval.post_eval_inline_direct.tqdm')
    def test_main_no_failures_in_vuln(
        self, mock_tqdm, mock_datetime, mock_makedirs, mock_json_load, mock_file_open, mock_arg_parser
    ):
        mock_args = MagicMock()
        mock_args.json_input = "dummy.json"
        mock_arg_parser.return_value.parse_args.return_value = mock_args
        mock_datetime.now.return_value.strftime.return_value = "timestamp"

        mock_input_data = [{"id": "CVE-001", "failures": []}] # No failures
        mock_json_load.return_value = mock_input_data
        
        mock_tqdm_instance = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_tqdm_instance

        with patch('builtins.print') as mock_print:
            main()
            mock_print.assert_any_call("⚠️ Skipping CVE-001 — no failures listed.")
            mock_tqdm.assert_called_once_with(total=0, desc="Evaluating file conflicts") # total_conflicts is 0
            mock_tqdm_instance.update.assert_not_called()


    @patch('argparse.ArgumentParser')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('os.makedirs')
    @patch('post_eval.post_eval_inline_direct.datetime')
    @patch('post_eval.post_eval_inline_direct.tqdm')
    def test_main_no_file_conflicts_in_failure(
        self, mock_tqdm, mock_datetime, mock_makedirs, mock_json_load, mock_file_open, mock_arg_parser
    ):
        mock_args = MagicMock()
        mock_args.json_input = "dummy.json"
        mock_arg_parser.return_value.parse_args.return_value = mock_args
        mock_datetime.now.return_value.strftime.return_value = "timestamp"

        mock_input_data = [{
            "id": "CVE-002",
            "failures": [{"downstream_version": "v1", "file_conflicts": []}] # No file_conflicts
        }]
        mock_json_load.return_value = mock_input_data

        mock_tqdm_instance = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_tqdm_instance

        with patch('builtins.print') as mock_print:
            main()
            mock_print.assert_any_call("⚠️ Skipping CVE-002 — no file_conflicts for v1.")
            mock_tqdm.assert_called_once_with(total=0, desc="Evaluating file conflicts")
            mock_tqdm_instance.update.assert_not_called()

    @patch('argparse.ArgumentParser')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('json.dump') # Mock to prevent actual file writing
    @patch('csv.DictWriter') # Mock to prevent actual file writing
    @patch('os.makedirs')
    @patch('post_eval.post_eval_inline_direct.compute_metrics')
    @patch('post_eval.post_eval_inline_direct.datetime')
    @patch('post_eval.post_eval_inline_direct.tqdm')
    def test_main_missing_content_in_conflict(
        self, mock_tqdm, mock_datetime, mock_compute_metrics, mock_makedirs,
        mock_csv_writer, mock_json_dump, mock_json_load, mock_file_open, mock_arg_parser
    ):
        mock_args = MagicMock()
        mock_args.json_input = "dummy.json"
        mock_arg_parser.return_value.parse_args.return_value = mock_args
        mock_datetime.now.return_value.strftime.return_value = "timestamp"

        mock_input_data = [{
            "id": "CVE-003",
            "failures": [{
                "downstream_version": "v2",
                "file_conflicts": [{
                    "file_name": "test.py"
                    # upstream_file_content and downstream_file_content are missing
                }]
            }]
        }]
        mock_json_load.return_value = mock_input_data
        mock_compute_metrics.return_value = {"metric": 0.1, "token_count_total": 5} # Dummy metrics

        mock_tqdm_instance = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_tqdm_instance

        main()
        # clean_code("") results in "", so compute_metrics is called with empty strings
        mock_compute_metrics.assert_called_once_with("", "", "test.py")
        mock_tqdm.assert_called_once_with(total=1, desc="Evaluating file conflicts")
        mock_tqdm_instance.update.assert_called_once_with(1)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

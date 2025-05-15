# Vanir Report Parser

This module parses security vulnerability reports from Vanir into a structured format for further processing.

## Files
- `vanir_report_parser.py`: Main parser for Vanir security reports

## Usage


## Input
- `reports/vanir_output.json`: Raw Vanir vulnerability report

## Output
- `reports/parsed_report.json`: Structured data with patch information

## How It Works
1. Loads the Vanir report from JSON
2. Extracts patch URLs and affected files/functions
3. Organizes data by patch with file/function mappings
4. Saves the structured data for use by other modules

## Class: VanirParser
- `__init__(file_path, output_path)`: Initialize parser with input/output paths
- `load_vanir_report()`: Loads raw JSON file
- `extract_patch_hash(patch_url)`: Extracts commit hash from URL
- `parse_vanir_report()`: Restructures data for easier processing
- `write_output_to_json()`: Saves processed data to JSON
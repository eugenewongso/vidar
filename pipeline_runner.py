r"""Main pipeline orchestrator for the Vanir Patch Management System.

This script serves as the master runner for the entire patch processing workflow.
It executes a series of modules in a predefined sequence to parse vulnerability
reports, fetch required patches, attempt to apply them, and use an LLM to
correct any patches that fail.

The pipeline is designed to be run from the root of the `vidar` directory.

Usage:
  python pipeline_runner.py
"""

import argparse
import json
import os
import subprocess
import sys
import yaml
import logging
import click
import asyncio
from rich.console import Console
from rich.spinner import Spinner
from rich.progress import (
    Progress,
    BarColumn,
    MofNCompleteColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from pathlib import Path

# --- Vidar Scripts ---
from patch_adopter import run_adoption_step
from patch_fetcher import run_fetcher_step
from llm_patch_runner import run_llm_correction_step

# --- Global Objects ---
console = Console()

def setup_logging(verbose=False):
    """Sets up a configured logger for the application."""
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_file_path = os.path.join(logs_dir, 'vidar_pipeline.log')

    # Determine the console log level
    console_log_level = logging.INFO if verbose else logging.WARNING

    # Use basicConfig with force=True. This is the key to the solution.
    # It removes any existing handlers (which cause the cluttered console)
    # and sets up our desired file handler with the correct format for ALL modules.
    logging.basicConfig(
        force=True,  # This is essential.
        level=logging.DEBUG,
        format='%(asctime)s - %(name)-25s - %(levelname)-8s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w')
        ]
    )

    # Now, add a second handler ONLY for the console with a simple format.
    # The progress bar silencing logic will target this specific handler.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_log_level)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger().addHandler(console_handler)
    
    logging.info("Vidar pipeline logging initiated.")

def run_command(command, cwd=None):
    """Runs a command, redirecting its output to the main log file."""
    try:
        # Using PIPE to capture output and log it
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding='utf-8'
        )
        
        # Log command output line by line at DEBUG level
        if process.stdout:
            for line in process.stdout:
                logging.debug(line.strip())

        process.wait()

        if process.returncode != 0:
            logging.error(f"Command failed with exit code {process.returncode}")
            # The error will be visible on console if level is INFO/ERROR
            console.print(f"[bold red]Error:[/bold red] Command failed. See {os.path.join('logs', 'vidar_pipeline.log')} for details.")
            sys.exit(process.returncode)
            
    except FileNotFoundError:
        logging.error(f"Command not found. Is '{command[0]}' in your PATH?")
        console.print(f"[bold red]Error:[/bold red] Command '{command[0]}' not found.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        console.print(f"[bold red]Error:[/bold red] An unexpected error occurred. See log for details.")
        sys.exit(1)

class PipelineRunner:
    """Orchestrates the entire patch processing pipeline from start to finish.

    This class manages the sequence of operations, handles the flow of data
    between scripts, and provides descriptive output for each stage.
    """

    def __init__(self, config_path: str):
        """Initializes the PipelineRunner, setting up necessary paths."""
        self.vidar_dir = Path(__file__).resolve().parent
        self.python_executable = sys.executable
        
        # Load configuration
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            console.print(f"[bold red]Error:[/bold red] Configuration file not found at '{config_path}'")
            sys.exit(1)
        except yaml.YAMLError as e:
            console.print(f"[bold red]Error:[/bold red] Error parsing YAML configuration file: {e}")
            sys.exit(1)
            
        self.paths_config = self.config.get("paths", {})

    def run_spinner_step(self, title, command):
        """Prints a descriptive header and runs a pipeline step with a spinner."""
        with console.status(f"[bold green]{title}", spinner="dots"):
            # Detailed logging goes to the file
            logging.info(f"STARTING STEP: {title}")
            logging.info(f"Command: {' '.join(command)}")
            run_command(command, cwd=self.vidar_dir)
            logging.info(f"COMPLETED STEP: {title}")
        console.print(f"✅ {title}")
        
    def run_progress_step(self, title: str, step_generator, *args, **kwargs):
        """Runs a pipeline step that yields progress and displays a progress bar."""
        console_handler = next((h for h in logging.getLogger().handlers if isinstance(h, logging.StreamHandler)), None)
        original_formatter = None
        if console_handler:
            original_formatter = console_handler.formatter
            console_handler.setFormatter(logging.Formatter(''))

        step_iterator = step_generator(*args, **kwargs)
        summary_data = None
        total_items = 0
        
        progress_columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        ]
        
        try:
            with Progress(*progress_columns, console=console) as progress:
                task_id = None
                for update in step_iterator:
                    if update.get('type') == 'error':
                        if console_handler and original_formatter is not None:
                            console_handler.setFormatter(original_formatter)
                        console.print(f"[bold red]Error in '{title}':[/bold red] {update['message']}")
                        sys.exit(1)
                    
                    if update.get('type') == 'progress':
                        total = update.get('total', 0)
                        total_items = total
                        completed = update.get('completed', 0)
                        
                        if total == 0: # Handle case with no items to process
                            if task_id is None:
                               task_id = progress.add_task(f"[green]{title}", total=1)
                            progress.update(task_id, completed=1)
                            break

                        if task_id is None:
                            task_id = progress.add_task(f"[green]{title}", total=total)
                        
                        progress.update(task_id, completed=completed)
                    
                    elif update.get('type') == 'summary':
                        summary_data = update.get('data')
                        total_items += 1
        finally:
            if console_handler and original_formatter is not None:
                console_handler.setFormatter(original_formatter)
        
        if summary_data:
            failures = 0
            if 'failed' in summary_data and isinstance(summary_data['failed'], list):
                failures = len(summary_data['failed'])
            elif 'failed_patches' in summary_data:
                failures = summary_data['failed_patches']

            if failures > 0:
                console.print(f"⚠️  {title} completed with {failures} failures out of {total_items}. See log for details.")
            else:
                console.print(f"✅ {title}")
        else:
            console.print(f"✅ {title}")

    async def run_async_progress_step(self, title: str, step_generator, *args, **kwargs):
        """Runs an async pipeline step that yields progress and displays a progress bar."""
        console_handler = next((h for h in logging.getLogger().handlers if isinstance(h, logging.StreamHandler)), None)
        original_formatter = None
        if console_handler:
            original_formatter = console_handler.formatter
            console_handler.setFormatter(logging.Formatter(''))
            
        step_iterator = step_generator(*args, **kwargs)
        summary_data = None
        total_items = 0

        progress_columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        ]

        try:
            with Progress(*progress_columns, console=console) as progress:
                task_id = None
                async for update in step_iterator:
                    if update.get('type') == 'error':
                        # Restore logging to show the critical error
                        if console_handler and original_formatter is not None:
                            console_handler.setFormatter(original_formatter)
                        console.print(f"[bold red]Error in '{title}':[/bold red] {update['message']}")
                        sys.exit(1)
                    
                    if update.get('type') == 'progress':
                        total = update.get('total', 0)
                        total_items = total
                        completed = update.get('completed', 0)
                        
                        if total == 0:
                            if task_id is None:
                               task_id = progress.add_task(f"[green]{title}", total=1)
                            progress.update(task_id, completed=1)
                            break

                        if task_id is None:
                            task_id = progress.add_task(f"[green]{title}", total=total)
                        
                        progress.update(task_id, completed=completed)

                    elif update.get('type') == 'summary':
                        summary_data = update.get('data')
                        total_items += 1
        finally:
            if console_handler and original_formatter is not None:
                console_handler.setFormatter(original_formatter)

        if summary_data:
            failures = 0
            if 'failed' in summary_data and isinstance(summary_data['failed'], list):
                failures = len(summary_data['failed'])
            elif 'failed_patches' in summary_data:
                failures = summary_data['failed_patches']
            
            if failures > 0:
                console.print(f"⚠️  {title} completed with {failures} failures out of {total_items}. See log for details.")
            else:
                console.print(f"✅ {title}")
        else:
            console.print(f"✅ {title}")

    def prepare_llm_input(self) -> bool:
        """
        Filters the application report to prepare the input for the LLM.
        This step is silent on the console, logging details to the file.
        """
        title = "Preparing input for LLM"
        
        # Silence console logging for this internal step
        console_handler = next((h for h in logging.getLogger().handlers if isinstance(h, logging.StreamHandler)), None)
        original_formatter = None
        if console_handler:
            original_formatter = console_handler.formatter
            console_handler.setFormatter(logging.Formatter(''))

        skipped_count = 0
        failed_patches = []
        
        try:
            with console.status(f"[bold green]{title}", spinner="dots"):
                logging.info(f"STARTING STEP: {title}")
                
                report_path = self.vidar_dir / self.paths_config.get("vanir_patch_application_report")
                llm_input_path = self.vidar_dir / self.paths_config.get("llm_input_report")
                fetched_patches_dir = self.vidar_dir / self.paths_config.get("fetched_patches_dir")
                
                os.makedirs(os.path.dirname(llm_input_path), exist_ok=True)

                try:
                    with open(report_path, 'r', encoding='utf-8') as f:
                        adoption_report = json.load(f)
                except FileNotFoundError:
                    logging.error(f"Cannot find patch application report at '{report_path}'.")
                    # This is a critical error, so we will show it on the console.
                    if console_handler and original_formatter is not None:
                        console_handler.setFormatter(original_formatter)
                    console.print(f"[bold red]Error:[/bold red] Cannot find patch application report at '{report_path}'.")
                    sys.exit(1)

                for patch in adoption_report.get('patches', []):
                    if (
                        patch.get('status') == 'Rejected'
                        or 'Failed Hunks' in patch.get('detailed_status', '')
                    ):
                        new_patch_entry = patch.copy()
                        original_diff_path = os.path.join(fetched_patches_dir, patch['patch_file'])
                        if os.path.exists(original_diff_path):
                            with open(original_diff_path, 'r', encoding='utf-8') as f:
                                new_patch_entry['upstream_patch_content'] = f.read()
                        else:
                            logging.info(f"Could not find original diff file: {original_diff_path}. Skipping patch for LLM.")
                            skipped_count += 1
                            continue
                        failed_patches.append(new_patch_entry)
                
                with open(llm_input_path, 'w', encoding='utf-8') as f:
                    json.dump({"patches": failed_patches}, f, indent=4)
                
                logging.info(f"Created '{os.path.basename(llm_input_path)}' with {len(failed_patches)} failed patches.")
                logging.info(f"COMPLETED STEP: {title}")
        finally:
            if console_handler and original_formatter is not None:
                console_handler.setFormatter(original_formatter)
        
        if not failed_patches:
            console.print("✅ Preparing input for LLM (No failed patches to process)")
            return False
            
        summary_message = f"✅ {title} ({len(failed_patches)} patches need correction"
        if skipped_count > 0:
            summary_message += f", {skipped_count} skipped"
        summary_message += ")"
        console.print(summary_message)
        
        return True

    def run_pipeline(self):
        """Executes the full pipeline in the correct sequence."""
        console.print("[bold cyan]Starting Vidar Security Patch Processing Pipeline...[/bold cyan]")

        self.run_spinner_step(
            title="1. Parsing Vanir Report",
            command=[self.python_executable, os.path.join(self.vidar_dir, 'vanir_report_parser.py')]
        )
        
        self.run_progress_step(
            title="2. Fetching Patches",
            step_generator=run_fetcher_step
        )

        self.run_progress_step(
            title="3. Applying Original Patches",
            step_generator=run_adoption_step,
            source='Vanir',
            target_source_path=os.getenv("TARGET_SOURCE_PATH")
        )

        if self.prepare_llm_input():
            asyncio.run(self.run_async_progress_step(
                title="4. Running LLM Patch Correction",
                step_generator=run_llm_correction_step
            ))
            
            self.run_progress_step(
                title="5. Applying LLM-Generated Patches",
                step_generator=run_adoption_step,
                source='LLM',
                target_source_path=os.getenv("TARGET_SOURCE_PATH")
            )

        self.generate_final_summary()

        console.print("\n[bold green]🎉🎉🎉 Pipeline completed successfully! 🎉🎉🎉[/bold green]")
        console.print(f"📄 Full details logged to: [cyan]{os.path.join('logs', 'vidar_pipeline.log')}[/cyan]")

    def generate_final_summary(self):
        """
        Reads all relevant reports and generates a single, comprehensive summary.
        """
        title = "6. Generating Final Summary Report"
        with console.status(f"[bold green]{title}", spinner="dots"):
            logging.info(f"STARTING STEP: {title}")

            vidar_dir = Path(self.vidar_dir)
            parsed_report_path = vidar_dir / self.paths_config.get("parsed_vanir_report")
            fetch_failures_path = vidar_dir / self.paths_config.get("fetch_failures_report")
            initial_apply_report_path = vidar_dir / self.paths_config.get("vanir_patch_application_report")
            failed_patch_path = vidar_dir / self.paths_config.get("llm_input_report")
            llm_apply_report_path = vidar_dir / self.paths_config.get("llm_patch_application_report")
            llm_detailed_report_path = vidar_dir / self.paths_config.get("llm_detailed_output_report")
            final_summary_path = vidar_dir / self.paths_config.get("final_summary_report")

            # Initialize data structures
            summary = {}
            analysis = {
                "fetch_failures": {"total": 0, "failed_urls": []},
                "upstream_patch_application_failures": {"total": 0, "breakdown_by_error": {}, "failed_patches": []},
                "llm_processing_failures": {"total": 0, "breakdown_by_error": {}, "failed_patches": []}
            }

            # Helper to read JSON files safely
            def _read_json(path, default=None):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    logging.warning(f"Could not read or parse JSON file at '{path}'. Using default value.")
                    return default if default is not None else {}

            # 1. Get total patches from Vanir
            parsed_report = _read_json(parsed_report_path)
            total_from_vanir = len(parsed_report.get('patches', []))

            # 2. Process Fetch Failures
            fetch_failures = _read_json(fetch_failures_path, default=[])
            analysis["fetch_failures"]["total"] = len(fetch_failures)
            analysis["fetch_failures"]["failed_urls"] = fetch_failures

            # 3. Process Upstream Patch Application Report
            initial_apply_report = _read_json(initial_apply_report_path)
            patches_for_application = len(initial_apply_report.get("patches", []))
            
            directly_applied = 0
            for result in initial_apply_report.get("patches", []):
                if result.get("detailed_status") and "Applied Successfully" in result.get("detailed_status"):
                    directly_applied += 1

            failed_patch_report = _read_json(failed_patch_path)
            patches_forwarded_to_llm = len(failed_patch_report.get('patches', []))
            
            # Populate detailed upstream failures
            upstream_failures = analysis["upstream_patch_application_failures"]
            upstream_failures["total"] = patches_forwarded_to_llm
            for patch in failed_patch_report.get("patches", []):
                error_type = patch.get("detailed_status", "Unknown Error")
                upstream_failures["breakdown_by_error"][error_type] = upstream_failures["breakdown_by_error"].get(error_type, 0) + 1
                upstream_failures["failed_patches"].append({
                    "patch_file": patch.get("patch_file"),
                    "project": patch.get("project"),
                    "error_type": error_type,
                    "error_message": patch.get("message_output", "").strip()
                })

            # 4. Process LLM Results
            llm_applied = 0
            llm_application_failures = 0
            llm_report = _read_json(llm_apply_report_path)
            llm_processed_patches = llm_report.get("patches", [])
            
            llm_failures = analysis["llm_processing_failures"]
            
            # Get LLM application failures
            for result in llm_processed_patches:
                if result.get("detailed_status") and "Applied Successfully" in result.get("detailed_status"):
                    llm_applied += 1
                elif result.get("status") == "Rejected":
                    llm_application_failures += 1
                    error_type = "LLM_patch_rejected"
                    llm_failures["breakdown_by_error"][error_type] = llm_failures["breakdown_by_error"].get(error_type, 0) + 1
                    llm_failures["failed_patches"].append({
                        "patch_file": result.get("patch_file"),
                        "project": result.get("project"),
                        "error_type": error_type,
                        "error_message": result.get("message_output", "").strip()
                    })
            
            # Get LLM generation failures
            llm_detailed_report = _read_json(llm_detailed_report_path)
            llm_summary = llm_detailed_report.get("summary", {})
            llm_generation_failures_list = llm_summary.get("failed_patches_list", [])
            llm_generation_failures = len(llm_generation_failures_list)
            llm_total_input_tokens = llm_summary.get("total_input_tokens", 0)
            llm_total_output_tokens = llm_summary.get("total_output_tokens", 0)
            llm_total_tokens = llm_summary.get("total_tokens", 0)

            error_type = "LLM_generation_failed"
            if llm_generation_failures > 0:
                llm_failures["breakdown_by_error"][error_type] = llm_generation_failures
                for patch_info in llm_generation_failures_list:
                     llm_failures["failed_patches"].append({
                        "patch_file": patch_info.get("patch_file"),
                        "project": patch_info.get("project"),
                        "error_type": error_type,
                        "error_message": patch_info.get("error", "LLM did not produce a usable patch.")
                    })
            
            llm_failures["total"] = llm_application_failures + llm_generation_failures
            
            # 5. Compile Final Summary
            total_successful = directly_applied + llm_applied
            summary["pipeline_summary"] = {
                "total_patches_from_vanir": total_from_vanir,
                "total_fetch_failures": analysis["fetch_failures"]["total"],
                "total_patches_for_application": patches_for_application,
                "patches_applied_directly": directly_applied,
                "patches_forwarded_to_llm": patches_forwarded_to_llm,
                "patches_successfully_fixed_by_llm": llm_applied,
                "llm_application_failures": llm_application_failures,
                "llm_generation_failures": llm_generation_failures,
                "llm_total_input_tokens": llm_total_input_tokens,
                "llm_total_output_tokens": llm_total_output_tokens,
                "llm_total_tokens": llm_total_tokens,
                "total_successful_patches": total_successful,
                "total_failed_patches": patches_forwarded_to_llm - llm_applied,
                "application_success_rate": f"{(total_successful / patches_for_application * 100):.2f}%" if patches_for_application > 0 else "0.00%",
                "pipeline_success_rate": f"{(total_successful / total_from_vanir * 100):.2f}%" if total_from_vanir > 0 else "0.00%"
            }
            
            summary["detailed_error_analysis"] = analysis
            
            # 6. Write to file and log to console
            final_summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(final_summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=4)
            
            logging.info(f"Final summary report saved to '{final_summary_path.name}'")
            
            # Log a clean summary to the console log.
            s = summary['pipeline_summary']
            summary_text = "\n--- Final Pipeline Summary ---\n"
            summary_text += f"  - Total Patches from Vanir:         {s['total_patches_from_vanir']}\n"
            summary_text += f"  - Fetch Failures:                   {s['total_fetch_failures']}\n"
            summary_text += "  ---------------------------------\n"
            summary_text += f"  - Patches for Application:          {s['total_patches_for_application']}\n"
            summary_text += f"  -   Applied Directly:               {s['patches_applied_directly']}\n"
            summary_text += f"  -   Forwarded to LLM:             {s['patches_forwarded_to_llm']}\n"
            summary_text += "  ---------------------------------\n"
            summary_text += "  - LLM Processing:\n"
            summary_text += f"  -   Successfully Fixed by LLM:    {s['patches_successfully_fixed_by_llm']}\n"
            summary_text += f"  -   LLM Application Failures:     {s['llm_application_failures']}\n"
            summary_text += f"  -   LLM Generation Failures:      {s['llm_generation_failures']}\n"
            summary_text += f"  -   LLM Total Tokens Used:        {s['llm_total_tokens']} (Input: {s['llm_total_input_tokens']}, Output: {s['llm_total_output_tokens']})\n"
            summary_text += "  ---------------------------------\n"
            summary_text += f"  - Total Successfully Applied:       {s['total_successful_patches']}\n"
            summary_text += f"  - Overall Application Success Rate: {s['application_success_rate']}\n"
            summary_text += f"  - End-to-End Pipeline Success Rate: {s['pipeline_success_rate']}\n"
            summary_text += "--------------------------------\n"
            logging.info(summary_text)

        logging.info(f"COMPLETED STEP: {title}")
        console.print(f"✅ {title}")

@click.command()
@click.option('--source_path', required=True, type=click.Path(exists=True), help='Path to the target source directory.')
@click.option('--verbose', is_flag=True, default=False, help="Enable verbose console output.")
def main(source_path, verbose):
    """Main entry point for the Vidar Patch Management Pipeline."""
    setup_logging(verbose=verbose)
    os.environ['TARGET_SOURCE_PATH'] = source_path
    
    vidar_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(vidar_dir, 'config.yaml')
    
    runner = PipelineRunner(config_path)
    runner.run_pipeline()

if __name__ == "__main__":
    main() 
r"""Main pipeline orchestrator for the Vanir Patch Management System.

This script serves as the master runner for the entire patch processing workflow.
It executes a series of modules in a predefined sequence to parse vulnerability
reports, fetch required patches, attempt to apply them, and use an LLM to
correct any patches that fail.

The pipeline is designed to be run from the root of the `vidar` directory.

Usage:
  python pipeline_runner.py
"""

import json
import os
import subprocess
import sys

def run_command(command, cwd=None):
    """Runs a command, displaying live output and checking for errors.

    Args:
        command: A list of strings representing the command to run.
        cwd: The working directory in which to run the command.
    """
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding='utf-8'
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='')

        process.wait()

        if process.returncode != 0:
            print(f"\n❌ Error: Command failed with exit code {process.returncode}")
            sys.exit(process.returncode)
            
    except FileNotFoundError:
        print(f"❌ Error: Command not found. Is '{command[0]}' in your PATH?")
        sys.exit(1)
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        sys.exit(1)

class PipelineRunner:
    """Orchestrates the entire patch processing pipeline from start to finish.

    This class manages the sequence of operations, handles the flow of data
    between scripts, and provides descriptive output for each stage.
    """

    def __init__(self):
        """Initializes the PipelineRunner, setting up necessary paths."""
        self.vidar_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.vidar_dir)
        self.python_executable = sys.executable
        self.reports_dir = os.path.join(self.vidar_dir, 'reports')
        self.fetched_patches_dir = os.path.join(
            self.project_root, 'fetch_patch_output', 'diff_output')

    def run_step(self, title, description, command):
        """Prints a descriptive header and runs a pipeline step.

        Args:
            title: The title of the pipeline step.
            description: A brief description of what the step does.
            command: The command to execute as a list of strings.
        """
        print(f"\n{'='*80}")
        print(f"🚀 STEP: {title}")
        print(f"{'-'*80}")
        print(f"ℹ️  Description: {description}")
        print(f"💻 Command: {' '.join(command)}")
        print(f"{'-'*80}\n")
        run_command(command, cwd=self.vidar_dir)
        print(f"\n✅ STEP COMPLETED: {title}")

    def prepare_llm_input(self) -> bool:
        """Filters the application report to prepare the input for the LLM.

        This function reads the `patch_application_report.json`, finds all
        patches marked as 'Rejected', and creates `failed_patch.json` containing
        the necessary details for the LLM runner.

        Returns:
            True if there are failed patches to process, False otherwise.
        """
        title = "Prepare for LLM Correction"
        print(f"\n{'='*80}")
        print(f"⚙️  INTERMEDIATE STEP: {title}")
        print(f"{'-'*80}")
        print("ℹ️  Description: Reads the report from the initial patch application, "
              "filters for 'Rejected' patches, and creates 'failed_patch.json' "
              "as input for the LLM runner.")
        print(f"{'-'*80}\n")

        report_path = os.path.join(self.reports_dir,
                                   'patch_application_report.json')
        llm_input_path = os.path.join(self.project_root, 'failed_patch.json')

        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                adoption_report = json.load(f)
        except FileNotFoundError:
            print(f"❌ Error: Cannot find patch application report at '{report_path}'.")
            print("   Did 'patch_adopter.py --source Vanir' run successfully?")
            sys.exit(1)

        # Filter for patches that were rejected
        failed_patches = []
        for patch in adoption_report.get('patches', []):
            if patch.get('status') == 'Rejected':
                new_patch_entry = patch.copy()
                
                # Load the content of the original failed diff file
                original_diff_path = os.path.join(self.fetched_patches_dir,
                                                  patch['patch_file'])
                if os.path.exists(original_diff_path):
                    with open(original_diff_path, 'r', encoding='utf-8') as f:
                        new_patch_entry['upstream_patch_content'] = f.read()
                else:
                    print(f"⚠️ Warning: Could not find original diff file: "
                          f"{original_diff_path}. Skipping patch for LLM.")
                    continue
                failed_patches.append(new_patch_entry)
        
        with open(llm_input_path, 'w', encoding='utf-8') as f:
            json.dump({"patches": failed_patches}, f, indent=4)
            
        print(f"✅ Created '{os.path.basename(llm_input_path)}' with "
              f"{len(failed_patches)} failed patches.")
        
        if not failed_patches:
            print("✅ STEP COMPLETED: No failed patches to process with the LLM. "
                  "The pipeline will stop here.")
            return False
        
        print(f"✅ STEP COMPLETED: {title}")
        return True

    def run_pipeline(self):
        """Executes the full pipeline in the correct sequence."""
        print("Starting Vanir Security Patch Processing Pipeline...")

        # Step 1: Parse the initial Vanir report.
        self.run_step(
            title="1. Parse Vanir Report",
            description=("Parses 'reports/vanir_output.json' to create a "
                         "structured 'reports/parsed_report.json' listing "
                         "unique patches."),
            command=[self.python_executable,
                     os.path.join(self.vidar_dir, 'vanir_report_parser.py')]
        )
        
        # Step 2: Download the patches specified in the parsed report.
        self.run_step(
            title="2. Fetch Patches",
            description=(f"Downloads original patches from repositories into "
                         f"'{os.path.basename(self.fetched_patches_dir)}/'."),
            command=[self.python_executable,
                     os.path.join(self.vidar_dir, 'patch_fetcher.py')]
        )
        
        # Step 3: Attempt to apply the original downloaded patches.
        self.run_step(
            title="3. Apply Original (Vanir) Patches",
            description=("Attempts to apply the downloaded patches and generates "
                         "'reports/patch_application_report.json' with the "
                         "results."),
            command=[self.python_executable,
                     os.path.join(self.vidar_dir, 'patch_adopter.py'),
                     '--source', 'Vanir']
        )

        # Intermediate Step: Check for failures and prepare for the LLM.
        if self.prepare_llm_input():
            # Step 4: Use the LLM to correct the patches that failed.
            self.run_step(
                title="4. Run LLM Patch Correction",
                description=("Uses the Gemini LLM to generate corrected patches "
                             "for failed ones, saving them to "
                             "'patch_adoption/generated_patches/'."),
                command=[self.python_executable,
                         os.path.join(self.vidar_dir, 'llm_patch_runner.py')]
            )
            
            # Step 5: Attempt to apply the new, LLM-generated patches.
            self.run_step(
                title="5. Apply LLM-Generated Patches",
                description=("Attempts to apply the new, LLM-generated patches "
                             "and updates the final report."),
                command=[self.python_executable,
                         os.path.join(self.vidar_dir, 'patch_adopter.py'),
                         '--source', 'LLM']
            )

        print(f"\n{'='*80}")
        print("🎉🎉🎉 Pipeline completed successfully! 🎉🎉🎉")
        print(f"{'='*80}")

def main():
    """Main entry point."""
    runner = PipelineRunner()
    runner.run_pipeline()

if __name__ == "__main__":
    main() 
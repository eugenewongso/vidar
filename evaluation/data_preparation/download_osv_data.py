r"""Downloads and extracts a complete OSV vulnerability database for an ecosystem.

This script is the primary tool for bootstrapping the local vulnerability
database needed for analysis. It connects to the OSV.dev Google Cloud Storage
bucket, downloads a zip archive containing all known vulnerabilities for a
specified ecosystem (e.g., "Android"), and extracts the individual JSON files
into a local directory.

The process is as follows:
1.  Constructs the download URL based on the specified ecosystem name.
2.  Initiates a streaming download of the `all.zip` archive.
3.  Displays a rich progress bar showing download speed, size, and estimated
    time remaining.
4.  Once downloaded, it extracts all JSON files from the zip archive into the
    specified output directory.
5.  Handles common errors such as invalid ecosystem names (HTTP 404), network
    issues, and corrupted zip files.

Usage:
  python download_osv_data.py --ecosystem <EcosystemName> --output_dir <path_to_dir>

Example:
  python download_osv_data.py --ecosystem Android --output_dir osv_data_android
"""
import argparse
import os
import requests
import zipfile
import io
import sys
from rich.progress import Progress, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn

# The base URL for OSV data dumps on Google Cloud Storage.
OSV_GCS_URL = "https://osv-vulnerabilities.storage.googleapis.com"

def download_and_extract(ecosystem: str, output_dir: str):
    """
    Downloads and extracts the OSV data dump for a given ecosystem.

    Args:
        ecosystem: The name of the ecosystem (e.g., "Android", "PyPI").
        output_dir: The directory to extract the vulnerability files into.
    """
    zip_url = f"{OSV_GCS_URL}/{ecosystem}/all.zip"
    print(f"⬇️  Downloading OSV data for ecosystem '{ecosystem}' from:")
    print(f"   {zip_url}")

    try:
        response = requests.get(zip_url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        # Create the output directory if it doesn't exist.
        os.makedirs(output_dir, exist_ok=True)

        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            "•",
            TimeRemainingColumn(),
        ) as progress:
            task_id = progress.add_task(f"Downloading {ecosystem}.zip", total=total_size)
            zip_content = io.BytesIO()
            for chunk in response.iter_content(chunk_size=8192):
                if chunk: # filter out keep-alive new chunks
                    zip_content.write(chunk)
                    progress.update(task_id, advance=len(chunk))

        print(f"📦 Extracting files to '{output_dir}'...")
        zip_content.seek(0)
        with zipfile.ZipFile(zip_content) as zf:
            zf.extractall(output_dir)

        print(f"✅ Successfully downloaded and extracted {len(zf.namelist())} files.")

    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e.status_code} {e.response.reason}", file=sys.stderr)
        print(f"   Please check if the ecosystem '{ecosystem}' is correct and data is available at the URL.", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"❌ Download failed: {e}", file=sys.stderr)
        sys.exit(1)
    except zipfile.BadZipFile:
        print("❌ Error: Downloaded file is not a valid zip file.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Download and extract OSV data dumps.")
    parser.add_argument(
        "--ecosystem",
        type=str,
        default="Android",
        help="The OSV ecosystem to download data for (e.g., 'Android', 'PyPI')."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="osv_data_android",
        help="The directory to extract the vulnerability JSON files into."
    )
    args = parser.parse_args()

    download_and_extract(args.ecosystem, args.output_dir)

if __name__ == "__main__":
    main() 
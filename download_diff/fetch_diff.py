import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime

class DiffFetcher:
    """Handles fetching and extracting diff files from Android source repositories."""

    def __init__(self, commit_hash, output_dir_html="Fetch_patch_output_html", output_dir_diff="Fetch_patch_output_diff"):
        self.commit_hash = commit_hash
        self.diff_url = f"https://android.googlesource.com/platform/frameworks/base/+/{commit_hash}%5E%21/"
        self.output_dir_html = output_dir_html
        self.output_dir_diff = output_dir_diff
        os.makedirs(self.output_dir_html, exist_ok=True)
        os.makedirs(self.output_dir_diff, exist_ok=True)
    
    def fetch_diff(self):
        """Fetches the diff file from the source repository."""
        print(f"Fetching diff from: {self.diff_url}")
        response = requests.get(self.diff_url)
        if response.status_code != 200:
            print(f"Failed to fetch diff. HTTP Status: {response.status_code}")
            return None
        return response.text
    
    def save_prettified_html(self, html_content):
        """Saves the prettified HTML content of the diff page."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        html_filename = os.path.join(self.output_dir_html, f"{self.commit_hash}_{timestamp}.html")
        with open(html_filename, "w", encoding="utf-8") as file:
            file.write(html_content)
        print(f"Prettified Diff file saved as: {html_filename}")
        return html_filename
    
    def extract_text_diff(self, html_filename):
        """Extracts plain text diff content from the HTML file."""
        with open(html_filename, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "lxml")
        text_content = soup.get_text(separator="\n", strip=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        text_filename = os.path.join(self.output_dir_diff, f"{self.commit_hash}_{timestamp}.txt")
        with open(text_filename, "w", encoding="utf-8") as output_file:
            output_file.write(text_content)
        print(f"Extracted content saved to: {text_filename}")
        return text_filename
    
    def process_diff(self):
        """Orchestrates the full process of fetching, saving, and extracting the diff."""
        html_content = self.fetch_diff()
        if html_content:
            html_filename = self.save_prettified_html(html_content)
            text_filename = self.extract_text_diff(html_filename)
            return text_filename
        return None

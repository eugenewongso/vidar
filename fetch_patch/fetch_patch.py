import os
import requests 
from bs4 import BeautifulSoup 
from datetime import datetime

# logic here form Eugene to traverse 
diff_url = "https://android.googlesource.com/platform/frameworks/base/+/cde345a7ee06db716e613e12a2c218ce248ad1c4" 
# diff_url = "https://git.codelinaro.org/clo/la/kernel/msm-5.10/-/commit/c1a7b4b4a736fa175488122cca9743cff2ae72e8"

if "android.googlesource.com" in diff_url:
    diff_url += "^!"
elif "git.codelinaro.org" in diff_url:
    diff_url += ".diff"

# TODO: need to change this later on to use Eugene's Map

print(f"Fetching diff from: {diff_url}")

output_dir_html = "Fetch_patch_output_html"
output_dir_diff = "Fetch_patch_output_diff"

os.makedirs(output_dir_html, exist_ok=True)
response = requests.get(diff_url)
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  


if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")

    diff_filename = os.path.join(output_dir_html, f"{timestamp}.html")
    with open(diff_filename, "w", encoding="utf-8") as f:
        f.write(soup.prettify())  

    print(f"Prettified Diff file saved as: {diff_filename}")
    diff_filename_html = os.path.join(output_dir_html, f"{timestamp}.html") # current html file opened
    with open(diff_filename_html, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "lxml")
    
    text_content = soup.get_text(separator="\n", strip=True)
    output_filename = os.path.join(output_dir_diff, f"{timestamp}.txt") # might neeed to change file extension later 
    with open(output_filename, "w", encoding="utf-8") as output_file:
        output_file.write(text_content)

    print("Extracted content saved to:", output_filename)

    # title_tag = soup.find('title')
    # if title_tag:
    #     print("Title Commit Metadata:", title_tag.text.strip())
    # metadata_tag = soup.find('pre', class_='MetadataMessage')
    # if metadata_tag:
    #     print("Metadata Tags:", metadata_tag.text.strip()) # only print one metadata
    # # Get per hunk diff:
    # if soup.find('span', class_='Diff-hunk'):
    #     diff_hunks = soup.find_all('span', class_='Diff-hunk')
    #     for index, hunk in enumerate(diff_hunks, start=1): 
    #         print(f"Hunk {index} found: {hunk.text.strip()}")
    
else:
    print(f"Failed to fetch diff. HTTP Status: {response.status_code}")

with open(diff_filename_html, "r", encoding="utf-8") as file:
    pass


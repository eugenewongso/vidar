from gemini_function import GeminiPatchGenerator
import os

print(f"Current Working Directory: {os.getcwd()}")
json_file = os.path.abspath("reports/xiaomi_parsed.json")
print(f"Resolved JSON Path: {json_file}")

with open(json_file, "r") as f:
    print("File opened successfully!")

# Path to the parsed JSON file
json_file = os.path.abspath("reports/xiaomi_parsed.json")

# Initialize the generator
generator = GeminiPatchGenerator()

# Generate the patch
patch_file = generator.generate_patch(json_file)

print(f"Generated patch is saved at: {patch_file}")

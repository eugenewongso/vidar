from unidiff import PatchSet

def validate_diff(diff_file_path):
    """
    Validates whether the given diff file is a valid unified diff format.
    
    Args:
        diff_file_path (str): Path to the saved diff file.
    
    Returns:
        tuple: (bool, str) where True means valid, False means invalid with an error message.
    """
    try:
        with open(diff_file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        # Extract diff content starting from "==== DIFF CONTENT ===="
        in_diff_section = False
        diff_lines = []

        for line in lines:
            if line.strip() == "==== DIFF CONTENT ====":
                in_diff_section = True  # Start capturing diff content
                continue  # Skip this header line itself

            if in_diff_section:
                diff_lines.append(line)

        # Convert back to string format
        diff_str = "".join(diff_lines).strip()

        print("diff string to parse:", diff_str)

        # Validate diff format
        PatchSet(diff_str)
        return True, "Valid unified diff."
    
    except Exception as e:
        return False, str(e)

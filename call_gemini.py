import os
from google import genai
from google.genai import types

def patch_port_diff(diff_file: str, vulnerable_file: str, error_message: str) -> str:
    """
    Takes a diff file and a vulnerable file as input, processes them using Google's Gemini API,
    and returns an updated diff file that can be applied cleanly.
    
    Parameters:
    - diff_file (str): The content of the original diff file.
    - vulnerable_file (str): The content of the vulnerable file to which the diff will be applied.
    
    Returns:
    - str: The updated diff file that applies cleanly.
    """
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY")) # or make sure to do export GEMINI_API_KEY="AIzaSyAtUfjEH-Mrvjq7COBItoAWBoDGzSx-gVo"

    model = "gemini-2.0-pro-exp-02-05"
    
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=f"""I am going to give you a diff file and a vulnerable file.
                    
                    This is the diff file:

                    {diff_file}

                    This diff file cannot be applied cleanly, and it results in the following error:
                    {error_message}

                    This is the vulnerable file:

                    I want you to 'patch port' this diff file so that it applies cleanly to the given vulnerable file.
                    Strictly return only the updated diff file so that I can copy and paste it and apply it using 'patch apply'.
                    Also make sure that there is an additional blank line break at the end of the file so that it works.  
                    
                    {vulnerable_file}

					I want you to return the new, updated diff file so that I can apply it to the vulnerable files cleanly.
                    """
                ),
            ],
        ),
    ]

    response = client.models.generate_content(model=model, contents=contents)

    return response.text  # Returns the updated diff file
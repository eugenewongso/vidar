def request_clean_patch(original_diff: str, error_message: str, vulnerable_source: str) -> str:
    """
    Automates the LLM query process for generating a modified patch file.
    
    Parameters:
    - original_diff (str): The diff file content that failed to apply.
    - error_message (str): The error output from `git apply --check`.
    - vulnerable_source (str): The content of the vulnerable file before applying the patch.
    
    Returns:
    - str: The modified diff file that should apply cleanly.
    """
    prompt = f"""
    I have a `.diff` file that fails to apply to a vulnerable source file. Here is the `.diff` file:

    ```
    {original_diff}
    ```

    Here is the error message when applying the patch:
    ```
    git apply --check <PATCH_FILENAME>.diff
    {error_message}
    ```

    I also have the vulnerable source file before patching:
    ```
    {vulnerable_source}
    ```

    I need you to modify the `.diff` file so that it applies cleanly to the vulnerable source file while maintaining the intended fix. Ensure that all necessary changes (such as file structure, copyright headers, or line offsets) are considered.

    Just give me the modified `.diff` file that will apply cleanly.
    Ensure that it includes a blank line at the end.
    """

    return prompt
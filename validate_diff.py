from unidiff import PatchSet

def validate_diff(diff_str):
    try:
        PatchSet(diff_str)
        return True, ""
    except Exception as e:
        return False, str(e)
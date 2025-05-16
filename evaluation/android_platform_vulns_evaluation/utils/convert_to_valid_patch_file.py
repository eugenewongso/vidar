patch_str = """Example patch"""

with open("test_patch.diff", "w") as f:
    f.write(patch_str.encode("utf-8").decode("unicode_escape"))
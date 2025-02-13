from patch_adopter import PatchAdopter

# Simple test to check if PatchAdopter initializes
def test_patch_application():
    adopter = PatchAdopter()
    assert adopter is not None
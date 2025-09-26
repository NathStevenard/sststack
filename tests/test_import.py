"""
This is a quick step to ensure the sststack package is installed in your environment.

Just run the script. If it fails, try to install again the environment (see the README.md file).
"""

def test_import():
    import sststack
    assert hasattr(sststack, "Stacker")
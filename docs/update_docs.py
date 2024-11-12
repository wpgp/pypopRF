# docs/update_docs.py
import subprocess
import sys


def update_docs():
    """Update Sphinx documentation."""
    try:
        # Clean build directory
        subprocess.run(["make", "clean"], cwd="docs")

        # Build HTML documentation
        subprocess.run(["make", "html"], cwd="docs")

        print("Documentation updated successfully!")

    except subprocess.CalledProcessError as e:
        print(f"Error updating documentation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    update_docs()
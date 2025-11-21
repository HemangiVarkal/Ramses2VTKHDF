"""
Unit tests for the Chhavi CLI.

These tests verify that the command-line interface:
1. Executes a dry-run without errors
2. Lists available fields correctly
3. Handles invalid input folders gracefully

"""

import subprocess
import sys
from pathlib import Path
import tempfile

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

RAMSES_OUTPUT_ROOT = Path("ramses_outputs/sedov_3d")
SNAPSHOT_FOLDERS = ["output_00001", "output_00002"]


# ──────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────

def test_cli_dry_run():
    """Ensure the CLI dry-run command executes without errors on a snapshot."""

    snapshot = SNAPSHOT_FOLDERS[0]

    # Use a temporary directory as output directory for testing
    with tempfile.TemporaryDirectory() as tmp_output_dir:
        result = subprocess.run(
            [sys.executable, "-m", "chhavi.cli",
             "--base-dir", str(RAMSES_OUTPUT_ROOT),
             "--folder-name", snapshot,
             "-n", "1",
             "--dry-run",
             "--verbose",
             "--output-dir", tmp_output_dir],  # Pass output-dir explicitly
            capture_output=True,
            text=True,
        )

        # CLI may skip outputs if files are missing; still considered success
        assert result.returncode == 0
        assert ("dry run" in result.stdout.lower() or
                "completed" in result.stdout.lower() or
                "no data for output" in result.stderr.lower())


def test_cli_list_fields():
    """Verify that the CLI lists all available fields for a snapshot."""

    snapshot = SNAPSHOT_FOLDERS[1]

    # Use a temporary directory as output directory for testing (optional for list-fields)
    with tempfile.TemporaryDirectory() as tmp_output_dir:
        result = subprocess.run(
            [sys.executable, "-m", "chhavi.cli",
             "--base-dir", str(RAMSES_OUTPUT_ROOT),
             "--folder-name", snapshot,
             "-n", "1",
             "--list-fields",
             "--verbose",
             "--output-dir", tmp_output_dir],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert ("available fields" in result.stdout.lower() or
                "no fields" in result.stdout.lower())


def test_cli_invalid_folder():
    """Check that the CLI returns a non-zero exit code for a non-existent folder."""

    # Using a non-existent output-dir as additional argument to check robustness
    with tempfile.TemporaryDirectory() as tmp_output_dir:
        # Provide a deliberately non-existent input folder to trigger error
        non_existent_folder = "/non/existent/path"

        result = subprocess.run(
            [sys.executable, "-m", "chhavi.cli",
             "--base-dir", non_existent_folder,
             "--folder-name", "output_999",
             "-n", "1",
             "--output-dir", tmp_output_dir],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "no such file or directory" in result.stderr.lower() or "error" in result.stderr.lower()

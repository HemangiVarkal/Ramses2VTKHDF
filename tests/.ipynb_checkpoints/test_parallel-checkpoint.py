
"""

Parallel conversion tests for Ramses2VTKHDF.

Uses real RAMSES outputs and verifies dry-run mode for single and multiple snapshots.

"""

from pathlib import Path
from ramses_to_vtkhdf.parallel import run_parallel_conversion

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

RAMSES_OUTPUT_ROOT = Path("ramses_outputs/sedov_3d")
SNAPSHOT_FOLDERS = ["output_00001", "output_00002"]


# ──────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────

def test_parallel_dry_run_single():
    
    """Run parallel conversion in dry-run mode for a single real snapshot."""
    
    snapshot = SNAPSHOT_FOLDERS[0]
    output_num = int(snapshot.split("_")[-1])
    input_folder = RAMSES_OUTPUT_ROOT / snapshot

    run_parallel_conversion(
        output_numbers=[output_num],
        input_folder=str(input_folder),
        output_prefix="test_amr",
        fields=None,
        dry_run=True,
        verbose=False,
    )


def test_parallel_dry_run_multiple():
    
    """Run parallel conversion in dry-run mode for multiple real snapshots."""
    
    output_numbers = [int(s.split("_")[-1]) for s in SNAPSHOT_FOLDERS]
    input_folder = RAMSES_OUTPUT_ROOT

    run_parallel_conversion(
        output_numbers=output_numbers,
        input_folder=str(input_folder),
        output_prefix="test_amr",
        fields=None,
        dry_run=True,
        verbose=False,
    )

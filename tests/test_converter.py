"""
Unit tests for the Chhavi converter.

These tests verify that the converter:
1. Initializes correctly
2. Filters AMR levels properly
3. Builds masks for spatial filtering
4. Collects fields from a mesh dictionary
5. Performs a dry-run on real snapshots without writing files

"""

import numpy as np
from chhavi.converter import ChhaviConverter

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

RAMSES_OUTPUT_ROOT = "ramses_outputs/sedov_3d"
SNAPSHOT_FOLDERS = ["output_00001", "output_00002"]


# ──────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────

def test_converter_init():
    """Ensure converter initializes with correct input folder and prefix."""
    conv = ChhaviConverter(
        input_folder=RAMSES_OUTPUT_ROOT,
        output_prefix="test_amr",
        fields=["density"],
        dry_run=True,
    )
    assert conv.input_folder == RAMSES_OUTPUT_ROOT
    assert conv.output_prefix == "test_amr"


def test_filter_levels():
    """Verify filtering of AMR levels using level_start and level_end."""
    conv = ChhaviConverter(input_folder=RAMSES_OUTPUT_ROOT)
    # By default, all levels included
    assert conv._filter_levels([0, 1, 2, 3, 4]) == [0, 1, 2, 3, 4]

    conv.level_start = 2
    conv.level_end = 3
    # Only levels 2 and 3 included
    assert conv._filter_levels([0, 1, 2, 3, 4]) == [2, 3]


def test_build_mask_all_true():
    """Ensure mask building returns all True when no physical bounds are set."""
    conv = ChhaviConverter(input_folder=RAMSES_OUTPUT_ROOT)
    px = np.array([0.1, 0.5, 0.9])
    py = np.array([0.2, 0.6, 0.8])
    pz = np.array([0.0, 0.5, 1.0])
    mask = conv._build_mask(px, py, pz)
    assert mask.all()


def test_collect_fields_dict():
    """Verify that fields can be collected from a dummy mesh dictionary."""
    conv = ChhaviConverter(input_folder=RAMSES_OUTPUT_ROOT)
    dummy_mesh = {"density": [1.0], "pressure": [1.0]}
    fields = conv._collect_fields_from_mesh(dummy_mesh)
    assert "density" in fields
    assert "pressure" in fields


def test_dry_run_real_snapshot():
    """Ensure dry-run conversion works on a real snapshot without writing files."""
    snapshot = SNAPSHOT_FOLDERS[0]
    conv = ChhaviConverter(
        input_folder=RAMSES_OUTPUT_ROOT,
        output_prefix=f"dryrun_{snapshot}",
        fields=["density", "velocity"],
        dry_run=True,
    )
    try:
        output_num = int(snapshot.split("_")[-1])
        conv.process_output(output_num)
        success = True
    except Exception as e:
        print(f"⚠️ Dry-run failed for snapshot {snapshot}: {e}")
        success = False

    assert success

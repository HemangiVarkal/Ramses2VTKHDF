#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

─────────────────────────────────────────────────────────────
Example Usage of Ramses2VTKHDF
─────────────────────────────────────────────────────────────

This script demonstrates how to use the `RamsesToVtkHdfConverter`
class to explore and convert RAMSES simulation outputs into
VTKHDF OverlappingAMR files.

Features demonstrated:
1. Listing available fields in snapshots
2. Inspecting basic dataset info
3. Performing a dry-run conversion (no files written)
4. Showing physical fields that will be added to the output
─────────────────────────────────────────────────────────────

"""

import os
from typing import List
from ramses_to_vtkhdf.converter import RamsesToVtkHdfConverter, list_fields_for_snapshot

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

# Parent folder where all RAMSES snapshot folders are stored
RAMSES_OUTPUT_ROOT = "ramses_outputs/sedov_3d"

# Snapshot folders (used for naming/output prefix)
SNAPSHOT_FOLDERS = ["output_00001", "output_00002"]

# Fields to preview
FIELDS_TO_INSPECT = ["density", "velocity", "pressure"]

# Dry-run prevents actual file writing; set to False to convert for real
DRY_RUN = True


# ──────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────

def print_snapshot_info(snapshot_num: int, snapshot_name: str, fields: List[str]):
    
    """
    Print a summary of a snapshot for the user.

    Args:
        snapshot_num: Snapshot number (1, 2, ...)
        snapshot_name: Folder name of the snapshot
        fields: List of fields detected in this snapshot
    """
    
    print(f"\n🔹 Processing snapshot {snapshot_num} ({snapshot_name})...")
    if fields:
        print(f"Detected fields: {', '.join(fields)}")
    else:
        print("Detected fields: None")


def print_physical_fields(fields: List[str]):
    
    """
    Print which physical fields will be written, classifying them
    as scalar or vector fields.

    Args:
        fields: List of requested fields
    """
    
    # Example classification; adjust based on your simulation
    scalar_fields = [f for f in fields if f.lower() in {"density", "pressure", "grav_potential"}]
    vector_fields = [f for f in fields if f.lower() in {"velocity", "grav_acceleration", "magnetic_field"}]

    print("Fields that will be added to VTKHDF:")
    if scalar_fields:
        print("  Scalars:", ", ".join(scalar_fields))
    if vector_fields:
        print("  Vectors:", ", ".join(vector_fields))
    if not scalar_fields and not vector_fields:
        print("  None (no standard fields detected)")


# ──────────────────────────────────────────────────────────────
# Main Example Workflow
# ──────────────────────────────────────────────────────────────

def main():
    
    """
    Main function demonstrating usage of RamsesToVtkHdfConverter.
    Lists available fields, performs a dry-run conversion, and shows
    physical fields that will be written.
    """
    
    print("=== Ramses2VTKHDF Example Usage ===")
    print("This example inspects snapshots and performs a dry-run conversion.\n")

    for idx, snapshot_name in enumerate(SNAPSHOT_FOLDERS, start=1):
        # 1️⃣ List available fields in the snapshot
        try:
            available_fields = list_fields_for_snapshot(RAMSES_OUTPUT_ROOT, idx)
        except Exception as e:
            print(f"⚠️ Failed to list fields for snapshot {idx}: {e}")
            available_fields = []

        print_snapshot_info(idx, snapshot_name, available_fields)

        # 2️⃣ Create converter object
        converter = RamsesToVtkHdfConverter(
            input_folder=RAMSES_OUTPUT_ROOT,
            output_prefix=f"example_{snapshot_name}",
            fields=FIELDS_TO_INSPECT,
            dry_run=DRY_RUN,
        )

        # 3️⃣ Show which physical fields will be added
        print_physical_fields(converter.requested_fields)

        # 4️⃣ Load and convert snapshot (dry-run)
        try:
            converter.process_output(idx)
            print(f"✅ Snapshot {idx} dry-run conversion completed successfully!")
        except Exception as e:
            print(f"❌ Failed to process snapshot {idx}: {e}")

    print("\n🎉 Example usage finished!")
    print("Set `dry_run=False` to write actual VTKHDF files.")


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()

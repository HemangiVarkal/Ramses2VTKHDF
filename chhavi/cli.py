#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

──────────────────────────────────────────────────────────────────────────────
CLI QUICK START (copy–paste, then tweak)
──────────────────────────────────────────────────────────────────────────────
Example run (filters a subvolume, includes extra fields):

    python3 chhavi.py \
        --base-dir ./simulations \
        --folder-name output_dir \
        --numbers 1,3,5-7 \
        --output-prefix overlapping_amr \
        --level-start 2 --level-end 5 \
        --x-range 0.0:1.0 --y-range 0.0:1.0 --z-range 0.0:1.0 \
        --fields density,pressure \
        --verbose

Exploration mode :

    # Lists fields Osyris sees in the mesh (no conversion happens)
    python3 chhavi.py --base-dir ./simulations --folder-name output_dir \
        -n 5 --list-fields

    # Dry-run: show counts after filters, but don’t write .vtkhdf
    python3 chhavi.py --base-dir ./simulations --folder-name output_dir \
        -n 5 --level-start 1 --dry-run --verbose

Required args:

    --base-dir         Path to your RAMSES run root directory.
    --folder-name      Subfolder inside base-dir containing outputs.
    -n / --numbers     Output numbers to process. Formats:
                       "7" or "3,5,9" or "10-15"

Optional args:

    --output-prefix / -o   Prefix for output files (default: overlapping_amr)
    --level-start / --level-end     AMR level filter (inclusive)
    --x-range / --y-range / --z-range   Normalized ranges [0,1] over box length
    --fields           Physical fields to be included (default: density, pressure, velocity)
    --verbose          step-by-step narration
    --list-fields      Only list available mesh fields and exit
    --dry-run          Run everything except the actual write step

Tip on “normalized ranges”:
    RAMSES coordinates are in code/physical units. We divide by the simulation
    box length (from metadata) so [0,1] always spans the full domain, regardless
    of units.

"""


import os
import argparse
import logging

from .converter import parse_output_numbers, parse_norm_range, parse_fields_arg, list_fields_for_snapshot
from .parallel import run_parallel_conversion, setup_logging

logger = logging.getLogger("chhavi")


def main() -> None:

    """
    Parse CLI args and run the conversion pipeline.
    """

    parser = argparse.ArgumentParser(description="VTKHDF AMR Generator from RAMSES Data (refactored)")

    # Required inputs
    parser.add_argument("--base-dir", type=str, required=True, help="Base directory containing simulation folders (REQUIRED)")
    parser.add_argument("--folder-name", type=str, required=True, help="Folder inside base_dir to process (REQUIRED)")
    parser.add_argument("-n", "--numbers", type=parse_output_numbers, required=True, help="Output numbers like '1', '1,3,5' or '2-7' (REQUIRED)")

    # Output and level selection
    parser.add_argument("-o", "--output-prefix", dest="output_prefix", default="overlapping_amr", help="Output file prefix (default: overlapping_amr)")
    parser.add_argument("--level-start", type=int, default=None, help="Minimum AMR level to include (inclusive). Optional.")
    parser.add_argument("--level-end", type=int, default=None, help="Maximum AMR level to include (inclusive). Optional.")

    # Normalized ranges (single arg per axis)
    parser.add_argument("--x-range", type=parse_norm_range, default=None, help="Normalized x range 'min:max' (e.g., 0.2:0.8, :0.7, 0.1:, :).")
    parser.add_argument("--y-range", type=parse_norm_range, default=None, help="Normalized y range 'min:max'.")
    parser.add_argument("--z-range", type=parse_norm_range, default=None, help="Normalized z range 'min:max'.")

    # Field selection (replace per-field enable flags with a single argument)
    parser.add_argument("--fields", type=parse_fields_arg, default=None, help="Comma-separated list of fields to include (e.g. density,velocity,magnetic_field). If omitted, sensible defaults are used.")

    parser.add_argument("--list-fields", action="store_true", help="List available fields in the first requested snapshot and exit.")

    # Utility flags
    parser.add_argument("--dry-run", action="store_true", help="Print plan without writing files.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")

    args = parser.parse_args()

    # Configure logging early
    setup_logging(args.verbose)

    # Build absolute input folder path and validate
    input_folder = os.path.join(os.path.abspath(args.base_dir), args.folder_name)

    if not os.path.exists(input_folder):
        logger.error("Input folder not found: %s", input_folder)
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    # Check level ranges
    def positive_int(val):
        try:
            iv = int(val)
            if iv < 0:
                raise argparse.ArgumentTypeError(f"Invalid value: {val}. Must be non-negative.")
            return iv
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid integer value: {val}")

    if args.level_start is not None and args.level_end is not None:
        if args.level_end < args.level_start:
            parser.error(f"Invalid level range: end ({args.level_end}) < start ({args.level_start}).")

    # Normalize default ranges: None -> (None,None)
    def norm_default(r):
        return (None, None) if r is None else r

    x_range = norm_default(args.x_range)
    y_range = norm_default(args.y_range)
    z_range = norm_default(args.z_range)

    # If user requested to list fields, inspect the first snapshot in args.numbers
    if args.list_fields:
        # pick first number (safe because parser ensured args.numbers exists)
        first_num = args.numbers[0]
        logger.info("Listing fields for snapshot %s in folder '%s'...", first_num, input_folder)
        fields = list_fields_for_snapshot(input_folder, first_num)

        if fields:
            print("Available fields (best-effort):")
            for f in fields:
                print(" -", f)
        else:
            print("No fields discovered (see logs for details).")
        return

    # Run conversion (fields argument may be None meaning use defaults+auto-detect)
    try:
        run_parallel_conversion(
            output_numbers=args.numbers,
            input_folder=input_folder,
            output_prefix=args.output_prefix,
            fields=args.fields,
            level_start=args.level_start,
            level_end=args.level_end,
            x_range_norm=x_range,
            y_range_norm=y_range,
            z_range_norm=z_range,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    except Exception as e:
        logger.exception("FATAL: Unexpected error: %s", e)
        raise


if __name__ == "__main__":
    main()

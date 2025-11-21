#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Parallel execution utilities for chhavi.

"""

from __future__ import annotations

from functools import partial
from typing import List, Optional, Tuple

import logging
import time
import concurrent.futures

from .converter import ChhaviConverter, setup_logging

logger = logging.getLogger("chhavi")


def process_single_output(
    output_num: int,
    input_folder: str,
    output_prefix: str,
    fields: Optional[List[str]],
    level_start: Optional[int],
    level_end: Optional[int],
    x_range_norm: Tuple[Optional[float], Optional[float]],
    y_range_norm: Tuple[Optional[float], Optional[float]],
    z_range_norm: Tuple[Optional[float], Optional[float]],
    dry_run: bool,
    verbose: bool,
    output_directory: Optional[str] = None,
) -> None:
    """
    Worker function executed in each process. It configures logging and runs conversion
    for a single snapshot number.

    Args:
        output_num: Snapshot/output number being processed.
        input_folder: Root input directory containing RAMSES outputs.
        output_prefix: File prefix for output files.
        fields: Optional list of scalar/vector fields to export.
        level_start: Optional AMR level filtering start.
        level_end: Optional AMR level filtering end.
        x_range_norm: Optional normalized filtering range on X.
        y_range_norm: Optional normalized filtering range on Y.
        z_range_norm: Optional normalized filtering range on Z.
        dry_run: Flag to skip writing output files.
        verbose: Flag for verbose logging.
        output_directory: Optional directory to save output files. If None, uses input_folder.
    """
    setup_logging(verbose)

    try:
        conv = ChhaviConverter(
            input_folder=input_folder,
            output_prefix=output_prefix,
            fields=fields,
            level_start=level_start,
            level_end=level_end,
            x_range_norm=x_range_norm,
            y_range_norm=y_range_norm,
            z_range_norm=z_range_norm,
            dry_run=dry_run,
            output_directory=output_directory,
        )
        conv.process_output(output_num)
    except Exception:
        # Catch any unexpected worker-level exceptions and log them
        # Do not re-raise because we want other workers to continue
        logger.exception("[worker %s] Unexpected worker error", output_num)


def run_parallel_conversion(
    output_numbers: List[int],
    input_folder: str,
    output_prefix: str,
    fields: Optional[List[str]] = None,
    level_start: Optional[int] = None,
    level_end: Optional[int] = None,
    x_range_norm: Tuple[Optional[float], Optional[float]] = (None, None),
    y_range_norm: Tuple[Optional[float], Optional[float]] = (None, None),
    z_range_norm: Tuple[Optional[float], Optional[float]] = (None, None),
    dry_run: bool = False,
    verbose: bool = False,
    nproc: Optional[int] = None,
    output_directory: Optional[str] = None,
) -> None:
    """
    High-level parallel runner that dispatches conversion of multiple outputs.

    Parameters:
    - output_numbers: List of snapshot numbers to convert.
    - input_folder: Input directory containing RAMSES outputs.
    - output_prefix: Prefix for output files.
    - fields: Optional list of scalar/vector fields to export.
    - level_start, level_end: Optional AMR level filtering.
    - x_range_norm, y_range_norm, z_range_norm: Optional spatial filtering ranges (normalized).
    - dry_run: If True, run without writing output files.
    - verbose: Enable detailed logging.
    - nproc: Number of CPU cores to use for parallel execution.
             If None or not provided, defaults to 1 (serial execution).
             When provided and positive, uses up to min(nproc, number of outputs) parallel workers.
    - output_directory: Optional directory path to store generated output files.
                        If None, defaults to input_folder.

    Behavior:
    - Attempts parallel execution using the specified number of workers.
    - On parallel execution failure, falls back to serial processing per output,
      continuing on errors without stopping the entire process.

    Returns:
    - None
    """

    if nproc is not None and nproc > 0:
        nworkers = min(nproc, len(output_numbers))
    else:
        nworkers = 1

    logger.info("Starting on %d worker(s) for outputs %s", nworkers, output_numbers)
    t0 = time.time()

    worker = partial(
        process_single_output,
        input_folder=input_folder,
        output_prefix=output_prefix,
        fields=fields,
        level_start=level_start,
        level_end=level_end,
        x_range_norm=x_range_norm,
        y_range_norm=y_range_norm,
        z_range_norm=z_range_norm,
        dry_run=dry_run,
        verbose=verbose,
        output_directory=output_directory,
    )

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=nworkers) as ex:
            list(ex.map(worker, output_numbers))
    except Exception as e:
        logger.error("Parallel execution failed: %s", e)
        logger.info("Falling back to serial execution...")

        for num in output_numbers:
            try:
                worker(num)
            except Exception as ew:
                logger.exception("Serial worker failed for output %s: %s", num, ew)

    logger.info("Total elapsed: %.2fs", time.time() - t0)

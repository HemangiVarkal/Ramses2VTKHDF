#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Parallel execution utilities for ramses_to_vtkhdf.

"""

# ─────────────────────────────────────────────────────────────────────────────
# Library imports
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from functools import partial
from multiprocessing import cpu_count
from typing import List, Optional, Tuple

import logging
import sys
import time
import concurrent.futures

from .converter import RamsesToVtkHdfConverter, setup_logging

logger = logging.getLogger("ramses2vtkhdf")


# ─────────────────────────────────────────────────────────────────────────────
# Parallel driver
# ─────────────────────────────────────────────────────────────────────────────

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
) -> None:
    
    """
    Worker function executed in each process. It configures logging and runs conversion
    for a single snapshot number.
    """
    
    setup_logging(verbose)
   
    try:
        conv = RamsesToVtkHdfConverter(
            input_folder=input_folder,
            output_prefix=output_prefix,
            fields=fields,
            level_start=level_start,
            level_end=level_end,
            x_range_norm=x_range_norm,
            y_range_norm=y_range_norm,
            z_range_norm=z_range_norm,
            dry_run=dry_run,
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
) -> None:
    
    """
    High-level parallel runner that dispatches conversion of multiple outputs.

    If parallel execution fails, falls back to serial execution and continues on per-snapshot errors.
    """
    
    nworkers = max(1, min(cpu_count(), len(output_numbers)))
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
    )

    # Try parallel execution first
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=nworkers) as ex:
            # ex.map will propagate exceptions from worker; worker itself logs and swallows; so this should be safe
            list(ex.map(worker, output_numbers))
    except Exception as e:
        logger.error("Parallel execution failed: %s", e)
        logger.info("Falling back to serial execution...")
        
        for num in output_numbers:
            try:
                worker(num)
            except Exception as ew:
                # Worker should already catch; this is a last-resort guard
                logger.exception("Serial worker failed for output %s: %s", num, ew)

    logger.info("Total elapsed: %.2fs", time.time() - t0)

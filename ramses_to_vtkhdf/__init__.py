# -*- coding: utf-8 -*-

"""

RAMSES → VTKHDF (Overlapping AMR) Converter
===========================================

Author: Hemangi C. Varkal

Affiliation: Space Applications Centre, ISRO, Ahmedabad, Gujarat, India.

──────────────────────────────────────────────────────────────────────────────
Description
──────────────────────────────────────────────────────────────────────────────
This script converts RAMSES simulation outputs into the VTKHDF format that
ParaView and other VTK-based tools can read.

──────────────────────────────────────────────────────────────────────────────
WHY THIS EXISTS
──────────────────────────────────────────────────────────────────────────────
- RAMSES uses an Adaptive Mesh Refinement (AMR) grid that’s great for HPC, but
  not directly plug-and-play for most visualization pipelines.
- VTKHDF’s **OverlappingAMR** stores data by refinement level, with cell-centered
  fields and AMR indexing metadata. That’s perfect for ParaView.

"""

from .converter import (
    RamsesToVtkHdfConverter,
    parse_output_numbers,
    parse_norm_range,
    parse_fields_arg,
    list_fields_for_snapshot,
)

from .parallel import (
    process_single_output,
    run_parallel_conversion,
)

__version__ = "1.0.0"
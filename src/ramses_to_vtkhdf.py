#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

RAMSES → VTKHDF (Overlapping AMR) Converter
===========================================

Author: Hemangi Varkal

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

──────────────────────────────────────────────────────────────────────────────
IT SUPPORTS:
──────────────────────────────────────────────────────────────────────────────
 - Vectorized spatial filtering with normalized ranges (--x-range/--y-range/--z-range)
 - Per-snapshot parallel processing (ProcessPoolExecutor)
 - Field discovery (--list-fields) and explicit field selection (--fields)
 - Dry-run mode (--dry-run) to print the plan without writing files
 - Embedding run metadata (CLI command, timestamp, code version) in each output file

──────────────────────────────────────────────────────────────────────────────
CLI QUICK START (copy–paste, then tweak)
──────────────────────────────────────────────────────────────────────────────
Example run (filters a subvolume, includes extra fields):

    python3 ramses_to_vtkhdf.py \
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
    python3 ramses_to_vtkhdf.py --base-dir ./simulations --folder-name output_dir \
        -n 5 --list-fields

    # Dry-run: show counts after filters, but don’t write .vtkhdf
    python3 ramses_to_vtkhdf.py --base-dir ./simulations --folder-name output_dir \
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


# ─────────────────────────────────────────────────────────────────────────────
# Library imports
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

# Standard Library Imports
import argparse
import logging
import os
import shlex
import sys
import time
from functools import partial
from multiprocessing import cpu_count
from typing import Dict, Iterable, List, Optional, Tuple

import concurrent.futures

# External Libraries
import numpy as np
import h5py as h5
import osyris

# Tooling/version
__version__ = "1.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(verbose: bool) -> None:
    
    """
    Configure global logging.

    Args:
        verbose: If True, set DEBUG level, otherwise INFO.
    """
    
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Reduce noise from libraries (adjustable)
    logging.getLogger("h5py").setLevel(logging.WARNING)

logger = logging.getLogger("ramses2vtkhdf")


# ─────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_output_numbers(arg: str) -> List[int]:
    
    """
    Parse output numbers strings like '5', '1,3,5', or '2-7' into a list of ints.

    Args:
        arg: user-provided string

    Returns:
        List of ints representing snapshot/output numbers.

    Raises:
        argparse.ArgumentTypeError on invalid format.
    """
    
    if "-" in arg and "," in arg:
        raise argparse.ArgumentTypeError("Do not mix ranges and lists; use either 'a-b' or 'a,b,c'.")
    
    if "-" in arg:
        try:
            start, end = map(int, arg.split("-", 1))
        except ValueError:
            raise argparse.ArgumentTypeError("Invalid range; use 'start-end'.")
        if end < start:
            raise argparse.ArgumentTypeError("Range end must be >= start.")
        return list(range(start, end + 1))
    
    if "," in arg:
        nums = []
        for x in arg.split(","):
            x = x.strip()
            if x == "":
                continue
            try:
                nums.append(int(x))
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid integer in list: '{x}'")
        return nums
    
    try:
        return [int(arg)]
    except ValueError:
        raise argparse.ArgumentTypeError("Output number must be an integer.")


def parse_norm_range(arg: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    
    """
    Parse normalized axis range spec: 'min:max', ':max', 'min:', or ':'.

    Returns (min_norm, max_norm) where each entry is a float in [0,1], or None when not provided.

    Args:
        arg: string or None
    """
    
    if arg is None:
        return (None, None)
    
    s = arg.strip()
    
    if s == "":
        return (None, None)
   
    if ":" not in s:
        raise argparse.ArgumentTypeError("Axis range must be 'min:max' (e.g., 0.2:0.8, :0.6, 0.1:, :).")
    
    left, right = s.split(":", 1)
    minv = float(left) if left.strip() != "" else 0.0
    maxv = float(right) if right.strip() != "" else 1.0
    
    if not (0.0 <= minv <= 1.0 and 0.0 <= maxv <= 1.0):
        raise argparse.ArgumentTypeError("Axis normalized bounds must be within [0, 1].")
    
    if minv > maxv:
        raise argparse.ArgumentTypeError("Axis min cannot be greater than axis max.")
    
    return (minv, maxv)


def parse_fields_arg(arg: Optional[str]) -> Optional[List[str]]:
    
    """
    Parse the --fields argument which is a comma-separated list of field names.

    Returns None if user didn't pass anything (means use defaults or auto-detect).
    """
    
    if arg is None:
        return None
    
    fields = [f.strip() for f in arg.split(",") if f.strip() != ""]
    
    return fields if fields else None


# ─────────────────────────────────────────────────────────────────────────────
# Core converter
# ─────────────────────────────────────────────────────────────────────────────

class RamsesToVtkHdfConverter:

    """
    Convert RAMSES (osyris) outputs into VTKHDF OverlappingAMR files.

    This class is intentionally small and unit-test friendly: separate methods perform
    small responsibilities (boxlength inference, mask building, vector extraction, HDF5 writes).
    """
    
    def __init__(
        self,
        input_folder: str,
        output_prefix: str = "overlapping_amr",
        fields: Optional[List[str]] = None,
        level_start: Optional[int] = None,
        level_end: Optional[int] = None,
        x_range_norm: Tuple[Optional[float], Optional[float]] = (None, None),
        y_range_norm: Tuple[Optional[float], Optional[float]] = (None, None),
        z_range_norm: Tuple[Optional[float], Optional[float]] = (None, None),
        dry_run: bool = False,
    ):
        
        # Inputs & configuration
        self.input_folder = input_folder
        self.output_prefix = output_prefix

        # User-requested fields (None = auto-detect later)
        self.requested_fields = fields

        # Level bounds (None => no filter)
        self.level_start = level_start
        self.level_end = level_end

        # normalized ranges (None/None => no filter)
        self.x_range_norm = x_range_norm
        self.y_range_norm = y_range_norm
        self.z_range_norm = z_range_norm

        self.dry_run = dry_run

        # dtype aliases for HDF5 datasets
        self.float_dtype = "f"  # float32
        self.int_dtype = "i8"

        # physical bounds converted after boxlength known (dict with 'x','y','z' -> (lo,hi) or None)
        self._phys_bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None

    
    # --------------- boxlength & bound conversion ---------------

    def _infer_boxlength_from_data(self, data) -> Optional[float]:
        
        """
        Try to infer simulation box length from osyris dataset `data`.

        Returns:
            box length as float, or None if not found.
        """
        
        # Common location in osyris metadata
        try:
            if hasattr(data, "meta") and isinstance(data.meta, dict) and "boxlen" in data.meta:
                return float(data.meta["boxlen"])
        except Exception:
            # Code stops here if boxlength is not found
            logger.debug("Error while reading data.meta for boxlen")
            pass
                         
        # Not found
        return None

        
    # --------------- Computation of Spatial Bounds ---------------

    def _compute_physical_bounds(self, data) -> None:
        
        """
        Convert normalized filters to physical coordinates using inferred box length.

        If any axis filter is requested but boxlength cannot be inferred, a ValueError is raised.
        """
        
        # Decide whether any filter was requested
        any_filter = any(
            [
                self.x_range_norm != (None, None),
                self.y_range_norm != (None, None),
                self.z_range_norm != (None, None),
            ]
        )
        
        if not any_filter:
            # No filtering requested; set phys bounds to None and exit
            self._phys_bounds = None
            return

        boxlen = self._infer_boxlength_from_data(data)
        
        if boxlen is None:
            # Warn early: spatial filters requested but boxlen not available
            logger.warning(
                "Spatial filtering was requested but boxlength could not be inferred from dataset. "
                "Filtering will fail. Ensure your RAMSES/osyris data includes box length metadata."
            )
            # Raise so caller knows they explicitly requested filtering that cannot be satisfied
            raise ValueError(
                "Spatial filtering requested but boxlength could not be inferred from dataset. "
                "Ensure your RAMSES/osyris data includes box length metadata."
            )

        # Convert normalized ranges to physical (0..boxlen)
        def conv(r: Tuple[Optional[float], Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
            if r == (None, None):
                return (None, None)
            
            lo, hi = r
            
            return (lo * boxlen, hi * boxlen)

        xb = conv(self.x_range_norm)
        yb = conv(self.y_range_norm)
        zb = conv(self.z_range_norm)

        self._phys_bounds = {"x": xb, "y": yb, "z": zb}

        # Info logs so users know exactly what filtering was applied
        logger.info("Using boxlength = %s", boxlen)

        if xb != (None, None):
            logger.info("x filter (physical): [%.6g, %.6g]", xb[0], xb[1])
        if yb != (None, None):
            logger.info("y filter (physical): [%.6g, %.6g]", yb[0], yb[1])
        if zb != (None, None):
            logger.info("z filter (physical): [%.6g, %.6g]", zb[0], zb[1])

    
    # --------------- Loading Data ---------------

    def read_data(self, output_num: int):
        
        """
        Load a RAMSES snapshot using osyris.RamsesDataset and return the loaded dataset.
        On failure, logs the error and returns None (so caller can handle).
        """
        
        try:
            ds = osyris.RamsesDataset(output_num, path=self.input_folder).load()
            return ds
        except Exception as e:
            logger.error("Failed to load output %s from '%s': %s", output_num, self.input_folder, e)
            logger.debug("Exception details:", exc_info=True)
            return None


    # --------------- Conversion Pipeline ---------------

    def _filter_levels(self, levels: Iterable[int]) -> List[int]:
        
        """
        Apply level_start/level_end filters to the list/iterable of levels.
        """
        
        levels = list(levels)
        
        if self.level_start is not None:
            levels = [lvl for lvl in levels if lvl >= self.level_start]
        
        if self.level_end is not None:
            levels = [lvl for lvl in levels if lvl <= self.level_end]
        
        return levels


    def _build_mask(self, px: np.ndarray, py: np.ndarray, pz: np.ndarray) -> np.ndarray:
        
        """
        Build boolean mask to apply spatial filtering.

        If no physical bounds are set (self._phys_bounds is None) returns all True mask.
        """
        
        if self._phys_bounds is None:
            return np.ones_like(px, dtype=bool)

        mask = np.ones_like(px, dtype=bool)
        
        (xmin, xmax) = self._phys_bounds["x"]
        (ymin, ymax) = self._phys_bounds["y"]
        (zmin, zmax) = self._phys_bounds["z"]

        if xmin is not None and xmax is not None:
            mask &= (px >= xmin) & (px <= xmax)
        if ymin is not None and ymax is not None:
            mask &= (py >= ymin) & (py <= ymax)
        if zmin is not None and zmax is not None:
            mask &= (pz >= zmin) & (pz <= zmax)

        return mask

        
    def _extract_vector(self, vec_field) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        """
        Extract vector components x,y,z from a vector-like osyris field.

        Tries an efficient fast path (vec_field.x.values / vec_field['x'].values) and falls back
        to safe iteration if necessary.

        Returns:
            tuple of 1D numpy arrays (vx, vy, vz)
        """
        
        # Fast path 1: mapping access (common in structured array style)
        try:
            # try dict/column style first
            if hasattr(vec_field, "__getitem__"):
                # Many wrappers expose fields as vec_field['x'].values
                try:
                    vx = np.asarray(vec_field["x"].values, dtype=float)
                    vy = np.asarray(vec_field["y"].values, dtype=float)
                    vz = np.asarray(vec_field["z"].values, dtype=float)
                    return vx, vy, vz
                except Exception:
                    # ignore and try attribute style next
                    pass
            
            # Fast path 2: attribute style vec_field.x.values
            try:
                vx = np.asarray(vec_field.x.values, dtype=float)
                vy = np.asarray(vec_field.y.values, dtype=float)
                vz = np.asarray(vec_field.z.values, dtype=float)
                return vx, vy, vz
            except Exception:
                # fall through to safe iteration
                pass
        except Exception:
            # defensive: do not fail here; go to fallback
            logger.debug("Fast path vector extraction failed; using safe fallback.", exc_info=True)

        # Safe fallback: iterate and extract component values (guaranteed to work)
        vx_list = []
        vy_list = []
        vz_list = []
        
        for v in vec_field:
            # Each `v` might be an object with attributes x,y,z (which themselves may have `.values`)
            try:
                # v.x.values is common in osyris
                vx_list.append(float(v.x.values))
                vy_list.append(float(v.y.values))
                vz_list.append(float(v.z.values))
            except Exception:
                # fallback to attribute numeric value (if any), or raise if totally unexpected
                try:
                    vx_list.append(float(v.x))
                    vy_list.append(float(v.y))
                    vz_list.append(float(v.z))
                except Exception as ee:
                    logger.debug("Vector element extraction failed for element %r: %s", v, ee)
                    raise RuntimeError("Unable to extract vector components from osyris field; incompatible format.")
                    
        return np.asarray(vx_list, dtype=float), np.asarray(vy_list, dtype=float), np.asarray(vz_list, dtype=float)

        
    # --------------- Writing Data ---------------
        
    def _write_level_to_hdf5(self, level_group, spacing: float, corner_coords: np.ndarray, scalars_dict: Dict[str, np.ndarray], vectors_dict: Dict[str, np.ndarray]) -> None:
        
        """
        Write one AMR level group to HDF5 following VTKHDF OverlappingAMR expectations.

        Args:
            level_group: h5py Group for level (already created)
            spacing: cell spacing (dx) for this level
            corner_coords: Nx3 array of cell corner coordinates (min corner)
            scalars_dict: mapping name -> 1D array (length N) of scalar values
            vectors_dict: mapping name -> Nx3 array of vector components
        """
        
        # Spacing attribute (three components for XYZ)
        level_group.attrs.create("Spacing", [spacing, spacing, spacing], dtype=self.float_dtype)
        level_group.attrs["NumberOfBlocks"] = len(corner_coords)

        # Build AMRBox: degenerate per-cell boxes encoded as [iMin, iMax, jMin, jMax, kMin, kMax]
        inv = 1.0 / spacing
        corner_coords = np.asarray(corner_coords)
        i_min = (corner_coords[:, 0] * inv).astype(np.int32)
        j_min = (corner_coords[:, 1] * inv).astype(np.int32)
        k_min = (corner_coords[:, 2] * inv).astype(np.int32)
        amr_values = np.column_stack([i_min, i_min, j_min, j_min, k_min, k_min]).astype(np.int32)
        level_group.create_dataset("AMRBox", data=amr_values)

        # CellData group for scalars & vectors
        celldata = level_group.create_group("CellData")

        # Scalars: store as Nx1 arrays (VTK expects 2D with second dimension 1)
        for name, arr in scalars_dict.items():
            # reshape to (N,1)
            celldata.create_dataset(name, data=arr.reshape(-1, 1).astype(self.float_dtype))

        # Vectors: store as Nx3 arrays
        for name, arr in vectors_dict.items():
            celldata.create_dataset(name, data=arr.astype(self.float_dtype))

        # Create placeholder groups (VTK readers expect them)
        level_group.create_group("PointData")
        level_group.create_group("FieldData")


    def _collect_fields_from_mesh(self, mesh) -> List[str]:
        
        """
        Try to collect available field names from the mesh object.
        This attempts multiple access patterns to be robust across osyris versions.
        """
    
        # mesh often behaves like a pandas DataFrame or a dict-like mapping
        try:
            keys = list(mesh.keys())
            return keys
        except Exception as e:
            logger.error("Cannot extract fields from mesh: %s", e)
            return []
     

    def convert_one(self, output_num: int, data) -> None:
        
        """
        Convert one loaded dataset to VTKHDF and write to disk (unless dry-run).

        Args:
            output_num: snapshot number
            data: object returned by osyris.RamsesDataset(...).load()
        """
        
        if data is None:
            logger.warning("No data for output %s; skipping.", output_num)
            return

         # Verify mesh presence
        if "mesh" not in data:
            logger.error("Dataset %s does not contain a 'mesh' key; skipping.", output_num)
            return

        mesh = data["mesh"]
        
        # Compute physical bounds if spatial filtering requested (may raise ValueError)
        try:
            self._compute_physical_bounds(data)
        except ValueError as e:
            # bubble up as warning + stop this snapshot (user explicitly requested filter but cannot compute)
            logger.error("Skipping output %s: %s", output_num, e)
            return
            
        # Track fields not found anywhere in this dataset    
        missing_fields = set()

        # Optional: if requested_fields is None, pick sensible defaults (density, pressure, velocity)
        if self.requested_fields is None:
            # Default fields commonly present in RAMSES outputs
            self.requested_fields = ["density", "pressure", "velocity"]

        # Determine integer AMR levels present and filter them
        try:
            levels = sorted(np.unique(mesh["level"].values))
        except Exception:
            # fallback if mesh indexing differs
            try:
                levels = sorted(np.unique(np.asarray([row["level"] for row in mesh])))
            except Exception as e:
                logger.error("Unable to determine AMR levels for output %s: %s", output_num, e)
                logger.debug("mesh sample: %r", mesh)
                return

        levels = self._filter_levels(levels)
        logger.info("Levels to write for output %s: %s", output_num, levels)

        # Dry-run: only print plan
        if self.dry_run:
            logger.info(
                "[dry-run] Would write file '%s_%05d.vtkhdf' with fields: %s (levels: %s), if available.",
                self.output_prefix,
                output_num,
                self.requested_fields,
                levels,
            )
            return

        # Build filename and begin writing
        t0 = time.time()
        output_filename = f"{self.output_prefix}_{output_num:05d}.vtkhdf"

        # Container to hold processed level data
        level_data = []
        
        try:   
            for new_level, actual_level in enumerate(levels):
                level_sel = mesh[mesh["level"].values == actual_level]

                # spacing (dx) assumed uniform within a level
                dx_arr = level_sel["dx"].values
               
                if len(dx_arr) == 0:
                    logger.debug("Level %s has zero cells; skipping.", actual_level)
                    continue
                
                spacing = float(dx_arr[0])

                # positions → arrays (fast extraction)
                pos = level_sel["position"]
                px = np.asarray([float(p.x.values) for p in pos])
                py = np.asarray([float(p.y.values) for p in pos])
                pz = np.asarray([float(p.z.values) for p in pos])

                logger.debug("Level %s: %d cells before filtering", actual_level, len(px))

                # mask for spatial filtering (vectorized)
                mask = self._build_mask(px, py, pz)
                kept = int(np.count_nonzero(mask))
                logger.debug("Level %s: %d cells after spatial filtering", actual_level, kept)
                if kept == 0:
                    # nothing to write at this level
                    continue

                # corner coords (min corner per cell)
                l = level_sel["dx"].values.astype(float)
            
                # apply mask to l (reduce memory before stacking)
                l = l[mask]
                cx = px[mask] - l / 2.0
                cy = py[mask] - l / 2.0
                cz = pz[mask] - l / 2.0
                corner_coords = np.column_stack([cx, cy, cz])

                # Build scalars & vectors dictionaries based on requested_fields
                scalars: Dict[str, np.ndarray] = {}
                vectors: Dict[str, np.ndarray] = {}
   
                # Helper to safely read a scalar field if available
                def get_scalar(name: str) -> Optional[np.ndarray]:
                    try:
                        if name in level_sel:
                            arr = np.asarray(level_sel[name].values, dtype=float)[mask]
                            return arr
                    except Exception:
                        logger.debug("Scalar field '%s' reading failed on level %s", 
                                     name, 
                                     actual_level, 
                                     exc_info=True
                                    )
                    return None
                    
                # Helper to safely read a vector field if available
                def get_vector(name: str) -> Optional[np.ndarray]:
                    try:
                        if name in level_sel:
                            vf = level_sel[name]
                            vx, vy, vz = self._extract_vector(vf)
                            vec = np.column_stack([vx[mask], vy[mask], vz[mask]])
                            return vec
                    except Exception:
                        logger.debug("Vector field '%s' reading failed on level %s", 
                                     name, 
                                     actual_level, 
                                     exc_info=True
                                    )
                    return None
                    
                # Core + optional fields
                for fieldname in self.requested_fields:
                    if fieldname in {"density", "pressure", "grav_potential"}:
                        arr = get_scalar(fieldname)
                        if arr is not None:
                            scalars[fieldname.capitalize()] = arr
                        else:
                            missing_fields.add(fieldname)
                            
                    elif fieldname in {"velocity", "grav_acceleration", "magnetic_field"}:
                        vec = get_vector(fieldname)
                        if vec is not None and vec.size:
                            vectors[fieldname.replace("_", " ").title().replace(" ", "_")] = vec
                        else:
                            missing_fields.add(fieldname)
                            
                    else:
                        # Any other user-provided field
                        arr = get_scalar(fieldname)
                        if arr is not None:
                            scalars[fieldname] = arr
                        else:
                            vec = get_vector(fieldname)
                            if vec is not None:
                                vectors[fieldname] = vec
                            else:
                                missing_fields.add(fieldname)

                # Save this level’s results if it has *any* data
                if scalars or vectors:
                    level_data.append((new_level, spacing, corner_coords, scalars, vectors))

            # If no levels survived → skip file creation
            if not level_data:
                # After all levels: report missing fields (once per output)
                for mf in sorted(missing_fields):
                    logger.warning(
                    "Requested field '%s' not found in output %s; skipping.",
                        mf,
                        output_num,
                        )
                logger.warning("Skipping output %s: no file created (all cells filtered out or requested fields missing).", output_num)
                return
            
            with h5.File(output_filename, "w") as f:
                # Root group for VTKHDF - follow expected attributes for OverlappingAMR
                root = f.create_group("VTKHDF", track_order=True)
                root.attrs["Version"] = (2, 2)
                root.attrs.create(
                    "Type", b"OverlappingAMR", dtype=h5.string_dtype("ascii", len(b"OverlappingAMR"))
                )
                root.attrs.create(
                    "GridDescription", b"XYZ", dtype=h5.string_dtype("ascii", len(b"XYZ"))
                )
                root.attrs.create(
                    "Origin", [0.0, 0.0, 0.0], dtype=self.float_dtype
                )
                root.attrs["NumberOfLevels"] = len(level_data)

                # Embed metadata about the conversion run (command used, timestamp, version)
                root.attrs["generator_command"] = shlex.join(sys.argv)
                root.attrs["generator_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                root.attrs["generator_version"] = __version__

                # Actually write each prepared level
                for new_level, spacing, corner_coords, scalars, vectors in level_data:
                    # Write level group
                    level_group = root.create_group(f"Level{new_level}")
                    self._write_level_to_hdf5(
                        level_group=level_group,
                        spacing=spacing,
                        corner_coords=corner_coords,
                        scalars_dict=scalars,
                        vectors_dict=vectors,
                    )

            # Completed writing this file
            logger.info("DONE: Saved '%s_%05d.vtkhdf' in %.2fs", 
                        self.output_prefix, 
                        output_num, 
                        time.time() - t0
                       )
        except Exception as e:
            # log full exception for debugging but continue processing other snapshots
            logger.exception("Failed to convert/write output %s: %s", output_num, e)

    
    # --------------- External entry points ---------------

    def process_output(self, output_num: int) -> None:
        
        """
        Read and convert a single output (load + convert_one).
        This wrapper isolates exceptions so callers (parallel runner) can continue on failure.
        """
        
        data = self.read_data(output_num)

        try:
            self.convert_one(output_num, data)
        except Exception as e:
            # convert_one should generally handle errors gracefully; this is a last-resort guard
            logger.exception("Unexpected failure converting output %s: %s", output_num, e)


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


# ─────────────────────────────────────────────────────────────────────────────
# Utility: list fields
# ─────────────────────────────────────────────────────────────────────────────

def list_fields_for_snapshot(input_folder: str, output_num: int) -> List[str]:
    
    """
    Load one snapshot and return a best-effort list of available fields found in mesh.

    This function is tolerant: when structures differ across osyris versions it will attempt
    a few strategies and return the union of discovered names.
    """
    
    ds = None
    try:
        ds = osyris.RamsesDataset(output_num, path=input_folder).load()
    except Exception as e:
        logger.error("Failed to load output %s for field listing: %s", output_num, e)
        return []

    if ds is None or "mesh" not in ds:
        logger.warning("No 'mesh' found in snapshot %s; cannot list fields.", output_num)
        return []

    mesh = ds["mesh"]
    
    conv = RamsesToVtkHdfConverter(input_folder=input_folder)
    
    fields = conv._collect_fields_from_mesh(mesh)
    
    # Provide a short, sane list (unique)
    unique_fields = []
    
    for f in fields:
        if f not in unique_fields:
            unique_fields.append(f)
    return unique_fields


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()

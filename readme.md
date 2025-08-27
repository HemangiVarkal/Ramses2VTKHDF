# 🚀 RAMSES → VTKHDF (Overlapping AMR) Converter 

**Author:** Hemangi Varkal  
**Affiliation:** Space Applications Centre, ISRO, Ahmedabad, Gujarat, India  
**Version:** 1.0.0  

---

## 📃 Overview 

This Python-based converter transforms **RAMSES simulation outputs** into **VTKHDF OverlappingAMR files**, ready for visualization in **ParaView** and other VTK-compatible tools.  

RAMSES uses **Adaptive Mesh Refinement (AMR)**, which is excellent for high-performance simulations but not directly compatible with standard visualization pipelines. This script bridges that gap, enabling you to convert hierarchical AMR meshes into VTKHDF format while preserving spatial refinement, scalar fields, and vector fields.  

---

## ✨ Key Features 

- **Vectorized Spatial Filtering**: Filter your simulation subvolume using normalized ranges `--x-range/--y-range/--z-range` (0–1 relative to box length).  
- **Parallel Conversion ⚡**: Process multiple snapshots in parallel using all available CPU cores.  
- **Field Management**:
  - List available fields with `--list-fields` 🔍  
  - Include only specific fields with `--fields density,velocity,…`  
- **Dry-Run Mode 👀**: Preview conversion plans without writing files.  
- **Metadata Embedding**: Stores CLI command, timestamp, and code version in output HDF5.  
- **AMR-Level Selection**: Convert only a subset of AMR levels with `--level-start` and `--level-end`.  

---

## 🛠️ Installation Requirements 

- Python ≥ 3.8  
- Packages: `numpy`, `h5py`, `osyris`  
- Optional: `concurrent.futures` (built-in for parallel processing)  

---

## Usage Guide

### Quick Start Example

Convert snapshots 1, 3, and 5–7, including density and pressure, within a subvolume:

```bash
python3 ramses_to_vtkhdf.py \
    --base-dir ./simulations \
    --folder-name output_dir \
    --numbers 1,3,5-7 \
    --output-prefix overlapping_amr \
    --level-start 2 --level-end 5 \
    --x-range 0.0:1.0 --y-range 0.0:1.0 --z-range 0.0:1.0 \
    --fields density,pressure \
    --verbose
```

---

### Exploration Modes

- ** 🔍 Field Inspection Mode **: List available fields in a snapshot:

```bash
python3 ramses_to_vtkhdf.py --base-dir ./simulations --folder-name output_dir -n 5 --list-fields
```

- ** 👀 Preview Mode (Dry-Run) **: Show counts and plan without writing files:

```bash
python3 ramses_to_vtkhdf.py --base-dir ./simulations --folder-name output_dir -n 5 --level-start 1 --dry-run --verbose
```

---

## CLI Arguments Overview

| Argument | Type | Required | Default | Notes |
|----------|------|----------|---------|-------|
| `--base-dir` | str | Yes | - | Base directory containing simulation folders |
| `--folder-name` | str | Yes | - | Folder inside base-dir with outputs |
| `-n / --numbers` | list[int] | Yes | - | Snapshot numbers like `1`, `1,3,5` or `2-7` |
| `-o / --output-prefix` | str | No | `overlapping_amr` | Prefix for output files |
| `--level-start` | int | No | None | Minimum AMR level (inclusive) |
| `--level-end` | int | No | None | Maximum AMR level (inclusive) |
| `--x-range` | min:max | No | None | Normalized X-axis range [0,1] |
| `--y-range` | min:max | No | None | Normalized Y-axis range [0,1] |
| `--z-range` | min:max | No | None | Normalized Z-axis range [0,1] |
| `--fields` | list[str] | No | Defaults | Comma-separated field names to include |
| `--list-fields` | flag | No | False | List fields and exit |
| `--dry-run` | flag | No | False | Preview plan without writing files |
| `--verbose` | flag | No | False | Enable detailed logging |

> ⚠️ **Note:** Normalized ranges are relative to the simulation box. `0.0:0.5` corresponds to the first half of the domain.  

---

## 🛠️ Conversion Pipeline Overview 

1. **Load Data**: Uses `osyris.RamsesDataset` to read snapshots.  
2. **Determine AMR Levels**: Detects all levels, applies `--level-start`/`--level-end` filters.  
3. **Spatial Filtering**: Converts normalized ranges to physical coordinates.  
4. **Field Extraction**: Separates scalar fields (density, pressure, etc.) and vector fields (velocity, magnetic_field).  
5. **Mask & Prepare Data**: Apply filtering mask; compute cell corner coordinates.  
6. **HDF5 Writing**:  
   - Writes each AMR level as a group.  
   - Stores `CellData` with scalars and vectors.  
   - Includes placeholder `PointData` and `FieldData` groups for VTK compatibility.  
   - Embeds metadata: command, timestamp, version.  

> ⚠️ **Important:** AMRBox stores degenerate per-cell blocks. VTK expects this format for OverlappingAMR. Do **not** collapse boxes unless familiar with VTK’s data structure.  

---

## 🗓️ Notes & Best Practices 

- Default fields if `--fields` is not specified: `density`, `pressure`, `velocity`.  
- Use `--dry-run` to confirm filter settings and snapshot selection before committing to disk.  
- Logging: Verbose mode (`--verbose`) gives step-by-step information including number of cells retained per level.  
- Parallel execution automatically uses available CPU cores; falls back to serial if needed.  
- If no cells survive filtering or fields are missing, the output file is skipped, with warnings logged.  

---

## 📝 License & Acknowledgements 

- Script developed at **Space Applications Centre, ISRO, Ahmedabad, Gujarat, India.**.  
- Proper citation recommended when used in publications.  
- Adapted to VTKHDF OverlappingAMR following ParaView conventions.


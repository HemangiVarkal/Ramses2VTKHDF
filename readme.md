# Ramses2VTKHDF: A Python tool for converting RAMSES outputs to VTKHDF

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-brightgreen)]()
[![Build Status](https://img.shields.io/badge/tests-passing-brightgreen)]()


## Overview

**Ramses2VTKHDF** is a Python package that converts **[RAMSES](https://ramses-organisation.readthedocs.io/en/latest/) simulation outputs** into
**[VTKHDF](https://vtk.org/documentation/) OverlappingAMR** format.  

It provides both a **command-line interface (CLI)** and a **Python API**, making it
easy to visualize an analyze in [ParaView](https://docs.paraview.org/en/latest/) and other compatible tools.

The package also includes test coverage, example scripts, and clear documentation,
ensuring it is reproducible and accessible for scientific use.

---

## Features

- Convert RAMSES AMR outputs into VTKHDF OverlappingAMR files
- Uses [Osyris](https://osyris.readthedocs.io/en/stable/) for data analysis and extraction
- Support for both **scalar fields** (density, pressure, grav_potential) and
  **vector fields** (velocity, magnetic_field, grav_acceleration)
- Dry-run mode to preview what would be written without creating files
- CLI and Python API for flexible use
- Parallel conversion support for multiple outputs
- Fully tested with `pytest`

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/HemangiVarkal/Ramses2VTKHDF.git
cd Ramses2VTKHDF
pip install -r requirements.txt
```

---

## Usage

### Command-Line Interface (CLI)

Example:

```bash
python -m ramses_to_vtkhdf.cli  --base-dir ramses_outputs/    
        --folder-name sedov_3d/  -n 1   /
        --output-prefix sedov_test     /
        --fields density,velocity,pressure  --dry-run
```

Key options:
- `--base-dir` → Parent folder   
- `--folder-name` → Folder containing RAMSES outputs 
- `-n` → Snapshot number(s)  
- `--output-prefix` → Prefix for generated `.vtkhdf` files  
- `--fields` → Comma-separated list of fields (scalars/vectors)  
- `--dry-run` → Run without writing files  

---

### Python API

```python
from ramses_to_vtkhdf.converter import RamsesToVtkHdfConverter

converter = RamsesToVtkHdfConverter(
    input_folder="ramses_outputs/sedov_3d",
    output_prefix="sedov_test",
    fields=["density", "velocity"],
    dry_run=True
)

converter.process_output(1)
```

---

### Example Script

An example is provided in `examples/example_usage.py`:

```bash
python -m examples.example_usage   
```

This script:
1. Lists available fields in snapshots  
2. Prints dataset info  
3. Performs a dry-run conversion  
4. Displays scalar/vector fields to be written  

---

## Output File Structure

```lua
<output-prefix>_00001.vtkhdf
<output-prefix>_00002.vtkhdf
...
```

---

## Tests

Run the test suite with:

```bash
pytest tests/
```

---

## Repository Structure

```
Ramses2VTKHDF/
├── ramses_to_vtkhdf/       # Core Python package
│   ├── __init__.py
│   ├── cli.py
│   ├── converter.py
│   └── parallel.py
│
├── tests/                  # Unit tests
│   ├── __init__.py
    ├── test_cli.py
│   ├── test_converter.py
│   ├── test_import.py
│   ├── test_parallel.py
│   └── test_parser.py
│
├── examples/               # Example usage scripts
│   └── example_usage.py
│
├── ramses_outputs/  # Sample real RAMSES outputs
│   └── sedov_3d/
│       ├── output_00001/
│       └── output_00002/
│
├── LICENSE
├── README.md
├── requirements.txt
└── .gitignore
```

---



## Notes & Best Practices 

- Default fields if `--fields` is not specified: `density`, `pressure`, `velocity`.    
- Logging: Verbose mode (`--verbose`) gives step-by-step information including number of cells retained per level.  
- Parallel execution automatically uses available CPU cores; falls back to serial if needed.  
- If no cells survive filtering or fields are missing, the output file is skipped, with warnings logged.  

---


## License

This project is licensed under the terms of the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

## Authors

- Hemangi Varkal (https://github.com/HemangiVarkal)

---

## Acknowledgements 

- Script developed at **Space Applications Centre, ISRO, Ahmedabad, Gujarat, India.**.  
- Adapted to VTKHDF OverlappingAMR following ParaView conventions.

---

## Citation

If you use **Ramses2VTKHDF** in your work, please cite it as:

> Varkal, H. (2025). *Ramses2VTKHDF: A Python tool for converting RAMSES outputs to VTKHDF*.  
> GitHub repository: https://github.com/HemangiVarkal/Ramses2VTKHDF

---

from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ramses-to-vtkhdf",
    version="1.0.0",
    author="Hemangi Varkal",
    author_email="hemangivarkal1612@gmail.com", 
    description="Convert RAMSES simulation outputs into VTKHDF Overlapping AMR format for visualization in ParaView and other VTK-based tools.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HemangiVarkal/Ramses2VTKHDF",
    install_requires=[
        "numpy>=1.20",
        "h5py>=3.0",
        "osyris>=1.2", 
    ],
    entry_points={
        "console_scripts": [
            "ramses-to-vtkhdf=ramses_to_vtkhdf.cli:main",
        ],
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False,
    keywords="RAMSES VTK VTKHDF astrophysics AMR visualization ParaView",
)

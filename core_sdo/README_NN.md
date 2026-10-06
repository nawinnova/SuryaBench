# Core SDO Machine Learning Ready Dataset Generation

## Description

This repository provides scripts to download **Atmospheric Imaging Assembly (AIA)** & **Helioseismic and Magnetic Imager(HMI)** data from **NASA's Solar Dynamics Observatory ([SDO](https://sdo.gsfc.nasa.gov/))** and preprocess it to create a **homogenized, ML–ready, core-sdo dataset** used in the training of the solar wind prediction model

The data is sourced from **Joint Science Operations Center ([JSOC](http://jsoc.stanford.edu/))**, and several preprocessing steps are applied to bring AIA and HMI data into Level 1.5, ready for machine learning applications.

- **Source Data:** JSOC Full Disk FITS files 
- **Cadence:** 60 minutes 
- **Channels:** 
  - **1 of 8 AIA wavelengths:** 94, 131, 171, 193, 211, 304, 335, 1600 Å 
  <!-- - **5 HMI variables:** LOS Magnetogram, LOS Dopplergram, Bx, By, Bz  -->
- **Purpose:** Training machine learning models on SDO observations.

---

##  Repository Structure

```bash
core_sdo/
│
├── core_sdo_download.py              # Script to download AIA/HMI FITS data via JSOC (modified for 1 passband)
├── core_sdo_mlready_processing_multi.py  # Preprocessing pipeline to create ML-ready NetCDF4 files
├── core_sdo_mlready_processing_1pb.py # (New-file) Preprocessing pipeline to create ML-ready NetCDF4 files for 1 passband 
├── helio.py                          # Utility functions for solar disk scaling, HMI vector conversion, etc.
├── plot_nc.py                        # Quicklook plotter for processed NetCDF4 files
├── requirements.txt                  # Python dependencies
├── README.md                         # Original Documentation
└── README_NN.md                      # Original Documentation
```


## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```
---

## Usage

### 1. Download a sample dataset from JSOC

Make sure to provide your email address in the script for JSOC access.
If it's your first time, JSOC will send a confirmation email — reply with "yes" to activate access.

```bash
cd core_sdo
python core_sdo_download.py
```

### 2. Process downloaded fits files (basic process)
```bash
cd core_sdo
python core_sdo_mlready_processing_1pb.py --superpixel 8 --superpixel-func mean

```
Several argument possible, check the python file or run 
```bash
python core_sdo_mlready_processing_1pb.py --help
```
### 3 Plot a processed maps in the NetCDF4 file
```bash
cd core_sdo
python plot_nc.py 'filename.nc'
```

## Contact (this version)
<!-- Dinesha Hegde, [dinesha.hegde@uah.edu] -->
Nawin Ngampoopun, [ngampoopun@mps.nmpg.de]


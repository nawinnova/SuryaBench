"""
Convert SDO/AIA FITS files to AI/ML-ready NetCDF4 (4K) files, ONE passband only. With optional superpixel binning.

Modified from core_sdo_mlready_processing_multi.py (originally 8 AIA + 5 HMI = 13 variables).
Each output NetCDF4 file now contains a single variable: aia<PASSBAND>.

Valid AIA passbands: 94, 131, 171, 193, 211, 304, 335, 1600 (Angstrom)

Usage:
    python core_sdo_mlready_processing_single.py --passband 193
    python core_sdo_mlready_processing_single.py --passband 171 --compression LZ4
    python core_sdo_mlready_processing_single.py --passband 193 --superpixel 4
    python core_sdo_mlready_processing_single.py --passband 193 --superpixel 16 --superpixel-func mean

Original author: Dinesha Hegde, dinesha.hegde@gmail.com
Modified by: Nawin Ngampoopun, MPS with help from Claude
"""

import os
import gc
import json
import argparse
import datetime as dt
from glob import glob

import numpy as np
import xarray as xr
import astropy.units as u
from astropy.io import fits
from sunpy.map import Map
import hdf5plugin  # for LZ4 compression / filter 32004

import helio

# CONFIG
VALID_PASSBANDS = (94, 131, 171, 193, 211, 304, 335, 1600)
DEFAULT_PASSBAND = 211

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FITS_PATH = os.path.join(BASE_DIR, "../core_sdo_fits")
DEFAULT_OUT_DIR = os.path.join(BASE_DIR, "../mlready_core_sdo_nc4")


# -----------------------------------------------------------------------------
def read_wavelength(fname):
    """Read WAVELNTH from the FITS header only (no image data loaded)."""
    for hdu in (1, 0):  # Level-1 AIA files are usually Rice-compressed (header in HDU 1)
        try:
            hdr = fits.getheader(fname, hdu)
        except (IndexError, OSError):
            continue
        if "WAVELNTH" in hdr:
            return int(hdr["WAVELNTH"])
    return None


def apply_superpixel(aia_map, superpixel, func_name="sum"):
    """
    Reduce the resolution of a sunpy Map by combining `superpixel` x `superpixel`
    blocks of pixels (see sunpy's "Resampling Maps" gallery example).

    - func_name='sum'  : sunpy default; each superpixel is the sum of its pixels.
    - func_name='mean' : each superpixel is the average (keeps DN/s units per pixel).
    """
    if superpixel == 1:
        return aia_map

    ny, nx = aia_map.data.shape
    if ny % superpixel != 0 or nx % superpixel != 0:
        raise ValueError(
            f"Image shape {aia_map.data.shape} is not divisible by superpixel size {superpixel}"
        )

    func = {"sum": np.sum, "mean": np.mean}[func_name]
    return aia_map.superpixel([superpixel, superpixel] * u.pixel, func=func)


def sdo_fits_to_mlready_netcdf_for_timestamp(
    fits_path,
    timestamp,
    passband,
    out_dir,
    nc_outfile=None,
    nc_compression_type="LZ4",
    superpixel=1,
    superpixel_func="sum",
):
    """
    Process the AIA FITS file for ONE passband at ONE timestamp and write a
    NetCDF4 (4K) file containing a single variable.

    Parameters:
    - fits_path: Path to the folder containing SDO FITS files.
    - timestamp: Timestamp string in 'YYYYMMDD_HHMM' format.
    - passband: AIA wavelength in Angstrom (e.g. 193).
    - out_dir: Output folder.
    - nc_outfile: Optional output NetCDF4 file path. If None, a default name is used.
    - nc_compression_type: 'LZ4' or 'zlib'.
    - superpixel: Superpixel size N (N x N pixels combined). 1 = no binning.
    - superpixel_func: 'sum' (sunpy default) or 'mean'.

    Returns:
    - None
    """

    # AIA files for this timestamp, then keep only the requested passband
    candidates = sorted(
        f
        for f in glob(os.path.join(fits_path, "aia*.fits"))
        if helio.compute_timestamp(os.path.basename(f)) == timestamp
    )
    aia_files = [f for f in candidates if read_wavelength(f) == passband]

    print(f"\n=== Processing timestamp {timestamp} (AIA {passband} A) ===")
    print("AIA files at this timestamp:", len(candidates))
    print(f"AIA {passband} A files:", len(aia_files))

    if len(aia_files) == 0:
        print(f"No AIA {passband} A FITS found for {timestamp}, skipping.")
        return
    if len(aia_files) > 1:
        print(f"Multiple AIA {passband} A files for {timestamp}; using {aia_files[0]}")

    fname = aia_files[0]

    if nc_outfile is None:
        sp_tag = f"_sp{superpixel}" if superpixel > 1 else ""
        nc_outfile = os.path.join(out_dir, f"{timestamp}_aia{passband}{sp_tag}.nc")

    data_arrays = {}
    encoding = {}

    # -------------------------------------------------------------------------
    # AIA (single passband)
    # -------------------------------------------------------------------------
    m = Map(fname)
    wavelnth = m.meta["wavelnth"]

    if m.meta["quality"] != 0:
        print(f"Bad AIA file {fname}, quality={m.meta['quality']} -> skipping this timestamp")
        return

    # original meta
    meta0 = m.meta

    # process AIA map
    aia_map = helio.process_aia_map(m)

    # optional superpixel binning (after Level-1.5 processing)
    if superpixel > 1:
        print(f"applying {superpixel}x{superpixel} superpixel ({superpixel_func}): "
              f"{aia_map.data.shape} -> ", end="")
        aia_map = apply_superpixel(aia_map, superpixel, superpixel_func)
        print(aia_map.data.shape)

    # processed meta (includes updated pixel scale / reference pixel if binned)
    meta1 = aia_map.meta

    # get data as float32, replace NaN with 0.0
    xdata = aia_map.data.astype(np.float32)
    np.nan_to_num(xdata, copy=False, nan=0.0)

    var_name = f"aia{wavelnth}"
    description = f"Level-1.5 AIA image for wavelength {wavelnth} Angstrom"
    if superpixel > 1:
        description += f", {superpixel}x{superpixel} superpixels ({superpixel_func})"

    # chunk size follows the (possibly binned) image shape
    chunks = tuple(int(n) for n in xdata.shape)

    print(f"add var: {var_name} (AIA) to data_arrays...")
    data_arrays[var_name] = xr.DataArray(
        xdata,
        name=var_name,
        dims=["y", "x"],
        attrs={
            "unit": "DN/s",
            "t_obs": meta0.get("t_obs", ""),
            "qflag": meta0.get("quality", 0),
            "description": description,
            "meta_0": json.dumps(meta0),
            "meta_1": json.dumps(meta1),
        },
    )

    # Apply compression
    if nc_compression_type == "zlib":
        encoding[var_name] = {"zlib": True, "complevel": 5, "chunksizes": chunks}
    else:  # default to LZ4
        encoding[var_name] = {
            "compression": 32004,  # LZ4
            "compression_opts": (),
            "chunksizes": chunks,
            "_FillValue": 0.0,
        }

    del xdata

    # sanity check
    if len(data_arrays) != 1:
        print(f"Expected 1 variable but got {len(data_arrays)} for {timestamp}. Skipping.")
        return

    # Global attributes
    attrs = {
        "title": (
            f"AI/ML ready level-1.5 SDO/AIA {passband} A data in 4K Resolution"
            if superpixel == 1
            else f"AI/ML ready level-1.5 SDO/AIA {passband} A data, {superpixel}x{superpixel} superpixels"
        ),
        "original author": (
            "Dinesha Hegde,dinesha.hegde@uah.edu;"
            "Sujit Roy, sujit.roy@uah.edu; "
            "Amy Lin, amy.lin@uah.edu"
        ),
        "institution": "NASA MSFC IMPACT Project; ESSC & CSPAR, University of Alabama in Huntsville",
        "modifier": "Nawin Ngampoopun, MPS",
        "data_time": timestamp,
        "passband": passband,
        "superpixel": superpixel,
        "superpixel_func": superpixel_func if superpixel > 1 else "none",
        "production_date": dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # wrap into Dataset
    ds = xr.Dataset(data_arrays, attrs=attrs)

    gc.collect()

    # write to NetCDF4 file
    ds.to_netcdf(
        path=nc_outfile,
        format="NETCDF4",
        engine="h5netcdf",
        encoding=encoding,
        mode="w",
    )

    ds.close()
    del ds

    print(f"made {nc_outfile}")


def process_multi_timestamps(fits_path, passband, out_dir, compression="zlib",
                             superpixel=1, superpixel_func="sum"):
    """
    Find all timestamps present in the AIA FITS files in this folder and build
    one single-passband NetCDF per timestamp.
    """
    aia_files = glob(os.path.join(fits_path, "aia*.fits"))
    timestamps = sorted({helio.compute_timestamp(os.path.basename(f)) for f in aia_files})
    print(f"Found {len(timestamps)} timestamps: {timestamps}")

    for ts in timestamps:
        sdo_fits_to_mlready_netcdf_for_timestamp(
            fits_path,
            ts,
            passband=passband,
            out_dir=out_dir,
            nc_outfile=None,
            nc_compression_type=compression,
            superpixel=superpixel,
            superpixel_func=superpixel_func,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-passband AIA ML-ready NetCDF4 processing")
    parser.add_argument("--passband", type=int, default=DEFAULT_PASSBAND, choices=VALID_PASSBANDS,
                        help=f"AIA wavelength in Angstrom (default: {DEFAULT_PASSBAND})")
    parser.add_argument("--fits-dir", default=DEFAULT_FITS_PATH, help="Folder with input FITS files")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Folder for output NetCDF files")
    parser.add_argument("--compression", default="zlib", choices=("zlib", "LZ4"),
                        help="NetCDF compression (default: zlib, as in the original script's main)")
    parser.add_argument("--superpixel", type=int, default=1,
                        help="Superpixel size N: combine N x N pixels (default: 1 = no binning)")
    parser.add_argument("--superpixel-func", default="sum", choices=("sum", "mean"),
                        help="How to combine pixels in a superpixel (default: sum, sunpy default)")
    args = parser.parse_args()

    if args.superpixel < 1:
        parser.error("--superpixel must be >= 1")

    if not os.path.exists(args.fits_dir):
        raise FileNotFoundError(f"FITS root folder not found: {args.fits_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    process_multi_timestamps(args.fits_dir, args.passband, args.out_dir, args.compression,
                             args.superpixel, args.superpixel_func)
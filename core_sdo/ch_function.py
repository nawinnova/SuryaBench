"""
Module for coronal hole detection and properties extraction. To use as potential input for SW forecasting model

Note: coronal hole parameters are A_CH and P_CH based on Ahn et al. (2026)
"""

import numpy as np
import astropy.units as u
import sunpy.map
from astropy.coordinates import SkyCoord
from sunpy.map.maputils import coordinate_is_on_solar_disk, all_coordinates_from_map
from skimage.filters import threshold_multiotsu
import copy
from scipy import ndimage

def segment_coord(map, lat_range, lon_range, coord_system=None):
    """
    Segment a region of interest in the solar map based on latitude and longitude ranges.

    Parameters
    ----------
    map : sunpy.map.Map
        The input solar map.
    lat_range : tuple
        A tuple specifying the latitude range (min_lat, max_lat) in degrees.
    lon_range : tuple
        A tuple specifying the longitude range (min_lon, max_lon) in degrees.

    Returns
    -------
    segmented_map : sunpy.map.Map
        A new solar map containing only the specified region of interest.
    """
    # Create a mask for the specified latitude and longitude ranges
    hpc_coords = all_coordinates_from_map(map)
    if coord_system is not None:
        hpc_coords = hpc_coords.transform_to(coord_system)
    
    lat_mask = (hpc_coords.lat >= lat_range[0]) & (hpc_coords.lat < lat_range[1])
    lon_mask = (hpc_coords.lon >= lon_range[0]) & (hpc_coords.lon < lon_range[1])
    roi_mask = ~(lat_mask & lon_mask)

    segmented_map = sunpy.map.Map(map.data, map.meta, mask = roi_mask)

    return segmented_map

def find_ch_otsu(map, show=False):
    """
    Find coronal holes using Otsu's thresholding method.

    Parameters
    ----------
    map : sunpy.map.Map
        The input solar map.
        
    Returns
    -------
    ch_mask : np.ndarray
        A boolean mask indicating the coronal hole regions.
    """
    # Get the data from the map
    data = copy.deepcopy(map.data)
    # select only on disk region
    hpc_coords = all_coordinates_from_map(map)
    data[~coordinate_is_on_solar_disk(hpc_coords)] = np.nan
    
    ## apply initial thresholding, such that CH intensity will never exceed median
    data[data > np.nanmedian(data)] = np.nan
    # Apply Otsu's thresholding method
    thresholds = threshold_multiotsu(data[~np.isnan(data)], classes=3, nbins=100)
    # Create a mask for coronal holes (regions below the first threshold)
    ch_mask = data < thresholds[0]
    ch_mask = ndimage.binary_opening(ch_mask)

    if show == True:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 5))
        plt.hist(data.flatten(), bins=100, color='gray', alpha=0.7)
        for thres in thresholds:
            plt.axvline(thres, linestyle='dashed', linewidth=2, label=f'Threshold = {thres:.2f}')   
        plt.legend()
        plt.show()

    return ch_mask

def find_ch_thres(map, thr='classic'):
    """
    Find coronal holes using a fixed threshold.

    Parameters
    ----------
    map : sunpy.map.Map
        The input solar map.
    thr : float
        The threshold value. Default is 'classic'. which is 0.35 * np.nanmedian(data)

    Returns
    -------
    ch_mask : np.ndarray
        A boolean mask indicating the coronal hole regions.
    """
    # Get the data from the map
    data = copy.deepcopy(map.data)
    # select only on disk region
    hpc_coords = all_coordinates_from_map(map)
    data[~coordinate_is_on_solar_disk(hpc_coords)] = np.nan

    # Create a mask for coronal holes (regions below the threshold)
    if thr == 'classic':
        thr = 0.35 * np.nanmedian(data)
        # print('median = ', np.nanmedian(data))
        # print('Threshold = ', thr)
    ch_mask = data < thr
    ch_mask = ndimage.binary_opening(ch_mask)

    return ch_mask

def calculate_chparam(ch_mask, segmented_map):
    """
    Calculate coronal hole parameters from the coronal hole mask and segmented map.

    Parameters
    ----------
    ch_mask : np.ndarray
        A boolean mask indicating the coronal hole regions.
    segmented_map : sunpy.map.Map
        The segmented solar map.

    Returns
    -------
    A_CH : float
        The area ratio of the coronal hole to the segmented area.
    P_CH : float
        The perimeter of the coronal hole.
    """

    # find overlap between segmented_map.mask and ch_area_otsu
    overlap_mask = np.logical_and(~segmented_map.mask, ch_mask)
    ## A_CH = ratio of CH area pixel to segmented area pixel
    A_CH = len(np.nonzero(overlap_mask)[0]) / len(np.nonzero(~segmented_map.mask)[0])
    
    ## P_CH = np.sum(1/b), where b is brightness of each pixel
    segmented_map_data = np.ma.array(segmented_map.data, mask=segmented_map.mask)
    segmented_map_data_1d = segmented_map_data.compressed()
    lower_bound = np.nanpercentile(segmented_map_data_1d, 5)
    # print(lower_bound)
    upper_bound = np.nanpercentile(segmented_map_data_1d, 95)
    # print(upper_bound)
    filtered_data = segmented_map_data_1d[(segmented_map_data_1d >= lower_bound) & (segmented_map_data_1d <= upper_bound)]
    P_CH = np.sum(1/filtered_data)

    return A_CH, P_CH

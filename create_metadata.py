import os
import gc
import time
import numpy as np
import xarray as xr
import pandas as pd

MASK_VALUE = 65535
TIMESTAMP_STR_LEN = 12

def compute_metrics(images, convert_to_mmh):
    max_intensities = []
    mean_intensities = []

    for image in images:
        valid_pixels = image[image != 65535]

        if convert_to_mmh:
            valid_pixels = (valid_pixels/100)*12

        max_intensity = np.percentile(valid_pixels, 99) if valid_pixels.size > 0 else 0
        mean_intensity = np.mean(valid_pixels) if valid_pixels.size > 0 else 0

        max_intensities.append(max_intensity)
        mean_intensities.append(mean_intensity)

    return np.array(max_intensities), np.array(mean_intensities)

def process_nc_file(file_path, convert_to_mmh=True):
    """
    Process one netCDF file:
      - Convert the timestamp character array to a list of strings.
      - Read the image data and compute the metrics using the Numba-accelerated function.
    Returns timestamps, max intensities, and mean intensities.
    """
    with xr.open_dataset(file_path) as ds:
        # Convert the fixed-length character array to strings.
        ts_char_array = ds['timestamp'].values  # shape (time, str_dim)
        timestamps = ts_char_array.view('S12').ravel().astype('U12')
        images = ds['image_data'].values  # expected shape: (time, y, x)
    max_intensities, mean_intensities = compute_metrics(images, convert_to_mmh)
    del images
    return timestamps, max_intensities, mean_intensities

def gather_file_paths(data_dir, storage_type):
    """
    Traverse the directory structure and return a list of file paths.
    For storage_type 'nc', assume all files in the directory are netCDF files.
    """
    file_paths = []
    if storage_type == 'nested_h5':
        for year in sorted(os.listdir(data_dir)):
            year_path = os.path.join(data_dir, year)
            for month in sorted(os.listdir(year_path)):
                month_path = os.path.join(year_path, month)
                month_files = [f for f in os.listdir(month_path) if f.endswith('.h5')]
                for file_name in sorted(month_files):
                    file_paths.append(os.path.join(month_path, file_name))
    elif storage_type == 'nc':
        for file_name in sorted(os.listdir(data_dir)):
            file_paths.append(os.path.join(data_dir, file_name))
    else:
        print(f"Storage type {storage_type} is not recognized. Use 'nested_h5' or 'nc'.")
    return file_paths

def main():
    data_dir = '/vol/csedu-nobackup/project/mrobben/nowcasting/dataset/test'
    output_csv = '/vol/csedu-nobackup/project/mrobben/nowcasting/output/intensities_full.csv'
    storage_type = 'nc'
    
    file_paths = gather_file_paths(data_dir, storage_type=storage_type)
    results = []
    
    for file_path in file_paths:
        
        st = time.time()
        timestamps, max_intensities, mean_intensities = process_nc_file(file_path)
        et = time.time()
        print(f'{os.path.basename(file_path)}, computed in {et-st}s')
        
        if timestamps is None:
            continue
        # Append metrics from this file to the overall results.
        for t, max_val, mean_val in zip(timestamps, max_intensities, mean_intensities):
            results.append({'timestamp': t, 'max_intensity': max_val, 'mean_intensity': mean_val})
        # Explicitly call garbage collection after each file.

        del timestamps
        del max_intensities
        del mean_intensities
        gc.collect()
    
    # Convert results to a DataFrame and save as CSV.
    df = pd.DataFrame(results)
    df.sort_values(by='timestamp', inplace=True)
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")


if __name__ == '__main__':
    main()
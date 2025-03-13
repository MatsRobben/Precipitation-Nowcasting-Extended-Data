import os
import h5py
import numpy as np
import netCDF4
import pandas as pd

# Global constants
MASK_VALUE = 65535
TIMESTAMP_STR_LEN = 12  # Fixed length for timestamp strings

def gather_files(root_dir, unique_filenames_by_month=None):
    """
    Gather .h5 files grouped by (year, month).

    If unique_filenames_by_month is provided, it should be a dictionary with keys like "YYYY-MM"
    and values as lists of filenames (e.g. as saved in unique_filenames_by_month.npy). In that case,
    file paths are constructed as os.path.join(root_dir, year, month, filename).

    If unique_filenames_by_month is None, recursively walk the root_dir and gather all .h5 files,
    assuming the folder structure is <root_dir>/<year>/<month>/<file>.h5.

    Returns:
        files_by_year_month: A dictionary with keys as (year, month) tuples and values as lists of full file paths.
    """
    files_by_year_month = {}
    if unique_filenames_by_month is not None:
        # Use the provided unique filenames to construct full file paths.
        for key, file_list in unique_filenames_by_month.items():
            # key is expected to be "YYYY-MM"
            try:
                year, month = key.split("-")
            except ValueError:
                print(f"Invalid key format: {key}. Expected format 'YYYY-MM'. Skipping.")
                continue
            # Construct full paths using the known folder structure.
            paths = [os.path.join(root_dir, year, month, filename) for filename in file_list]
            files_by_year_month[(year, month)] = paths
    else:
        # Recursively gather files from the root directory.
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith(".h5"):
                    full_path = os.path.join(dirpath, filename)
                    # Assuming folder structure: .../<year>/<month>/<file>.h5
                    parts = full_path.split(os.sep)
                    if len(parts) >= 3:
                        year = parts[-3]
                        month = parts[-2]
                        if year.isdigit() and month.isdigit():
                            key = (year, month)
                            files_by_year_month.setdefault(key, []).append(full_path)
    return files_by_year_month


def process_h5_file(file_path, mask_value=MASK_VALUE, convert_to_mmh=True):
    """
    Process a single HDF5 file:
      - Extract image from 'image1/image_data'
      - Set mask pixels (mask_value) to 0
      - Compute the 99th percentile (max rainfall) from valid (non-masked) pixels
      - Extract the timestamp from the 'overview' group attribute 'product_datetime_end'
    
    Returns:
        timestamp (str): the product end timestamp.
        processed_image (np.ndarray): the processed image as uint8.
        max_intensity (float): the 99th percentile of valid pixel values.
    """
    with h5py.File(file_path, "r") as h5_file:
        # Read the raw image data
        raw_image = h5_file["image1/image_data"][()]
        
        # Compute 99th percentile and mean over valid pixels (ignoring mask pixels)
        valid_pixels = raw_image[raw_image != mask_value]

        # Convert the pixels to mm of rainfall per hour
        if convert_to_mmh:
            valid_pixels = (valid_pixels/100)*12

        max_intensity = np.percentile(valid_pixels, 99) if valid_pixels.size > 0 else 0
        mean_intensity = np.mean(valid_pixels) if valid_pixels.size > 0 else 0
        
        # Process image: set masked pixels to 0
        processed_image = np.where(raw_image == mask_value, 0, raw_image).astype(np.uint16)
    
    # Extract the timestamp from the filename.
    # Assumes the filename format: RAD_NL25_RAC_5M_202101010005.h5
    base_name = os.path.basename(file_path)                  # "RAD_NL25_RAC_5M_202101010005.h5"
    name_without_ext = os.path.splitext(base_name)[0]         # "RAD_NL25_RAC_5M_202101010005"
    timestamp = name_without_ext.split("_")[-1]               # "202101010005"

    if len(timestamp) != 12:
        print(timestamp)
    
    return timestamp, processed_image, max_intensity, mean_intensity


def write_netcdf_for_shard(year, month, file_list, output_dir=".", min_intensity=0.01, print_output=False, use_unique=False):
    """
    Process all HDF5 files for a given year-month shard and write their data to a netCDF file.
    
    The netCDF file will contain:
      - 'image_data': uint8 array of images (time, y, x)
      - 'timestamp': fixed-length character array of timestamps (time, TIMESTAMP_STR_LEN)
      - 'max_rainfall': float array of 99th percentile intensity values (time)
    
    The output netCDF file is named 'output_<year>_<month>.nc' and is saved in output_dir.
    """
    if use_unique:
        min_intensity=0

    file_list.sort()
    print(f"Processing {len(file_list)} files for {year}-{month}...")
    
    # Determine image dimensions using the first file
    with h5py.File(file_list[0], "r") as sample_file:
        sample_image = sample_file["image1/image_data"][()]
        ny, nx = sample_image.shape

    # Lists to accumulate data
    timestamps = []
    images = []
    max_rainfalls = []
    
    for file_path in file_list:
        timestamp, processed_image, max_intensity, mean_intensity = process_h5_file(file_path)

        # # If there is almost no precipitation we do not include the record
        # if mean_intensity < min_intensity:
        #     continue
        
        timestamps.append(timestamp)
        images.append(processed_image)
        max_rainfalls.append(max_intensity)

        if print_output:
            print(f"Processed file: {os.path.basename(file_path)} | Timestamp: {timestamp} | Max rainfall: {max_intensity} | Mean rainfall: {mean_intensity}")
    
    if len(file_list)-len(max_rainfalls) != 0:
        print(f"{len(file_list)-len(max_rainfalls)} observations are excluded, as they are <{min_intensity} mm h^-1")

    # Stack images into a 3D array (time, y, x)
    images_array = np.stack(images, axis=0)
    max_rainfalls_array = np.array(max_rainfalls, dtype=np.float16)
    timestamps_array = netCDF4.stringtochar(np.array(timestamps, 'S12'))

    # Create the netCDF file
    output_filename = os.path.join(output_dir, f"output_{year}_{month}.nc")
    nc_file = netCDF4.Dataset(output_filename, "w", format="NETCDF4")
    
    # Create dimensions
    nc_file.createDimension("time", len(max_rainfalls))
    nc_file.createDimension("y", ny)
    nc_file.createDimension("x", nx)
    nc_file.createDimension("str_dim", TIMESTAMP_STR_LEN)
    
    # Create variables
    timestamp_var = nc_file.createVariable("timestamp", "S1", ("time", "str_dim"))
    image_var = nc_file.createVariable("image_data", "u2", ("time", "y", "x"))
    max_rainfall_var = nc_file.createVariable("max_rainfall", "f4", ("time",))
    
    # Assign data to variables
    image_var[:] = images_array
    max_rainfall_var[:] = max_rainfalls_array

    timestamp_var[:] = timestamps_array

    
    nc_file.close()
    print(f"Created netCDF file: {output_filename} | Selected {len(max_rainfalls)} observations. \n")

def main():
    # Set your root directory containing the nested .h5 files
    root_dir = "/vol/csedu-nobackup/project/mrobben/nowcasting/dataset/radar"
    # Output directory for the netCDF files (can be changed as needed)
    output_dir = "/vol/csedu-nobackup/project/mrobben/nowcasting/dataset/test"

    # Set use_unique_only = True to load the unique filenames and construct file paths accordingly.
    use_unique_only = False
    unique_filenames_by_month = None
    if use_unique_only:
        unique_filenames_path = "output/unique_filenames_by_month.npy"  # Adjust path if needed
        unique_filenames_by_month = np.load(unique_filenames_path, allow_pickle=True).item()
        # Expecting a dict with keys like "2020-01" and values as lists of filenames.

    # Gather files grouped by (year, month)
    files_by_year_month = gather_files(root_dir, unique_filenames_by_month=unique_filenames_by_month)

    # Process each year-month shard and create a corresponding netCDF file
    for (year, month), file_list in files_by_year_month.items():
        write_netcdf_for_shard(year, month, file_list, output_dir=output_dir, use_unique=use_unique_only)

if __name__ == '__main__':
    main()

import os
import h5py
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_image_file(file_path, convert_to_mmh=True):
    """
    Load an HDF5 file, compute the 99th percentile of the valid pixels
    (excluding mask values of 65535), and return the base filename and intensity.
    """
    try:
        with h5py.File(file_path, 'r') as f:
            image = f['image1/image_data'][...]
        # Exclude masked pixels (value 65535)
        valid_pixels = image[image != 65535]

        if convert_to_mmh:
            valid_pixels = (valid_pixels/100)*12
    
        max_intensity = np.percentile(valid_pixels, 99) if valid_pixels.size > 0 else 0
        mean_intensity = np.mean(valid_pixels) if valid_pixels.size > 0 else 0
        
        base_filename = os.path.basename(file_path)
        return base_filename, max_intensity, mean_intensity
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None, None

def gather_file_paths(data_dir):
    """
    Traverse the directory structure and return a list of paths to all .h5 files.
    """
    file_paths = []
    # Loop over each year folder
    for year in sorted(os.listdir(data_dir)):
        year_path = os.path.join(data_dir, year)
        
        # Loop over each month folder inside the year
        for month in sorted(os.listdir(year_path)):
            month_path = os.path.join(year_path, month)
            
            # Get all .h5 files in the month folder
            month_files = [f for f in os.listdir(month_path) if f.endswith('.h5')]
            for file_name in sorted(month_files):
                file_paths.append(os.path.join(month_path, file_name))
    return file_paths

def main():
    # data_dir = '/vol/knmimo-nobackup/users/pkools/thesis-forecasting/data/knmi_archived_raw/RAD_NL25_RAP_5min/'
    data_dir = '/vol/csedu-nobackup/project/mrobben/nowcasting/dataset/radar'
    output_csv = '/vol/csedu-nobackup/project/mrobben/nowcasting/intensities_test.csv'
    
    # Gather all file paths from the directory
    file_paths = gather_file_paths(data_dir)
    results = []
    
    # Use ProcessPoolExecutor to process files in parallel.
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_image_file, file_path): file_path for file_path in file_paths}
        for i, future in enumerate(as_completed(futures)):
            if i % 1000 == 0:
                print(i, end=', ', flush=True)
            base_filename, max_intensity, mean_intensity = future.result()
            if base_filename is not None:
                results.append({'filename': base_filename, 'max_intensity': max_intensity, 'mean_intensity': mean_intensity})
    
    # Convert results to a DataFrame and save as CSV.
    df = pd.DataFrame(results)
    df.sort_values(by='filename', inplace=True)
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")

if __name__ == '__main__':
    main()

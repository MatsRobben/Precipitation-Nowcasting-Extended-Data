import numpy as np
import pandas as pd
import random
import os

# -------------------------------
# Configuration
# -------------------------------
csv_path = '/vol/csedu-nobackup/project/mrobben/nowcasting/output/intensities_test.csv'
output_samples_path = 'output/training_samples_by_month.npy'
output_filenames_path = 'output/unique_filenames_by_month.npy'
window_size = 22            # number of consecutive frames per sample
num_samples = 1000          # desired total number of samples for training

# Define bins based on precipitation (max_intensity)
# Here we use exponential spacing between 0.2 and 50, then set 0 as lower and np.inf as upper edge.
bins = np.exp(np.linspace(np.log(0.2), np.log(10), 6))
bin_edges = np.concatenate(([0], bins, [np.inf]))
num_bins = len(bin_edges) - 1

# Set a random seed for reproducibility (optional)
random.seed(42)

# -------------------------------
# Helper function to extract year and month from a filename
# -------------------------------
def get_timestamp(filename):
    """
    Extract the 12-digit timestamp from a filename.
    Assumes the filename is formatted like: RAD_NL25_RAC_5M_202101051745.h5
    Returns:
        ts (str): e.g. "202101051745"
    """
    base = os.path.basename(filename)            # "RAD_NL25_RAC_5M_202101051745.h5"
    name_without_ext = os.path.splitext(base)[0]  # "RAD_NL25_RAC_5M_202101051745"
    ts = name_without_ext.split('_')[-1]          # "202101051745"
    return ts

def get_year_month_from_timestamp(ts):
    """
    Given a 12-digit timestamp string (YYYYMMDDHHMM),
    return a tuple (year, month) as strings.
    """
    return ts[:4], ts[4:6]

# -------------------------------
# Load CSV and create valid sliding window samples
# -------------------------------
df = pd.read_csv(csv_path)

# Create samples from sliding windows that do not cross month boundaries.
# For each valid window, store the timestamp (extracted from the first file),
# the original filename (from the first file), and the aggregated max intensity.
samples = []
for i in range(len(df) - window_size + 1):
    window_df = df.iloc[i:i + window_size]
    filenames = window_df['filename'].tolist()
    
    # Extract timestamps from the filenames in the window.
    timestamps = [get_timestamp(fname) for fname in filenames]
    
    # Get the year and month from the first timestamp.
    year, month = get_year_month_from_timestamp(timestamps[0])
    
    # Ensure all filenames in this window belong to the same month.
    if not all(get_year_month_from_timestamp(ts) == (year, month) for ts in timestamps):
        continue  # Skip windows crossing month boundaries.
    
    # Compute the aggregated precipitation metric (maximum max_intensity in the window).
    agg_max = window_df['max_intensity'].max()
    
    # Store both the extracted timestamp and the original filename.
    samples.append({
        'timestamps': timestamps,
        'filenames': filenames,
        'agg_max_intensity': agg_max,
        'year': year,
        'month': month
    })

# -------------------------------
# Bin the samples based on precipitation
# -------------------------------
agg_values = [sample['agg_max_intensity'] for sample in samples]
bin_indices = np.digitize(agg_values, bin_edges) - 1  # adjust to 0-based index

# Organize samples into bins.
binned_samples = {i: [] for i in range(num_bins)}
for sample, b_idx in zip(samples, bin_indices):
    if 0 <= b_idx < num_bins:
        binned_samples[b_idx].append(sample)

for key, value in binned_samples.items():
    print(f"Bin {key}: {len(value)} samples")

# -------------------------------
# Stratified subsampling: uniform across precipitation bins using sampling with replacement
# -------------------------------
base_count = num_samples // num_bins
remainder = num_samples % num_bins

selected_samples = []
for bin_idx in range(num_bins):
    available = binned_samples[bin_idx]
    desired = base_count + (1 if bin_idx < remainder else 0)
    if available:
        # Use sampling with replacement.
        chosen = random.choices(available, k=desired)
        selected_samples.extend(chosen)
    else:
        print(f'Bin {bin_idx} is empty, no samples are included.')

print(f"Total selected samples: {len(selected_samples)}")

# -------------------------------
# Create dictionaries grouping samples and filenames by year-month combination
# -------------------------------
samples_by_month = {}
filenames_by_month = {}

for sample in selected_samples:
    key = f"{sample['year']}-{sample['month']}"
    
    # Group samples by month
    samples_by_month.setdefault(key, []).append(sample['timestamps'])
    
    # Group filenames by month
    filenames_by_month.setdefault(key, set()).update(sample['filenames'])

# Save the samples grouped by month
np.save(output_samples_path, samples_by_month)
print(f"Saved samples grouped by month to {output_samples_path}")

# Save the unique filenames grouped by month
np.save(output_filenames_path, filenames_by_month)
print(f"Saved unique filenames by month to {output_filenames_path}")
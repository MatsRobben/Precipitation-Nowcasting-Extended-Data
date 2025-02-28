import numpy as np

# Load the dictionary of samples grouped by month
samples_by_month = np.load('output/training_samples_by_month.npy', allow_pickle=True).item()

# Load the dictionary of unique filenames grouped by month
filenames_by_month = np.load('output/unique_filenames.npy', allow_pickle=True).item()

# Loop through all months of 2020
for month in range(1, 13):
    key = f'2020-{month:02d}'  # Format month as '2020-01', '2020-02', ..., '2020-12'
    
    # Get samples for the month
    samples = samples_by_month.get(key, [])
    
    # Get unique filenames for the month
    filenames = filenames_by_month.get(key, [])
    
    print(f"Found {len(samples)} samples and {len(filenames)} unique filenames for {key}.")

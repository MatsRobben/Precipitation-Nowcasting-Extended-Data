import h5py
import os
import numpy as np
import matplotlib.pyplot as plt

file_path = "/vol/csedu-nobackup/project/mrobben/nowcasting/dataset/radar/2021/01/RAD_NL25_RAC_5M_202101010005.h5"

def print_h5_structure(name, obj):
    """Print the structure of an HDF5 file including groups, datasets, attributes, shape, and type."""
    if isinstance(obj, h5py.Group):
        print(f"Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name} | Shape: {obj.shape} | Type: {obj.dtype}")
    
    for key, value in obj.attrs.items():
        print(f"  Attribute - {key}: {value}")

with h5py.File(file_path, "r") as h5_file:
    h5_file.visititems(print_h5_structure)

# # File path to your HDF5 file
# file_path = "/vol/csedu-nobackup/project/mrobben/nowcasting/dataset/radar/2021/01/01/RAD_NL25_RAC_5M_202101010005.h5"
# output_folder = "figures"

# # Define the out-of-image mask value
# OUT_OF_IMAGE_VALUE = 65535  # Given in attributes

# # Create the output folder if it doesn't exist
# os.makedirs(output_folder, exist_ok=True)

# # Open the HDF5 file
# with h5py.File(file_path, "r") as h5_file:
#     # List the datasets we are interested in
#     datasets = ["image1/image_data", "image2/image_data"]
    
#     for dataset_name in datasets:
#         if dataset_name in h5_file:
#             data = h5_file[dataset_name][()]  # Load dataset as NumPy array
            
#             # Create a mask for valid radar data (excluding 65535 values)
#             mask = data != OUT_OF_IMAGE_VALUE  
#             valid_data = data[mask]  # Extract only valid values
            
#             # Initialize output image with 255 (for masked areas)
#             scaled_data = np.full_like(data, 255, dtype=np.uint8)  

#             if valid_data.size > 0:
#                 # Normalize only valid data to 0-255
#                 min_val, max_val = valid_data.min(), valid_data.max()
#                 scaled_data[mask] = ((valid_data - min_val) * (255 / (max_val - min_val))).astype(np.uint8)

#                 # Save as an image
#                 fig_path = os.path.join(output_folder, f"{dataset_name.replace('/', '_')}.png")
#                 plt.imsave(fig_path, scaled_data, cmap="gray")

#                 print(f"Saved {dataset_name} as {fig_path}")
#             else:
#                 print(f"Warning: {dataset_name} has no valid data.")
#         else:
#             print(f"Dataset {dataset_name} not found in the file.")

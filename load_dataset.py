from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import lightning as L

from PIL import Image, ImageOps
import xarray as xr
import numpy as np

import random
import os
import re
from typing import Optional, List
import time

class RadarDataset(IterableDataset):
    def __init__(
        self,
        samples_by_month_path: str,
        month_to_nc: dict,
        input_size: int = 4,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = True,
        months: Optional[List[str]] = None,
    ):
        """
        Args:
            samples_by_month_path (str): Path to the .npy file containing a dictionary of samples
                                         grouped by month (keys like '2020-01').
            month_to_nc (dict): Dictionary mapping month keys to netCDF file paths.
            shuffle_shards (bool): Whether to shuffle the order of shards (months).
            shuffle_within_shard (bool): Whether to shuffle the samples within each shard.
        """
        self.samples_by_month = np.load(samples_by_month_path, allow_pickle=True).item()
        # If a list of months is specified, filter to keep only those.
        if months is not None:
            self.samples_by_month = {
                month: samples 
                for month, samples in self.samples_by_month.items() 
                if month in months
            }

        self.month_to_nc = month_to_nc
        self.input_size = input_size
        self.shuffle_shards = shuffle_shards
        self.shuffle_within_shard = shuffle_within_shard

        # Create a list of shards: each element is a tuple (month_key, samples)
        self.shards = list(self.samples_by_month.items())

    def process_images(self, selected_images: np.ndarray) -> np.ndarray:
        """
        Processes the selected images:
          1. Scales the values using the formula (x/100)*12 (i.e. multiplies by 0.12).
          2. For each time frame:
             - Converts the frame to a PIL image (mode "F" for floating point),
             - Pads the image to a square (by computing the necessary borders),
             - Resizes it to 256x256 using bilinear interpolation.
          3. Stacks all frames and adds a channel axis, resulting in shape [Time, 1, 256, 256].

        Args:
            selected_images (np.ndarray): Array of shape [Time, Height, Width].

        Returns:
            np.ndarray: Processed array of shape [Time, 1, 256, 256].
        """
        processed_frames = []
        for frame in selected_images:
            # Convert to mm h^-1 by apply the scaling and multipying by 12.
            frame = frame.astype(np.float32) * 0.12

            # Convert numpy array to PIL image. Mode "F" supports floating point.
            img = Image.fromarray(frame, mode="F")

            # Get original dimensions (PIL uses (width, height)).
            w, h = img.size

            # Determine necessary padding to make image square.
            if h > w:
                pad_total = h - w
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                border = (pad_left, 0, pad_right, 0)
            elif w > h:
                pad_total = w - h
                pad_top = pad_total // 2
                pad_bottom = pad_total - pad_top
                border = (0, pad_top, 0, pad_bottom)
            else:
                border = (0, 0, 0, 0)

            # Pad the image.
            if any(border):
                img = ImageOps.expand(img, border=border, fill=0)

            # Resize the image to 256x256.
            img = img.resize((256, 256), resample=Image.BILINEAR)

            # Convert back to numpy array.
            frame_processed = np.array(img)
            processed_frames.append(frame_processed)

        # Stack frames to shape [Time, 256, 256]
        processed = np.stack(processed_frames, axis=0)
        # Add a channel dimension to get shape [Time, 1, 256, 256]
        processed = np.expand_dims(processed, axis=1)
        return processed

    def __iter__(self):
        # Retrieve worker info to split shards among workers
        worker_info = get_worker_info()
        shards = self.shards.copy()

        # Optionally shuffle the shard order.
        if self.shuffle_shards:
            random.shuffle(shards)

        # If multiple workers are used, split the shards among them.
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            shards = shards[worker_id::num_workers]

        # Iterate over each assigned shard.
        for month_key, samples in shards:
            nc_path = self.month_to_nc.get(month_key)
            if nc_path is None:
                print(f"Warning: No netCDF file provided for month {month_key}. Skipping.")
                continue

            # Open the netCDF file for this shard.
            ds = xr.open_dataset(nc_path)

            # Create list of timestamps
            ts_char_array = ds['timestamp'].values  # shape (time, str_dim)
            timestamps = ts_char_array.view('S12').ravel().astype('U12') # Char array to string array
            timestamps = timestamps.tolist()

            # Optionally shuffle the samples within the shard.
            shard_samples = samples.copy()
            if self.shuffle_within_shard:
                random.shuffle(shard_samples)

            # Process each sample in the shard.
            for sample_timestamps in shard_samples:

                indices = [timestamps.index(ts) for ts in sample_timestamps if ts in timestamps]
                
                selected_images = ds['image_data'].isel(time=indices).values

                # Process the selected images using PIL.
                processed_images = self.process_images(selected_images)

                # Split processed images into input sequence and future targets.
                context_images = processed_images[:self.input_size]
                future_images = processed_images[self.input_size:]
                yield context_images, future_images

            ds.close()


def build_month_to_nc(root_dir):
    """
    Scans the given root directory for netCDF files with names in the format:
      output_YYYY_MM.nc
    and returns a dictionary mapping keys "YYYY-MM" to the corresponding full file paths.

    Args:
        root_dir (str): Path to the directory containing the netCDF files.

    Returns:
        dict: Dictionary with keys in the format "YYYY-MM" and values as full file paths.
    """
    month_to_nc = {}
    # Regex pattern to match filenames like: output_2020_01.nc
    pattern = re.compile(r"output_(\d{4})_(\d{2})\.nc$")
    
    # List all files in the root directory
    for file_name in os.listdir(root_dir):
        match = pattern.match(file_name)
        if match:
            year, month = match.groups()
            key = f"{year}-{month}"
            full_path = os.path.join(root_dir, file_name)
            month_to_nc[key] = full_path

    return month_to_nc


class RadarDataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        samples_by_month_path: str,
        input_size: int = 4,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = True,
        batch_size: int = 4,
        num_workers: int = 8,
        val_month: Optional[str] = None,  # e.g. "2020-03"
    ):
        """
        Args:
            root_dir (str): Directory containing the netCDF files.
            samples_by_month_path (str): Path to the .npy file with samples grouped by month.
            input_size (int): Number of images per sample used as input.
            shuffle_shards (bool): Whether to shuffle the order of shards.
            shuffle_within_shard (bool): Whether to shuffle samples within each shard.
            batch_size (int): Batch size for the DataLoader.
            num_workers (int): Number of workers for the DataLoader.
            val_month (Optional[str]): If provided, this month (e.g., "2020-03") will be used exclusively for validation.
        """
        super().__init__()
        self.root_dir = root_dir
        self.samples_by_month_path = samples_by_month_path
        self.input_size = input_size
        self.shuffle_shards = shuffle_shards
        self.shuffle_within_shard = shuffle_within_shard
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_month = val_month

        self.month_to_nc = None
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        # Optionally, add any data downloading or pre-processing logic here.
        pass

    def setup(self, stage: Optional[str] = None):
        # Build the mapping from month keys to netCDF file paths.
        self.month_to_nc = build_month_to_nc(self.root_dir)
        
        # Load the full samples dictionary to determine available months.
        full_samples = np.load(self.samples_by_month_path, allow_pickle=True).item()
        all_months = list(full_samples.keys())

        if self.val_month is not None:
            # Ensure the validation month is present.
            if self.val_month not in all_months:
                raise ValueError(f"Validation month {self.val_month} not found in samples.")

            # Training months are all except the validation month.
            train_months = [m for m in all_months if m != self.val_month]

            self.train_dataset = RadarDataset(
                samples_by_month_path=self.samples_by_month_path,
                month_to_nc=self.month_to_nc,
                input_size=self.input_size,
                shuffle_shards=self.shuffle_shards,
                shuffle_within_shard=self.shuffle_within_shard,
                months=train_months,
            )
            self.val_dataset = RadarDataset(
                samples_by_month_path=self.samples_by_month_path,
                month_to_nc=self.month_to_nc,
                input_size=self.input_size,
                shuffle_shards=False,  # Usually no shuffling for validation
                shuffle_within_shard=False,
                months=[self.val_month],
            )
        else:
            # If no validation month is specified, use the full dataset for training.
            self.train_dataset = RadarDataset(
                samples_by_month_path=self.samples_by_month_path,
                month_to_nc=self.month_to_nc,
                input_size=self.input_size,
                shuffle_shards=self.shuffle_shards,
                shuffle_within_shard=self.shuffle_within_shard,
            )
            self.val_dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise ValueError("No validation dataset provided. Please set val_month in the DataModule.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

def build_month_to_nc(root_dir: str) -> dict:
    """
    Scans the given root directory for netCDF files with names like:
      output_YYYY_MM.nc
    and returns a dictionary mapping "YYYY-MM" to full file paths.
    """
    month_to_nc = {}
    pattern = re.compile(r"output_(\d{4})_(\d{2})\.nc$")
    for file_name in os.listdir(root_dir):
        match = pattern.match(file_name)
        if match:
            year, month = match.groups()
            key = f"{year}-{month}"
            full_path = os.path.join(root_dir, file_name)
            month_to_nc[key] = full_path
    return month_to_nc


# Example usage:
if __name__ == '__main__':
    root_directory = "dataset/test"  # Change to your root directory
    month_to_nc = build_month_to_nc(root_directory)
    # for key, path in month_to_nc.items():
    #     print(f"{key}: {path}")

    # Path to your saved samples_by_month file.
    samples_by_month_path = 'output/training_samples_by_month.npy'

    # Create the IterableDataset.
    dataset = RadarDataset(
        samples_by_month_path, 
        month_to_nc, 
        input_size=4,
        shuffle_shards=True, 
        shuffle_within_shard=True
    )

    # Create a DataLoader with multiple workers.
    # Each worker will load its own shards in parallel.
    dataloader = DataLoader(dataset, batch_size=4, num_workers=8)

    st = time.time()

    # Iterate over the DataLoader.
    for batch_idx, (images, future_images) in enumerate(dataloader):
        print(f"Batch {batch_idx}: images shape: {images.shape}, timestamps shape: {future_images.shape}, type: {type(images)}")
        # Pass the batch to your model, or perform additional processing.
        # For demonstration, we'll break after one batch.
        if batch_idx == 0:
            break

    et = time.time()

    print(et-st)



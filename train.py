import torch
import lightning as L
from torch.utils.data import DataLoader, Dataset
from dgmr import DGMR

from load_dataset import RadarDataModule


if __name__ == "__main__":

    data_module = RadarDataModule(
        root_dir="dataset/test",  # Directory containing netCDF files (e.g., "data/nc_files")
        samples_by_month_path="output/training_samples_by_month.npy",  # .npy file with samples grouped by month
        input_size=4,
        shuffle_shards=True,
        shuffle_within_shard=True,
        batch_size=4,
        num_workers=2,
        val_month="2020-03",  # Use the month "2020-03" exclusively for validation
    )
    data_module.setup()

    # Set overfit_batches=1 to repeatedly train on the same batch.
    trainer = L.Trainer(
        max_epochs=100,         # You may want more epochs to clearly overfit.
        accelerator="auto",
        precision=32,
        overfit_batches=1,      # Only one batch is used for both training and validation.
        log_every_n_steps=1,
    )

    # Parameters for the dummy data.
    num_train_samples = 100
    num_val_samples = 20
    context_frames = 4     # Adjust based on your model's expectation
    forecast_steps = 18    # Should match DGMR's forecast_steps parameter
    channels = 1
    height = 256
    width = 256

    # Create the model.
    model = DGMR(
        forecast_steps=forecast_steps,
        output_shape=height,
        input_channels=channels,
        latent_channels=768,
        context_channels=384,
        generation_steps=1,
    )

    # Run training; this will call your training_step and validation_step.
    trainer.fit(model, data_module)

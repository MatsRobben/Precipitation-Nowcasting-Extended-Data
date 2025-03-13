import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.callbacks import ModelCheckpoint

from dgmr import DGMR

from load_dataset import RadarDataModule
from load_random_dataset import RandomRadarDataModule


if __name__ == "__main__":

    val = [
        # "2020-11", "2020-12", "2021-01", "2021-02", "2021-03"
        "2020-05", "2020-11", "2021-05", "2021-11", "2022-05"
        ]
    test = [
        # "2021-01", "2021-02", "2021-03", "2021-04", "2021-05", "2021-06", "2021-07", "2021-08", "2021-09", "2021-10", "2021-11", "2021-12",
        # "2022-01", "2022-02", "2022-03", "2022-04", "2022-05", "2022-06", "2022-07", "2022-08", "2022-09", "2022-10", "2022-11", "2022-12", 
        "2023-01", "2023-02", "2023-03", "2023-04", "2023-05", "2023-06", "2023-07", "2023-08", "2023-09", "2023-10", "2023-11", "2023-12", "2022-11"
        ]
    
    # val = ["2020-11", "2020-12"]
    # test = []

    # Parameters
    context_frames = 4
    forecast_steps = 12
    channels = 1
    height = 256//1
    width = 256//1

    data_module = RadarDataModule(
        root_dir="dataset/RTCOR_processed",  # Directory containing netCDF files (e.g., "data/nc_files")
        samples_by_month_path="output/training_samples_by_month.npy",  # .npy file with samples grouped by month
        context_len=context_frames,
        forecast_len=forecast_steps,
        img_size=(height, width),
        shuffle_shards=True,
        shuffle_within_shard=True,
        batch_size=16, # 7 with discriminators
        num_train_workers=3,
        num_val_workers=0,
        val_months=val,
        test_months=test
    )
    data_module.setup()

    # # Parameters for synthetic data matching your original settings.
    # context_frames = 4
    # forecast_steps = 1
    # channels = 1
    # height = 256//2
    # width = 256//2
    # num_samples_train = 6720  # Adjust as needed for benchmarking
    # num_samples_val = 1120
    # batch_size = 7
    # num_workers = 2

    # data_module = RandomRadarDataModule(
    #     num_samples_train=num_samples_train,
    #     num_samples_val=num_samples_val,
    #     context_len=context_frames,
    #     forecast_len=forecast_steps,
    #     channels=channels,
    #     height=height,
    #     width=width,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    # )
    # data_module.setup()


    logger = TensorBoardLogger("./results/tb_logs/", name="DGMR_experiment")

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath="./results/checkpoints",
    #     filename="best-{epoch}-{val/grid_loss:.2f}",  # Best checkpoint name
    #     monitor="val/grid_loss",  # Your validation metric (from validation_step logs)
    #     mode="min",  # "min" for loss, "max" for accuracy
    #     save_top_k=1,  # Save only the best checkpoint
    #     save_last=True,  # Additional "last.ckpt" file every epoch
    #     every_n_epochs=1,  # Check every epoch
    #     save_on_train_epoch_end=False,  # Critical for DDP sync
    # )

    # profiler = PyTorchProfiler(
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler(logger.log_dir),
    #     schedule=torch.profiler.schedule(
    #         wait=1,  # Skip first batch
    #         warmup=10,  # Warmup for 1 batch
    #         active=3,  # Profile only 3 batches (originally 5)
    #         repeat=1   # Don't loop
    #     ),
    #     record_shapes=True,  # Disable input/output shape recording
    #     profile_memory=True,  # Disable memory profiling
    #     with_stack=True,  # Disable stack tracing
    # )

    # Set overfit_batches=1 to repeatedly train on the same batch.
    trainer = L.Trainer(
        max_epochs=250,
        accelerator="auto",
        strategy='ddp', #'ddp_find_unused_parameters_true',
        log_every_n_steps=1,
        precision="16-mixed",
        logger=logger,
        benchmark=True,
        profiler="simple"
    )

    # Create the model.
    model = DGMR(
        forecast_steps=forecast_steps,
        output_shape=height,
        input_channels=channels,
        latent_channels=int(768//1),
        context_channels=int(384//1),
        use_discriminators=False
    )

    # Run training; this will call your training_step and validation_step.
    trainer.fit(model, data_module)
from typing import List
import torch
from torch.utils.checkpoint import checkpoint
import lightning as L

from dgmr.conditioning import ConditioningStack, LatentCondStack
from dgmr.discriminators import TemporalDiscriminator, SpatialDiscriminator
from dgmr.generators import Generator, Sampler
from dgmr.losses import loss_hinge_disc, loss_hinge_gen, loss_grid_regularizer


class DGMR(L.LightningModule):
    """Deep Generative Model of Radar"""

    def __init__(
        self,
        forecast_steps: int = 4,
        input_channels: int = 1,
        output_shape: int = 256,
        latent_channels: int = 768,
        context_channels: int = 384,
        # variable_channels: List[int] = None,
        generation_steps: int = 1,
        scale_channels=False,
        gen_lr: float = 5e-5,
        disc_lr: float = 2e-4,
        grid_lambda: float = 20.0,
        beta1: float = 0.0,
        beta2: float = 0.999,
        

    ):
        super().__init__()
        self.generation_steps = generation_steps
        self.gen_lr = gen_lr
        self.disc_lr = disc_lr
        self.grid_lambda = grid_lambda
        self.beta1 = beta1
        self.beta2 = beta2

        self.conditioning_stack = ConditioningStack(
            input_channels=input_channels,
            output_channels=context_channels,
            scale_channels=scale_channels,
        )
        self.latent_stack = LatentCondStack(
            shape=(8, output_shape // 32, output_shape // 32),
            output_channels=latent_channels,
        )

        self.sampler = Sampler(
            forecast_steps=forecast_steps,
            latent_channels=latent_channels,
            context_channels=(
                context_channels * input_channels
                if scale_channels
                else context_channels
            ),
        )
        self.generator = Generator(
            self.conditioning_stack, self.latent_stack, self.sampler
        )

        self.spatial_discriminator = SpatialDiscriminator(
            input_channels=input_channels,
            num_timesteps=8,
        )
        self.temporal_discriminator = TemporalDiscriminator(
            input_channels=input_channels
        )

        self.save_hyperparameters()

        self.global_iteration = 0

        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        torch.autograd.set_detect_anomaly(True)

    def forward(self, x):
        x = self.generator(x)
        return x

    def training_step(self, batch, batch_idx):
        images, future_images = batch

        self.global_iteration += 1

        opt_g, opt_ds, opt_dt = self.optimizers()

        ##########################
        # Optimize Discriminators #
        ##########################
        # Two discriminator steps per generator step
        for _ in range(2):
            predictions = checkpoint(self.forward, images, use_reentrant=False).detach()

            # Optimize spatial Discriminator
            spatial_loss = self.discriminator_step(
                future_images,
                predictions,
                self.spatial_discriminator,
                loss_f=loss_hinge_disc,
                opt=opt_ds,
            )

            # Optimize Temporal Discriminator
            # Cat along time dimension [B, T, C, H, W]
            generated_sequence = torch.cat([images, predictions], dim=1)
            real_sequence = torch.cat([images, future_images], dim=1)

            temporal_loss = self.discriminator_step(
                real_sequence,
                generated_sequence,
                self.temporal_discriminator,
                loss_f=loss_hinge_disc,
                opt=opt_dt,
            )

        ######################
        # Optimize Generator #
        ######################
        
        predictions = [
            checkpoint(self.forward, images, use_reentrant=False)
            for _ in range(self.generation_steps)
        ]

        gen_mean = torch.stack(predictions, dim=0).mean(dim=0)
        grid_cell_reg = loss_grid_regularizer(gen_mean, future_images)

        real_sequence = torch.cat([images, future_images], dim=1)

        generated_scores = []
        for prediction in predictions:

            _, spatial_score_generated = self.discriminator_step(
                future_images,
                prediction,
                self.spatial_discriminator,
            )

            generated_sequence = torch.cat([images, prediction], dim=1)
            _, temporal_score_generated = self.discriminator_step(
                real_sequence,
                generated_sequence,
                self.temporal_discriminator,
            )

            generated_scores.append(spatial_score_generated + temporal_score_generated)

        generator_disc_loss = loss_hinge_gen(torch.cat(generated_scores, dim=0))

        generator_loss = self.grid_lambda * grid_cell_reg - generator_disc_loss

        opt_g.zero_grad()
        self.manual_backward(generator_loss)
        opt_g.step()

        self.log_dict(
            {
                "train/ds_loss": spatial_loss,
                "train/dt_loss": temporal_loss,
                "train/g_loss": generator_loss,
            },
            prog_bar=True,
            on_step=True,
        )

        # possibly log sampled images (TODO)


    def validation_step(self, batch, batch_idx):
        images, future_images = batch

        ##########################
        # Optimize Discriminators #
        ##########################
        # Two discriminator steps per generator step
        for _ in range(1):
            predictions = self.forward(images)

            # Optimize spatial Discriminator
            spatial_loss = self.discriminator_step(
                future_images,
                predictions,
                self.spatial_discriminator,
                loss_f=loss_hinge_disc,
            )

            # Optimize Temporal Discriminator
            # Cat along time dimension [B, T, C, H, W]
            generated_sequence = torch.cat([images, predictions], dim=1)
            real_sequence = torch.cat([images, future_images], dim=1)

            temporal_loss = self.discriminator_step(
                real_sequence,
                generated_sequence,
                self.temporal_discriminator,
                loss_f=loss_hinge_disc,
            )

        ######################
        # Optimize Generator #
        ######################
        # Maybe predict only one sequence for validation.
        predictions = [
            self.forward(images) for _ in range(self.generation_steps)
        ]

        gen_mean = torch.stack(predictions, dim=0).mean(dim=0)
        grid_cell_reg = loss_grid_regularizer(gen_mean, future_images)

        real_sequence = torch.cat([images, future_images], dim=1)

        generated_scores = []
        for prediction in predictions:

            _, spatial_score_generated = self.discriminator_step(
                future_images,
                prediction,
                self.spatial_discriminator,
            )

            generated_sequence = torch.cat([images, prediction], dim=1)
            _, temporal_score_generated = self.discriminator_step(
                real_sequence,
                generated_sequence,
                self.temporal_discriminator,
            )

            generated_scores.append(spatial_score_generated + temporal_score_generated)

        generator_disc_loss = loss_hinge_gen(torch.cat(generated_scores, dim=0))

        generator_loss = generator_disc_loss - self.grid_lambda * grid_cell_reg

        self.log_dict(
            {
                "val/ds_loss": spatial_loss,
                "val/dt_loss": temporal_loss,
                "val/g_loss": generator_loss,
            },
            on_step=True,
            prog_bar=True,
        )


    def discriminator_step(
        self, real_sequence, generated_sequence, discriminator, loss_f=None, opt=None
    ):
        # Cat long batch for the real+generated
        concatenated_inputs = torch.cat([real_sequence, generated_sequence], dim=0)

        concatenated_outputs = discriminator(concatenated_inputs)
        score_real, score_generated = torch.split(
            concatenated_outputs,
            [real_sequence.shape[0], generated_sequence.shape[0]],
            dim=0,
        )

        if loss_f is None:
            return score_real, score_generated

        loss = loss_f(score_generated, score_real)

        if opt is not None:
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()

        return loss

    def configure_optimizers(self):
        b1 = self.beta1
        b2 = self.beta2

        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=self.gen_lr, betas=(b1, b2)
        )
        opt_ds = torch.optim.Adam(
            self.spatial_discriminator.parameters(), lr=self.disc_lr, betas=(b1, b2)
        )
        opt_dt = torch.optim.Adam(
            self.temporal_discriminator.parameters(), lr=self.disc_lr, betas=(b1, b2)
        )

        return [opt_g, opt_ds, opt_dt], []  # First optimizers, second schedulers

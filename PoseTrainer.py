import torch
import os
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import comet_ml
from comet_ml import Experiment
from comet_ml.integration.pytorch import watch
from models.PoseLossExtension import PoseLossExtension

class PoseTrainer:
    """Trainer class for the Pose Estimation model.
    """

    def __init__(self, model, class_names, train_loader, val_loader, device=torch.device("cpu"), config=None, experiment=None):
        """

        Args:
            model (torch model): Model to be trained
            train_loader (dataloader): train dataloader
            val_loader (dataloader): validation dataloader
            device (str, optional): cuda or cpu. Defaults to 'cpu'.
            config (dict, optional): configuration file. Defaults to None.
        """
        self.model = model.to(device)
        self.class_names = class_names
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config or {}

        # loss function and optimizer
        self.criterion = PoseLossExtension(alpha=1.0, beta=1.0, class_names=self.class_names)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max= config['num_epochs'], eta_min=1e-6)
        # self.grad_accum_steps = self.config.get('grad_accum_steps', 8)

        # log metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.step = 0
        self.experiment = experiment
        self.num_batches = len(self.train_loader)

    def train_epoch(self):
        """Train the model for one epoch.

        Returns:
            avg_loss (float): average total loss of the epoch
            avg_trans_loss (float): average translation loss of the epoch
            avg_rot_loss (float): average rotation loss of the epoch
        """
        self.model.train()
        total_loss = 0
        total_trans_loss = 0
        total_rot_loss = 0

        pbar = tqdm(self.train_loader, desc='Training')
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            gt_trans = batch['translation']
            gt_rot = batch['rotation']
            obj_id = batch['obj_id']

            # forward pass
            pixel_rotations_norm, pixel_translations, pixel_confidences = self.model(batch)

            # compute loss (not normalized values)
            loss, r, t= self.criterion(pixel_rotations_norm, pixel_translations, pixel_confidences, gt_trans, gt_rot, obj_id)

            # normalize ONLY for backward (gradient accumulation)
            # loss_normalized = loss / self.grad_accum_steps
            loss.backward()

            # accumulate loss for statistics
            total_loss += loss.item()
            # total_trans_loss += trans_loss.item()
            # total_rot_loss += rot_loss.item()

            # step each grad_accum_steps or at the last batch
            # if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == total_batches:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.experiment is not None:
                if batch_idx % 100 == 0:
                    self.experiment.log_metrics({
                        "batch_loss": loss.item(), # not normalized
                        # "batch_trans_loss": trans_loss.item(), # not normalized
                        # "batch_rot_loss": rot_loss.item(), # normalized
                        "learning_rate": self.optimizer.param_groups[0]['lr'],
                        "step": self.step
                    })

            self.step += 1

            # update progress bar (not normalized)
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                # 'Trans': f'{trans_loss.item():.4f}',
                # 'Rot': f'{rot_loss.item():.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })

        # compute average on the total batches (not on accumulation steps)
        avg_loss = total_loss / self.num_batches
        # avg_trans_loss = total_trans_loss / self.num_batches
        # avg_rot_loss = total_rot_loss / self.num_batches

        return avg_loss
    
    def validate(self):
        """Validate the model on the validation set after each epoch.

        Returns:
            avg_loss (float): average total loss of the epoch
            avg_trans_loss (float): average translation loss of the epoch
            avg_rot_loss (float): average rotation loss of the epoch
        """
        self.model.eval()
        total_loss = 0
        total_trans_loss = 0
        total_rot_loss = 0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                gt_trans = batch['translation']
                gt_rot = batch['rotation']
                obj_id = batch['obj_id']

                # forward pass
                pixel_rotations_norm, pixel_translations, pixel_confidences = self.model(batch)

                # compute loss
                loss, r, t = self.criterion(pixel_rotations_norm, pixel_translations, pixel_confidences, gt_trans, gt_rot, obj_id)

                total_loss += loss.item()
                # total_trans_loss += trans_loss.item()
                # total_rot_loss += rot_loss.item()

        avg_loss = total_loss / num_batches
        # avg_trans_loss = total_trans_loss / num_batches
        # avg_rot_loss = total_rot_loss / num_batches

        if self.experiment is not None:
            self.experiment.log_metrics({
                "val_loss": avg_loss,
                # "val_trans_loss": avg_trans_loss,
                # "val_rot_loss": avg_rot_loss,
                "epoch": len(self.train_losses)+1
            })

        return avg_loss
        # avg_trans_loss, avg_rot_loss

    def train(self, num_epochs):
        """Train the model for the specified number of epochs.

        Args:
            num_epochs (int): number of epochs to train the model for
        """
        print(f"Starting training for {num_epochs} epochs...")

        if self.experiment is not None:
            with self.experiment.train():
                watch(self.model)

        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')

            # training
            train_loss= self.train_epoch()

            # validation
            val_loss = self.validate()

            # scheduler step
            self.scheduler.step()

            # save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # log epoch metrics
            if self.experiment is not None:
                self.experiment.log_metrics({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    # "train_trans_loss": train_trans_loss,
                    # "train_rot_loss": train_rot_loss,
                    "val_loss": val_loss,
                    # "val_trans_loss": val_trans_loss,
                    # "val_rot_loss": val_rot_loss,
                })
                  
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            # print(f'Train Loss: {train_loss:.4f} (Trans: {train_trans_loss:.4f}, Rot: {train_rot_loss:.4f})')
            # print(f'Val Loss: {val_loss:.4f} (Trans: {val_trans_loss:.4f}, Rot: {val_rot_loss:.4f})')

            # save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

                os.makedirs(f"./checkpoints/extension/", exist_ok=True)

                lr = self.optimizer.param_groups[0]['lr']
                batch_size = self.config.get('batch_size', 32)
                best_model_path = (
                    f"./checkpoints/extension/{self.config['name_saved_file']}_{self.config['backbone']}" # quaternion
                    f"_bs{batch_size}.pth"
                )

                # save the model
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'config': self.config
                }, best_model_path)

                # log model
                if self.experiment is not None:
                    self.experiment.log_metric("best_val_loss", val_loss)

                print(f'Saved best model with val_loss: {val_loss:.4f}')

        print("Training completed!")

    def plot_losses(self):
        """Plot the training and validation losses.
        """
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', color='blue')
        plt.plot(self.val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        # log plot
        if self.experiment is not None:
            self.experiment.log_image(image_data= plt, name= "Plot losses")

        plt.tight_layout()
        plt.show()
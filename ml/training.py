"""
Training module for AlphaZero Xiangqi model

This module handles the training of the neural network model using data
generated from self-play games.
"""

import os
import time
import logging
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from .config import *
from .model import XiangqiNetwork, AlphaZeroTrainer
from .utils import ensure_dir, load_training_data, get_timestamp

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("xiangqi.training")


class XiangqiDataset(Dataset):
    """Dataset for Xiangqi training examples"""

    def __init__(self, examples):
        """
        Initialize the dataset with training examples

        Args:
            examples: List of (state, policy, value) tuples
        """
        self.states = []
        self.policies = []
        self.values = []

        for state, policy, value in examples:
            self.states.append(state)
            self.policies.append(policy)
            self.values.append(value)

        self.states = np.array(self.states)
        self.policies = np.array(self.policies)
        self.values = np.array(self.values).reshape(-1, 1)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]


class TrainingPipeline:
    """Training pipeline for Xiangqi AlphaZero"""

    def __init__(
        self, model=None, device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the training pipeline

        Args:
            model: Neural network model (optional, will create new if None)
            device: Device to use for training ('cuda' or 'cpu')
        """
        self.device = device
        if model is None:
            self.model = XiangqiNetwork()
        else:
            self.model = model

        self.trainer = AlphaZeroTrainer(self.model, self.device)

        # Make sure model directories exist
        ensure_dir(MODEL_DIR)

        # Training metrics
        self.loss_history = []
        self.policy_loss_history = []
        self.value_loss_history = []

    def load_latest_model(self):
        """Load the latest model from disk"""
        model_path = os.path.join(MODEL_DIR, LATEST_MODEL_NAME)
        if os.path.exists(model_path):
            logger.info(f"Loading latest model from {model_path}")
            success = self.trainer.load_checkpoint(model_path)
            if success:
                logger.info("Model loaded successfully")
                return True
            else:
                logger.warning("Failed to load model")
        return False

    def load_best_model(self):
        """Load the best model from disk"""
        model_path = os.path.join(MODEL_DIR, BEST_MODEL_NAME)
        if os.path.exists(model_path):
            logger.info(f"Loading best model from {model_path}")
            success = self.trainer.load_checkpoint(model_path)
            if success:
                logger.info("Model loaded successfully")
                return True
            else:
                logger.warning("Failed to load model")
        return False

    def save_model(self, is_best=False):
        """
        Save the current model

        Args:
            is_best: Whether to also save this as the best model
        """
        # Save as the latest model
        latest_path = os.path.join(MODEL_DIR, LATEST_MODEL_NAME)
        self.trainer.save_checkpoint(latest_path)
        logger.info(f"Saved latest model to {latest_path}")

        # If it's the best model, save a copy
        if is_best:
            best_path = os.path.join(MODEL_DIR, BEST_MODEL_NAME)
            self.trainer.save_checkpoint(best_path)
            logger.info(f"Saved best model to {best_path}")

        # Also save a timestamped version
        timestamp = get_timestamp()
        timestamped_path = os.path.join(MODEL_DIR, f"model_{timestamp}.pt")
        self.trainer.save_checkpoint(timestamped_path)
        logger.info(f"Saved timestamped model to {timestamped_path}")

    def load_training_data(self, data_files):
        """
        Load training data from files

        Args:
            data_files: List of data file paths

        Returns:
            List of (state, policy, value) tuples
        """
        all_examples = []

        for file_path in data_files:
            logger.info(f"Loading training data from {file_path}")
            examples = load_training_data(file_path)
            all_examples.extend(examples)
            logger.info(f"Loaded {len(examples)} examples")

        logger.info(f"Total training examples: {len(all_examples)}")
        return all_examples

    def train(self, training_examples, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE):
        """
        Train the model on the provided examples

        Args:
            training_examples: List of (state, policy, value) tuples
            num_epochs: Number of epochs to train for
            batch_size: Batch size for training

        Returns:
            Final loss value
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        start_time = time.time()

        # Prepare dataset and data loader
        dataset = XiangqiDataset(training_examples)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

        # Reset metrics
        epoch_losses = []
        epoch_policy_losses = []
        epoch_value_losses = []

        # Training loop
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_policy_loss = 0
            epoch_value_loss = 0
            num_batches = 0

            # Progress bar
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for states, policies, values in pbar:
                # Convert to PyTorch tensors
                states = torch.FloatTensor(states.numpy()).to(self.device)
                policies = torch.LongTensor(
                    torch.argmax(torch.FloatTensor(policies.numpy()), dim=1)
                ).to(self.device)
                values = torch.FloatTensor(values.numpy()).to(self.device)

                # Train on batch
                total_loss, policy_loss, value_loss = self.trainer.train_batch(
                    states, policies, values
                )

                # Update metrics
                epoch_loss += total_loss
                epoch_policy_loss += policy_loss
                epoch_value_loss += value_loss
                num_batches += 1

                # Update progress bar
                pbar.set_postfix(
                    {
                        "loss": total_loss,
                        "policy_loss": policy_loss,
                        "value_loss": value_loss,
                    }
                )

            # Average losses for this epoch
            avg_loss = epoch_loss / num_batches
            avg_policy_loss = epoch_policy_loss / num_batches
            avg_value_loss = epoch_value_loss / num_batches

            # Store loss history
            epoch_losses.append(avg_loss)
            epoch_policy_losses.append(avg_policy_loss)
            epoch_value_losses.append(avg_value_loss)

            logger.info(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, "
                f"Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}"
            )

            # Save model periodically
            if (epoch + 1) % EPOCHS_PER_SAVE == 0 or epoch == num_epochs - 1:
                self.save_model()

        # Update full history
        self.loss_history.extend(epoch_losses)
        self.policy_loss_history.extend(epoch_policy_losses)
        self.value_loss_history.extend(epoch_value_losses)

        # Calculate training time
        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"Training completed in {duration:.2f} seconds")
        logger.info(f"Final loss: {epoch_losses[-1]:.4f}")

        # Plot and save loss history
        self.plot_loss_history()

        return epoch_losses[-1]

    def plot_loss_history(self):
        """Plot and save the loss history"""
        plt.figure(figsize=(12, 8))

        # Plot total loss
        plt.subplot(3, 1, 1)
        plt.plot(self.loss_history)
        plt.title("Total Loss")
        plt.grid(True)

        # Plot policy loss
        plt.subplot(3, 1, 2)
        plt.plot(self.policy_loss_history)
        plt.title("Policy Loss")
        plt.grid(True)

        # Plot value loss
        plt.subplot(3, 1, 3)
        plt.plot(self.value_loss_history)
        plt.title("Value Loss")
        plt.grid(True)

        plt.tight_layout()

        # Save figure
        ensure_dir(os.path.join(MODEL_DIR, "plots"))
        timestamp = get_timestamp()
        plt.savefig(os.path.join(MODEL_DIR, "plots", f"loss_history_{timestamp}.png"))
        plt.close()

    def train_from_files(
        self, data_files, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE
    ):
        """
        Load data from files and train the model

        Args:
            data_files: List of data file paths
            num_epochs: Number of epochs to train for
            batch_size: Batch size for training

        Returns:
            Final loss value
        """
        # Load training examples
        training_examples = self.load_training_data(data_files)

        # Train the model
        return self.train(training_examples, num_epochs, batch_size)

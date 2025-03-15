"""
Neural Network model for AlphaZero-like Xiangqi implementation

This implements a dual-head neural network with:
1. A policy head to predict move probabilities
2. A value head to predict the game outcome
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

from .config import *


class ResidualBlock(nn.Module):
    """Residual block with batch normalization"""

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class XiangqiNetwork(nn.Module):
    """Neural network for Xiangqi using AlphaZero architecture"""

    def __init__(self, input_planes=19, board_height=10, board_width=9):
        super(XiangqiNetwork, self).__init__()

        # Input layer
        self.conv_input = nn.Conv2d(
            input_planes, NUM_CHANNELS, kernel_size=3, padding=1
        )
        self.bn_input = nn.BatchNorm2d(NUM_CHANNELS)

        # Residual blocks
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(NUM_CHANNELS) for _ in range(NUM_RESIDUAL_BLOCKS)]
        )

        # Policy Head
        self.policy_conv = nn.Conv2d(NUM_CHANNELS, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * board_height * board_width, ACTION_SIZE)

        # Value Head
        self.value_conv = nn.Conv2d(NUM_CHANNELS, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(
            32 * board_height * board_width, VALUE_HEAD_HIDDEN_SIZE
        )
        self.value_fc2 = nn.Linear(VALUE_HEAD_HIDDEN_SIZE, 1)

    def predict(self, state):
        """
        Predict policy and value for a single state

        Args:
            state: A numpy array representing the board state

        Returns:
            policy: Action probabilities
            value: Expected outcome (-1 to 1)
        """
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            policy, value = self.forward(state_tensor)
            policy = F.softmax(policy, dim=1).squeeze(0).cpu().numpy()
            value = value.squeeze().cpu().numpy()
        return policy, value

    def forward(self, x):
        # Common layers
        x = F.relu(self.bn_input(self.conv_input(x)))

        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value


class AlphaZeroLoss(nn.Module):
    """Loss function for AlphaZero, combining policy and value losses"""

    def __init__(self, l2_reg=L2_REGULARIZATION):
        super(AlphaZeroLoss, self).__init__()
        self.l2_reg = l2_reg

    def forward(
        self, policy_output, value_output, policy_targets, value_targets, model
    ):
        # Policy loss - cross entropy loss on the policy
        policy_loss = F.cross_entropy(policy_output, policy_targets)

        # Value loss - mean squared error on the value
        value_loss = F.mse_loss(value_output, value_targets)

        # L2 regularization
        l2_penalty = 0
        for param in model.parameters():
            l2_penalty += torch.norm(param) ** 2

        # Combined loss
        total_loss = policy_loss + value_loss + self.l2_reg * l2_penalty

        return total_loss, policy_loss, value_loss


class AlphaZeroTrainer:
    """Trainer for AlphaZero model"""

    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.device = device

        # Print which device we're using
        print(f"Training on device: {self.device}")

        self.model.to(device)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REGULARIZATION
        )

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=LEARNING_RATE_DECAY_STEPS,
            gamma=LEARNING_RATE_DECAY_RATE,
        )

        self.loss_fn = AlphaZeroLoss()
        self.train_step_count = 0

    def train_batch(self, states, policy_targets, value_targets):
        """Train the model on a batch of data"""
        # Convert to tensors and move to device
        states = torch.FloatTensor(states).to(self.device)
        policy_targets = torch.LongTensor(policy_targets).to(self.device)
        value_targets = torch.FloatTensor(value_targets).to(self.device)

        # Forward pass
        self.model.train()
        self.optimizer.zero_grad()
        policy_output, value_output = self.model(states)

        # Calculate loss
        loss, policy_loss, value_loss = self.loss_fn(
            policy_output, value_output, policy_targets, value_targets, self.model
        )

        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.train_step_count += 1

        return loss.item(), policy_loss.item(), value_loss.item()

    def predict(self, state):
        """Predict policy and value for a single state"""
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            policy, value = self.model(state_tensor)
            policy = F.softmax(policy, dim=1).squeeze(0).cpu().numpy()
            value = value.squeeze().cpu().numpy()

        return policy, value

    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_step_count": self.train_step_count,
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        if os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.train_step_count = checkpoint["train_step_count"]
            return True
        return False

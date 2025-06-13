"""
File: models.py
Role: Deep learning models for CerberusWAF request classification.

This module implements neural network models for classifying HTTP requests:
- CNN-based request classifier
- Model training and evaluation utilities
- Model saving and loading functionality
- Inference pipeline

Key Components:
- RequestClassifier: CNN model for request classification
- Model training utilities
- Model evaluation metrics
- Model persistence

Dependencies:
- torch: Deep learning framework
- torch.nn: Neural network modules
- torch.optim: Optimization algorithms
- numpy: Numerical operations
- sklearn: Metrics and data processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics import precision_recall_fscore_support
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class RequestDataset(Dataset):
    """Dataset class for HTTP request data."""

    def __init__(self, requests: List[Dict[str, Any]], labels: List[int]):
        """
        Initialize the dataset.

        Args:
            requests: List of request feature vectors
            labels: List of binary labels (0: benign, 1: malicious)
        """
        self.requests = torch.FloatTensor(requests)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.requests)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.requests[idx], self.labels[idx]


class RequestClassifier(nn.Module):
    """CNN-based classifier for HTTP requests."""

    def __init__(self, input_size: int, hidden_dims: List[int] = [128, 64, 32]):
        """
        Initialize the CNN classifier.

        Args:
            input_size: Size of input feature vector
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()

        # Store dimensions
        self.input_size = input_size
        self.hidden_dims = hidden_dims

        # Build CNN layers
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        # First conv layer
        self.conv_layers.append(nn.Conv1d(1, 32, kernel_size=3, padding=1))
        self.pool_layers.append(nn.MaxPool1d(2))

        # Additional conv layers
        self.conv_layers.append(nn.Conv1d(32, 64, kernel_size=3, padding=1))
        self.pool_layers.append(nn.MaxPool1d(2))

        # Calculate the size after conv layers
        conv_output_size = input_size // 4 * 64

        # Build fully connected layers
        self.fc_layers = nn.ModuleList()
        prev_size = conv_output_size

        for hidden_dim in hidden_dims:
            self.fc_layers.append(nn.Linear(prev_size, hidden_dim))
            prev_size = hidden_dim

        # Output layer
        self.output_layer = nn.Linear(
            hidden_dims[-1], 2)  # Binary classification

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, 2)
        """
        # Add channel dimension for CNN
        x = x.unsqueeze(1)  # Shape: (batch_size, 1, input_size)

        # Apply conv layers
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            x = F.relu(conv(x))
            x = pool(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Apply fully connected layers
        for fc in self.fc_layers:
            x = F.relu(fc(x))
            x = self.dropout(x)

        # Output layer
        x = self.output_layer(x)

        return x


class ModelTrainer:
    """Utility class for training and evaluating the request classifier."""

    def __init__(self, model: RequestClassifier, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the trainer.

        Args:
            model: RequestClassifier instance
            device: Device to use for training ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train the model for one epoch.

        Args:
            train_loader: DataLoader for training data

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.

        Args:
            eval_loader: DataLoader for evaluation data

        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in eval_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.numpy())

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def save_model(self, path: str):
        """
        Save the model and its configuration.

        Args:
            path: Path to save the model
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model state
        torch.save(self.model.state_dict(), save_dir / 'model.pt')

        # Save model configuration
        config = {
            'input_size': self.model.input_size,
            'hidden_dims': self.model.hidden_dims
        }

        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config, f)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> 'ModelTrainer':
        """
        Load a saved model.

        Args:
            path: Path to the saved model
            device: Device to load the model on

        Returns:
            ModelTrainer instance with loaded model
        """
        save_dir = Path(path)

        # Load configuration
        with open(save_dir / 'config.json', 'r') as f:
            config = json.load(f)

        # Create model
        model = RequestClassifier(
            input_size=config['input_size'],
            hidden_dims=config['hidden_dims']
        )

        # Load state
        model.load_state_dict(torch.load(
            save_dir / 'model.pt', map_location=device))

        return cls(model, device)


def prepare_training_data(requests: List[Dict[str, Any]], labels: List[int],
                          batch_size: int = 32, train_ratio: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare data loaders for training and evaluation.

    Args:
        requests: List of request feature vectors
        labels: List of binary labels
        batch_size: Batch size for training
        train_ratio: Ratio of data to use for training

    Returns:
        Tuple of (train_loader, eval_loader)
    """
    # Convert requests to feature vectors
    feature_vectors = []
    for request in requests:
        # Combine all request components into a single vector
        url_vector = request.get('url_vector', [])
        headers_vector = request.get('headers_vector', [])
        params_vector = request.get('params_vector', [])
        body_vector = request.get('body_vector', [])

        # Concatenate all vectors
        combined_vector = np.concatenate([
            url_vector, headers_vector, params_vector, body_vector
        ])
        feature_vectors.append(combined_vector)

    # Create dataset
    dataset = RequestDataset(feature_vectors, labels)

    # Split into train and eval
    train_size = int(len(dataset) * train_ratio)
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)

    return train_loader, eval_loader

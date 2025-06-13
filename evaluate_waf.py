"""
Evaluation script for CerberusWAF model.
Loads trained model and evaluates performance on test dataset.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from evaluation import ModelEvaluator
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class CNNModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(16 * (input_dim // 2), num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


def load_model(model_path, input_dim, num_classes):
    """Load PyTorch CNN model."""
    model = CNNModel(input_dim, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def load_test_data(data_path):
    """Load and preprocess test dataset."""
    # Load test data
    df = pd.read_csv(data_path)

    # Separate features and labels
    X_test = df.drop('label', axis=1).values
    y_test = df['label'].values

    # Convert to PyTorch tensor
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    X_test_tensor = X_test_tensor.unsqueeze(1)  # Add channel dimension for CNN

    return X_test_tensor, y_test


def get_predictions(model, X_test):
    """Get predictions from model."""
    with torch.no_grad():
        outputs = model(X_test)
        probabilities = torch.softmax(outputs, dim=1)
        _, predictions = torch.max(outputs, 1)

    return predictions.numpy(), probabilities.numpy()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix using ConfusionMatrixDisplay."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - CerberusWAF")

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Evaluate CerberusWAF model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model file (.pt)')
    parser.add_argument('--test-data', type=str, required=True,
                        help='Path to test dataset CSV')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save evaluation plots')

    args = parser.parse_args()

    # Load test data
    print(f"Loading test data from {args.test_data}...")
    X_test, y_test = load_test_data(args.test_data)

    # Get model parameters from data
    input_dim = X_test.shape[2]  # Remove batch and channel dimensions
    num_classes = len(np.unique(y_test))

    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model, input_dim, num_classes)

    # Get predictions
    print("Generating predictions...")
    y_pred, y_prob = get_predictions(model, X_test)

    # Evaluate model
    print("Evaluating model performance...")
    evaluator = ModelEvaluator("cerberus_waf")
    evaluator.load_predictions(y_test, y_pred, y_prob)
    evaluator.calculate_metrics()

    # Print summary
    evaluator.print_summary()

    # Plot confusion matrix
    if args.save_plots:
        print("Saving evaluation plots...")
        evaluator.save_results()
        plot_confusion_matrix(y_test, y_pred,
                              f"{evaluator.results_dir}/confusion_matrix_{evaluator.timestamp}.png")
    else:
        plot_confusion_matrix(y_test, y_pred)


if __name__ == "__main__":
    main()

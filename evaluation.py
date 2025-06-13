"""
Evaluation module for CerberusWAF model performance analysis.
Provides functions for calculating metrics and generating visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import seaborn as sns
from datetime import datetime
import json
import os


class ModelEvaluator:
    def __init__(self, model_name="cerberus_waf"):
        """Initialize the evaluator with model name and metrics storage."""
        self.model_name = model_name
        self.metrics = {}
        self.predictions = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create results directory if it doesn't exist
        self.results_dir = "evaluation_results"
        os.makedirs(self.results_dir, exist_ok=True)

    def load_predictions(self, y_true, y_pred, y_prob=None):
        """Load ground truth and predictions for evaluation."""
        self.predictions = {
            'y_true': np.array(y_true),
            'y_pred': np.array(y_pred),
            'y_prob': np.array(y_prob) if y_prob is not None else None
        }

    def calculate_metrics(self):
        """Calculate all performance metrics."""
        y_true = self.predictions['y_true']
        y_pred = self.predictions['y_pred']

        # Basic metrics
        self.metrics['accuracy'] = accuracy_score(y_true, y_pred)
        self.metrics['precision'] = precision_score(
            y_true, y_pred, average='weighted')
        self.metrics['recall'] = recall_score(
            y_true, y_pred, average='weighted')
        self.metrics['f1'] = f1_score(y_true, y_pred, average='weighted')

        # Detailed classification report
        self.metrics['classification_report'] = classification_report(
            y_true, y_pred, output_dict=True
        )

        # Confusion matrix
        self.metrics['confusion_matrix'] = confusion_matrix(
            y_true, y_pred).tolist()

        return self.metrics

    def plot_confusion_matrix(self, save=True):
        """Plot and save confusion matrix heatmap."""
        plt.figure(figsize=(10, 8))
        cm = np.array(self.metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        if save:
            plt.savefig(
                f'{self.results_dir}/confusion_matrix_{self.timestamp}.png')
            plt.close()
        else:
            plt.show()

    def plot_metrics_comparison(self, save=True):
        """Plot bar chart comparing main metrics."""
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        values = [self.metrics[m] for m in metrics]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values)
        plt.title('Model Performance Metrics')
        plt.ylim(0, 1)

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.3f}', ha='center', va='bottom')

        if save:
            plt.savefig(
                f'{self.results_dir}/metrics_comparison_{self.timestamp}.png')
            plt.close()
        else:
            plt.show()

    def plot_roc_curve(self, save=True):
        """Plot ROC curve if probability predictions are available."""
        if self.predictions['y_prob'] is None:
            print("ROC curve not available: probability predictions not provided")
            return

        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(
            self.predictions['y_true'], self.predictions['y_prob'])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")

        if save:
            plt.savefig(f'{self.results_dir}/roc_curve_{self.timestamp}.png')
            plt.close()
        else:
            plt.show()

    def save_results(self):
        """Save all metrics and plots to files."""
        # Save metrics to JSON
        metrics_file = f'{self.results_dir}/metrics_{self.timestamp}.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)

        # Generate all plots
        self.plot_confusion_matrix()
        self.plot_metrics_comparison()
        self.plot_roc_curve()

        print(f"Results saved to {self.results_dir}/")
        print(f"Metrics saved to {metrics_file}")

    def print_summary(self):
        """Print a summary of the evaluation results."""
        print("\nModel Evaluation Summary")
        print("=" * 50)
        print(f"Model: {self.model_name}")
        print(f"Timestamp: {self.timestamp}")
        print("\nMain Metrics:")
        print(f"Accuracy:  {self.metrics['accuracy']:.4f}")
        print(f"Precision: {self.metrics['precision']:.4f}")
        print(f"Recall:    {self.metrics['recall']:.4f}")
        print(f"F1-Score:  {self.metrics['f1']:.4f}")

        print("\nDetailed Classification Report:")
        print(classification_report(
            self.predictions['y_true'],
            self.predictions['y_pred']
        ))


def evaluate_model(y_true, y_pred, y_prob=None, model_name="cerberus_waf"):
    """
    Convenience function to run a complete evaluation.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        model_name: Name of the model being evaluated

    Returns:
        ModelEvaluator instance with calculated metrics
    """
    evaluator = ModelEvaluator(model_name)
    evaluator.load_predictions(y_true, y_pred, y_prob)
    evaluator.calculate_metrics()
    evaluator.print_summary()
    evaluator.save_results()
    return evaluator


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    # Generate sample data
    X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train a simple model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Get predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Run evaluation
    evaluator = evaluate_model(y_test, y_pred, y_prob, "example_model")

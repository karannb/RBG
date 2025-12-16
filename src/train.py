"""
Train and evaluate a SchNet model on protein structure data using Leave-One-Out Cross-Validation.
"""

import os
import json
import torch
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

from src.model import Model
from src.dataset import ProteinDataset
from src.metrics import compute_metrics


class Trainer:
    """
    Trainer class for model training and evaluation.
    """
    def __init__(self, model: Model, device: torch.device, train_loader: DataLoader, val_loader: DataLoader, optimizer: torch.optim.Optimizer):
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer

    def train_epoch(self):
        """
        Typical train loop.
        """
        self.model.train()
        total_loss = 0
        for batch in self.train_loader:
            data = batch.to(self.device)
            out = self.model(data.x, data.pos, data.batch)
            loss = torch.nn.functional.mse_loss(out.view(-1), data.y.view(-1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * data.num_graphs

        return total_loss / len(self.train_loader.dataset)

    def train(self, num_epochs: int, verbose: bool = True):
        """
        Full training loop.

        Returns:
            Dictionary with training history
        """
        history = {"train_loss": [], "val_loss": []}

        iterator = range(1, num_epochs + 1)
        if verbose:
            iterator = tqdm(iterator, desc="Training", leave=False)

        for epoch in iterator:
            train_loss = self.train_epoch()
            val_loss, _, _, _ = self.evaluate()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if verbose:
                iterator.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")

        return history

    @torch.no_grad()
    def evaluate(self):
        """
        Evaluate on validation set and return predictions.

        Returns:
            loss: Average loss over validation set
            predictions: List of predicted values
            targets: List of ground truth values
            pdb_names: List of PDB names
        """
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        pdb_names = []

        for batch in self.val_loader:
            data = batch.to(self.device)
            out = self.model(data.x, data.pos, data.batch)
            loss = torch.nn.functional.mse_loss(out.view(-1), data.y.view(-1))
            total_loss += loss.item() * data.num_graphs

            predictions.extend(out.view(-1).cpu().tolist())
            targets.extend(data.y.view(-1).cpu().tolist())
            pdb_names.extend(data.pdb_name)

        avg_loss = total_loss / len(self.val_loader.dataset)
        return avg_loss, predictions, targets, pdb_names


def plot_training_curves(history: dict, fold_idx: int, save_dir: str = "./plots/"):
    """
    Plot and save training and validation loss curves.

    Args:
        history: Dictionary with 'train_loss' and 'val_loss' lists
        fold_idx: Fold number for filename
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    plt.plot(epochs, history["val_loss"], label="Val Loss", marker="s", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Training Curves - Fold {fold_idx}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{fold_idx}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


def run_loocv(
    batch_size: int = 4,
    hidden_dim: int = 64,
    learning_rate: float = 1e-4,
    num_epochs: int = 20,
    cutoff: float = 20.0,
    save_dir: str = "./results/",
):
    """
    Run Leave-One-Out Cross-Validation.

    Args:
        batch_size: Batch size for training
        hidden_dim: Hidden dimension for the model
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs per fold
        cutoff: Cutoff distance for SchNet
        save_dir: Directory to save results

    Returns:
        Dictionary with all LOOCV results
    """
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n_samples = ProteinDataset.TOTAL_LENGTH_MINUS_LONG

    # Store results for each fold
    all_predictions = []
    all_targets = []
    all_pdb_names = []
    fold_results = []

    print(f"\nRunning LOOCV with {n_samples} folds...")
    print(f"Hyperparameters: batch_size={batch_size}, hidden_dim={hidden_dim}, lr={learning_rate}, epochs={num_epochs}, cutoff={cutoff}")
    print("-" * 60)

    for fold_idx in tqdm(range(n_samples), desc="LOOCV Folds"):
        # Prepare train/val split
        train_indices = [j for j in range(n_samples) if j != fold_idx]
        val_indices = [fold_idx]

        train_dataset = ProteinDataset(
            mode="training",
            data_path="./data/",
            embeddings_path="./embeddings/",
            indices=train_indices
        )
        val_dataset = ProteinDataset(
            mode="training",
            data_path="./data/",
            embeddings_path="./embeddings/",
            indices=val_indices
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1)

        # Initialize model and optimizer (fresh for each fold)
        model = Model(input_dim=1280, hidden_dim=hidden_dim, output_dim=1, cutoff=cutoff)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train
        trainer = Trainer(model, device, train_loader, val_loader, optimizer)
        history = trainer.train(num_epochs, verbose=False)

        # Plot and save training curves
        plot_training_curves(history, fold_idx, save_dir="./plots/")

        # Get predictions using evaluate
        _, preds, targs, names = trainer.evaluate()

        all_predictions.extend(preds)
        all_targets.extend(targs)
        all_pdb_names.extend(names)

        fold_results.append({
            "fold": fold_idx,
            "pdb_name": names[0],
            "prediction": preds[0],
            "target": targs[0],
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
        })

    # Compute overall metrics
    overall_metrics = compute_metrics(all_predictions, all_targets)

    # Print results
    print("\n" + "=" * 60)
    print("LOOCV Results Summary")
    print("=" * 60)
    print(f"  MSE:      {overall_metrics['mse']:.4f}")
    print(f"  RMSE:     {overall_metrics['rmse']:.4f}")
    print(f"  MAE:      {overall_metrics['mae']:.4f}")
    print(f"  R²:       {overall_metrics['r2']:.4f}")
    print(f"  Pearson:  {overall_metrics['pearson']:.4f}")
    print(f"  Spearman: {overall_metrics['spearman']:.4f}")
    print("=" * 60)

    # Compile full results
    results = {
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": {
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "cutoff": cutoff,
            "n_folds": n_samples,
        },
        "overall_metrics": overall_metrics,
        "fold_results": fold_results,
        "all_predictions": all_predictions,
        "all_targets": all_targets,
        "all_pdb_names": all_pdb_names,
    }

    # Save results
    results_file = os.path.join(save_dir, f"loocv_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Save predictions vs targets for plotting
    pred_file = os.path.join(save_dir, "predictions.csv")
    with open(pred_file, "w") as f:
        f.write("pdb_name,prediction,target\n")
        for name, pred, targ in zip(all_pdb_names, all_predictions, all_targets):
            f.write(f"{name},{pred},{targ}\n")
    print(f"Predictions saved to: {pred_file}")

    return results


def main():
    """
    Main entry point.
    """
    # Hyperparameters
    batch_size = 4
    hidden_dim = 64
    learning_rate = 1e-4
    num_epochs = 20
    cutoff = 20.0

    results = run_loocv(
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        cutoff=cutoff,
        save_dir="./results/",
    )

    return results


if __name__ == "__main__":
    main()

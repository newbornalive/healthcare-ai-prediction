"""PyTorch one-dimensional CNN for tabular diabetes risk prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class DiabetesCNN(nn.Module):
    """Compact experimental 1D CNN over a fixed feature ordering."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.n_features = n_features
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Conv1d(12, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return logits for a batch of standardised tabular features."""
        if features.ndim == 2:
            features = features.unsqueeze(1)
        return self.classifier(self.feature_extractor(features)).squeeze(1)


@dataclass
class CNNTrainingResult:
    """Trained model and epoch-level diagnostics."""

    model: DiabetesCNN
    history: list[dict[str, float]]
    best_epoch: int


def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def predict_cnn(
    model: DiabetesCNN,
    X: np.ndarray,
    batch_size: int = 8192,
    device: str = "cpu",
) -> np.ndarray:
    """Generate sigmoid probabilities in deterministic evaluation mode."""
    model.eval()
    model.to(device)
    loader = DataLoader(
        torch.tensor(X, dtype=torch.float32),
        batch_size=batch_size,
        shuffle=False,
    )
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch.to(device))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities)


def train_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    random_seed: int = 42,
    epochs: int = 4,
    batch_size: int = 4096,
    learning_rate: float = 2e-3,
    patience: int = 2,
    device: str = "cpu",
) -> CNNTrainingResult:
    """Train the CNN with class weighting, early stopping, and PR-AUC monitoring."""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))

    model = DiabetesCNN(n_features=X_train.shape[1]).to(device)
    positive_count = max(float(y_train.sum()), 1.0)
    negative_count = float(len(y_train) - y_train.sum())
    pos_weight = torch.tensor([negative_count / positive_count], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )

    train_loader = _make_loader(
        X_train, y_train, batch_size=batch_size, shuffle=True
    )
    validation_loader = _make_loader(
        X_validation, y_validation, batch_size=batch_size, shuffle=False
    )

    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_pr_auc = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimiser.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimiser.step()
            train_loss_sum += float(loss.item()) * len(labels)
            train_count += len(labels)

        model.eval()
        validation_loss_sum = 0.0
        validation_count = 0
        probabilities: list[np.ndarray] = []
        labels_list: list[np.ndarray] = []
        with torch.no_grad():
            for features, labels in validation_loader:
                features = features.to(device)
                labels = labels.to(device)
                logits = model(features)
                loss = criterion(logits, labels)
                validation_loss_sum += float(loss.item()) * len(labels)
                validation_count += len(labels)
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
                labels_list.append(labels.cpu().numpy())

        validation_probability = np.concatenate(probabilities)
        validation_target = np.concatenate(labels_list)
        validation_pr_auc = average_precision_score(
            validation_target, validation_probability
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss_sum / train_count,
                "validation_loss": validation_loss_sum / validation_count,
                "validation_pr_auc": float(validation_pr_auc),
                "learning_rate": float(optimiser.param_groups[0]["lr"]),
            }
        )

        if validation_pr_auc > best_pr_auc + 1e-4:
            best_pr_auc = float(validation_pr_auc)
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("CNN training did not produce a valid model state.")
    model.load_state_dict(best_state)
    model.to("cpu")
    return CNNTrainingResult(model=model, history=history, best_epoch=best_epoch)


def integrated_gradients(
    model: DiabetesCNN,
    X: np.ndarray,
    *,
    steps: int = 10,
    batch_size: int = 256,
) -> np.ndarray:
    """Calculate mean absolute integrated gradients from a zero baseline."""
    model.eval()
    baseline = np.zeros((1, X.shape[1]), dtype=np.float32)
    attributions: list[np.ndarray] = []
    for start in range(0, len(X), batch_size):
        batch = X[start : start + batch_size].astype(np.float32)
        batch_tensor = torch.tensor(batch)
        baseline_tensor = torch.tensor(np.repeat(baseline, len(batch), axis=0))
        total_gradients = torch.zeros_like(batch_tensor)
        for alpha in torch.linspace(0.0, 1.0, steps):
            interpolated = (
                baseline_tensor + alpha * (batch_tensor - baseline_tensor)
            ).requires_grad_(True)
            logits = model(interpolated)
            gradients = torch.autograd.grad(logits.sum(), interpolated)[0]
            total_gradients += gradients.detach()
        integrated = (
            batch_tensor - baseline_tensor
        ) * (total_gradients / steps)
        attributions.append(integrated.detach().numpy())
    return np.mean(np.abs(np.vstack(attributions)), axis=0)

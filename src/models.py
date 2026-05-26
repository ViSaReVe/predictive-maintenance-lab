import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# ── Conv1d Autoencoder (anomaly detection) ────────────────────────────────────

class ConvAutoencoder(nn.Module):
    """
    1D Convolutional Autoencoder for vibration signal reconstruction.

    Trained only on normal (healthy) data. High reconstruction error at
    inference time signals an anomaly — the motor is doing something it
    has never seen during training.

    Architecture: two encoder Conv1d+MaxPool blocks → linear bottleneck →
    two decoder Upsample+Conv1d blocks.
    """

    def __init__(self, seq_len: int = 1024, latent_dim: int = 32):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Flatten(),
            nn.Linear(32 * (seq_len // 16), latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * (seq_len // 16)), nn.ReLU(),
            nn.Unflatten(1, (32, seq_len // 16)),
            nn.Upsample(scale_factor=4),
            nn.Conv1d(32, 16, kernel_size=5, padding=2), nn.ReLU(),
            nn.Upsample(scale_factor=4),
            nn.Conv1d(16, 1, kernel_size=7, padding=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> np.ndarray:
        """
        Compute per-sample MSE reconstruction error.

        Parameters
        ----------
        x : Tensor, shape (batch, 1, seq_len)

        Returns
        -------
        errors : ndarray, shape (batch,)
        """
        self.eval()
        with torch.no_grad():
            recon = self(x)
            err = ((x - recon) ** 2).mean(dim=(1, 2))
        return err.cpu().numpy()


def train_autoencoder(
    model: ConvAutoencoder,
    X_normal: np.ndarray,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = 'cpu',
) -> list:
    """
    Train the autoencoder on normal-only windows.

    Parameters
    ----------
    X_normal : ndarray, shape (n_windows, seq_len)

    Returns
    -------
    losses : list of per-epoch mean loss values
    """
    from torch.utils.data import DataLoader, TensorDataset

    model = model.to(device)
    # Add channel dim: (n, seq_len) → (n, 1, seq_len)
    X_t = torch.tensor(X_normal[:, None, :]).float().to(device)
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses = []

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        avg = epoch_loss / len(X_normal)
        losses.append(avg)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  loss={avg:.6f}")
    return losses


# ── LSTM Regressor (RUL prediction) ──────────────────────────────────────────

class LSTMRegressor(nn.Module):
    """
    2-layer LSTM → linear head for RUL regression.

    Input:  (batch, seq_len, n_features)
    Output: (batch,) — predicted RUL
    """

    def __init__(self, input_size: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(1)


def train_lstm(
    model: LSTMRegressor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = 'cpu',
    val_fraction: float = 0.15,
    patience: int = 10,
) -> list:
    """
    Train the LSTM regressor with a validation split and early stopping.

    Holds out the last `val_fraction` of windows as a validation set.
    Training stops when val RMSE has not improved for `patience` epochs.
    Best weights (lowest val RMSE) are restored at the end.

    Returns
    -------
    train_losses : list of per-epoch train RMSE values (in cycle units)
    """
    import copy

    from torch.utils.data import DataLoader, TensorDataset

    model = model.to(device)

    # Validation split — last val_fraction of windows (preserves temporal order)
    n_val = max(1, int(len(X_train) * val_fraction))
    X_tr, X_val = X_train[:-n_val], X_train[-n_val:]
    y_tr, y_val = y_train[:-n_val], y_train[-n_val:]

    X_t  = torch.tensor(X_tr).float().to(device)
    y_t  = torch.tensor(y_tr).float().to(device)
    Xv_t = torch.tensor(X_val).float().to(device)
    yv_t = torch.tensor(y_val).float().to(device)

    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    optimizer  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-5
    )
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    best_state    = copy.deepcopy(model.state_dict())
    no_improve    = 0
    train_losses  = []

    for epoch in range(epochs):
        # ── train ──
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        train_rmse = float(np.sqrt(epoch_loss / len(X_tr)))
        train_losses.append(train_rmse)

        # ── validate ──
        model.eval()
        with torch.no_grad():
            val_pred = model(Xv_t)
            val_rmse = float(torch.sqrt(criterion(val_pred, yv_t)).item())

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  train={train_rmse:.2f}  val={val_rmse:.2f}")

        # ── LR scheduler ──
        scheduler.step(val_rmse)
        current_lr = optimizer.param_groups[0]['lr']
        if (epoch + 1) % 10 == 0:
            print(f"  lr={current_lr:.2e}")

        # ── early stopping ──
        if val_rmse < best_val_rmse - 0.01:
            best_val_rmse = val_rmse
            best_state    = copy.deepcopy(model.state_dict())
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch+1}  best val RMSE={best_val_rmse:.2f}")
                break

    model.load_state_dict(best_state)
    print(f"  Restored best weights  val RMSE={best_val_rmse:.2f} cycles")
    return train_losses


# ── Classical anomaly models ──────────────────────────────────────────────────

def fit_isolation_forest(X_normal: np.ndarray, contamination: float = 0.01) -> IsolationForest:
    """
    Train an Isolation Forest on normal-only feature vectors.
    contamination=0.01 → assume 1% of 'normal' training data is noisy.
    Score convention: lower (more negative) = more anomalous.
    """
    clf = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    clf.fit(X_normal)
    return clf


def fit_one_class_svm(X_normal: np.ndarray, nu: float = 0.05) -> OneClassSVM:
    """
    Train a One-Class SVM on normal-only feature vectors.
    nu ≈ upper bound on fraction of training outliers.
    Score convention: lower (more negative) = more anomalous.
    """
    clf = OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
    clf.fit(X_normal)
    return clf

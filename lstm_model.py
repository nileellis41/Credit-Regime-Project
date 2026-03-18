# lstm_model.py
"""
LSTM Credit Regime Classifier.

Architecture: 2-layer LSTM → LayerNorm → Dropout → Linear(N_REGIMES)
Training:     supervised on HMM Viterbi pseudo-labels (with optional label smoothing)
              class-weighted cross-entropy to handle regime imbalance
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
from config import (
    N_REGIMES, SEQUENCE_LEN, HIDDEN_SIZE, NUM_LAYERS, DROPOUT,
    BIDIRECTIONAL, BATCH_SIZE, MAX_EPOCHS, LR, WEIGHT_DECAY,
    PATIENCE, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED, MODEL_DIR,
)
from features import make_sequences


# ─── Model definition ─────────────────────────────────────────────────────────

class CreditRegimeLSTM(nn.Module):
    """
    Two-layer LSTM → LayerNorm → Dropout → 3-class softmax output.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
        bidirectional: bool = BIDIRECTIONAL,
        n_classes: int = N_REGIMES,
    ):
        super().__init__()
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        D = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * D)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * D, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_size)
        Returns logits: (batch, n_classes)
        """
        out, _ = self.lstm(x)           # (batch, seq_len, D*hidden)
        out    = out[:, -1, :]          # last timestep
        out    = self.layer_norm(out)
        out    = self.dropout(out)
        logits = self.classifier(out)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)


# ─── Train / eval helpers ─────────────────────────────────────────────────────

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _split_indices(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological (non-shuffled) train/val/test split."""
    n_test = int(n * TEST_SPLIT)
    n_val  = int(n * VAL_SPLIT)
    n_train = n - n_val - n_test
    train_idx = np.arange(0, n_train)
    val_idx   = np.arange(n_train, n_train + n_val)
    test_idx  = np.arange(n_train + n_val, n)
    return train_idx, val_idx, test_idx


def _make_loader(X_seq, y_seq, indices, shuffle=True) -> DataLoader:
    Xt = torch.tensor(X_seq[indices])
    yt = torch.tensor(y_seq[indices])
    ds = TensorDataset(Xt, yt)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=False)


# ─── Training loop ────────────────────────────────────────────────────────────

def train_lstm(
    X: np.ndarray,         # (T, n_features)  scaled
    y: np.ndarray,         # (T,) regime labels from HMM
    seq_len: int = SEQUENCE_LEN,
) -> tuple["CreditRegimeLSTM", dict]:
    """
    Build sequences, split train/val/test, train with early stopping.

    Returns
    -------
    model       : trained CreditRegimeLSTM
    history     : dict with loss/accuracy curves and test metrics
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = _get_device()
    print(f"[LSTM] Training on device: {device}")

    # ── Build sequences ────────────────────────────────────────────────────────
    X_seq, y_seq = make_sequences(X, y, seq_len)
    n, T, F = X_seq.shape
    print(f"[LSTM] Sequences: {n}  shape=({T}, {F})  labels unique={np.unique(y_seq)}")

    # ── Splits ────────────────────────────────────────────────────────────────
    train_idx, val_idx, test_idx = _split_indices(n)
    train_loader = _make_loader(X_seq, y_seq, train_idx, shuffle=True)
    val_loader   = _make_loader(X_seq, y_seq, val_idx,   shuffle=False)
    test_loader  = _make_loader(X_seq, y_seq, test_idx,  shuffle=False)

    # ── Class weights for imbalanced regimes ──────────────────────────────────
    y_train = y_seq[train_idx]
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    # Pad to N_REGIMES if any class missing in train split
    weights = np.ones(N_REGIMES)
    for i, c in enumerate(np.unique(y_train)):
        weights[c] = class_weights[i]
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    # ── Model, loss, optimizer ────────────────────────────────────────────────
    model = CreditRegimeLSTM(input_size=F).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    # ── Training loop ─────────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = np.inf
    patience_counter = 0
    best_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_loss    += loss.item() * len(yb)
            t_correct += (logits.argmax(1) == yb).sum().item()
            t_total   += len(yb)

        # Validate
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                loss = criterion(logits, yb)
                v_loss    += loss.item() * len(yb)
                v_correct += (logits.argmax(1) == yb).sum().item()
                v_total   += len(yb)

        scheduler.step()

        train_loss = t_loss / t_total
        val_loss   = v_loss / v_total
        train_acc  = t_correct / t_total
        val_acc    = v_correct / v_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{MAX_EPOCHS}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"val_acc={val_acc:.3f}")

        # Early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[LSTM] Early stopping at epoch {epoch}")
                break

    # Restore best weights
    model.load_state_dict(best_state)

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            probs = model.predict_proba(Xb)
            all_probs.append(probs.cpu().numpy())
            all_preds.extend(probs.argmax(1).cpu().numpy())
            all_labels.extend(yb.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.vstack(all_probs)

    test_acc = (all_preds == all_labels).mean()
    history["test_acc"]    = test_acc
    history["test_preds"]  = all_preds
    history["test_labels"] = all_labels
    history["test_probs"]  = all_probs
    history["test_idx"]    = test_idx
    print(f"[LSTM] Test accuracy: {test_acc:.4f}")

    return model, history


# ─── Full-sequence inference ──────────────────────────────────────────────────

def lstm_predict_full(
    model: "CreditRegimeLSTM",
    X: np.ndarray,
    seq_len: int = SEQUENCE_LEN,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference over the full time series using a sliding window.
    The first (seq_len - 1) timesteps get the label of the first window.

    Returns
    -------
    states      : (T,) integer regime labels
    posteriors  : (T, N_REGIMES) softmax probabilities
    """
    device = _get_device()
    model.eval().to(device)

    states, posteriors = [], []
    with torch.no_grad():
        for i in range(len(X) - seq_len + 1):
            window = torch.tensor(X[i : i + seq_len]).unsqueeze(0).to(device)
            probs  = model.predict_proba(window).cpu().numpy()[0]
            posteriors.append(probs)
            states.append(int(probs.argmax()))

    states     = np.array(states, dtype=np.int64)
    posteriors = np.array(posteriors)

    # Pad the front so length = T
    pad = len(X) - len(states)
    states     = np.concatenate([np.full(pad, states[0]), states])
    posteriors = np.vstack([np.tile(posteriors[0], (pad, 1)), posteriors])

    return states, posteriors


# ─── Persistence ──────────────────────────────────────────────────────────────

def save_lstm(model: "CreditRegimeLSTM", path: str = None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = path or os.path.join(MODEL_DIR, "lstm.pt")
    torch.save(model.state_dict(), path)
    print(f"[LSTM] Saved to {path}")


def load_lstm(input_size: int, path: str = None) -> "CreditRegimeLSTM":
    path = path or os.path.join(MODEL_DIR, "lstm.pt")
    model = CreditRegimeLSTM(input_size=input_size)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    print(f"[LSTM] Loaded from {path}")
    return model


if __name__ == "__main__":
    # Smoke test with synthetic data
    T, F = 200, 18
    X_fake = np.random.randn(T, F).astype(np.float32)
    y_fake = np.random.randint(0, N_REGIMES, T).astype(np.int64)
    model, history = train_lstm(X_fake, y_fake)
    print("Test acc:", history["test_acc"])

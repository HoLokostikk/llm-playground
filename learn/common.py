"""
common.py — підготовка даних Titanic і базова модель.
Використання:
    from common import load_titanic, Net, criterion
"""

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = ["age", "pclass", "sex", "fare", "sibsp", "parch"]
TARGET = "survived"
SEED = 42


def load_titanic(standardize=True, test_size=0.2):
    """Повертає (X_train, X_val, y_train, y_val) як float32-тензори."""
    df = sns.load_dataset("titanic")
    data = df[FEATURES + [TARGET]].copy()

    X = data[FEATURES]
    y = data[TARGET].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )

    # медіана рахується ТІЛЬКИ на трені — інакше витік даних
    age_median = X_train["age"].median()
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_train["age"] = X_train["age"].fillna(age_median)
    X_val["age"] = X_val["age"].fillna(age_median)

    # текст -> числа (map, а не ==, щоб повторний виклик нічого не ламав)
    for part in (X_train, X_val):
        part["sex"] = part["sex"].map({"male": 1, "female": 0}).astype(int)

    X_train = X_train.values.astype(np.float64)
    X_val = X_val.values.astype(np.float64)

    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    return to_t(X_train), to_t(X_val), to_t(y_train), to_t(y_val)


class Net(nn.Module):
    """Базова MLP для бінарної класифікації. Останній шар видає логіт."""

    def __init__(self, n_features, hidden=(32, 16)):
        super().__init__()
        layers = []
        prev = n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


criterion = nn.BCEWithLogitsLoss()


def accuracy(model, X, y):
    """Частка правильних відповідей. Поріг: логіт > 0."""
    model.eval()
    with torch.no_grad():
        return ((model(X) > 0).float() == y).float().mean().item()

from torch.utils.data import Dataset, DataLoader


class TabularDataset(Dataset):
    def __init__(self, X, y):
        assert len(X) == len(y), f"{len(X)} != {len(y)}"
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_model(X_train, y_train, X_val, y_val, *,
                batch_size=64, lr=0.1, epochs=200, seed=0,
                opt_cls=torch.optim.SGD, hidden=(32, 16),
                accum=1, shuffle=True, patience=10, verbose=False):
    """Тренує Net з early stopping. Повертає (model, best_val, stop_epoch)."""
    import copy

    torch.manual_seed(seed)
    model = Net(X_train.shape[1], hidden=hidden)
    optimizer = opt_cls(model.parameters(), lr=lr)

    loader = DataLoader(TabularDataset(X_train, y_train),
                        batch_size=batch_size, shuffle=shuffle)

    best_val, best_state, wait = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        for i, (xb, yb) in enumerate(loader):
            loss = criterion(model(xb), yb) / accum
            loss.backward()
            if (i + 1) % accum == 0:
                optimizer.step()
                optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()

        if val_loss < best_val:
            best_val, wait = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break

        if verbose and epoch % 10 == 0:
            print(f"epoch {epoch:3d}  val={val_loss:.4f}")

    model.load_state_dict(best_state)
    return model, best_val, epoch


def run_seeds(X_train, y_train, X_val, y_val, seeds=5, **kwargs):
    """Прогін по кількох сідах. Повертає (середнє, розкид)."""
    accs = []
    for s in range(seeds):
        model, _, _ = train_model(X_train, y_train, X_val, y_val, seed=s, **kwargs)
        accs.append(accuracy(model, X_val, y_val))
    return float(np.mean(accs)), float(np.std(accs))


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = load_titanic()
    print("train:", X_train.shape, "val:", X_val.shape)
    print("dtype:", X_train.dtype)
    print("середнє по ознаках:", X_train.mean(dim=0).round(decimals=3))
    print("частка виживших у трені:", y_train.mean().item())
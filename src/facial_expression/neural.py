"""Seeded CPU PyTorch MLP tuned and stopped using validation macro F1."""
from copy import deepcopy
import random

from imblearn.over_sampling import SMOTE
import numpy as np
import optuna
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import CLASSES


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def fit_network(X, y, X_val, y_val, params, seed, epochs, patience):
    set_seed(seed)
    model = nn.Sequential(
        nn.Linear(X.shape[1], params["hidden_dim"]), nn.ReLU(),
        nn.Dropout(params["dropout"]), nn.Linear(params["hidden_dim"], len(CLASSES)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)),
        batch_size=params["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
    )
    val_tensor = torch.tensor(X_val, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss()
    best_score, best_state, stale, best_epoch = -1., None, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            prediction = model(val_tensor).argmax(dim=1).numpy()
        score = f1_score(y_val, prediction, labels=range(len(CLASSES)), average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": loss_total / len(X), "validation_macro_f1": score})
        if score > best_score + 1e-10:
            best_score, best_state, best_epoch, stale = score, deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_score, best_epoch, history


class MLPClassifier:
    def __init__(self, scaler, model):
        self.scaler, self.model = scaler, model

    def predict(self, X):
        values = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            indices = self.model(values).argmax(dim=1).numpy()
        return np.asarray(CLASSES)[indices]


def tune_mlp(X, y, X_val, y_val, seed=42, quick=False):
    # Only training data fits scaler/SMOTE. Validation keeps its natural distribution.
    scaler = StandardScaler().fit(X)
    train_scaled, val_scaled = scaler.transform(X), scaler.transform(X_val)
    lookup = {name: i for i, name in enumerate(CLASSES)}
    encoded = np.asarray([lookup[name] for name in y])
    val_encoded = np.asarray([lookup[name] for name in y_val])
    balanced, labels = SMOTE(random_state=seed).fit_resample(train_scaled, encoded)

    def objective(trial):
        params = {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [16, 32, 64]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.4),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        }
        _, score, _, _ = fit_network(
            balanced, labels, val_scaled, val_encoded, params, seed,
            epochs=30 if quick else 200, patience=10 if quick else 40,
        )
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=2 if quick else 12, n_jobs=1)
    model, score, epoch, history = fit_network(
        balanced, labels, val_scaled, val_encoded, study.best_params, seed,
        epochs=30 if quick else 600, patience=10 if quick else 60,
    )
    details = {
        "parameters": study.best_params, "tuning_validation_macro_f1": study.best_value,
        "selected_epoch": epoch, "selected_validation_macro_f1": score,
        "trials": [{"number": t.number, "validation_macro_f1": t.value, "parameters": t.params}
                   for t in study.trials],
        "selection": "12 Optuna trials (2 in quick mode), validation macro F1; best validation checkpoint",
    }
    return MLPClassifier(scaler, model), details, history

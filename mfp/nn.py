"""A small PyTorch neural network wrapped as a scikit-learn classifier.

Wrapping it lets the network go through exactly the same pipeline, cross-validation and
metrics as the scikit-learn models, so the comparison is like for like.
torch is imported lazily: the serving API never needs it.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """Feed-forward network for binary classification with a class-weighted loss.

    pos_weight = (#negatives / #positives) makes each rare failure count as much as the
    many normal rows in total, which matters when only ~3% of rows are failures.
    """

    def __init__(self, hidden=(64, 32), dropout=0.1, lr=3e-3, weight_decay=1e-4,
                 epochs=150, batch_size=128, random_state=42):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def _build(self, n_in):
        import torch.nn as nn

        layers, width = [], n_in
        for h in self.hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(self.dropout)]
            width = h
        layers.append(nn.Linear(width, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        import torch

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        torch.manual_seed(self.random_state)
        gen = torch.Generator().manual_seed(self.random_state)
        model = self._build(X.shape[1])
        pos = max(float(y.sum()), 1.0)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y) - pos) / pos))
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
        self.loss_history_ = []
        model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(Xt), generator=gen)
            total = 0.0
            for i in range(0, len(Xt), self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(model(Xt[idx]).squeeze(1), yt[idx])
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            self.loss_history_.append(total / len(Xt))
        model.eval()
        self.model_ = model
        return self

    def predict_proba(self, X):
        import torch

        X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            p = torch.sigmoid(self.model_(X).squeeze(1)).numpy().astype(float)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

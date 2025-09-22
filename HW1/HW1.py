# Joe Ferguson - HW1 (finished)
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as skl
import torch
from torch._functorch.eager_transforms import linearize
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error

# -------- Load data ----------
df = pd.read_csv("cancer_reg.csv", encoding="ISO-8859-1")

# Dataset overview
print("Number of data points:", df.shape[0])  # 2. number of samples
print("The problem is to predict TARGET_deathRate")  # 2. regression target

# Check min/max values of columns
print("Min values (per-column):")
print(df.min(numeric_only=True))
print("The min value of TARGET_deathRate is", df["TARGET_deathRate"].min())
print("Max values (per-column):")
print(df.max(numeric_only=True))
print("The max value of TARGET_deathRate is", df["TARGET_deathRate"].max())

# Number of features
print("Features per data point:", df.shape[1] - 1)

# Missing values
print("Missing values per column:")
print(df.isnull().sum())


# ----------------- Data Preprocessing -----------------
def bin_to_midpoint(val):
    """
    Convert binned income strings like "(10000,20000]" into their midpoint value.
    Returns NaN if parsing fails.
    """
    if pd.isnull(val):
        return np.nan
    s = str(val).strip("()[]")  # remove brackets
    parts = s.split(",")
    try:
        a = float(parts[0])
        b = float(parts[1])
        return (a + b) / 2.0
    except Exception:
        return np.nan

# Apply transformation to income column
df["binnedInc"] = df["binnedInc"].apply(bin_to_midpoint)

# Encode categorical Geography column
df["Geography"] = skl.LabelEncoder().fit_transform(df["Geography"].astype(str))

# Fill missing values with column medians
df = df.fillna(df.median(numeric_only=True))

# Define label column
label_col = "TARGET_deathRate"

# Split into train (70%), val (15%), test (15%)
train_df, combine_df = train_test_split(df, test_size=0.25, random_state=104)
test_df, val_df = train_test_split(combine_df, test_size=0.5, random_state=104)

print("Train size:", len(train_df))
print("Validation size:", len(val_df))
print("Test size:", len(test_df))

# Separate features and labels
X_train_df = train_df.drop(columns=[label_col])
y_train = train_df[label_col].values.reshape(-1, 1)
X_val_df = val_df.drop(columns=[label_col])
y_val = val_df[label_col].values.reshape(-1, 1)
X_test_df = test_df.drop(columns=[label_col])
y_test = test_df[label_col].values.reshape(-1, 1)

# Fill NaNs using training set means
X_train_df = X_train_df.fillna(X_train_df.mean())
X_val_df = X_val_df.fillna(X_train_df.mean())
X_test_df = X_test_df.fillna(X_train_df.mean())

# Apply log1p transform to reduce skewness
X_train = np.log1p(X_train_df.values)
X_val = np.log1p(X_val_df.values)
X_test = np.log1p(X_test_df.values)

# Standardize features (zero mean, unit variance)
scaler = skl.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# Create Tensor datasets
train_ds = TensorDataset(X_train_t, y_train_t)
val_ds = TensorDataset(X_val_t, y_val_t)
test_ds = TensorDataset(X_test_t, y_test_t)

# Number of input features
dim = X_train.shape[1]

# Different neural net architectures to test
architectures = {
    "Linear Regression": [],
    "DNN-8-4": [8,4],
    "DNN-16": [16],
    "DNN-30-8": [30, 8],
    "DNN-30-16-8": [30, 16, 8],
    "DNN-30-16-8-4": [30, 16, 8, 4],
}

# Common activation functions
ReLU = nn.ReLU()
LeakyReLU = nn.LeakyReLU()
Sigmoid = nn.Sigmoid()
Tanh = nn.Tanh()


# ----------------- Model Classes -----------------
class LinearModel(nn.Module):
    """
    Simple linear regression model using PyTorch.
    """
    def __init__(self, dim, bs, lr, epochs,verbose):
        super().__init__()
        self.linear = nn.Linear(dim, 1)  # single linear layer
        # Train immediately after initialization
        self.train_losses, self.val_losses = train_model(self, bs, lr, epochs,verbose)

    def forward(self, x):
        return self.linear(x)

    def test(self):
        """
        Evaluate model on test set, print metrics, and plot results.
        """
        mse, r2, preds, y_true = evaluate_model(self, X_test_t, y_test)
        print(f"Test MSE: {mse:.4f}, Test R2: {r2:.4f}")

        plt.figure(figsize=(12, 4))

        # Left: training/validation loss curves
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label="train loss")
        plt.plot(self.val_losses, label="val loss")
        plt.title("Loss curves: Linear Regression")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()

        # Right: predictions vs actual values
        plt.subplot(1, 2, 2)
        plt.scatter(y_true, preds, alpha=0.5)
        min_val, max_val = y_true.min(), y_true.max()
        plt.plot([min_val, max_val], [min_val, max_val], "r--")
        plt.title(f"Predictions vs Actual\nMSE={mse:.2f}, R2={r2:.2f}")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.text(0.95, 0.02, f"Generated: {timestamp}",
             transform=plt.gca().transAxes,
             fontsize=9, color="gray",
             ha="right", va="bottom")
        plt.tight_layout()
        plt.show()


class DeepModel():
    """
    Deep neural network regression model with configurable architecture.
    """
    train_losses = []
    val_losses = []
    model = None
    model_arch = None
    actFunc = None

    def __init__(self, arch, activationFunction, bs, lr, epochs, verbose):
        self.actFunc = activationFunction
        self.model_arch = arch
        # Build network from architecture
        self.model = build_dnn(dim, architectures[arch], activationFunction)
        # Train the network
        self.train_losses, self.val_losses = train_model(self.model, bs, lr, epochs,verbose)

    def test(self):
        """
        Evaluate deep model on test set, print metrics, and plot results.
        """
        mse, r2, preds, y_true = evaluate_model(self.model, X_test_t, y_test)
        print(f"Test MSE: {mse:.4f}, Test R2: {r2:.4f}")

        plt.figure(figsize=(12, 4))

        # Left: loss curves
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label="train loss")
        plt.plot(self.val_losses, label="val loss")
        plt.title(f"Loss curves: {self.model_arch} ({self.actFunc})")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()

        # Right: predictions vs actual values
        plt.subplot(1, 2, 2)
        plt.scatter(y_true, preds, alpha=0.5)
        min_val, max_val = y_true.min(), y_true.max()
        plt.plot([min_val, max_val], [min_val, max_val], "r--")
        plt.title(f"Predictions vs Actual\nMSE={mse:.2f}, R2={r2:.2f}")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.text(0.95, 0.02, f"Generated: {timestamp}",
             transform=plt.gca().transAxes,
             fontsize=9, color="gray",
             ha="right", va="bottom")
        plt.tight_layout()
        plt.show()


# ----------------- Helper Functions -----------------
def build_dnn(input_dim, hidden_layers, activation):
    """
    Build a feedforward neural network given input dimension,
    list of hidden layer sizes, and activation function.
    """
    layers = []
    prev_dim = input_dim
    for h in hidden_layers:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(activation)
        prev_dim = h
    layers.append(nn.Linear(prev_dim, 1))  # output layer
    return nn.Sequential(*layers)


def train_model(model, bs, lr, epochs, verbose=True):
    """
    Train a given PyTorch model using SGD optimizer and MSE loss.
    Returns training and validation loss histories.
    """
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    train_losses, val_losses = [], []
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        train_losses.append(np.mean(batch_losses))

        # Validation step
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t).item()
            val_losses.append(val_loss)

        # Optional progress printing
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"Epoch {epoch}: Train Loss={train_losses[-1]:.4f}, Val Loss={val_loss:.4f}, Timestamp={timestamp}")

    return train_losses, val_losses


def evaluate_model(model, X_tensor, y_array):
    """
    Evaluate model predictions using MSE and R².
    Returns metrics, predictions, and ground truth.
    """
    model.eval()
    with torch.no_grad():
        preds_t = model(X_tensor)
    preds = preds_t.cpu().numpy().reshape(-1, 1)

    # Ensure ground truth is numpy
    if isinstance(y_array, torch.Tensor):
        y_np = y_array.cpu().numpy().reshape(-1, 1)
    else:
        y_np = np.array(y_array).reshape(-1, 1)

    # Compute metrics safely
    try:
        mse = mean_squared_error(y_np, preds)
    except Exception:
        mse = np.nan
    try:
        r2 = r2_score(y_np, preds)
    except Exception:
        r2 = np.nan

    # Handle invalid values
    if np.isnan(mse) or np.isinf(mse):
        mse = np.nan
    if np.isnan(r2) or np.isinf(r2):
        r2 = np.nan

    return mse, r2, preds, y_np


def test_model(model_architexture, activationFunction, batchsize, learningrate, epochs, verbose=False):
    """
    Train and evaluate either a LinearModel or DeepModel
    based on the chosen architecture.
    """
    if "Linear" in model_architexture:
        model = LinearModel(dim, batchsize, learningrate, epochs, verbose)
        model.test()
    else:
        model = DeepModel(model_architexture, activationFunction, batchsize, learningrate, epochs,verbose)
        model.test()


# ----------------- Main script -----------------
if __name__ == "__main__":
    # Run tests with same architecture but different learning rates
    test_model(model_architexture="DNN-8-4", activationFunction=LeakyReLU, batchsize=16, learningrate=0.001, epochs=100, verbose=True)

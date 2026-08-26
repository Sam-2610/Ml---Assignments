"""
ANN Regression - Power Plant Energy Prediction
(Corrected & commented version of 1_1__ANN_Regression.ipynb)

Fixes made to the original code:
1. Fixed typo: "crietrion" -> "criterion" (kept meaning same, just spelled correctly)
2. Converted tensors to numpy arrays explicitly before passing to sklearn's r2_score
   (sklearn expects numpy arrays / array-like objects, not raw torch tensors with grad history)
3. Used loss.item() consistently when accumulating validation loss, so we accumulate
   plain Python floats instead of tensors (avoids unnecessary memory build-up)
4. Added torch.manual_seed() for reproducible results every time the script is run
5. Added device handling (works on GPU automatically if available, otherwise CPU)
6. Added comments explaining each step of the pipeline
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Reproducibility: fix the random seed so results are consistent
# across runs (helps when comparing model performance / debugging)
# ------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)

# ------------------------------------------------------------------
# Use GPU if available, otherwise fall back to CPU
# ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# 1. Load the dataset
# ------------------------------------------------------------------
df = pd.read_csv("powerplant_data.csv")

# Preview the first 5 rows
print(df.head())

# Column meanings:
# AT => Ambient Temperature
# V  => Exhaust Vacuum
# AP => Ambient Pressure
# RH => Relative Humidity
# PE => Net hourly Electrical Energy output (target variable)

# ------------------------------------------------------------------
# 2. Check for missing values before doing anything else
# ------------------------------------------------------------------
print(df.isnull().sum())

# ------------------------------------------------------------------
# 3. Separate features (X) and target (y)
# ------------------------------------------------------------------
X = df.drop("PE", axis=1)
y = df["PE"]

print(y.head())

# ------------------------------------------------------------------
# 4. Split into training and testing sets (80% train, 20% test)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(df.shape)

# ------------------------------------------------------------------
# 5. Feature scaling (standardize features to mean=0, std=1)
#    Important for neural networks since they are sensitive to
#    the scale of input features. The scaler is FIT only on the
#    training data and then used to TRANSFORM both train and test
#    sets, to avoid data leakage from the test set.
# ------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_test_scaled)

# ------------------------------------------------------------------
# 6. Convert numpy arrays into PyTorch tensors
#    (neural networks in PyTorch operate on tensors, not
#    pandas/numpy objects directly)
# ------------------------------------------------------------------
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)

print(type(X_train_scaled))
print(type(y_train))
print(y_train.shape)

# ------------------------------------------------------------------
# 7. Wrap tensors into Dataset and DataLoader objects
#    DataLoader handles batching and shuffling automatically
# ------------------------------------------------------------------
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


# ------------------------------------------------------------------
# 8. Define the ANN (Artificial Neural Network) architecture
#    - Input layer size = number of features (4: AT, V, AP, RH)
#    - 2 hidden layers with 6 neurons each and ReLU activation
#    - Output layer with 1 neuron (regression -> single continuous value)
# ------------------------------------------------------------------
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()

        self.model = nn.Sequential(
            # 1st hidden layer
            nn.Linear(X_train.shape[1], 6),
            nn.ReLU(),

            # 2nd hidden layer
            nn.Linear(6, 6),
            nn.ReLU(),

            # output layer (no activation, since this is regression)
            nn.Linear(6, 1),
        )

    def forward(self, x):
        return self.model(x)


# ------------------------------------------------------------------
# 9. Initialize the model, loss function, and optimizer
# ------------------------------------------------------------------
model = ANN().to(device)

# Mean Squared Error is the standard loss function for regression tasks
criterion = nn.MSELoss()

# Adam optimizer adapts the learning rate automatically during training
optimizer = optim.Adam(model.parameters())

# ------------------------------------------------------------------
# 10. Train the ANN
#     For each epoch:
#       - Loop over training batches, compute loss, backpropagate,
#         and update weights
#       - After each epoch, evaluate on the test/validation set
#         (without updating weights) to track generalization
#       - Save the model whenever validation loss improves
#         (this is a simple form of "early checkpointing")
# ------------------------------------------------------------------
train_losses = []
val_losses = []

best_val_loss = float("inf")
epochs = 100

for epoch in range(epochs):
    model.train()  # set model to training mode
    running_loss = 0.0  # total training loss for this epoch

    for xb, yb in train_loader:
        # xb = features of 1 batch, yb = target values of 1 batch
        optimizer.zero_grad()          # clear gradients from previous step

        outputs = model(xb)            # forward pass: predicted outputs for this batch
        loss = criterion(outputs, yb)  # compute MSE loss
        loss.backward()                # backward pass: compute gradients
        optimizer.step()               # update model parameters using gradients

        running_loss += loss.item()    # accumulate loss as a plain Python float

    epoch_train_loss = running_loss / len(train_loader)
    train_losses.append(epoch_train_loss)

    # ---------------- Validation ----------------
    model.eval()  # set model to evaluation mode (disables dropout/batchnorm updates, if any)
    running_val_loss = 0.0

    with torch.no_grad():  # disable gradient computation (not needed for validation)
        for xb, yb in test_loader:
            outputs = model(xb)
            loss = criterion(outputs, yb)
            running_val_loss += loss.item()  # use .item() to store a float, not a tensor

    epoch_val_loss = running_val_loss / len(test_loader)
    val_losses.append(epoch_val_loss)

    print(f"epoch {epoch+1}/{epochs} ==> train loss = {epoch_train_loss:.4f} & val loss = {epoch_val_loss:.4f}")

    # Save the model only when validation loss improves (best model so far)
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), "best_model.pt")  # .pt or .pth extension

# ------------------------------------------------------------------
# 11. Plot training vs validation loss curves
#     Helps visually check for overfitting/underfitting
# ------------------------------------------------------------------
loss_df = pd.DataFrame({
    "Training Loss": train_losses,
    "Validation Loss": val_losses
})

plt.plot(loss_df["Training Loss"], label="Training Loss")
plt.plot(loss_df["Validation Loss"], label="Validation Loss")

plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()

# ------------------------------------------------------------------
# 12. Load the best saved model (lowest validation loss)
#     before final evaluation
# ------------------------------------------------------------------
model.load_state_dict(torch.load("best_model.pt"))

# ------------------------------------------------------------------
# 13. Final evaluation on both train and test sets
# ------------------------------------------------------------------
model.eval()
with torch.no_grad():
    train_preds = model(X_train_tensor)
    test_preds = model(X_test_tensor)

    train_mse_loss = criterion(train_preds, y_train_tensor)
    test_mse_loss = criterion(test_preds, y_test_tensor)

print("Training MSE:", train_mse_loss.item())
print("Testing MSE:", test_mse_loss.item())

# ------------------------------------------------------------------
# 14. R^2 score (coefficient of determination)
#     Explicitly convert tensors to numpy arrays first, since
#     sklearn's metrics expect numpy/array-like input, not
#     raw torch tensors. Also move to CPU in case GPU was used.
# ------------------------------------------------------------------
test_preds_np = test_preds.cpu().numpy()
print("r^2 score =", r2_score(y_test, test_preds_np))

# ------------------------------------------------------------------
# 15. Compare predicted vs actual values side by side
# ------------------------------------------------------------------
predicted_df = pd.DataFrame(test_preds_np, columns=["Predicted Values"])
actual_df = pd.DataFrame(y_test.values, columns=["Actual Values"])

comparison_df = pd.concat([predicted_df, actual_df], axis=1)
print(comparison_df)
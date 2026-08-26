"""
ANN Classification - Date Fruit Type Classification
(Corrected & commented version of 8_1__ANN_Classification.ipynb)

Fixes made to the original code:
1. Moved all imports to the top of the file (they were scattered across cells)
2. Renamed "criteria" -> "criterion" for consistency with standard PyTorch naming
   (functionally identical, just a cleaner/standard variable name)
3. Added torch.manual_seed() / np.random.seed() for reproducible results
4. Added device handling (automatically uses GPU if available, else CPU)
5. Added comments explaining each step of the pipeline
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

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
df = pd.read_csv("DateFruit_Dataset.csv")

# Preview the first 5 rows
print(df.head())  # 34 features, 1 category (target class)

print(df.shape)

# ------------------------------------------------------------------
# 2. Separate features (X) and target (y)
# ------------------------------------------------------------------
X = df.drop("Class", axis=1)
y = df["Class"]

# Check the unique fruit classes present in the dataset
print(df["Class"].unique())

# ------------------------------------------------------------------
# 3. Encode target labels
#    Neural networks need numeric labels, not text/string categories,
#    so we convert class names (e.g. "DOKOL", "SAFAVI", ...) into
#    integers (0, 1, 2, ...) using LabelEncoder.
# ------------------------------------------------------------------
le = LabelEncoder()
y = le.fit_transform(y)

# ------------------------------------------------------------------
# 4. Split into training and testing sets (80% train, 20% test)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------------
# 5. Feature scaling (standardize features to mean=0, std=1)
#    The scaler is FIT only on the training data and then used to
#    TRANSFORM both train and test sets, to avoid data leakage.
# ------------------------------------------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(y)

# ------------------------------------------------------------------
# 6. Convert numpy arrays into PyTorch tensors
#    Features -> float32 (required for nn.Linear layers)
#    Labels   -> long/int64 (required by nn.CrossEntropyLoss)
# ------------------------------------------------------------------
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# ------------------------------------------------------------------
# 7. Wrap tensors into Dataset and DataLoader objects
#    DataLoader handles batching and shuffling automatically
# ------------------------------------------------------------------
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


# ------------------------------------------------------------------
# 8. Build our Model
#    - Input layer size = number of features (34 measurements per fruit)
#    - 2 hidden layers with 64 neurons each and ReLU activation
#    - Output layer with 7 neurons (one per fruit class), since this
#      dataset has 7 possible date fruit types
# ------------------------------------------------------------------
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(X.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 7)   # 7 output classes (raw logits, no softmax here)
        )

    def forward(self, x):
        return self.model(x)


# ------------------------------------------------------------------
# 9. Initialize the model, loss function, and optimizer
# ------------------------------------------------------------------
model = ANN().to(device)

# CrossEntropyLoss expects raw logits + integer class labels,
# and internally applies softmax for us — that's why the model's
# output layer above does NOT have a softmax activation.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# ------------------------------------------------------------------
# 10. Train the ANN
#     For each epoch:
#       - Loop over training batches
#       - Forward pass -> compute loss -> backward pass -> update weights
#       - Track and print average training loss for the epoch
# ------------------------------------------------------------------
epochs = 100
for epoch in range(epochs):
    model.train()  # set model to training mode

    running_loss = 0.0

    for xb, yb in train_loader:
        optimizer.zero_grad()          # clear gradients from previous step

        outputs = model(xb)            # forward pass: raw class scores (logits)
        loss = criterion(outputs, yb)  # compute cross-entropy loss
        loss.backward()                # backward pass: compute gradients
        optimizer.step()               # update model parameters

        running_loss += loss.item()    # accumulate loss as a plain Python float

    train_loss = running_loss / len(train_loader)

    print(f"epoch = {epoch+1}/{epochs}, loss = {train_loss:.4f}")

# ------------------------------------------------------------------
# 11. Evaluate the trained model on the test set
#     For classification, we pick the class with the highest score
#     (logit) as the predicted class using torch.max, then compare
#     against the true labels to compute accuracy.
# ------------------------------------------------------------------
model.eval()

total = 0
correct = 0

with torch.no_grad():
    for xb, yb in test_loader:
        outputs = model(xb)  # e.g. [0.2, 0.5, 1.3, -0.5, ...] - 7 raw scores per sample
        _, predicted = torch.max(outputs, 1)  # index of the highest score = predicted class

        correct += (predicted == yb).sum().item()
        total += yb.size(0)  # number of actual samples in this batch

print("accuracy: ", correct / total * 100)

print(y_test.shape)
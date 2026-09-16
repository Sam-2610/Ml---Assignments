import re
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# NLTK needs these resources downloaded once per environment.
# Without this, word_tokenize() / stopwords.words() raise a LookupError.
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


# -----------------------------
# Load and inspect the dataset
# -----------------------------
df = pd.read_csv("IMDB Dataset.csv")
print(df.shape)
print(df.head())
print(df.isnull().sum())

df.drop_duplicates(inplace=True)
print(df.shape)


# -----------------------------
# Pre-processing
# -----------------------------

# 1. Convert all review text to lowercase for consistency
df["review"] = df["review"].str.lower()


# 2. Remove URLs (e.g. "http://..." or "https://...")
def remove_urls(text):
    return re.sub(r"http\S+", "", text)


df["review"] = df["review"].apply(remove_urls)


# 3. Remove punctuation, keeping only letters, digits and whitespace
def remove_punctuations(text):
    return re.sub(r"[^A-Za-z0-9\s]", "", text)


df["review"] = df["review"].apply(remove_punctuations)
print(df.head())


# 4. Remove HTML tags (e.g. "<br />")
def remove_html(text):
    return re.sub(r"<.*?>", "", text)


df["review"] = df["review"].apply(remove_html)


# 5. Remove stopwords (common words like "the", "is", "and" that carry
# little meaning for sentiment classification).
#
# NOTE: the original version used text.replace(word, "") inside a loop,
# which does substring replacement rather than removing whole tokens —
# e.g. removing "a" would also strip the "a" out of "cat" ("c t").
# Rebuilding the string from filtered tokens avoids that bug.
stop_words = set(stopwords.words("english"))


def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered)


df["review"] = df["review"].apply(remove_stopwords)


# 6. Stemming - reduce words to their root form (e.g. "running" -> "run")
ps = PorterStemmer()


def stemming(text):
    tokens = word_tokenize(text)
    stemmed_words = [ps.stem(token) for token in tokens]
    return " ".join(stemmed_words)


df["review"] = df["review"].apply(stemming)


# 7. Encode the target labels ("positive"/"negative") into 0/1
le = LabelEncoder()
df["sentiment"] = le.fit_transform(df["sentiment"])
y = df["sentiment"]


# 8. Vectorize the cleaned text using TF-IDF (top 5000 most informative terms)
tf = TfidfVectorizer(max_features=5000)
X = tf.fit_transform(df["review"])


# -----------------------------
# Train / test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape)
print(X_test.shape)

# Convert sparse TF-IDF matrices to dense arrays for PyTorch tensors
X_train = X_train.toarray()
X_test = X_test.toarray()

train_set = TensorDataset(
    torch.from_numpy(X_train).float(),
    torch.from_numpy(y_train.values).float(),
)

test_set = TensorDataset(
    torch.from_numpy(X_test).float(),
    torch.from_numpy(y_test.values).float(),
)

train_loader = DataLoader(train_set, shuffle=True, batch_size=64)
test_loader = DataLoader(test_set, shuffle=True, batch_size=64)


# -----------------------------
# RNN model definition
# -----------------------------
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Recurrent layer: processes the input sequence step by step
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer: maps the final hidden state to 1 output
        # (a single logit for binary sentiment classification)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Initial hidden state, shape: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.rnn(x, h0)
        # out: hidden state at every timestep -> (batch, seq_len, hidden_size)
        # we only need the final timestep's output for classification

        out = self.fc(out[:, -1, :])
        return out


input_size = X_train.shape[1]
model = RNN(input_size)

# NOTE: BCELoss expects probabilities in [0, 1] as input, which is why
# sigmoid is applied manually to the model's raw output before computing
# the loss below. For numerical stability, nn.BCEWithLogitsLoss (used on
# the raw, un-sigmoided output) is generally preferred over this pattern,
# but this works correctly as written.
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())


# -----------------------------
# Training loop
# -----------------------------
epochs = 10

for epoch in range(epochs):
    model.train()

    for Xb, yb in train_loader:
        optimizer.zero_grad()

        # Add a singleton "sequence length" dimension: (batch, 1, features)
        Xb = Xb.unsqueeze(1)

        outputs = model(Xb)  # shape: (batch_size, 1)
        outputs = torch.sigmoid(outputs.squeeze())  # shape: (batch_size,) -> probability

        loss = criterion(outputs, yb)  # compute loss
        loss.backward()  # backpropagate
        optimizer.step()  # update weights

    print(f"epoch = {epoch + 1}/{epochs} and loss = {loss.item()}")


# -----------------------------
# Evaluation
# -----------------------------
model.eval()

with torch.no_grad():
    correct_vals = 0
    tot_vals = 0

    for Xb, yb in test_loader:
        Xb = Xb.unsqueeze(1)

        outputs = model(Xb)
        predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()

        tot_vals += yb.size(0)
        correct_vals += (predicted == yb).sum().item()

    print(f"accuracy = {correct_vals / tot_vals * 100}")
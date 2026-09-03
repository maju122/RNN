import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. Dataset Generation
# ==========================================
def generate_digit_sequences(num_samples=5000, seq_len=10):
    X, y = [], []
    for _ in range(num_samples):
        step = np.random.randint(1, 4)
        start = np.random.randint(0, 50)
        
        # 50% probability for forward, 50% for reversed
        label = np.random.choice([0, 1])
        
        if label == 1:  # Forward / Correct Order
            seq = [start + i * step for i in range(seq_len)]
        else:          # Reversed Order
            seq = [start - i * step for i in range(seq_len)]
            
        X.append(seq)
        y.append(label)
        
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

X, y = generate_digit_sequences(num_samples=5000, seq_len=10)

# Normalize inputs
X_mean, X_std = X.mean(), X.std()
X_norm = (X - X_mean) / X_std

# Train/Test Split (80/20)
train_size = int(0.8 * len(X))
X_train, X_test = X_norm[:train_size], X_norm[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

# ==========================================
# 2. RNN Model Definition
# ==========================================
class DigitOrderRNN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=1, num_classes=2):
        super(DigitOrderRNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = x.unsqueeze(-1)  # Add feature dimension
        out, h_n = self.rnn(x)
        # Take the output from the last time-step
        last_out = out[:, -1, :]
        logits = self.fc(last_out)
        return logits

model = DigitOrderRNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# ==========================================
# 3. Model Training
# ==========================================
epochs = 15
train_losses, train_accs = [], []

for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc*100:.2f}%")

# ==========================================
# 4. Evaluation
# ==========================================
model.eval()
test_correct, test_total = 0, 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)

test_acc = test_correct / test_total
print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")

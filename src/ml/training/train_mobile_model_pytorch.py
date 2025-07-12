import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from datetime import datetime
import argparse

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

CLASS_LABELS = [
    'dog_bark', 'car_horn', 'alarm', 'glass_break',
    'door_slam', 'siren', 'footsteps', 'speech',
    'music', 'machinery', 'nature', 'silence'
]
NUM_CLASSES = len(CLASS_LABELS)
SAMPLE_RATE = 22050
N_MELS = 128
DURATION = 2.0
FEATURE_SIZE = N_MELS * int(SAMPLE_RATE * DURATION / 512)

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def generate_synthetic_data(num_samples=1000):
    X = np.zeros((num_samples, FEATURE_SIZE), dtype=np.float32)
    y = np.random.randint(0, NUM_CLASSES, num_samples)
    for i in range(num_samples):
        class_idx = y[i]
        if class_idx == 0:
            pattern = np.random.normal(0, 1, FEATURE_SIZE) + np.sin(np.linspace(0, 10, FEATURE_SIZE))
        elif class_idx == 1:
            pattern = np.random.normal(0, 0.5, FEATURE_SIZE) + np.sin(np.linspace(0, 20, FEATURE_SIZE))
        elif class_idx == 2:
            pattern = np.random.normal(0, 0.8, FEATURE_SIZE) + np.sin(np.linspace(0, 15, FEATURE_SIZE))
        else:
            pattern = np.random.normal(0, 1, FEATURE_SIZE)
        X[i] = pattern
    return X, y

def train_model(epochs=20, batch_size=32, lr=1e-3, output_dir='ml_models'):
    os.makedirs(output_dir, exist_ok=True)
    X_train, y_train = generate_synthetic_data(2000)
    X_val, y_val = generate_synthetic_data(500)
    
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val)
    y_val = torch.tensor(y_val, dtype=torch.long)

    model = SimpleMLP(FEATURE_SIZE, NUM_CLASSES)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(X_train.size(0))
        X_train, y_train = X_train[idx], y_train[idx]
        for i in range(0, X_train.size(0), batch_size):
            xb = X_train[i:i+batch_size]
            yb = y_train[i:i+batch_size]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_pred = val_logits.argmax(dim=1)
            val_acc = (val_pred == y_val).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save best model
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
        print(f"Epoch {epoch+1}/{epochs} - Val Acc: {val_acc:.4f}")
    # Load best model
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pt')))
    # Export TorchScript
    model.eval()
    example_input = torch.randn(1, FEATURE_SIZE)
    traced = torch.jit.trace(model, example_input)
    torchscript_path = os.path.join(output_dir, 'mlp_model_scripted.pt')
    traced.save(torchscript_path)
    # Save metadata
    metadata = {
        'class_labels': CLASS_LABELS,
        'sample_rate': SAMPLE_RATE,
        'n_mels': N_MELS,
        'duration': DURATION,
        'feature_size': FEATURE_SIZE,
        'created_at': datetime.now().isoformat(),
        'val_accuracy': best_val_acc,
        'privacy_status': 'local_only'
    }
    with open(os.path.join(output_dir, 'mlp_model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nTraining complete. TorchScript model saved to {torchscript_path}")
    print(f"Validation accuracy: {best_val_acc:.4f}")
    print(f"Metadata saved to mlp_model_metadata.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output-dir', default='ml_models')
    args = parser.parse_args()
    train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, output_dir=args.output_dir) 
"""
federated/client.py
Simulates a local device in the federated learning process:
- Loads preprocessed features and labels
- Trains the fusion model locally
- Encrypts weight updates before sending to the server
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from federated.encryption import encrypt_weights

class LocalClient:
    def __init__(self, client_id, model, features, graph_embeddings, labels, device="cpu"):
        self.client_id = client_id
        self.model = copy.deepcopy(model)  # local copy of global model
        self.features = torch.tensor(features, dtype=torch.float32).to(device)
        self.graph_embeddings = torch.tensor(graph_embeddings, dtype=torch.float32).to(device)
        self.labels = torch.tensor(labels, dtype=torch.long).to(device)
        self.device = device

    def train_local(self, epochs=2, lr=0.001):
        self.model.to(self.device)
        self.model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(self.features, self.graph_embeddings)
            loss = criterion(outputs, self.labels)
            loss.backward()
            optimizer.step()

            print(f"[CLIENT {self.client_id}] Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

    def get_encrypted_weights(self):
        """Return encrypted model weights after training"""
        weights = {k: v.cpu().detach().numpy() for k, v in self.model.state_dict().items()}
        encrypted_weights = encrypt_weights(weights)
        return encrypted_weights


if __name__ == "__main__":
    # This is just for standalone testing — in practice, run_prototype.py will call this
    from models.transformer_model import TransformerModel
    import numpy as np

    # Dummy test
    feat_dim = 8
    graph_dim = 8
    batch_size = 10
    num_classes = 3

    class DummyFusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(feat_dim + graph_dim, 32)
            self.fc2 = nn.Linear(32, num_classes)

        def forward(self, features, graph_embeddings):
            x = torch.cat((features, graph_embeddings), dim=1)
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    # Create dummy model and client
    model = DummyFusion()
    features = np.random.rand(batch_size, feat_dim)
    graph_embeds = np.random.rand(batch_size, graph_dim)
    labels = np.random.randint(0, num_classes, batch_size)

    client = LocalClient(client_id=1, model=model, features=features, graph_embeddings=graph_embeds, labels=labels)
    client.train_local()
    encrypted = client.get_encrypted_weights()
    print("[DEBUG] Example encrypted weight keys:", list(encrypted.keys())[:3])


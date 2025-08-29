"""
federated/server.py
Prototype Federated Server:
- Receives encrypted model updates from clients
- Decrypts (prototype: simple placeholder)
- Aggregates updates via Federated Averaging
- Updates global model
"""

import copy
import numpy as np
import torch
from federated.encryption import decrypt_weights

class FederatedServer:
    def __init__(self, model, device="cpu"):
        self.global_model = model
        self.device = device

    def aggregate_updates(self, encrypted_client_weights_list):
        """
        Aggregate encrypted weights from multiple clients using FedAvg
        encrypted_client_weights_list: list of encrypted weight dicts
        """
        decrypted_weights_list = [decrypt_weights(w) for w in encrypted_client_weights_list]

        # Initialize average weights with zeros
        avg_weights = {}
        for key in decrypted_weights_list[0]:
            avg_weights[key] = np.zeros_like(decrypted_weights_list[0][key])

        # Sum weights
        for weights in decrypted_weights_list:
            for key in weights:
                avg_weights[key] += weights[key]

        # Average
        num_clients = len(decrypted_weights_list)
        for key in avg_weights:
            avg_weights[key] /= num_clients

        # Load averaged weights into the global model
        avg_weights_torch = {k: torch.tensor(v, dtype=torch.float32) for k, v in avg_weights.items()}
        self.global_model.load_state_dict(avg_weights_torch, strict=False)

        print(f"[SERVER] Aggregated model updated from {num_clients} clients.")

    def get_global_model(self):
        """Return a deep copy of the current global model"""
        return copy.deepcopy(self.global_model)


if __name__ == "__main__":
    # Dummy test with fake client updates
    import torch.nn as nn

    # Simple model
    model = nn.Linear(4, 2)
    server = FederatedServer(model)

    # Fake client weights (simulate encrypted->decrypted)
    fake_weights_1 = {k: np.random.rand(*v.shape) for k, v in model.state_dict().items()}
    fake_weights_2 = {k: np.random.rand(*v.shape) for k, v in model.state_dict().items()}

    # For prototype test: encrypt -> decrypt
    from federated.encryption import encrypt_weights
    enc1 = encrypt_weights(fake_weights_1)
    enc2 = encrypt_weights(fake_weights_2)

    server.aggregate_updates([enc1, enc2])


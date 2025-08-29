"""
federated/encryption.py
Prototype encryption/decryption utilities for model weights.
In production, replace with actual homomorphic encryption.
"""

import numpy as np

def encrypt_weights(weights_dict):
    """
    Simulate encryption by adding a wrapper.
    Args:
        weights_dict (dict): {param_name: numpy_array}
    Returns:
        dict: {param_name: {"enc": numpy_array}}
    """
    encrypted = {k: {"enc": v.copy()} for k, v in weights_dict.items()}
    return encrypted

def decrypt_weights(encrypted_weights_dict):
    """
    Simulate decryption by extracting numpy arrays.
    Args:
        encrypted_weights_dict (dict): {param_name: {"enc": numpy_array}}
    Returns:
        dict: {param_name: numpy_array}
    """
    decrypted = {k: v["enc"].copy() for k, v in encrypted_weights_dict.items()}
    return decrypted


if __name__ == "__main__":
    # Quick test
    fake_weights = {
        "layer1.weight": np.random.rand(4, 4),
        "layer1.bias": np.random.rand(4)
    }

    enc = encrypt_weights(fake_weights)
    print("[DEBUG] Encrypted format example:", list(enc.items())[0])

    dec = decrypt_weights(enc)
    print("[DEBUG] Decrypted matches original:", np.allclose(fake_weights["layer1.weight"], dec["layer1.weight"]))


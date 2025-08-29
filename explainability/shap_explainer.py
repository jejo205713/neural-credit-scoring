import shap
import numpy as np
import torch

class ShapExplainer:
    def __init__(self, model, background_features, feature_split_index=None):
        """
        SHAP explainer wrapper for FusionModel.
        
        Args:
            model: PyTorch model (FusionModel).
            background_features: Full feature matrix (NumPy or tensor) including tabular + graph.
            feature_split_index (int): Index where tabular features end and graph embeddings begin.
                                       If None, assumes half-half split.
        """
        self.model = model
        self.model.eval()

        # Ensure background data is NumPy
        self.background_features = (
            background_features.detach().cpu().numpy()
            if hasattr(background_features, "detach")
            else np.array(background_features)
        )

        # Feature split index for separating behavior and graph parts
        if feature_split_index is None:
            self.feature_split_index = self.background_features.shape[1] // 2
        else:
            self.feature_split_index = feature_split_index

    def wrapper(self, data):
        """
        SHAP calls this function with NumPy array.
        We split into behavior and graph parts for FusionModel.
        """
        data_tensor = torch.tensor(data, dtype=torch.float32)
        behavior_part = data_tensor[:, :self.feature_split_index]
        graph_part = data_tensor[:, self.feature_split_index:]

        with torch.no_grad():
            outputs = self.model(behavior_part, graph_part)

        return outputs.detach().cpu().numpy()

    def get_shap_values(self, instance):
        """
        Computes SHAP values for a given instance (full feature vector).
        """
        instance_np = (
            instance.detach().cpu().numpy().reshape(1, -1)
            if hasattr(instance, "detach")
            else np.array(instance).reshape(1, -1)
        )

        explainer = shap.KernelExplainer(self.wrapper, self.background_features)
        shap_values = explainer.shap_values(instance_np)
        return shap_values


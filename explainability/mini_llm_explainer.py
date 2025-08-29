"""
explainability/mini_llm_explainer.py
Prototype mini-LLM-style explanation generator.
Takes SHAP feature importance values and generates a natural-language reason for credit decision.
"""

import numpy as np

class MiniLLMExplainer:
    def __init__(self, feature_names=None):
        """
        Args:
            feature_names (list[str]): Names of features in order. 
                                       If None, will use generic names like 'Feature 1'.
        """
        self.feature_names = feature_names

    def generate_explanation(self, shap_values, top_k=3):
        """
        Args:
            shap_values (list/np.ndarray): SHAP values for one prediction.
            top_k (int): Number of top features to include in explanation.
        Returns:
            str: Human-readable explanation.
        """
        # Flatten in case shap_values is nested like [[...]] or shape (1, n)
        shap_values = np.array(shap_values, dtype=float).flatten()
        abs_vals = np.abs(shap_values)

        # Get top_k features by absolute SHAP value
        top_indices = np.argsort(abs_vals)[::-1][:top_k]

        explanations = []
        for idx in top_indices:
            idx_int = int(np.ravel(idx)[0])  # Ensure scalar integer
            fname = self.feature_names[idx_int] if self.feature_names else f"Feature {idx_int+1}"
            contribution = shap_values[idx_int]
            if contribution > 0:
                explanations.append(f"{fname} had a strong positive impact")
            else:
                explanations.append(f"{fname} lowered the score due to risk factors")

        # Combine into a readable sentence
        explanation_text = (
            "The credit decision was influenced mainly because: "
            + "; ".join(explanations)
            + "."
        )
        return explanation_text


if __name__ == "__main__":
    # Example test
    feature_names = [
        "Average Weekly Calls", "SMS to Call Ratio", "Finance App Hours",
        "Recharge Frequency", "On-time Bill Payments", "UPI Transactions",
        "E-commerce Spending", "Behavioral Score"
    ]
    shap_vals = np.array([[0.2, -0.1, 0.05, 0.3, 0.7, -0.4, 0.05, 0.1]])  # Nested shape test

    explainer = MiniLLMExplainer(feature_names)
    print(explainer.generate_explanation(shap_vals, top_k=3))


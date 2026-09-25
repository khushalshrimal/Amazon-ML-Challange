"""
Matching model training and inference wrapper.
"""

class EntityMatchingModel:
    """
    Model wrapper for entity resolution classification.
    Must comply with MIT/Apache 2.0 license and <= 8B parameters.
    """
    def __init__(self, model_type: str = "lgb"):
        self.model_type = model_type
        self.model = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        TODO: Train classifier (LightGBM/XGBoost/CatBoost or Logistic Regression baseline).
        """
        pass

    def predict_proba(self, X):
        """
        TODO: Return match probability for candidate pairs.
        """
        pass

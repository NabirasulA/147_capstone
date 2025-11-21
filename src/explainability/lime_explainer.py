# # src/explainability/lime_explainer.py
# import numpy as np
# import matplotlib.pyplot as plt

# class LIMEExplainer:
#     def __init__(self, model, feature_names, class_names):
#         self.model = model
#         self.feature_names = feature_names
#         self.class_names = class_names
#         self.explainer = None

#     def _import_lime(self):
#         try:
#             import lime
#             import lime.lime_tabular
#             return lime
#         except Exception as e:
#             print("Warning: failed to import lime:", e)
#             return None

#     def setup_explainer(self, training_data):
#         """Initialize the LIME explainer with training data (lazy import)."""
#         lime = self._import_lime()
#         if lime is None:
#             print("LIME not available; cannot setup explainer.")
#             return
#         self.explainer = lime.lime_tabular.LimeTabularExplainer(
#             training_data,
#             feature_names=self.feature_names,
#             class_names=self.class_names,
#             mode='classification'
#         )
    
#     def explain_instance(self, instance, num_features=10):
#         """Generate LIME explanation for a single instance"""
#         if self.explainer is None:
#             raise ValueError("Explainer not initialized. Call setup_explainer first.")

#         explanation = self.explainer.explain_instance(
#             instance,
#             self.model.predict_proba,
#             num_features=num_features
#         )
#         return explanation
    
#     def plot_explanation(self, explanation, class_idx=1):
#         """Plot LIME explanation for a specific class"""
#         fig = explanation.as_pyplot_figure(label=class_idx)
#         return fig
    
#     def get_feature_importance(self, explanation, class_idx=1):
#         """Get feature importance from LIME explanation"""
#         return explanation.as_list(label=class_idx)

import lime
import lime.lime_tabular
import numpy as np

class LIMEExplainer:
    def __init__(self, model, feature_names, class_names):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.explainer = None

    def predict_proba_wrapper(self, x):
        """
        LIME calls this instead of predict_proba.
        x comes in shape (n, features).
        CNN needs (n, features, 1)
        """
        x = np.array(x)
        x_cnn = x.reshape(x.shape[0], x.shape[1], 1)
        preds = self.model.predict(x_cnn, verbose=0)
        return np.hstack([1 - preds, preds])  # convert sigmoid to 2-class proba

    def setup_explainer(self, X_train):
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode="classification",
            discretize_continuous=True
        )

    def explain_instance(self, instance):
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call setup_explainer first.")

        instance = instance.reshape(1, -1)

        exp = self.explainer.explain_instance(
            instance[0],
            self.predict_proba_wrapper,
            num_features=10
        )

        return exp.as_pyplot_figure()

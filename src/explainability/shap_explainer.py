# src/explainability/shap_explainer.py
import numpy as np
import matplotlib.pyplot as plt

class SHAPExplainer:
    """Wrapper for SHAP explainers with lazy imports to avoid importing
    torch at module import time. This allows the module to be imported even
    when PyTorch is not correctly installed. The actual import of `shap`
    happens inside methods that need it.
    """
    def __init__(self, model, model_type='deep', sequence_length=None, num_features=None):
        self.model = model
        self.model_type = model_type
        self.explainer = None
        self.sequence_length = sequence_length
        self.num_features = num_features

    def _import_shap(self):
        try:
            import shap
            return shap
        except Exception as e:
            print("Warning: failed to import shap:", e)
            return None

    def setup_explainer(self, background_data):
        """Initialize the SHAP explainer with background data (lazy import)."""
        shap = self._import_shap()
        if shap is None:
            print("SHAP is not available; cannot setup explainer.")
            return

        # Always flatten background data for KernelExplainer compatibility
        if len(background_data.shape) == 3:
            # If 3D (batch, sequence, features), flatten sequence dimension
            background_data = background_data.reshape(background_data.shape[0], -1)
        elif len(background_data.shape) == 1:
            # If 1D, add batch dimension
            background_data = background_data.reshape(1, -1)

        def safe_predict(data):
            """Wrapper that ensures proper reshaping for sequence models."""
            if self.sequence_length and self.num_features:
                # Reshape to (batch, sequence, features) for model
                total_features = data.shape[-1]
                batch_size = data.shape[0]
                features_per_step = total_features // self.sequence_length
                reshaped = data.reshape(batch_size, self.sequence_length, features_per_step)
            else:
                reshaped = data
            return self.model.predict(reshaped)

        # Use KernelExplainer consistently - it's most robust with TF models
        self.explainer = shap.KernelExplainer(safe_predict, background_data)

    def explain_instance(self, instance):
        """Generate SHAP values for a single instance"""
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call setup_explainer first.")
        
        # Ensure instance is 2D for SHAP
        if len(instance.shape) == 3:
            # If 3D (batch, sequence, features), flatten sequence dimension
            instance = instance.reshape(instance.shape[0], -1)
        elif len(instance.shape) == 1:
            # If 1D, add batch dimension
            instance = instance.reshape(1, -1)

        shap = self._import_shap()
        if shap is None:
            print("SHAP is not available; cannot compute explanations.")
            return None

        if hasattr(self.explainer, 'shap_values'):
            shap_values = self.explainer.shap_values(instance)
            if isinstance(shap_values, list):
                return shap_values
            else:
                return [shap_values]
        else:
            print("Explainer has no shap_values method")
            return None

    def explain_dataset(self, data):
        """Generate SHAP values for a dataset"""
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call setup_explainer first.")

        shap = self._import_shap()
        if shap is None:
            print("SHAP is not available; cannot compute explanations.")
            return None

        shap_values = self.explainer.shap_values(data)
        if isinstance(shap_values, list):
            return shap_values
        else:
            return [shap_values]

    def plot_summary(self, shap_values, features, feature_names=None):
        """Plot SHAP summary plot (safe wrapper)."""
        # If features are 3D, flatten them to 2D for plotting
        if len(features.shape) == 3:
            features = features.reshape(features.shape[0], -1)
        shap = self._import_shap()
        if shap is None:
            print("SHAP is not available; cannot plot explanations.")
            return plt.figure()

        try:
            if isinstance(shap_values, list) and len(shap_values) > 0:
                shap_values_to_plot = shap_values[0]
            else:
                shap_values_to_plot = shap_values

            if shap_values_to_plot is None or len(shap_values_to_plot) == 0:
                print("Warning: No SHAP values to plot")
                return plt.figure()

            if feature_names is None:
                if len(np.array(features).shape) > 1:
                    feature_names = [f"Feature_{i}" for i in range(np.array(features).shape[1])]
                else:
                    feature_names = [f"Feature_{i}" for i in range(len(np.array(features)))]

            shap.summary_plot(shap_values_to_plot, features, feature_names=feature_names, show=False)
            fig = plt.gcf()
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error creating SHAP summary plot: {e}")
            import traceback
            traceback.print_exc()
            return plt.figure()

    def plot_dependence(self, shap_values, features, feature_idx, feature_names=None):
        shap = self._import_shap()
        if shap is None:
            print("SHAP is not available; cannot plot dependence.")
            return plt.figure()

        if isinstance(shap_values, list) and len(shap_values) > 0:
            shap_values_to_plot = shap_values[0]
        else:
            shap_values_to_plot = shap_values

        shap.dependence_plot(feature_idx, shap_values_to_plot, features, feature_names=feature_names)
        return plt.gcf()

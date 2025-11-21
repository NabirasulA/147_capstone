# src/explainability/ensemble_explainer.py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from .shap_explainer import SHAPExplainer
from .permutation_explainer import PermutationExplainer

class EnsembleExplainer:
    def __init__(self, cnn_model, lstm_model, cnn_explainer=None, lstm_explainer=None, use_permutation=False):
        self.cnn_model = cnn_model
        self.lstm_model = lstm_model
        self.ensemble_weights = [0.5, 0.5]  # Equal weights for CNN and LSTM
        self.use_permutation = use_permutation

        if use_permutation:
            # Use permutation explainer (more reliable with large datasets)
            self.cnn_explainer = cnn_explainer or PermutationExplainer(cnn_model)
            self.lstm_explainer = lstm_explainer or PermutationExplainer(lstm_model)
        else:
            # Try SHAP explainer first (may hit TF errors with large datasets)
            self.cnn_explainer = cnn_explainer or SHAPExplainer(cnn_model, model_type='kernel')
            self.lstm_explainer = lstm_explainer or SHAPExplainer(lstm_model, model_type='kernel')

    def explain_ensemble_prediction(self, cnn_input, lstm_input, true_label=None):
        """Explain how the ensemble combines CNN and LSTM predictions"""

        # Get individual model predictions
        cnn_pred = self._safe_predict(self.cnn_model, cnn_input)
        lstm_pred = self._safe_predict(self.lstm_model, lstm_input)

        # Get individual model importances (flatten the input)
        cnn_flattened = cnn_input.reshape(cnn_input.shape[0], -1)
        lstm_flattened = lstm_input.reshape(lstm_input.shape[0], -1)
        
        # Get importance values
        cnn_imp = self.cnn_explainer.explain_instance(cnn_flattened)
        lstm_imp = self.lstm_explainer.explain_instance(lstm_flattened)

        # Calculate ensemble prediction
        ensemble_pred = self.ensemble_weights[0] * cnn_pred + self.ensemble_weights[1] * lstm_pred

        # Calculate weighted importance values for ensemble
        if isinstance(cnn_imp, list) and len(cnn_imp) > 0:
            cnn_imp_weighted = self.ensemble_weights[0] * cnn_imp[0]
        else:
            cnn_imp_weighted = self.ensemble_weights[0] * cnn_imp

        if isinstance(lstm_imp, list) and len(lstm_imp) > 0:
            lstm_imp_weighted = self.ensemble_weights[1] * lstm_imp[0]
        else:
            lstm_imp_weighted = self.ensemble_weights[1] * lstm_imp

        ensemble_imp = cnn_imp_weighted + lstm_imp_weighted

        return {
            'cnn_prediction': cnn_pred,
            'lstm_prediction': lstm_pred,
            'ensemble_prediction': ensemble_pred,
            'cnn_importance': cnn_imp,
            'lstm_importance': lstm_imp,
            'ensemble_importance': ensemble_imp,
            'true_label': true_label,
            'model_agreement': self._calculate_agreement(cnn_pred, lstm_pred)
        }

    def _calculate_agreement(self, cnn_pred, lstm_pred):
        """Calculate agreement between CNN and LSTM predictions"""
        cnn_class = np.argmax(cnn_pred, axis=1)[0]
        lstm_class = np.argmax(lstm_pred, axis=1)[0]

        return {
            'same_prediction': cnn_class == lstm_class,
            'cnn_confidence': np.max(cnn_pred),
            'lstm_confidence': np.max(lstm_pred),
            'confidence_gap': abs(np.max(cnn_pred) - np.max(lstm_pred))
        }

    def _safe_predict(self, model, X):
        """Try model.predict(X) and fall back to calling the model in eager mode.

        This avoids TF `predict_function` issues in some saved/compiled models.
        Returns a numpy array of predictions.
        """
        try:
            preds = model.predict(X)
            # Ensure numpy array
            if hasattr(preds, 'numpy'):
                return preds.numpy()
            return np.asarray(preds)
        except Exception:
            # Fallback: try calling the model in eager mode
            try:
                tf.config.run_functions_eagerly(True)
                preds = model(X, training=False)
                if hasattr(preds, 'numpy'):
                    return preds.numpy()
                return np.asarray(preds)
            finally:
                try:
                    tf.config.run_functions_eagerly(False)
                except Exception:
                    pass

    def compare_model_explanations(self, cnn_input, lstm_input, feature_names=None):
        """Compare feature importance across CNN, LSTM, and ensemble"""

        # Get explanations for each model
        cnn_result = self.explain_ensemble_prediction(cnn_input, lstm_input)

        # Extract importance values for comparison
        cnn_imp = cnn_result['cnn_importance']
        lstm_imp = cnn_result['lstm_importance'] 
        ensemble_imp = cnn_result['ensemble_importance']

        # Handle list format 
        if isinstance(cnn_imp, list) and len(cnn_imp) > 0:
            cnn_imp_plot = cnn_imp[0]
        else:
            cnn_imp_plot = cnn_imp

        if isinstance(lstm_imp, list) and len(lstm_imp) > 0:
            lstm_imp_plot = lstm_imp[0]
        else:
            lstm_imp_plot = lstm_imp

        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Model predictions comparison
        models = ['CNN', 'LSTM', 'Ensemble']
        predictions = [
            cnn_result['cnn_prediction'].flatten(),
            cnn_result['lstm_prediction'].flatten(),
            cnn_result['ensemble_prediction'].flatten()
        ]

        axes[0, 0].bar(models, [np.max(p) for p in predictions])
        axes[0, 0].set_title('Prediction Confidence by Model')
        axes[0, 0].set_ylabel('Confidence')

        # Feature importance comparison (top 10 features)
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(cnn_imp_plot))]

        # Get top 10 most important features for each model
        cnn_top_features = np.argsort(np.abs(cnn_imp_plot))[-10:]
        lstm_top_features = np.argsort(np.abs(lstm_imp_plot))[-10:]
        ensemble_top_features = np.argsort(np.abs(ensemble_imp))[-10:]

        # Feature overlap analysis
        overlap_cnn_lstm = len(set(cnn_top_features) & set(lstm_top_features))
        overlap_all = len(set(cnn_top_features) & set(lstm_top_features) & set(ensemble_top_features))

        axes[0, 1].bar(['CNN vs LSTM', 'All Models'],
                      [overlap_cnn_lstm, overlap_all])
        axes[0, 1].set_title('Feature Importance Overlap')
        axes[0, 1].set_ylabel('Number of Overlapping Features')

        # Prediction agreement
        agreement = cnn_result['model_agreement']
        axes[1, 0].bar(['Agreement', 'CNN Confidence', 'LSTM Confidence'],
                      [1 if agreement['same_prediction'] else 0,
                       agreement['cnn_confidence'],
                       agreement['lstm_confidence']])
        axes[1, 0].set_title('Model Agreement Analysis')
        axes[1, 0].set_ylabel('Score')

        # Feature importance correlation
        correlation = np.corrcoef(cnn_imp_plot, lstm_imp_plot)[0, 1]
        importance_type = "Permutation" if self.use_permutation else "SHAP"
        axes[1, 1].text(0.5, 0.5, f'CNN-LSTM {importance_type}\nCorrelation: {correlation:.3f}',
                        ha='center', va='center', transform=axes[1, 1].transAxes,
                        bbox=dict(boxstyle='round', facecolor='lightblue'))
        axes[1, 1].set_title('Model Explanation Correlation')
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])

        plt.tight_layout()
        return fig, cnn_result

    def explain_ensemble_dataset(self, cnn_data, lstm_data, sample_indices=None):
        """Explain multiple instances from the ensemble"""

        if sample_indices is None:
            # Use just a few samples for the explanation
            sample_indices = np.random.choice(len(cnn_data), min(3, len(cnn_data)), replace=False)

        results = []
        for idx in sample_indices:
            cnn_sample = cnn_data[idx:idx+1]  # Keep batch dimension
            lstm_sample = lstm_data[idx:idx+1]

            result = self.explain_ensemble_prediction(cnn_sample, lstm_sample)
            results.append(result)

        return results

    def plot_ensemble_comparison(self, cnn_data, lstm_data, feature_names=None):
        """Create comprehensive ensemble comparison visualization"""

        # Get sample explanations
        sample_indices = np.random.choice(len(cnn_data), min(5, len(cnn_data)), replace=False)
        explanations = self.explain_ensemble_dataset(cnn_data, lstm_data, sample_indices)

        # Create comparison plot
        fig, axes = plt.subplots(3, len(sample_indices), figsize=(20, 12))

        for i, (sample_idx, explanation) in enumerate(zip(sample_indices, explanations)):
            # CNN importance values
            cnn_imp = explanation['cnn_importance']
            if isinstance(cnn_imp, list) and len(cnn_imp) > 0:
                cnn_imp_plot = cnn_imp[0]
            else:
                cnn_imp_plot = cnn_imp

            # LSTM importance values
            lstm_imp = explanation['lstm_importance']
            if isinstance(lstm_imp, list) and len(lstm_imp) > 0:
                lstm_imp_plot = lstm_imp[0]
            else:
                lstm_imp_plot = lstm_imp

            # Ensemble importance values  
            ensemble_imp = explanation['ensemble_importance']

            # Top features for each model
            if feature_names is None:
                feature_names = [f'Feature_{j}' for j in range(len(cnn_imp_plot))]

            # Get top 5 features for each model
            cnn_top = np.argsort(np.abs(cnn_imp_plot))[-5:]
            lstm_top = np.argsort(np.abs(lstm_imp_plot))[-5:]  
            ensemble_top = np.argsort(np.abs(ensemble_imp))[-5:]

            # Plot for this sample
            y_pos = np.arange(5)

            # CNN importance
            axes[0, i].barh(y_pos, [abs(cnn_imp_plot[j]) for j in cnn_top])
            axes[0, i].set_yticks(y_pos)
            axes[0, i].set_yticklabels([feature_names[j] for j in cnn_top])
            axes[0, i].set_title(f'Sample {sample_idx}\nCNN Top Features')
            axes[0, i].set_xlabel('Importance')

            # LSTM importance
            axes[1, i].barh(y_pos, [abs(lstm_imp_plot[j]) for j in lstm_top])
            axes[1, i].set_yticks(y_pos)
            axes[1, i].set_yticklabels([feature_names[j] for j in lstm_top])
            axes[1, i].set_title('LSTM Top Features')
            axes[1, i].set_xlabel('Importance')

            # Ensemble importance
            axes[2, i].barh(y_pos, [abs(ensemble_imp[j]) for j in ensemble_top])
            axes[2, i].set_yticks(y_pos)
            axes[2, i].set_yticklabels([feature_names[j] for j in ensemble_top])
            axes[2, i].set_title('Ensemble Top Features')
            axes[2, i].set_xlabel('Importance')

        plt.tight_layout()
        return fig

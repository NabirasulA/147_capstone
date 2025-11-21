"""
Simple permutation-based feature importance explainer that works reliably with large datasets.
Uses feature permutation to assess importance by measuring prediction changes.
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class PermutationExplainer:
    def __init__(self, model, sequence_length=None, num_features=None):
        self.model = model
        self.sequence_length = sequence_length
        self.num_features = num_features

    def explain_instance(self, instance, n_samples=10):
        """Calculate feature importance by permuting features."""
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        
        # Get base prediction
        base_pred = self.model.predict(instance)
        importances = np.zeros(instance.shape[1])
        
        # For each feature
        for i in range(instance.shape[1]):
            scores = []
            # Permute the feature n_samples times
            for _ in range(n_samples):
                # Create copy and permute feature
                X_permuted = instance.copy()
                X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
                
                # Get prediction with permuted feature
                new_pred = self.model.predict(X_permuted)
                
                # Score is decrease in prediction probability
                score = np.abs(base_pred - new_pred).mean()
                scores.append(score)
            
            # Average importance across samples
            importances[i] = np.mean(scores)
        
        return importances

    def explain_dataset(self, data, n_samples=5, max_instances=100):
        """Calculate feature importance across multiple instances."""
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        # Sample instances if needed
        if len(data) > max_instances:
            indices = np.random.choice(len(data), max_instances, replace=False)
            data = data[indices]
        
        all_importances = []
        for instance in tqdm(data, desc="Computing permutation importance"):
            imp = self.explain_instance(instance.reshape(1, -1), n_samples)
            all_importances.append(imp)
        
        return np.array(all_importances)

    def plot_summary(self, importances, features=None, feature_names=None):
        """Create summary plot of feature importances."""
        # Average importance across instances
        mean_imp = np.mean(importances, axis=0)
        std_imp = np.std(importances, axis=0)
        
        # Sort by importance
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(mean_imp))]
        
        sort_idx = np.argsort(mean_imp)
        mean_imp = mean_imp[sort_idx]
        std_imp = std_imp[sort_idx]
        feature_names = np.array(feature_names)[sort_idx]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(mean_imp))
        ax.barh(y_pos, mean_imp, xerr=std_imp, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance (Mean Impact on Prediction)')
        ax.set_title('Permutation Feature Importance')
        
        plt.tight_layout()
        return fig
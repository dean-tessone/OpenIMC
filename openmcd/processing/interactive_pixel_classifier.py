"""
Interactive pixel level classification pixel classifier using Random Forest.

This module implements the training and inference pipeline for pixel-level
classification using user-provided scribbles and Random Forest.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
import joblib
import os

# Optional dependencies
try:
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    _HAVE_SKLEARN = True
except ImportError:
    _HAVE_SKLEARN = False


class InteractivePixelClassifier:
    """
    Pixel-level classifier for Interactive pixel level classification segmentation.
    
    Uses Random Forest or Extra Trees to classify pixels into:
    - Background (0)
    - Nucleus (1) 
    - Cytoplasm/Membrane (2)
    """
    
    def __init__(self, 
                 classifier_type: str = "random_forest",
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 random_state: int = 42):
        """
        Initialize the pixel classifier.
        
        Args:
            classifier_type: Type of classifier ("random_forest" or "extra_trees")
            n_estimators: Number of trees in the ensemble
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            random_state: Random state for reproducibility
        """
        if not _HAVE_SKLEARN:
            raise ImportError("scikit-learn is required for Interactive pixel level classification")
        
        self.classifier_type = classifier_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        
        # Initialize classifier
        self._init_classifier()
        
        # Training data storage
        self.training_features = None
        self.training_labels = None
        self.feature_names = None
        self.is_trained = False
        
        # Class names (will be set based on training data)
        self.class_names = None
        self.n_classes = None
    
    def _init_classifier(self):
        """Initialize the classifier based on type."""
        if self.classifier_type == "random_forest":
            self.classifier = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1  # Use all available cores
            )
        elif self.classifier_type == "extra_trees":
            self.classifier = ExtraTreesClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
    
    def add_training_data(self, 
                         features: np.ndarray, 
                         labels: np.ndarray,
                         feature_names: List[str] = None):
        """
        Add training data from user scribbles.
        
        Args:
            features: Feature array of shape (N_pixels, N_features)
            labels: Label array of shape (N_pixels,) with values 0, 1, 2
            feature_names: List of feature names
        """
        if self.training_features is None:
            self.training_features = features
            self.training_labels = labels
            self.feature_names = feature_names
            
            # Set class names based on actual labels present
            unique_labels = np.unique(labels)
            unique_labels = np.sort(unique_labels)  # Sort to ensure consistent ordering
            
            # Map labels to class names
            label_to_name = {0: "background", 1: "nucleus", 2: "cytoplasm"}
            self.class_names = [label_to_name[label] for label in unique_labels]
            self.n_classes = len(self.class_names)
            
            print(f"[DEBUG] Set class names based on training data: {self.class_names}")
        else:
            # Append to existing training data
            self.training_features = np.vstack([self.training_features, features])
            self.training_labels = np.hstack([self.training_labels, labels])
            
            # Update class names if new labels are present
            unique_labels = np.unique(self.training_labels)
            unique_labels = np.sort(unique_labels)
            label_to_name = {0: "background", 1: "nucleus", 2: "cytoplasm"}
            new_class_names = [label_to_name[label] for label in unique_labels]
            if new_class_names != self.class_names:
                self.class_names = new_class_names
                self.n_classes = len(self.class_names)
                print(f"[DEBUG] Updated class names: {self.class_names}")
    
    def train(self, 
              cross_validate: bool = True,
              cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train the classifier on accumulated training data.
        
        Args:
            cross_validate: Whether to perform cross-validation
            cv_folds: Number of folds for cross-validation
            
        Returns:
            Dictionary with training metrics
        """
        if self.training_features is None or len(self.training_features) == 0:
            raise ValueError("No training data available. Add training data first.")
        
        print(f"[DEBUG] Classifier training data: {self.training_features.shape}")
        print(f"[DEBUG] Classifier training labels: {self.training_labels.shape}")
        
        # Check for sufficient training data
        unique_labels = np.unique(self.training_labels)
        print(f"[DEBUG] Unique labels in training: {unique_labels}")
        if len(unique_labels) < 2:
            raise ValueError("Need at least 2 classes for training")
        
        # Train the classifier
        print("[DEBUG] Fitting classifier...")
        self.classifier.fit(self.training_features, self.training_labels)
        self.is_trained = True
        print("[DEBUG] Classifier fitted successfully")
        
        # Compute training metrics
        metrics = {}
        metrics["n_training_samples"] = len(self.training_features)
        metrics["n_features"] = self.training_features.shape[1]
        metrics["classes_present"] = unique_labels.tolist()
        metrics["class_counts"] = {
            self.class_names[i]: int(np.sum(self.training_labels == i))
            for i in unique_labels
        }
        
        # Cross-validation if requested
        if cross_validate and len(self.training_features) >= cv_folds * 2:
            try:
                print(f"[DEBUG] Starting cross-validation with {cv_folds} folds...")
                cv_scores = cross_val_score(
                    self.classifier, 
                    self.training_features, 
                    self.training_labels, 
                    cv=cv_folds,
                    scoring='accuracy'
                )
                metrics["cv_accuracy_mean"] = float(np.mean(cv_scores))
                metrics["cv_accuracy_std"] = float(np.std(cv_scores))
                print(f"[DEBUG] Cross-validation completed: {metrics['cv_accuracy_mean']:.3f} ± {metrics['cv_accuracy_std']:.3f}")
            except Exception as e:
                print(f"[DEBUG] Cross-validation failed: {e}")
                metrics["cv_error"] = str(e)
        
        # Feature importance
        if hasattr(self.classifier, 'feature_importances_'):
            metrics["feature_importance"] = self.classifier.feature_importances_.tolist()
        
        return metrics
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for given features.
        
        Args:
            features: Feature array of shape (N_pixels, N_features)
            
        Returns:
            Probability array of shape (N_pixels, N_classes)
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained. Call train() first.")
        
        return self.classifier.predict_proba(features)
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class labels for given features.
        
        Args:
            features: Feature array of shape (N_pixels, N_features)
            
        Returns:
            Label array of shape (N_pixels,)
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained. Call train() first.")
        
        return self.classifier.predict(features)
    
    def predict_image_proba(self, 
                           image_features: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for an entire image.
        
        Args:
            image_features: Feature array of shape (H, W, N_features)
            
        Returns:
            Probability array of shape (H, W, N_classes)
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained. Call train() first.")
        
        height, width, n_features = image_features.shape
        
        # Reshape for prediction
        features_flat = image_features.reshape(-1, n_features)
        
        # Predict probabilities
        proba_flat = self.predict_proba(features_flat)
        
        # Reshape back to image dimensions
        proba_image = proba_flat.reshape(height, width, self.n_classes)
        
        return proba_image
    
    def predict_image(self, image_features: np.ndarray) -> np.ndarray:
        """
        Predict class labels for an entire image.
        
        Args:
            image_features: Feature array of shape (H, W, N_features)
            
        Returns:
            Label array of shape (H, W)
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained. Call train() first.")
        
        height, width, n_features = image_features.shape
        
        # Reshape for prediction
        features_flat = image_features.reshape(-1, n_features)
        
        # Predict labels
        labels_flat = self.predict(features_flat)
        
        # Reshape back to image dimensions
        labels_image = labels_flat.reshape(height, width)
        
        return labels_image
    
    def get_class_probability_maps(self, 
                                 image_features: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get separate probability maps for each class.
        
        Args:
            image_features: Feature array of shape (H, W, N_features)
            
        Returns:
            Dictionary with class names as keys and probability maps as values
        """
        proba_image = self.predict_image_proba(image_features)
        
        probability_maps = {}
        for i, class_name in enumerate(self.class_names):
            probability_maps[class_name] = proba_image[:, :, i]
        
        return probability_maps
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'classifier': self.classifier,
            'classifier_type': self.classifier_type,
            'class_names': self.class_names,
            'feature_names': self.feature_names,
            'training_metrics': {
                'n_training_samples': len(self.training_features),
                'n_features': self.training_features.shape[1] if self.training_features is not None else 0
            }
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.classifier = model_data['classifier']
        self.classifier_type = model_data['classifier_type']
        self.class_names = model_data['class_names']
        self.feature_names = model_data['feature_names']
        self.is_trained = True
    
    def reset_training_data(self):
        """Reset all training data."""
        self.training_features = None
        self.training_labels = None
        self.feature_names = None
        self.is_trained = False
        self._init_classifier()
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of current training data."""
        if self.training_features is None:
            return {"status": "no_training_data"}
        
        unique_labels = np.unique(self.training_labels)
        summary = {
            "status": "has_training_data",
            "n_samples": len(self.training_features),
            "n_features": self.training_features.shape[1],
            "classes_present": unique_labels.tolist(),
            "class_counts": {
                self.class_names[i]: int(np.sum(self.training_labels == i))
                for i in unique_labels
            },
            "is_trained": self.is_trained
        }
        
        return summary


def create_interactive_pixel_classifier(classifier_type: str = "random_forest",
                            n_estimators: int = 100,
                            random_state: int = 42) -> InteractivePixelClassifier:
    """
    Convenience function to create an Interactive pixel level classification pixel classifier.
    
    Args:
        classifier_type: Type of classifier ("random_forest" or "extra_trees")
        n_estimators: Number of trees in the ensemble
        random_state: Random state for reproducibility
        
    Returns:
        Initialized InteractivePixelClassifier instance
    """
    return InteractivePixelClassifier(
        classifier_type=classifier_type,
        n_estimators=n_estimators,
        random_state=random_state
    )

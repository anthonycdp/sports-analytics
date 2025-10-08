"""
Predictive Models Module for Sports Analytics

Implements machine learning models for win probability prediction
and player performance forecasting.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, classification_report, confusion_matrix
)
import joblib


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    log_loss: float
    
    def to_dict(self) -> Dict:
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'auc_roc': self.auc_roc,
            'log_loss': self.log_loss
        }


class WinProbabilityModel:
    """
    Match outcome prediction model.
    
    Predicts the probability of home win, draw, or away win
    based on team statistics and historical performance.
    
    Example:
    --------
    >>> model = WinProbabilityModel()
    >>> model.fit(X_train, y_train)
    >>> probs = model.predict_proba(X_test)
    """
    
    def __init__(
        self,
        model_type: str = 'gradient_boosting',
        random_state: int = 42
    ):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.metrics: Optional[ModelMetrics] = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the underlying model."""
        if self.model_type == 'logistic':
            self.model = LogisticRegression(
                multi_class='multinomial',
                max_iter=1000,
                random_state=self.random_state
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        scale_features: bool = True
    ) -> 'WinProbabilityModel':
        """
        Fit the model to training data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable (H/D/A)
        scale_features : bool
            Whether to scale features
            
        Returns:
        --------
        self
        """
        self.feature_names = list(X.columns)
        
        if scale_features:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict match outcomes."""
        X_scaled = self.scaler.transform(X[self.feature_names])
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict outcome probabilities.
        
        Returns:
        --------
        np.ndarray
            Array of shape (n_samples, 3) with columns [A, D, H]
        """
        X_scaled = self.scaler.transform(X[self.feature_names])
        return self.model.predict_proba(X_scaled)
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> ModelMetrics:
        """
        Evaluate model performance.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Test features
        y : pd.Series
            True labels
            
        Returns:
        --------
        ModelMetrics
            Evaluation metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        self.metrics = ModelMetrics(
            accuracy=accuracy_score(y, y_pred),
            precision=precision_score(y, y_pred, average='weighted'),
            recall=recall_score(y, y_pred, average='weighted'),
            f1=f1_score(y, y_pred, average='weighted'),
            auc_roc=roc_auc_score(y, y_proba, multi_class='ovr'),
            log_loss=log_loss(y, y_proba)
        )
        
        return self.metrics
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Returns:
        --------
        Dict with mean and std of accuracy scores
        """
        X_scaled = self.scaler.fit_transform(X)
        scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='accuracy')
        
        return {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'all_scores': scores
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance (for tree-based models)."""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_).mean(axis=0)
        else:
            return pd.DataFrame()
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def save(self, filepath: str):
        """Save model to disk."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }, filepath)
    
    def load(self, filepath: str):
        """Load model from disk."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.model_type = data['model_type']


class PlayerPerformanceModel:
    """
    Player performance prediction model.
    
    Predicts match rating and goal contributions based on
    player attributes and recent form.
    """
    
    def __init__(
        self,
        target: str = 'match_rating',
        random_state: int = 42
    ):
        self.target = target
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> 'PlayerPerformanceModel':
        """Fit the performance model."""
        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)
        
        # Use Ridge regression for continuous target
        self.model = Ridge(alpha=1.0)
        self.model.fit(X_scaled, y)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict player performance."""
        X_scaled = self.scaler.transform(X[self.feature_names])
        return self.model.predict(X_scaled)
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        y_pred = self.predict(X)
        
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - y_pred))
        r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2))
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def get_top_performers(
        self,
        players_df: pd.DataFrame,
        features_df: pd.DataFrame,
        n: int = 10
    ) -> pd.DataFrame:
        """Predict top performers for upcoming matches."""
        predictions = self.predict(features_df)
        
        results = players_df.copy()
        results['predicted_rating'] = predictions
        
        return results.nlargest(n, 'predicted_rating')


def prepare_training_data(
    matches: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare match data for model training.
    
    Parameters:
    -----------
    matches : pd.DataFrame
        Feature-engineered match data
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed
        
    Returns:
    --------
    Tuple of X_train, X_test, y_train, y_test
    """
    # Define feature columns
    numeric_cols = matches.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['match_id', 'home_score', 'away_score', 'home_points', 'away_points',
                    'total_goals', 'goal_difference', 'result']
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    X = matches[feature_cols].fillna(0)
    y = matches['result']
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

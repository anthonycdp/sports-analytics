"""Tests for predictive models module."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    WinProbabilityModel,
    PlayerPerformanceModel,
    ModelMetrics,
    prepare_training_data
)
from src.data_generation import generate_match_data, generate_player_data
from src.features import FeatureEngineer


class TestWinProbabilityModel:
    """Tests for win probability model."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        matches, teams = generate_match_data(n_matches=200, seed=42)
        fe = FeatureEngineer()
        matches_featured = fe.create_match_features(matches, teams)
        X_train, X_test, y_train, y_test = prepare_training_data(matches_featured)
        return X_train, X_test, y_train, y_test
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = WinProbabilityModel(model_type='logistic')
        assert model.model_type == 'logistic'
        assert model.model is not None
    
    def test_model_invalid_type_raises_error(self):
        """Test that invalid model type raises error."""
        with pytest.raises(ValueError):
            WinProbabilityModel(model_type='invalid_type')
    
    def test_model_fit(self, sample_data):
        """Test model fitting."""
        X_train, X_test, y_train, y_test = sample_data
        model = WinProbabilityModel(model_type='logistic')
        
        fitted = model.fit(X_train, y_train)
        
        assert fitted is model  # Returns self
        assert model._fitted is not False
    
    def test_model_predict(self, sample_data):
        """Test model prediction."""
        X_train, X_test, y_train, y_test = sample_data
        model = WinProbabilityModel(model_type='logistic')
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(p in ['H', 'D', 'A'] for p in predictions)
    
    def test_model_predict_proba(self, sample_data):
        """Test probability prediction."""
        X_train, X_test, y_train, y_test = sample_data
        model = WinProbabilityModel(model_type='logistic')
        model.fit(X_train, y_train)
        
        probas = model.predict_proba(X_test)
        
        assert probas.shape == (len(X_test), 3)
        assert np.allclose(probas.sum(axis=1), 1.0)  # Probabilities sum to 1
        assert (probas >= 0).all() and (probas <= 1).all()
    
    def test_model_evaluate(self, sample_data):
        """Test model evaluation."""
        X_train, X_test, y_train, y_test = sample_data
        model = WinProbabilityModel(model_type='logistic')
        model.fit(X_train, y_train)
        
        metrics = model.evaluate(X_test, y_test)
        
        assert isinstance(metrics, ModelMetrics)
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.f1 <= 1
        assert 0 <= metrics.auc_roc <= 1
    
    def test_model_cross_validate(self, sample_data):
        """Test cross-validation."""
        X_train, X_test, y_train, y_test = sample_data
        model = WinProbabilityModel(model_type='logistic')
        
        cv_results = model.cross_validate(X_train, y_train, cv=3)
        
        assert 'mean_accuracy' in cv_results
        assert 'std_accuracy' in cv_results
        assert 0 <= cv_results['mean_accuracy'] <= 1
    
    def test_get_feature_importance(self, sample_data):
        """Test feature importance extraction."""
        X_train, X_test, y_train, y_test = sample_data
        model = WinProbabilityModel(model_type='gradient_boosting')
        model.fit(X_train, y_train)
        
        importance = model.get_feature_importance()
        
        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns


class TestPlayerPerformanceModel:
    """Tests for player performance model."""
    
    @pytest.fixture
    def sample_player_data(self):
        """Generate sample player data for testing."""
        matches, teams = generate_match_data(n_matches=100, seed=42)
        players, performances = generate_player_data(
            n_players=100, 
            matches_df=matches,
            seed=42
        )
        
        fe = FeatureEngineer()
        performances_featured = fe.create_player_features(performances, players)
        
        # Prepare features
        exclude_cols = ['performance_id', 'match_id', 'date', 'player_id', 
                       'team_id', 'opponent_id', 'match_rating', 'season']
        feature_cols = [c for c in performances_featured.columns 
                       if c not in exclude_cols and 
                       performances_featured[c].dtype in [np.float64, np.int64]]
        
        X = performances_featured[feature_cols].fillna(0)
        y = performances_featured['match_rating']
        
        return X, y
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = PlayerPerformanceModel()
        assert model.target == 'match_rating'
    
    def test_model_fit(self, sample_player_data):
        """Test model fitting."""
        X, y = sample_player_data
        model = PlayerPerformanceModel()
        
        fitted = model.fit(X, y)
        
        assert fitted is model
    
    def test_model_predict(self, sample_player_data):
        """Test model prediction."""
        X, y = sample_player_data
        model = PlayerPerformanceModel()
        model.fit(X, y)
        
        predictions = model.predict(X)
        
        assert len(predictions) == len(X)
    
    def test_model_evaluate(self, sample_player_data):
        """Test model evaluation."""
        X, y = sample_player_data
        model = PlayerPerformanceModel()
        model.fit(X, y)
        
        metrics = model.evaluate(X, y)
        
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert metrics['rmse'] >= 0


class TestModelMetrics:
    """Tests for ModelMetrics dataclass."""
    
    def test_metrics_creation(self):
        """Test metrics object creation."""
        metrics = ModelMetrics(
            accuracy=0.8,
            precision=0.75,
            recall=0.85,
            f1=0.8,
            auc_roc=0.9,
            log_loss=0.5
        )
        
        assert metrics.accuracy == 0.8
        assert metrics.f1 == 0.8
    
    def test_metrics_to_dict(self):
        """Test metrics conversion to dictionary."""
        metrics = ModelMetrics(
            accuracy=0.8,
            precision=0.75,
            recall=0.85,
            f1=0.8,
            auc_roc=0.9,
            log_loss=0.5
        )
        
        d = metrics.to_dict()
        
        assert isinstance(d, dict)
        assert 'accuracy' in d
        assert d['accuracy'] == 0.8


class TestPrepareTrainingData:
    """Tests for data preparation function."""
    
    def test_prepare_data_returns_correct_types(self):
        """Test that prepare_training_data returns correct types."""
        matches, teams = generate_match_data(n_matches=100, seed=42)
        fe = FeatureEngineer()
        matches_featured = fe.create_match_features(matches, teams)
        
        X_train, X_test, y_train, y_test = prepare_training_data(matches_featured)
        
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
    
    def test_prepare_data_correct_split_ratio(self):
        """Test that data is split correctly."""
        matches, teams = generate_match_data(n_matches=100, seed=42)
        fe = FeatureEngineer()
        matches_featured = fe.create_match_features(matches, teams)
        
        X_train, X_test, y_train, y_test = prepare_training_data(
            matches_featured, test_size=0.3
        )
        
        total = len(X_train) + len(X_test)
        assert len(X_test) / total == pytest.approx(0.3, rel=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for feature engineering module."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import FeatureEngineer, FeatureConfig
from src.data_generation import generate_match_data, generate_player_data


class TestFeatureConfig:
    """Tests for FeatureConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = FeatureConfig()
        
        assert config.rolling_windows == [3, 5, 10]
        assert config.include_form is True
        assert config.include_h2h is True
        assert config.include_home_advantage is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = FeatureConfig(
            rolling_windows=[5, 10],
            include_form=False
        )
        
        assert config.rolling_windows == [5, 10]
        assert config.include_form is False


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        matches, teams = generate_match_data(n_matches=200, seed=42)
        return matches, teams
    
    def test_initialization(self):
        """Test FeatureEngineer initialization."""
        fe = FeatureEngineer()
        assert fe._fitted is False
    
    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = FeatureConfig(rolling_windows=[5])
        fe = FeatureEngineer(config=config)
        
        assert fe.config.rolling_windows == [5]
    
    def test_create_match_features_returns_dataframe(self, sample_data):
        """Test that match features returns a DataFrame."""
        matches, teams = sample_data
        fe = FeatureEngineer()
        
        result = fe.create_match_features(matches, teams)
        
        assert isinstance(result, pd.DataFrame)
    
    def test_create_match_features_adds_columns(self, sample_data):
        """Test that feature engineering adds new columns."""
        matches, teams = sample_data
        fe = FeatureEngineer()
        
        result = fe.create_match_features(matches, teams)
        
        # Should have more columns than original
        assert len(result.columns) > len(matches.columns)
    
    def test_create_match_features_no_nan_in_key_columns(self, sample_data):
        """Test that key columns don't have NaN after feature engineering."""
        matches, teams = sample_data
        fe = FeatureEngineer()
        
        result = fe.create_match_features(matches, teams)
        
        # Check no NaN in result column
        assert result['result'].notna().all()
    
    def test_create_match_features_with_teams(self, sample_data):
        """Test feature engineering with team data."""
        matches, teams = sample_data
        fe = FeatureEngineer()
        
        result = fe.create_match_features(matches, teams)
        
        # Should have team-related features
        assert 'home_attack_strength' in result.columns
        assert 'away_attack_strength' in result.columns
    
    def test_create_match_features_without_teams(self, sample_data):
        """Test feature engineering without team data."""
        matches, teams = sample_data
        fe = FeatureEngineer()
        
        result = fe.create_match_features(matches, teams=None)
        
        # Should still work
        assert isinstance(result, pd.DataFrame)
    
    def test_create_h2h_features(self, sample_data):
        """Test head-to-head feature creation."""
        matches, teams = sample_data
        config = FeatureConfig(include_h2h=True)
        fe = FeatureEngineer(config=config)
        
        result = fe.create_match_features(matches, teams)
        
        assert 'h2h_home_win_rate' in result.columns
    
    def test_create_form_features(self, sample_data):
        """Test form feature creation."""
        matches, teams = sample_data
        config = FeatureConfig(include_form=True)
        fe = FeatureEngineer(config=config)
        
        result = fe.create_match_features(matches, teams)
        
        assert 'home_form_points' in result.columns
        assert 'away_form_points' in result.columns
    
    def test_get_feature_columns(self):
        """Test feature column list retrieval."""
        fe = FeatureEngineer()
        feature_cols = fe.get_feature_columns()
        
        assert isinstance(feature_cols, list)
        assert len(feature_cols) > 0


class TestPlayerFeatures:
    """Tests for player feature engineering."""
    
    @pytest.fixture
    def sample_player_data(self):
        """Generate sample player data."""
        matches, teams = generate_match_data(n_matches=100, seed=42)
        players, performances = generate_player_data(
            n_players=50, 
            matches_df=matches,
            seed=42
        )
        return players, performances
    
    def test_create_player_features_returns_dataframe(self, sample_player_data):
        """Test that player features returns a DataFrame."""
        players, performances = sample_player_data
        fe = FeatureEngineer()
        
        result = fe.create_player_features(performances, players)
        
        assert isinstance(result, pd.DataFrame)
    
    def test_create_player_features_adds_columns(self, sample_player_data):
        """Test that player features adds new columns."""
        players, performances = sample_player_data
        fe = FeatureEngineer()
        
        result = fe.create_player_features(performances, players)
        
        # Should have rolling features
        assert 'rolling_goals_5' in result.columns
        assert 'rolling_rating_5' in result.columns
    
    def test_create_player_features_maintains_rows(self, sample_player_data):
        """Test that number of rows is maintained."""
        players, performances = sample_player_data
        fe = FeatureEngineer()
        
        result = fe.create_player_features(performances, players)
        
        assert len(result) == len(performances)
    
    def test_create_player_features_season_stats(self, sample_player_data):
        """Test that season statistics are calculated."""
        players, performances = sample_player_data
        fe = FeatureEngineer()
        
        result = fe.create_player_features(performances, players)
        
        assert 'season_goals' in result.columns
        assert 'season_minutes' in result.columns
        assert 'goals_per_90' in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

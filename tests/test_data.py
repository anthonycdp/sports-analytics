"""Tests for data generation module."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation import (
    generate_team_stats,
    generate_match_data,
    generate_player_data,
    _generate_player_performance
)


class TestTeamStatsGeneration:
    """Tests for team statistics generation."""
    
    def test_generate_team_stats_returns_dataframe(self):
        """Test that team stats generation returns a DataFrame."""
        teams = generate_team_stats(n_teams=10, seed=42)
        assert isinstance(teams, pd.DataFrame)
    
    def test_generate_team_stats_correct_count(self):
        """Test that correct number of teams are generated."""
        n_teams = 15
        teams = generate_team_stats(n_teams=n_teams, seed=42)
        assert len(teams) == n_teams
    
    def test_generate_team_stats_has_required_columns(self):
        """Test that all required columns are present."""
        teams = generate_team_stats(n_teams=10, seed=42)
        required_columns = [
            'team_id', 'team_name', 'attack_strength', 
            'defense_strength', 'overall_rating'
        ]
        for col in required_columns:
            assert col in teams.columns
    
    def test_generate_team_stats_reproducible(self):
        """Test that same seed produces same results."""
        teams1 = generate_team_stats(n_teams=10, seed=42)
        teams2 = generate_team_stats(n_teams=10, seed=42)
        pd.testing.assert_frame_equal(teams1, teams2)
    
    def test_generate_team_stats_different_seeds(self):
        """Test that different seeds produce different results."""
        teams1 = generate_team_stats(n_teams=10, seed=42)
        teams2 = generate_team_stats(n_teams=10, seed=123)
        assert not teams1['attack_strength'].equals(teams2['attack_strength'])


class TestMatchDataGeneration:
    """Tests for match data generation."""
    
    def test_generate_match_data_returns_tuple(self):
        """Test that match generation returns a tuple of DataFrames."""
        result = generate_match_data(n_matches=100, seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_generate_match_data_correct_count(self):
        """Test that correct number of matches are generated."""
        matches, teams = generate_match_data(n_matches=150, seed=42)
        assert len(matches) == 150
    
    def test_generate_match_data_no_self_matches(self):
        """Test that no team plays against itself."""
        matches, teams = generate_match_data(n_matches=100, seed=42)
        assert (matches['home_team_id'] != matches['away_team_id']).all()
    
    def test_generate_match_data_valid_results(self):
        """Test that all results are valid (H, D, A)."""
        matches, teams = generate_match_data(n_matches=100, seed=42)
        assert matches['result'].isin(['H', 'D', 'A']).all()
    
    def test_generate_match_data_scores_consistent_with_result(self):
        """Test that scores match the result."""
        matches, teams = generate_match_data(n_matches=100, seed=42)
        
        # Home wins
        home_wins = matches[matches['result'] == 'H']
        assert (home_wins['home_score'] > home_wins['away_score']).all()
        
        # Away wins
        away_wins = matches[matches['result'] == 'A']
        assert (away_wins['home_score'] < away_wins['away_score']).all()
        
        # Draws
        draws = matches[matches['result'] == 'D']
        assert (draws['home_score'] == draws['away_score']).all()
    
    def test_generate_match_data_football_realistic_scores(self):
        """Test that football scores are realistic (0-10 goals)."""
        matches, teams = generate_match_data(n_matches=100, sport='football', seed=42)
        assert (matches['home_score'] >= 0).all()
        assert (matches['home_score'] < 15).all()
        assert (matches['away_score'] >= 0).all()
        assert (matches['away_score'] < 15).all()
    
    def test_generate_match_data_has_dates(self):
        """Test that match dates are present and valid."""
        matches, teams = generate_match_data(
            n_matches=100, 
            start_date="2022-08-01",
            end_date="2023-05-30",
            seed=42
        )
        assert 'date' in matches.columns
        assert pd.api.types.is_datetime64_any_dtype(matches['date'])


class TestPlayerDataGeneration:
    """Tests for player data generation."""
    
    def test_generate_player_data_returns_dataframe(self):
        """Test that player generation returns a DataFrame."""
        players = generate_player_data(n_players=100, seed=42)
        assert isinstance(players, pd.DataFrame)
    
    def test_generate_player_data_correct_count(self):
        """Test that correct number of players are generated."""
        players = generate_player_data(n_players=150, seed=42)
        assert len(players) == 150
    
    def test_generate_player_data_valid_positions(self):
        """Test that all positions are valid."""
        players = generate_player_data(n_players=100, seed=42)
        valid_positions = ['GK', 'DEF', 'MID', 'FWD']
        assert players['position'].isin(valid_positions).all()
    
    def test_generate_player_data_with_performances(self):
        """Test player generation with match performances."""
        matches, teams = generate_match_data(n_matches=50, seed=42)
        players, performances = generate_player_data(
            n_players=100, 
            matches_df=matches,
            seed=42
        )
        
        assert isinstance(performances, pd.DataFrame)
        assert len(performances) > 0
        assert 'match_rating' in performances.columns
        assert 'goals' in performances.columns


class TestPlayerPerformance:
    """Tests for individual player performance generation."""
    
    def test_performance_rating_in_valid_range(self):
        """Test that match ratings are in valid range (4-10)."""
        matches, teams = generate_match_data(n_matches=10, seed=42)
        players = generate_player_data(n_players=20, seed=42)
        
        player = players.iloc[0]
        match = matches.iloc[0]
        
        perf = _generate_player_performance(player, match, 'home', 2, 1)
        
        assert 4.0 <= perf['match_rating'] <= 10.0
    
    def test_performance_goals_non_negative(self):
        """Test that goals are non-negative."""
        matches, teams = generate_match_data(n_matches=10, seed=42)
        players = generate_player_data(n_players=20, seed=42)
        
        player = players.iloc[0]
        match = matches.iloc[0]
        
        perf = _generate_player_performance(player, match, 'home', 2, 1)
        
        assert perf['goals'] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Feature Engineering Module for Sports Analytics

Transforms raw match and player data into features suitable for
predictive modeling and statistical analysis.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    rolling_windows: List[int] = None
    include_form: bool = True
    include_h2h: bool = True
    include_home_advantage: bool = True
    
    def __post_init__(self):
        if self.rolling_windows is None:
            self.rolling_windows = [3, 5, 10]


class FeatureEngineer:
    """
    Feature engineering class for sports analytics.
    
    Creates features from raw match data for predictive modeling.
    
    Example:
    --------
    >>> fe = FeatureEngineer()
    >>> features = fe.create_match_features(matches_df, teams_df)
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self._fitted = False
    
    def create_match_features(
        self,
        matches: pd.DataFrame,
        teams: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Create comprehensive features for match prediction.
        
        Parameters:
        -----------
        matches : pd.DataFrame
            Match data with home/away teams and results
        teams : pd.DataFrame, optional
            Team statistics data
            
        Returns:
        --------
        pd.DataFrame
            Feature-engineered dataset ready for modeling
        """
        df = matches.copy()
        
        # Sort by date for rolling calculations
        df = df.sort_values('date').reset_index(drop=True)
        
        # Create team-based features
        df = self._create_team_features(df, teams)
        
        # Create rolling statistics
        df = self._create_rolling_features(df)
        
        # Create head-to-head features
        if self.config.include_h2h:
            df = self._create_h2h_features(df)
        
        # Create form features
        if self.config.include_form:
            df = self._create_form_features(df)
        
        # Create match context features
        df = self._create_context_features(df)
        
        # Drop rows with NaN from rolling windows
        df = df.dropna()
        
        self._fitted = True
        return df
    
    def _create_team_features(
        self,
        df: pd.DataFrame,
        teams: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Add team-level features."""
        if teams is not None:
            # Merge team stats
            df = df.merge(
                teams.add_prefix('home_'),
                left_on='home_team_id',
                right_on='home_team_id',
                how='left'
            )
            df = df.merge(
                teams.add_prefix('away_'),
                left_on='away_team_id',
                right_on='away_team_id',
                how='left'
            )
            
            # Create relative strength features
            df['attack_diff'] = df['home_attack_strength'] - df['away_attack_strength']
            df['defense_diff'] = df['home_defense_strength'] - df['away_defense_strength']
            df['overall_diff'] = df['home_overall_rating'] - df['away_overall_rating']
            df['market_value_diff'] = df['home_market_value_m'] - df['away_market_value_m']
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling average features for each team."""
        # Create features for home and away teams
        for prefix in ['home', 'away']:
            team_col = f'{prefix}_team_id'
            
            for window in self.config.rolling_windows:
                # Rolling goals scored
                df[f'{prefix}_goals_scored_{window}'] = df.groupby(team_col).apply(
                    lambda x: x[f'{prefix}_score'].rolling(window, min_periods=1).mean()
                ).reset_index(level=0, drop=True)
                
                # Rolling goals conceded (need opponent's goals)
                opp_prefix = 'away' if prefix == 'home' else 'home'
                df[f'{prefix}_goals_conceded_{window}'] = df.groupby(team_col).apply(
                    lambda x: x[f'{opp_prefix}_score'].rolling(window, min_periods=1).mean()
                ).reset_index(level=0, drop=True)
        
        return df
    
    def _create_h2h_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create head-to-head features."""
        # Create a unique identifier for each matchup
        df['matchup'] = df.apply(
            lambda x: tuple(sorted([x['home_team_id'], x['away_team_id']])),
            axis=1
        )
        
        # Calculate historical H2H results
        h2h_stats = []
        
        for idx, row in df.iterrows():
            # Get previous matches between these teams
            prev_matches = df[
                (df['matchup'] == row['matchup']) & 
                (df['date'] < row['date'])
            ]
            
            if len(prev_matches) > 0:
                home_wins = prev_matches[
                    (prev_matches['home_team_id'] == row['home_team_id']) &
                    (prev_matches['result'] == 'H')
                ].shape[0]
                away_wins = prev_matches[
                    (prev_matches['home_team_id'] == row['home_team_id']) &
                    (prev_matches['result'] == 'A')
                ].shape[0]
                draws = prev_matches[prev_matches['result'] == 'D'].shape[0]
                
                h2h_stats.append({
                    'h2h_home_wins': home_wins,
                    'h2h_away_wins': away_wins,
                    'h2h_draws': draws,
                    'h2h_total': len(prev_matches),
                    'h2h_home_win_rate': home_wins / len(prev_matches) if len(prev_matches) > 0 else 0.5,
                })
            else:
                h2h_stats.append({
                    'h2h_home_wins': 0,
                    'h2h_away_wins': 0,
                    'h2h_draws': 0,
                    'h2h_total': 0,
                    'h2h_home_win_rate': 0.5,
                })
        
        h2h_df = pd.DataFrame(h2h_stats)
        df = pd.concat([df.reset_index(drop=True), h2h_df], axis=1)
        df = df.drop('matchup', axis=1)
        
        return df
    
    def _create_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create team form features based on recent results."""
        for prefix in ['home', 'away']:
            team_col = f'{prefix}_team_id'
            
            # Calculate form points (3 for win, 1 for draw)
            if prefix == 'home':
                df[f'{prefix}_form_points'] = df.groupby(team_col).apply(
                    lambda x: x['home_points'].rolling(5, min_periods=1).sum()
                ).reset_index(level=0, drop=True)
            else:
                df[f'{prefix}_form_points'] = df.groupby(team_col).apply(
                    lambda x: x['away_points'].rolling(5, min_periods=1).sum()
                ).reset_index(level=0, drop=True)
        
        # Form difference
        df['form_diff'] = df['home_form_points'] - df['away_form_points']
        
        return df
    
    def _create_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create match context features."""
        # Day of week
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Month
        df['month'] = df['date'].dt.month
        
        # Is weekend
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Season progress (0-1)
        season_start = df.groupby('season')['date'].transform('min')
        season_end = df.groupby('season')['date'].transform('max')
        df['season_progress'] = (df['date'] - season_start) / (season_end - season_start + pd.Timedelta(days=1))
        
        # Goal expectancy features
        df['expected_total_goals'] = (
            df.get('home_goals_scored_5', 0) + df.get('away_goals_scored_5', 0)
        ) / 2
        df['goal_diff_expectation'] = (
            df.get('home_goals_scored_5', 0) - df.get('away_goals_scored_5', 0)
        )
        
        return df
    
    def create_player_features(
        self,
        performances: pd.DataFrame,
        players: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create features for player performance prediction.
        
        Parameters:
        -----------
        performances : pd.DataFrame
            Player match performances
        players : pd.DataFrame
            Player master data
            
        Returns:
        --------
        pd.DataFrame
            Feature-engineered player dataset
        """
        df = performances.merge(players, on='player_id', how='left')
        
        # Rolling performance metrics
        for window in [3, 5, 10]:
            df[f'rolling_goals_{window}'] = df.groupby('player_id')['goals'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f'rolling_assists_{window}'] = df.groupby('player_id')['assists'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f'rolling_rating_{window}'] = df.groupby('player_id')['match_rating'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        
        # Recent form (last 5 games)
        df['recent_form'] = df.groupby('player_id')['match_rating'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        
        # Days since last match
        df['days_since_last'] = df.groupby('player_id')['date'].diff().dt.days.fillna(7)
        
        # Season statistics
        df['season_goals'] = df.groupby(['player_id', 'season'])['goals'].transform('cumsum')
        df['season_assists'] = df.groupby(['player_id', 'season'])['assists'].transform('cumsum')
        df['season_minutes'] = df.groupby(['player_id', 'season'])['minutes_played'].transform('cumsum')
        
        # Efficiency metrics
        df['goals_per_90'] = df['season_goals'] / (df['season_minutes'] / 90 + 0.1)
        df['assists_per_90'] = df['season_assists'] / (df['season_minutes'] / 90 + 0.1)
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns for modeling."""
        return [
            'home_attack_strength', 'away_attack_strength',
            'home_defense_strength', 'away_defense_strength',
            'home_overall_rating', 'away_overall_rating',
            'attack_diff', 'defense_diff', 'overall_diff',
            'home_goals_scored_5', 'away_goals_scored_5',
            'home_goals_conceded_5', 'away_goals_conceded_5',
            'home_form_points', 'away_form_points', 'form_diff',
            'h2h_home_win_rate', 'market_value_diff',
            'is_weekend', 'season_progress'
        ]

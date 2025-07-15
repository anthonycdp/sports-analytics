"""
Data Generation Module for Sports Analytics

Generates realistic simulated data for football/basketball matches and players.
Uses statistical distributions to create believable sports statistics.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from datetime import datetime, timedelta


def generate_team_stats(
    n_teams: int = 20,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate realistic team statistics.
    
    Parameters:
    -----------
    n_teams : int
        Number of teams to generate
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        Team statistics including attack, defense, form ratings
    """
    if seed is not None:
        np.random.seed(seed)
    
    team_names = [
        "Eagles", "Tigers", "Lions", "Wolves", "Bears",
        "Hawks", "Falcons", "Panthers", "Wildcats", "Bulldogs",
        "Mustangs", "Stallions", "Raptors", "Cobras", "Vipers",
        "Thunder", "Lightning", "Storm", "Blaze", "Inferno"
    ][:n_teams]
    
    # Generate correlated team attributes
    base_strength = np.random.uniform(60, 95, n_teams)
    
    teams = pd.DataFrame({
        'team_id': range(1, n_teams + 1),
        'team_name': team_names,
        'attack_strength': np.clip(base_strength + np.random.normal(0, 8, n_teams), 50, 100),
        'defense_strength': np.clip(100 - base_strength + np.random.normal(0, 8, n_teams), 50, 100),
        'home_advantage': np.random.uniform(1.05, 1.25, n_teams),
        'form_rating': np.random.uniform(0.3, 0.8, n_teams),
        'market_value_m': np.random.exponential(150, n_teams) + 50,
        'avg_age': np.random.normal(26.5, 2, n_teams),
    })
    
    teams['overall_rating'] = (teams['attack_strength'] + teams['defense_strength']) / 2
    
    return teams


def generate_match_data(
    n_matches: int = 1000,
    n_teams: int = 20,
    start_date: str = "2022-08-01",
    end_date: str = "2024-05-30",
    sport: str = "football",
    seed: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate realistic match data for football or basketball.
    
    Parameters:
    -----------
    n_matches : int
        Number of matches to generate
    n_teams : int
        Number of teams
    start_date : str
        Start date for matches
    end_date : str
        End date for matches
    sport : str
        'football' or 'basketball'
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        Match data and team statistics
    """
    if seed is not None:
        np.random.seed(seed)
    
    teams = generate_team_stats(n_teams, seed)
    
    # Generate dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    date_range = (end - start).days
    
    matches = []
    
    for match_id in range(1, n_matches + 1):
        # Select home and away teams (no self-matches)
        home_idx = np.random.randint(0, n_teams)
        away_idx = np.random.randint(0, n_teams - 1)
        if away_idx >= home_idx:
            away_idx += 1
        
        home_team = teams.iloc[home_idx]
        away_team = teams.iloc[away_idx]
        
        # Calculate expected goals/points based on team strengths
        if sport == "football":
            # Football: typically 0-5 goals per team
            home_attack = home_team['attack_strength'] / 100
            away_defense = away_team['defense_strength'] / 100
            home_expected = (home_attack * (1 - away_defense/200)) * home_team['home_advantage'] * 2.2
            away_expected = (away_team['attack_strength']/100 * (1 - home_team['defense_strength']/200)) * 1.6
            
            home_score = np.random.poisson(home_expected)
            away_score = np.random.poisson(away_expected)
            
            # Possession based on team strength
            home_possession = 50 + (home_team['overall_rating'] - away_team['overall_rating']) * 0.3
            home_possession = np.clip(home_possession + np.random.normal(0, 5), 30, 70)
            
            # Shots
            home_shots = int(np.random.exponential(12) + 5)
            away_shots = int(np.random.exponential(10) + 4)
            
            # Shots on target
            home_shots_ot = int(home_shots * np.random.uniform(0.3, 0.5))
            away_shots_ot = int(away_shots * np.random.uniform(0.3, 0.5))
            
            # Corners and fouls
            home_corners = np.random.poisson(5)
            away_corners = np.random.poisson(4)
            home_fouls = np.random.poisson(12)
            away_fouls = np.random.poisson(13)
            
        else:  # Basketball
            # Basketball: typically 80-130 points per team
            home_expected = (home_team['attack_strength'] * 1.2) + np.random.normal(0, 8)
            away_expected = (away_team['attack_strength'] * 1.1) + np.random.normal(0, 8)
            
            home_score = int(np.clip(home_expected, 75, 140))
            away_score = int(np.clip(away_expected, 75, 140))
            
            # Additional basketball stats
            home_possession = np.random.uniform(48, 52)
            home_shots = np.random.randint(70, 95)
            away_shots = np.random.randint(70, 95)
            home_shots_ot = int(home_shots * np.random.uniform(0.45, 0.55))
            away_shots_ot = int(away_shots * np.random.uniform(0.45, 0.55))
            home_corners = np.random.randint(8, 15)  # 3-pointers made
            away_corners = np.random.randint(8, 15)
            home_fouls = np.random.randint(15, 25)
            away_fouls = np.random.randint(15, 25)
        
        # Determine result
        if home_score > away_score:
            result = 'H'
            home_points = 3 if sport == 'football' else 1
            away_points = 0
        elif home_score < away_score:
            result = 'A'
            home_points = 0
            away_points = 3 if sport == 'football' else 1
        else:
            result = 'D'
            home_points = 1
            away_points = 1
        
        # Generate match date
        match_date = start + timedelta(days=np.random.randint(0, date_range))
        
        matches.append({
            'match_id': match_id,
            'date': match_date,
            'season': f"{match_date.year}-{match_date.year + 1}" if match_date.month >= 8 else f"{match_date.year - 1}-{match_date.year}",
            'home_team_id': home_team['team_id'],
            'home_team': home_team['team_name'],
            'away_team_id': away_team['team_id'],
            'away_team': away_team['team_name'],
            'home_score': home_score,
            'away_score': away_score,
            'result': result,
            'home_points': home_points,
            'away_points': away_points,
            'home_possession': round(home_possession, 1),
            'home_shots': home_shots,
            'away_shots': away_shots,
            'home_shots_on_target': home_shots_ot,
            'away_shots_on_target': away_shots_ot,
            'home_corners': home_corners,
            'away_corners': away_corners,
            'home_fouls': home_fouls,
            'away_fouls': away_fouls,
            'total_goals': home_score + away_score,
            'goal_difference': abs(home_score - away_score),
            'attendance': int(np.random.exponential(30000) + 10000),
        })
    
    matches_df = pd.DataFrame(matches)
    matches_df['date'] = pd.to_datetime(matches_df['date'])
    matches_df = matches_df.sort_values('date').reset_index(drop=True)
    
    return matches_df, teams


def generate_player_data(
    n_players: int = 400,
    n_teams: int = 20,
    matches_df: Optional[pd.DataFrame] = None,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate player performance data.
    
    Parameters:
    -----------
    n_players : int
        Number of players to generate
    n_teams : int
        Number of teams
    matches_df : pd.DataFrame, optional
        Match data to generate performances from
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        Player data and match performances
    """
    if seed is not None:
        np.random.seed(seed)
    
    positions = ['GK', 'DEF', 'MID', 'FWD']
    position_weights = [0.1, 0.35, 0.35, 0.2]
    
    first_names = ['James', 'John', 'Michael', 'David', 'Robert', 'William', 'Carlos', 'Luis', 'Marco', 'Ahmed',
                   'Lucas', 'Mateo', 'Bruno', 'Leo', 'Kai', 'Ethan', 'Noah', 'Oliver', 'Elijah', 'Mateus']
    last_names = ['Smith', 'Johnson', 'Silva', 'Santos', 'Garcia', 'Martinez', 'Muller', 'Rossi', 'Anderson', 'Kim',
                  'Tanaka', 'Nguyen', 'Patel', 'Kowalski', 'Johansson', 'Petrov', 'Hansen', 'Santos', 'Costa', 'Fernandez']
    
    players = []
    for player_id in range(1, n_players + 1):
        position = np.random.choice(positions, p=position_weights)
        team_id = np.random.randint(1, n_teams + 1)
        
        # Position-specific base ratings
        if position == 'GK':
            base_rating = np.random.normal(70, 10)
        elif position == 'DEF':
            base_rating = np.random.normal(68, 12)
        elif position == 'MID':
            base_rating = np.random.normal(70, 11)
        else:  # FWD
            base_rating = np.random.normal(72, 13)
        
        age = int(np.clip(np.random.normal(26, 4), 18, 38))
        
        players.append({
            'player_id': player_id,
            'player_name': f"{np.random.choice(first_names)} {np.random.choice(last_names)}",
            'team_id': team_id,
            'position': position,
            'age': age,
            'overall_rating': np.clip(base_rating, 50, 95),
            'pace': np.clip(np.random.normal(70, 12), 40, 95),
            'shooting': np.clip(np.random.normal(65 if position == 'FWD' else 55, 15), 30, 95),
            'passing': np.clip(np.random.normal(68 if position in ['MID', 'FWD'] else 55, 12), 30, 95),
            'dribbling': np.clip(np.random.normal(67, 14), 30, 95),
            'defending': np.clip(np.random.normal(55 if position in ['DEF', 'GK'] else 70, 15), 30, 95),
            'physical': np.clip(np.random.normal(68, 10), 40, 95),
            'market_value_m': np.random.exponential(20) + 1,
            'wage_k': np.random.exponential(50) + 10,
        })
    
    players_df = pd.DataFrame(players)
    
    # Generate match performances if matches provided
    if matches_df is not None:
        performances = []
        for _, match in matches_df.iterrows():
            # Sample players from both teams (starting XI)
            home_players = players_df[players_df['team_id'] == match['home_team_id']].sample(
                min(11, len(players_df[players_df['team_id'] == match['home_team_id']])),
                replace=True
            )
            away_players = players_df[players_df['team_id'] == match['away_team_id']].sample(
                min(11, len(players_df[players_df['team_id'] == match['away_team_id']])),
                replace=True
            )
            
            for _, player in home_players.iterrows():
                perf = _generate_player_performance(
                    player, match, 'home', match['home_score'], match['away_score']
                )
                performances.append(perf)
            
            for _, player in away_players.iterrows():
                perf = _generate_player_performance(
                    player, match, 'away', match['away_score'], match['home_score']
                )
                performances.append(perf)
        
        performances_df = pd.DataFrame(performances)
        return players_df, performances_df
    
    return players_df


def _generate_player_performance(
    player: pd.Series,
    match: pd.Series,
    side: str,
    team_goals: int,
    opp_goals: int
) -> dict:
    """Generate a single player performance record."""
    is_starter = np.random.random() > 0.15
    minutes_played = 90 if is_starter else np.random.randint(0, 45)
    
    base_rating = player['overall_rating']
    
    # Position-specific stats
    if player['position'] == 'GK':
        goals = 0
        assists = 0 if np.random.random() > 0.01 else 1
        saves = np.random.poisson(3 + opp_goals * 0.5)
        tackles = np.random.poisson(0.5)
    elif player['position'] == 'DEF':
        goals = np.random.poisson(0.05) if minutes_played > 30 else 0
        assists = np.random.poisson(0.08) if minutes_played > 30 else 0
        saves = 0
        tackles = np.random.poisson(3)
    elif player['position'] == 'MID':
        goals = np.random.poisson(0.12) if minutes_played > 30 else 0
        assists = np.random.poisson(0.15) if minutes_played > 30 else 0
        saves = 0
        tackles = np.random.poisson(2)
    else:  # FWD
        goals = np.random.poisson(0.25) if minutes_played > 30 else 0
        assists = np.random.poisson(0.12) if minutes_played > 30 else 0
        saves = 0
        tackles = np.random.poisson(0.8)
    
    # Adjust goals based on team performance
    if team_goals > 0 and player['position'] in ['MID', 'FWD']:
        goal_bonus = np.random.binomial(team_goals, 0.15)
        goals = min(goals + goal_bonus, 4)
    
    # Calculate match rating
    perf_score = 6.5 + (goals * 0.8) + (assists * 0.5) - (opp_goals * 0.1)
    if side == 'home':
        perf_score += 0.1
    perf_score += np.random.normal(0, 0.5)
    match_rating = np.clip(perf_score, 4.0, 10.0)
    
    return {
        'performance_id': None,  # Will be assigned later
        'match_id': match['match_id'],
        'date': match['date'],
        'player_id': player['player_id'],
        'team_id': player['team_id'],
        'opponent_id': match['away_team_id'] if side == 'home' else match['home_team_id'],
        'side': side,
        'position': player['position'],
        'is_starter': is_starter,
        'minutes_played': minutes_played,
        'goals': goals,
        'assists': assists,
        'saves': saves,
        'tackles': tackles,
        'passes': np.random.poisson(25 if player['position'] in ['MID', 'DEF'] else 15),
        'pass_accuracy': np.clip(np.random.normal(78, 10), 50, 100),
        'shots': np.random.poisson(1.5 if player['position'] == 'FWD' else 0.5),
        'shots_on_target': np.random.poisson(0.5),
        'match_rating': round(match_rating, 1),
    }


if __name__ == "__main__":
    # Example usage
    matches, teams = generate_match_data(n_matches=500, seed=42)
    players, performances = generate_player_data(n_players=300, matches_df=matches, seed=42)
    
    print("Matches shape:", matches.shape)
    print("Teams shape:", teams.shape)
    print("Players shape:", players.shape)
    print("Performances shape:", performances.shape)
    
    # Save to CSV
    matches.to_csv("data/raw/matches.csv", index=False)
    teams.to_csv("data/raw/teams.csv", index=False)
    players.to_csv("data/raw/players.csv", index=False)
    performances.to_csv("data/raw/performances.csv", index=False)

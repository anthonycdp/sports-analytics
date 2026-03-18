"""
Generate sample sports analytics datasets for portfolio project.
This creates realistic synthetic data for demonstration purposes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

def generate_nba_player_stats():
    """Generate synthetic NBA player statistics for 2023-24 season."""
    n_players = 150

    # Player positions
    positions = ['PG', 'SG', 'SF', 'PF', 'C']
    position_weights = [0.2, 0.2, 0.2, 0.2, 0.2]

    # First names and last names for realistic player names
    first_names = ['James', 'Michael', 'David', 'Kevin', 'Chris', 'Marcus', 'Anthony', 'Jason',
                   'Tyler', 'Brandon', 'Jayson', 'Devin', 'Trae', 'Luka', 'Giannis', 'Nikola',
                   'Joel', 'LeBron', 'Stephen', 'Klay', 'Draymond', 'Anthony', 'Karl', 'Damian',
                   'Jaylen', 'Jayson', 'Donovan', 'Shai', 'Paolo', 'Victor']
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
                  'Rodriguez', 'Martinez', 'Anderson', 'Taylor', 'Thomas', 'Moore', 'Jackson',
                  'Martin', 'Lee', 'Thompson', 'White', 'Harris', 'Clark', 'Lewis', 'Robinson',
                  'Walker', 'Young', 'Allen', 'King', 'Wright', 'Scott', 'Green']

    players = []
    for i in range(n_players):
        position = np.random.choice(positions, p=position_weights)

        # Base stats vary by position
        if position == 'PG':
            base_ppg = np.random.normal(15, 5)
            base_ast = np.random.normal(7, 2)
            base_reb = np.random.normal(4, 1.5)
            base_fg = np.random.normal(0.45, 0.05)
        elif position == 'SG':
            base_ppg = np.random.normal(16, 5)
            base_ast = np.random.normal(3, 1.5)
            base_reb = np.random.normal(4, 1.5)
            base_fg = np.random.normal(0.44, 0.05)
        elif position == 'SF':
            base_ppg = np.random.normal(14, 4)
            base_ast = np.random.normal(3, 1.5)
            base_reb = np.random.normal(5, 2)
            base_fg = np.random.normal(0.46, 0.04)
        elif position == 'PF':
            base_ppg = np.random.normal(13, 4)
            base_ast = np.random.normal(2, 1)
            base_reb = np.random.normal(7, 2)
            base_fg = np.random.normal(0.48, 0.04)
        else:  # C
            base_ppg = np.random.normal(12, 4)
            base_ast = np.random.normal(1.5, 0.8)
            base_reb = np.random.normal(9, 2.5)
            base_fg = np.random.normal(0.52, 0.04)

        # Generate correlated stats
        mpg = np.clip(np.random.normal(28, 8), 5, 42)
        games_played = int(np.clip(np.random.normal(65, 15), 20, 82))

        player = {
            'player_id': f'NBA{str(i+1).zfill(4)}',
            'player_name': f"{np.random.choice(first_names)} {np.random.choice(last_names)}",
            'team': np.random.choice(['LAL', 'BOS', 'MIA', 'GSW', 'PHX', 'MIL', 'DEN', 'PHI',
                                      'DAL', 'MEM', 'CLE', 'NYK', 'ATL', 'CHI', 'BKN', 'TOR',
                                      'SAC', 'MIN', 'NOP', 'POR', 'SAS', 'OKC', 'UTA', 'HOU',
                                                                      'DET', 'IND', 'ORL', 'WAS', 'CHA']),
            'position': position,
            'age': int(np.clip(np.random.normal(26, 4), 19, 40)),
            'games_played': games_played,
            'mpg': round(mpg, 1),
            'ppg': round(np.clip(base_ppg * (mpg/30), 0, 35), 1),
            'rpg': round(np.clip(base_reb * (mpg/30), 0, 15), 1),
            'apg': round(np.clip(base_ast * (mpg/30), 0, 12), 1),
            'spg': round(np.clip(np.random.normal(1, 0.5) * (mpg/30), 0, 3), 1),
            'bpg': round(np.clip(np.random.normal(0.8, 0.5) * (mpg/30) if position in ['C', 'PF'] else np.random.normal(0.4, 0.3) * (mpg/30), 0, 3), 1),
            'fg_pct': round(np.clip(base_fg, 0.35, 0.70), 3),
            'fg3_pct': round(np.clip(np.random.normal(0.35, 0.06), 0.20, 0.50), 3),
            'ft_pct': round(np.clip(np.random.normal(0.78, 0.08), 0.55, 0.95), 3),
            'tovpg': round(np.clip(np.random.normal(2, 1) * (mpg/30), 0, 5), 1),
            'salary': int(np.clip(np.random.lognormal(16, 0.8), 1_000_000, 55_000_000))
        }

        # Calculate advanced metrics
        player['ts_pct'] = round(player['fg_pct'] + 0.44 * player['ft_pct'] * 0.5, 3)
        player['per'] = round(np.clip(player['ppg'] * 0.8 + player['rpg'] * 0.5 + player['apg'] * 0.6 +
                                     player['spg'] * 1.5 + player['bpg'] * 1.5 - player['tovpg'] * 0.8, 5, 35), 1)
        player['ws'] = round(player['per'] * player['games_played'] / 82 * 0.8, 1)

        players.append(player)

    df = pd.DataFrame(players)
    df.to_csv('nba_player_stats_2023_24.csv', index=False)
    print(f"Generated NBA player stats: {len(df)} players")
    return df


def generate_mlb_team_stats():
    """Generate synthetic MLB team statistics for 2023 season."""
    teams = [
        'LAD', 'ATL', 'NYY', 'HOU', 'TEX', 'BAL', 'TB', 'TOR', 'SEA', 'MIN',
        'MIA', 'MIL', 'PHI', 'AZ', 'CHC', 'SF', 'SD', 'CLE', 'BOS', 'LAA',
        'CIN', 'PIT', 'NYM', 'STL', 'DET', 'WSH', 'KC', 'CWS', 'COL', 'OAK'
    ]

    team_data = []
    for team in teams:
        # Base winning percentage (some teams better than others)
        base_wpct = np.random.beta(5, 5)  # Centers around 0.500

        # Calculate runs scored and allowed based on winning percentage
        # Using Pythagorean expectation as a guide
        runs_scored = int(np.random.normal(750, 80))
        runs_allowed = int(runs_scored * (1 - base_wpct) / base_wpct * np.random.uniform(0.95, 1.05))

        wins = int(162 * base_wpct)
        losses = 162 - wins

        team_stats = {
            'team': team,
            'league': 'AL' if team in ['NYY', 'HOU', 'TEX', 'BAL', 'TB', 'TOR', 'SEA', 'MIN',
                                       'CLE', 'BOS', 'LAA', 'DET', 'KC', 'CWS', 'OAK'] else 'NL',
            'wins': wins,
            'losses': losses,
            'win_pct': round(wins / 162, 3),
            'runs_scored': runs_scored,
            'runs_allowed': runs_allowed,
            'run_diff': runs_scored - runs_allowed,
            'home_wins': int(wins * np.random.uniform(0.48, 0.55)),
            'home_losses': int(losses * np.random.uniform(0.45, 0.52)),
            'away_wins': 0,  # Will calculate
            'away_losses': 0,  # Will calculate
            'batting_avg': round(np.random.normal(0.248, 0.015), 3),
            'on_base_pct': round(np.random.normal(0.320, 0.015), 3),
            'slugging_pct': round(np.random.normal(0.420, 0.030), 3),
            'ops': round(np.random.normal(0.740, 0.040), 3),
            'era': round(np.random.normal(4.15, 0.50), 2),
            'whip': round(np.random.normal(1.28, 0.08), 3),
            'hr_hit': int(np.random.normal(180, 35)),
            'hr_allowed': int(np.random.normal(175, 35)),
            'stolen_bases': int(np.random.normal(95, 30)),
            'errors': int(np.random.normal(85, 15)),
            'pythagorean_wins': round(162 * (runs_scored**1.83) / ((runs_scored**1.83) + (runs_allowed**1.83)), 0),
        }

        team_stats['away_wins'] = wins - team_stats['home_wins']
        team_stats['away_losses'] = losses - team_stats['home_losses']
        team_stats['pythagorean_wins'] = int(team_stats['pythagorean_wins'])

        team_data.append(team_stats)

    df = pd.DataFrame(team_data)
    df.to_csv('mlb_team_stats_2023.csv', index=False)
    print(f"Generated MLB team stats: {len(df)} teams")
    return df


def generate_premier_league_matches():
    """Generate synthetic Premier League match data for 2023-24 season."""
    teams = [
        'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
        'Burnley', 'Chelsea', 'Crystal Palace', 'Everton', 'Fulham',
        'Liverpool', 'Luton Town', 'Man City', 'Man United', 'Newcastle',
        'Nottingham Forest', 'Sheffield United', 'Tottenham', 'West Ham', 'Wolves'
    ]

    # Team strength ratings (synthetic)
    team_strength = {
        'Man City': 0.85, 'Arsenal': 0.80, 'Liverpool': 0.78, 'Tottenham': 0.70,
        'Chelsea': 0.68, 'Newcastle': 0.67, 'Man United': 0.66, 'Brighton': 0.62,
        'Aston Villa': 0.61, 'West Ham': 0.58, 'Brentford': 0.55, 'Fulham': 0.54,
        'Crystal Palace': 0.52, 'Wolves': 0.51, 'Bournemouth': 0.48, 'Everton': 0.47,
        'Nottingham Forest': 0.45, 'Luton Town': 0.40, 'Burnley': 0.38, 'Sheffield United': 0.35
    }

    matches = []
    match_id = 1

    # Generate all matches (each team plays each other twice)
    start_date = datetime(2023, 8, 12)

    for gameweek in range(1, 39):  # 38 gameweeks
        for home_team in teams:
            for away_team in teams:
                if home_team == away_team:
                    continue

                # Only generate ~10 matches per gameweek (realistic)
                if np.random.random() > 0.1:
                    continue

                # Calculate expected goals based on team strength
                home_strength = team_strength[home_team]
                away_strength = team_strength[away_team]

                home_xg = np.clip(np.random.gamma(2.5 * home_strength / away_strength, 0.8), 0, 6)
                away_xg = np.clip(np.random.gamma(2.0 * away_strength / home_strength, 0.7), 0, 6)

                home_goals = int(np.round(home_xg + np.random.normal(0, 0.5)))
                away_goals = int(np.round(away_xg + np.random.normal(0, 0.5)))

                home_goals = max(0, home_goals)
                away_goals = max(0, away_goals)

                # Determine result
                if home_goals > away_goals:
                    result = 'H'
                    home_points, away_points = 3, 0
                elif away_goals > home_goals:
                    result = 'A'
                    home_points, away_points = 0, 3
                else:
                    result = 'D'
                    home_points, away_points = 1, 1

                match = {
                    'match_id': match_id,
                    'gameweek': gameweek,
                    'date': (start_date + timedelta(days=gameweek*7 + np.random.randint(-2, 3))).strftime('%Y-%m-%d'),
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_goals': home_goals,
                    'away_goals': away_goals,
                    'result': result,
                    'home_points': home_points,
                    'away_points': away_points,
                    'home_xg': round(home_xg, 2),
                    'away_xg': round(away_xg, 2),
                    'total_goals': home_goals + away_goals,
                    'goal_diff': home_goals - away_goals,
                    'home_possession': round(np.clip(np.random.normal(50 + (home_strength - away_strength) * 20, 8), 25, 80), 1),
                    'home_shots': int(np.clip(np.random.normal(14 + home_strength * 5, 4), 3, 30)),
                    'away_shots': int(np.clip(np.random.normal(12 + away_strength * 5, 4), 3, 30)),
                    'home_shots_on_target': int(np.clip(np.random.normal(5 + home_strength * 2, 2), 1, 15)),
                    'away_shots_on_target': int(np.clip(np.random.normal(4 + away_strength * 2, 2), 1, 15)),
                    'home_corners': int(np.clip(np.random.normal(6, 2.5), 1, 15)),
                    'away_corners': int(np.clip(np.random.normal(5, 2), 1, 12)),
                    'home_fouls': int(np.clip(np.random.normal(12, 3), 3, 25)),
                    'away_fouls': int(np.clip(np.random.normal(11, 3), 3, 25)),
                    'home_yellow_cards': int(np.clip(np.random.poisson(1.5), 0, 5)),
                    'away_yellow_cards': int(np.clip(np.random.poisson(1.4), 0, 5)),
                    'home_red_cards': int(np.clip(np.random.poisson(0.08), 0, 2)),
                    'away_red_cards': int(np.clip(np.random.poisson(0.08), 0, 2)),
                }

                matches.append(match)
                match_id += 1

    df = pd.DataFrame(matches)
    df.to_csv('premier_league_matches_2023_24.csv', index=False)
    print(f"Generated Premier League matches: {len(df)} matches")
    return df


def generate_player_injury_data():
    """Generate synthetic injury data for analysis."""
    n_injuries = 500

    injury_types = ['Hamstring', 'Knee ACL', 'Ankle Sprain', 'Groin', 'Shoulder',
                   'Back', 'Concussion', 'Quad', 'Calf', 'Hip', 'Wrist', 'Foot']

    sports = ['NBA', 'NFL', 'MLB', 'NHL', 'Soccer']
    sport_weights = [0.25, 0.30, 0.20, 0.10, 0.15]

    injuries = []
    for i in range(n_injuries):
        sport = np.random.choice(sports, p=sport_weights)
        injury_type = np.random.choice(injury_types)

        # Days missed varies by injury type
        if injury_type == 'Knee ACL':
            days_missed = int(np.clip(np.random.normal(250, 50), 120, 365))
        elif injury_type == 'Concussion':
            days_missed = int(np.clip(np.random.normal(12, 8), 3, 45))
        elif injury_type in ['Hamstring', 'Groin', 'Quad', 'Calf']:
            days_missed = int(np.clip(np.random.normal(21, 14), 5, 90))
        else:
            days_missed = int(np.clip(np.random.normal(28, 20), 3, 180))

        injury = {
            'injury_id': i + 1,
            'sport': sport,
            'injury_type': injury_type,
            'days_missed': days_missed,
            'player_age': int(np.clip(np.random.normal(27, 4), 18, 42)),
            'position_type': np.random.choice(['Offense', 'Defense', 'Specialist']),
            'season_phase': np.random.choice(['Preseason', 'Regular Season', 'Playoffs'], p=[0.15, 0.70, 0.15]),
            'recurring': np.random.choice([True, False], p=[0.25, 0.75]),
            'surgery_required': np.random.choice([True, False], p=[0.20, 0.80]),
            'recovery_success': np.random.choice([True, False], p=[0.92, 0.08]),
        }
        injuries.append(injury)

    df = pd.DataFrame(injuries)
    df.to_csv('player_injuries.csv', index=False)
    print(f"Generated injury data: {len(df)} records")
    return df


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    os.chdir('data')

    print("Generating sample sports datasets...")
    print("-" * 50)

    generate_nba_player_stats()
    generate_mlb_team_stats()
    generate_premier_league_matches()
    generate_player_injury_data()

    print("-" * 50)
    print("All datasets generated successfully!")

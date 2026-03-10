"""
Sports Analytics Interactive Dashboard

An interactive Streamlit dashboard for exploring sports analytics data,
viewing predictions, and analyzing team and player performance.

Run with: streamlit run dashboard/app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from src.data_generation import generate_match_data, generate_player_data
from src.features import FeatureEngineer
from src.models import WinProbabilityModel, prepare_training_data
from src.visualization import SportsVisualizer

# Page configuration
st.set_page_config(
    page_title="Sports Analytics Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stMetric > div {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load or generate data."""
    data_path = Path(__file__).parent.parent / "data" / "raw"
    
    if (data_path / "matches.csv").exists():
        matches = pd.read_csv(data_path / "matches.csv", parse_dates=['date'])
        teams = pd.read_csv(data_path / "teams.csv")
        players = pd.read_csv(data_path / "players.csv")
        performances = pd.read_csv(data_path / "performances.csv", parse_dates=['date'])
    else:
        # Generate data if not exists
        matches, teams = generate_match_data(n_matches=1000, seed=42)
        players, performances = generate_player_data(n_players=400, matches_df=matches, seed=42)
    
    return matches, teams, players, performances


@st.cache_resource
def train_model(matches, teams):
    """Train the win probability model."""
    fe = FeatureEngineer()
    matches_featured = fe.create_match_features(matches, teams)
    
    X_train, X_test, y_train, y_test = prepare_training_data(matches_featured)
    
    model = WinProbabilityModel(model_type='gradient_boosting')
    model.fit(X_train, y_train)
    
    return model, fe


def main():
    # Header
    st.markdown("<h1 class='main-header'>⚽ Sports Analytics Dashboard</h1>", unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        matches, teams, players, performances = load_data()
    
    # Train model
    with st.spinner("Training prediction model..."):
        model, feature_engineer = train_model(matches, teams)
    
    viz = SportsVisualizer()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Overview", "📊 Match Analysis", "👥 Teams", "🏃 Players", "🔮 Predictions"]
    )
    
    # Sidebar filters
    st.sidebar.markdown("---")
    st.sidebar.title("Filters")
    
    # Season filter
    seasons = sorted(matches['season'].unique())
    selected_season = st.sidebar.multiselect(
        "Season",
        seasons,
        default=seasons[-1:] if seasons else []
    )
    
    # Filter data
    if selected_season:
        filtered_matches = matches[matches['season'].isin(selected_season)]
    else:
        filtered_matches = matches
    
    # ============ PAGE: Overview ============
    if page == "🏠 Overview":
        st.header("Data Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Matches", len(filtered_matches))
        
        with col2:
            st.metric("Teams", len(teams))
        
        with col3:
            st.metric("Players", len(players))
        
        with col4:
            st.metric("Seasons", len(seasons))
        
        st.markdown("---")
        
        # Results distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Match Results Distribution")
            fig = viz.plot_match_results_distribution(filtered_matches)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Goals Distribution")
            fig = viz.plot_goals_distribution(filtered_matches)
            st.plotly_chart(fig, use_container_width=True)
        
        # Home advantage
        st.markdown("---")
        st.subheader("Home Advantage Analysis")
        
        home_wins = (filtered_matches['result'] == 'H').sum()
        total = len(filtered_matches)
        home_win_rate = home_wins / total if total > 0 else 0
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            **Key Finding:** Home teams win **{home_win_rate:.1%}** of matches.
            
            This demonstrates the classic "home advantage" phenomenon observed across 
            most sports - home teams consistently perform better due to factors like:
            - Crowd support
            - Familiarity with the venue
            - Reduced travel fatigue
            """)
        
        with col2:
            st.metric("Home Win Rate", f"{home_win_rate:.1%}")
            st.metric("Away Win Rate", f"{(filtered_matches['result'] == 'A').mean():.1%}")
            st.metric("Draw Rate", f"{(filtered_matches['result'] == 'D').mean():.1%}")
    
    # ============ PAGE: Match Analysis ============
    elif page == "📊 Match Analysis":
        st.header("Match Analysis")
        
        # Goal statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_home_goals = filtered_matches['home_score'].mean()
            st.metric("Avg Home Goals", f"{avg_home_goals:.2f}")
        
        with col2:
            avg_away_goals = filtered_matches['away_score'].mean()
            st.metric("Avg Away Goals", f"{avg_away_goals:.2f}")
        
        with col3:
            avg_total = filtered_matches['total_goals'].mean()
            st.metric("Avg Total Goals", f"{avg_total:.2f}")
        
        # Recent matches
        st.subheader("Recent Matches")
        
        recent_cols = ['date', 'home_team', 'home_score', 'away_score', 'away_team', 'result']
        st.dataframe(
            filtered_matches[recent_cols].sort_values('date', ascending=False).head(20),
            use_container_width=True
        )
        
        # H2H Heatmap
        st.subheader("Head-to-Head Results")
        fig = viz.plot_league_table_heatmap(filtered_matches.head(300))
        st.plotly_chart(fig, use_container_width=True)
    
    # ============ PAGE: Teams ============
    elif page == "👥 Teams":
        st.header("Team Analysis")
        
        # Team selector
        selected_team = st.selectbox(
            "Select Team",
            teams['team_name'].tolist()
        )
        
        if selected_team:
            team_info = teams[teams['team_name'] == selected_team].iloc[0]
            
            # Team metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Overall Rating", f"{team_info['overall_rating']:.1f}")
            
            with col2:
                st.metric("Attack", f"{team_info['attack_strength']:.1f}")
            
            with col3:
                st.metric("Defense", f"{team_info['defense_strength']:.1f}")
            
            with col4:
                st.metric("Market Value", f"€{team_info['market_value_m']:.0f}M")
            
            # Season progress
            st.subheader("Season Progress")
            fig = viz.plot_season_progress(filtered_matches, selected_team)
            st.plotly_chart(fig, use_container_width=True)
            
            # Team matches
            st.subheader("Team Matches")
            team_matches = filtered_matches[
                (filtered_matches['home_team'] == selected_team) |
                (filtered_matches['away_team'] == selected_team)
            ].sort_values('date', ascending=False)
            
            st.dataframe(
                team_matches[['date', 'home_team', 'home_score', 'away_score', 'away_team', 'result']].head(15),
                use_container_width=True
            )
        
        # All teams standings
        st.markdown("---")
        st.subheader("League Standings")
        fig = viz.plot_team_performance(teams, filtered_matches)
        st.plotly_chart(fig, use_container_width=True)
    
    # ============ PAGE: Players ============
    elif page == "🏃 Players":
        st.header("Player Analysis")
        
        # Top players
        st.subheader("Top Performers")
        
        # Calculate player stats
        player_stats = performances.groupby('player_id').agg({
            'match_rating': 'mean',
            'goals': 'sum',
            'assists': 'sum',
            'minutes_played': 'sum'
        }).merge(players[['player_id', 'player_name', 'position', 'team_id']], on='player_id')
        
        player_stats['goals_per_90'] = player_stats['goals'] / (player_stats['minutes_played'] / 90 + 0.1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top by Match Rating**")
            top_rated = player_stats.nlargest(10, 'match_rating')[
                ['player_name', 'position', 'match_rating', 'goals', 'assists']
            ]
            st.dataframe(top_rated, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**Top by Goals**")
            top_scorers = player_stats.nlargest(10, 'goals')[
                ['player_name', 'position', 'goals', 'goals_per_90']
            ]
            st.dataframe(top_scorers, use_container_width=True, hide_index=True)
        
        # Player selector for individual analysis
        st.markdown("---")
        st.subheader("Individual Player Analysis")
        
        selected_player = st.selectbox(
            "Select Player",
            player_stats['player_name'].tolist()
        )
        
        if selected_player:
            fig = viz.plot_player_performance(performances, selected_player)
            st.plotly_chart(fig, use_container_width=True)
    
    # ============ PAGE: Predictions ============
    elif page == "🔮 Predictions":
        st.header("Match Predictions")
        
        st.markdown("""
        Use our trained model to predict match outcomes. Select two teams to see 
        the predicted probabilities for home win, draw, and away win.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            home_team = st.selectbox("Home Team", teams['team_name'].tolist(), index=0)
        
        with col2:
            away_options = [t for t in teams['team_name'].tolist() if t != home_team]
            away_team = st.selectbox("Away Team", away_options, index=0)
        
        if st.button("Predict Outcome", type="primary"):
            # Get team IDs
            home_team_id = teams[teams['team_name'] == home_team]['team_id'].values[0]
            away_team_id = teams[teams['team_name'] == away_team]['team_id'].values[0]
            
            # Get historical matches between these teams
            h2h_matches = filtered_matches[
                ((filtered_matches['home_team_id'] == home_team_id) & 
                 (filtered_matches['away_team_id'] == away_team_id)) |
                ((filtered_matches['home_team_id'] == away_team_id) & 
                 (filtered_matches['away_team_id'] == home_team_id))
            ]
            
            # Create feature vector for prediction
            # Use recent averages and team stats
            home_team_data = teams[teams['team_id'] == home_team_id].iloc[0]
            away_team_data = teams[teams['team_id'] == away_team_id].iloc[0]
            
            # Get recent form
            recent_home = filtered_matches[
                filtered_matches['home_team_id'] == home_team_id
            ].tail(5)
            recent_away = filtered_matches[
                filtered_matches['away_team_id'] == away_team_id
            ].tail(5)
            
            # Prepare features for prediction
            feature_dict = {
                'home_attack_strength': home_team_data['attack_strength'],
                'away_attack_strength': away_team_data['attack_strength'],
                'home_defense_strength': home_team_data['defense_strength'],
                'away_defense_strength': away_team_data['defense_strength'],
                'home_overall_rating': home_team_data['overall_rating'],
                'away_overall_rating': away_team_data['overall_rating'],
                'attack_diff': home_team_data['attack_strength'] - away_team_data['attack_strength'],
                'defense_diff': home_team_data['defense_strength'] - away_team_data['defense_strength'],
                'overall_diff': home_team_data['overall_rating'] - away_team_data['overall_rating'],
                'market_value_diff': home_team_data['market_value_m'] - away_team_data['market_value_m'],
            }
            
            # Add default values for rolling features
            for col in model.feature_names:
                if col not in feature_dict:
                    feature_dict[col] = 0
            
            # Create feature DataFrame
            X_pred = pd.DataFrame([feature_dict])[model.feature_names]
            
            # Get predictions
            proba = model.predict_proba(X_pred)[0]
            prediction = model.predict(X_pred)[0]
            
            # Display results
            st.markdown("---")
            st.subheader("Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Home Win Probability", f"{proba[2]:.1%}")
            
            with col2:
                st.metric("Draw Probability", f"{proba[1]:.1%}")
            
            with col3:
                st.metric("Away Win Probability", f"{proba[0]:.1%}")
            
            # Predicted outcome
            outcome_map = {'H': f'{home_team} Win', 'D': 'Draw', 'A': f'{away_team} Win'}
            st.info(f"**Predicted Outcome:** {outcome_map[prediction]}")
            
            # Head-to-head record
            if len(h2h_matches) > 0:
                st.markdown("---")
                st.subheader("Head-to-Head Record")
                
                home_wins_h2h = len(h2h_matches[
                    (h2h_matches['home_team_id'] == home_team_id) & 
                    (h2h_matches['result'] == 'H') |
                    (h2h_matches['away_team_id'] == home_team_id) & 
                    (h2h_matches['result'] == 'A')
                ])
                away_wins_h2h = len(h2h_matches[
                    (h2h_matches['home_team_id'] == away_team_id) & 
                    (h2h_matches['result'] == 'H') |
                    (h2h_matches['away_team_id'] == away_team_id) & 
                    (h2h_matches['result'] == 'A')
                ])
                draws_h2h = len(h2h_matches[h2h_matches['result'] == 'D'])
                
                col1, col2, col3 = st.columns(3)
                col1.metric(f"{home_team} Wins", home_wins_h2h)
                col2.metric("Draws", draws_h2h)
                col3.metric(f"{away_team} Wins", away_wins_h2h)
            
            # Confidence visualization
            st.markdown("---")
            st.subheader("Probability Distribution")
            
            prob_df = pd.DataFrame({
                'Outcome': [f'{home_team} Win', 'Draw', f'{away_team} Win'],
                'Probability': [proba[2], proba[1], proba[0]]
            })
            
            fig = viz.plot_match_results_distribution(
                pd.DataFrame({'result': ['H', 'D', 'A']}),
                title="Predicted Outcome Probabilities"
            )
            fig.data[0].values = [proba[2] * 100, proba[1] * 100, proba[0] * 100]
            st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Sports Analytics Dashboard | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

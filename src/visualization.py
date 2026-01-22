"""
Visualization Module for Sports Analytics

Provides interactive and static visualizations for match analysis,
player performance, and model insights using Plotly and Matplotlib.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns


class SportsVisualizer:
    """
    Comprehensive visualization class for sports analytics.
    
    Provides methods for creating interactive plots, dashboards,
    and statistical visualizations.
    
    Example:
    --------
    >>> viz = SportsVisualizer()
    >>> fig = viz.plot_match_results_distribution(matches_df)
    >>> fig.show()
    """
    
    def __init__(self, theme: str = 'plotly_white'):
        self.theme = theme
        self.color_palette = {
            'home': '#1f77b4',
            'draw': '#7f7f7f',
            'away': '#d62728',
            'primary': '#2563eb',
            'secondary': '#10b981',
            'accent': '#f59e0b'
        }
    
    def plot_match_results_distribution(
        self,
        matches: pd.DataFrame,
        title: str = "Match Results Distribution"
    ) -> go.Figure:
        """Create a pie chart of match results."""
        results = matches['result'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=['Home Win', 'Draw', 'Away Win'],
            values=[results.get('H', 0), results.get('D', 0), results.get('A', 0)],
            hole=0.4,
            marker_colors=[self.color_palette['home'], 
                          self.color_palette['draw'], 
                          self.color_palette['away']],
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig.update_layout(
            title=title,
            template=self.theme,
            showlegend=True
        )
        
        return fig
    
    def plot_goals_distribution(
        self,
        matches: pd.DataFrame,
        title: str = "Goals Distribution"
    ) -> go.Figure:
        """Create histogram of goals scored."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Home Goals', 'Away Goals')
        )
        
        fig.add_trace(
            go.Histogram(
                x=matches['home_score'],
                name='Home Goals',
                marker_color=self.color_palette['home'],
                nbinsx=10
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(
                x=matches['away_score'],
                name='Away Goals',
                marker_color=self.color_palette['away'],
                nbinsx=10
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=title,
            template=self.theme,
            showlegend=False
        )
        
        return fig
    
    def plot_team_performance(
        self,
        teams: pd.DataFrame,
        matches: pd.DataFrame,
        metric: str = 'overall_rating',
        title: str = "Team Performance Overview"
    ) -> go.Figure:
        """Create a bar chart of team performance metrics."""
        # Calculate team points from matches
        home_points = matches.groupby('home_team')['home_points'].sum()
        away_points = matches.groupby('away_team')['away_points'].sum()
        total_points = home_points.add(away_points, fill_value=0)
        
        team_stats = teams.copy()
        team_stats['total_points'] = team_stats['team_name'].map(total_points)
        team_stats = team_stats.sort_values('total_points', ascending=True)
        
        fig = go.Figure(go.Bar(
            x=team_stats['total_points'],
            y=team_stats['team_name'],
            orientation='h',
            marker_color=self.color_palette['primary'],
            text=team_stats['total_points'],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Total Points',
            yaxis_title='Team',
            template=self.theme,
            height=600
        )
        
        return fig
    
    def plot_league_table_heatmap(
        self,
        matches: pd.DataFrame,
        title: str = "Head-to-Head Results Matrix"
    ) -> go.Figure:
        """Create a heatmap of head-to-head results."""
        teams = sorted(matches['home_team'].unique())
        n_teams = len(teams)
        
        # Create matrix of results
        # 3 = home win, 1 = draw, 0 = away win
        matrix = np.full((n_teams, n_teams), np.nan)
        
        for _, match in matches.iterrows():
            home_idx = teams.index(match['home_team'])
            away_idx = teams.index(match['away_team'])
            
            if match['result'] == 'H':
                matrix[home_idx, away_idx] = 3
            elif match['result'] == 'D':
                matrix[home_idx, away_idx] = 1
            else:
                matrix[home_idx, away_idx] = 0
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=teams,
            y=teams,
            colorscale=[
                [0, self.color_palette['away']],
                [0.5, self.color_palette['draw']],
                [1, self.color_palette['home']]
            ],
            hoverongaps=False,
            colorbar=dict(
                title='Result',
                tickvals=[0, 1, 3],
                ticktext=['Away Win', 'Draw', 'Home Win']
            )
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Away Team',
            yaxis_title='Home Team',
            template=self.theme,
            height=700,
            width=800
        )
        
        return fig
    
    def plot_win_probability_timeline(
        self,
        probabilities: pd.DataFrame,
        actual_result: str,
        title: str = "Win Probability Over Time"
    ) -> go.Figure:
        """Plot win probabilities over time."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=probabilities['date'],
            y=probabilities['home_prob'],
            name='Home Win',
            line=dict(color=self.color_palette['home'], width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=probabilities['date'],
            y=probabilities['draw_prob'],
            name='Draw',
            line=dict(color=self.color_palette['draw'], width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=probabilities['date'],
            y=probabilities['away_prob'],
            name='Away Win',
            line=dict(color=self.color_palette['away'], width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Probability',
            template=self.theme,
            hovermode='x unified',
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def plot_player_performance(
        self,
        performances: pd.DataFrame,
        player_name: str,
        title: Optional[str] = None
    ) -> go.Figure:
        """Plot individual player performance over time."""
        player_data = performances[
            performances['player_name'] == player_name
        ].sort_values('date')
        
        if title is None:
            title = f"{player_name} Performance Over Time"
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Match Rating', 'Goals', 'Assists', 'Minutes Played'),
            vertical_spacing=0.15
        )
        
        fig.add_trace(
            go.Scatter(x=player_data['date'], y=player_data['match_rating'],
                      mode='lines+markers', name='Rating', line=dict(color=self.color_palette['primary'])),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=player_data['date'], y=player_data['goals'],
                  name='Goals', marker_color=self.color_palette['secondary']),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=player_data['date'], y=player_data['assists'],
                  name='Assists', marker_color=self.color_palette['accent']),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=player_data['date'], y=player_data['minutes_played'],
                      mode='lines', name='Minutes', line=dict(color=self.color_palette['home'])),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            template=self.theme,
            showlegend=False,
            height=600
        )
        
        return fig
    
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        title: str = "Feature Importance"
    ) -> go.Figure:
        """Create horizontal bar chart for feature importance."""
        importance_df = importance_df.sort_values('importance', ascending=True)
        
        fig = go.Figure(go.Bar(
            x=importance_df['importance'],
            y=importance_df['feature'],
            orientation='h',
            marker_color=self.color_palette['primary']
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Importance',
            yaxis_title='Feature',
            template=self.theme,
            height=max(400, len(importance_df) * 25)
        )
        
        return fig
    
    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        title: str = "Feature Correlation Matrix"
    ) -> go.Figure:
        """Create correlation heatmap."""
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu_r',
            zmid=0,
            text=corr.round(2).values,
            texttemplate='%{text}',
            textfont={'size': 8}
        ))
        
        fig.update_layout(
            title=title,
            template=self.theme,
            width=800,
            height=700
        )
        
        return fig
    
    def plot_model_performance_comparison(
        self,
        metrics_dict: dict,
        title: str = "Model Performance Comparison"
    ) -> go.Figure:
        """Compare performance metrics across models."""
        models = list(metrics_dict.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
        
        fig = go.Figure()
        
        for metric in metrics:
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=models,
                y=[metrics_dict[m].get(metric, 0) for m in models],
                text=[f'{metrics_dict[m].get(metric, 0):.3f}' for m in models],
                textposition='outside'
            ))
        
        fig.update_layout(
            title=title,
            barmode='group',
            yaxis=dict(range=[0, 1.1]),
            template=self.theme,
            height=500
        )
        
        return fig
    
    def plot_season_progress(
        self,
        matches: pd.DataFrame,
        team_name: str,
        title: Optional[str] = None
    ) -> go.Figure:
        """Plot team's points accumulation over the season."""
        team_matches = matches[
            (matches['home_team'] == team_name) | 
            (matches['away_team'] == team_name)
        ].sort_values('date')
        
        points = []
        cumulative = 0
        
        for _, match in team_matches.iterrows():
            if match['home_team'] == team_name:
                cumulative += match['home_points']
            else:
                cumulative += match['away_points']
            points.append(cumulative)
        
        if title is None:
            title = f"{team_name} Season Progress"
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=team_matches['date'],
            y=points,
            mode='lines+markers',
            name='Cumulative Points',
            line=dict(color=self.color_palette['primary'], width=2),
            fill='tozeroy',
            fillcolor='rgba(37, 99, 235, 0.1)'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Cumulative Points',
            template=self.theme
        )
        
        return fig
    
    def create_dashboard(
        self,
        matches: pd.DataFrame,
        teams: pd.DataFrame,
        performances: Optional[pd.DataFrame] = None
    ) -> Tuple[go.Figure, ...]:
        """Create a multi-panel dashboard."""
        # Panel 1: Results distribution
        fig1 = self.plot_match_results_distribution(matches)
        
        # Panel 2: Goals distribution
        fig2 = self.plot_goals_distribution(matches)
        
        # Panel 3: Team standings
        fig3 = self.plot_team_performance(teams, matches)
        
        # Panel 4: H2H heatmap
        fig4 = self.plot_league_table_heatmap(matches.head(500))  # Limit for performance
        
        return fig1, fig2, fig3, fig4


def set_matplotlib_style():
    """Configure matplotlib for consistent styling."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.figsize': (12, 6),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 100
    })

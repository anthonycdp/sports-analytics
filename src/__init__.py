"""
Sports Analytics Package

A comprehensive toolkit for sports data analysis, hypothesis testing,
predictive modeling, and visualization.
"""

__version__ = "1.0.0"
__author__ = "Sports Analytics Team"

from .data_generation import generate_match_data, generate_player_data
from .features import FeatureEngineer
from .models import WinProbabilityModel, PlayerPerformanceModel
from .visualization import SportsVisualizer

__all__ = [
    "generate_match_data",
    "generate_player_data",
    "FeatureEngineer",
    "WinProbabilityModel",
    "PlayerPerformanceModel",
    "SportsVisualizer",
]

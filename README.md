# Sports Analytics: Deep Dive Analysis with Hypotheses, Modeling & Storytelling

A comprehensive sports analytics project demonstrating data science best practices through statistical hypothesis testing, predictive modeling, and narrative-driven analysis.

## 🎯 Project Overview

This project provides a complete end-to-end sports analytics pipeline for football (soccer) data, including:

- **Data Generation**: Realistic simulated match and player data
- **Statistical Analysis**: Hypothesis testing to validate sports analytics assumptions
- **Predictive Modeling**: Machine learning models for win probability and player performance
- **Interactive Dashboard**: Streamlit-based visualization tool
- **Storytelling**: Narrative-driven analysis notebooks

## 📊 Key Features

### Statistical Hypothesis Testing
- **Home Advantage**: Testing whether home teams win significantly more than random chance
- **Goal Scoring**: Analyzing home vs away goal distributions
- **Team Quality Effect**: Validating correlation between team ratings and outcomes
- **Position Performance**: ANOVA testing for player position differences

### Predictive Models
- **Win Probability Model**: Gradient Boosting classifier for match outcome prediction (H/D/A)
- **Player Performance Model**: Regression model for predicting match ratings
- **Feature Engineering**: Rolling averages, form metrics, H2H records

### Interactive Visualizations
- Match results distribution
- Team performance comparisons
- Head-to-head matrices
- Player performance timelines
- Season progress tracking

## 🏗️ Project Structure

```
08-sports-analytics/
├── data/
│   ├── raw/                    # Generated raw data
│   └── processed/              # Feature-engineered data
├── notebooks/
│   ├── 01_data_exploration.ipynb    # EDA and data overview
│   ├── 02_hypothesis_testing.ipynb  # Statistical tests
│   ├── 03_predictive_modeling.ipynb # ML model training
│   └── 04_storytelling_analysis.ipynb # Narrative analysis
├── src/
│   ├── __init__.py
│   ├── data_generation.py      # Data generation functions
│   ├── features.py             # Feature engineering
│   ├── models.py               # Predictive models
│   └── visualization.py        # Plotting utilities
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── tests/
│   ├── test_data.py           # Data generation tests
│   ├── test_models.py         # Model tests
│   └── test_features.py       # Feature engineering tests
├── models/                     # Saved model files
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🚀 Quick Start

### Option 1: Local Installation

```bash
# Clone the repository
git clone <repository-url>
cd 08-sports-analytics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate data
python -c "from src.data_generation import generate_match_data, generate_player_data; \
    matches, teams = generate_match_data(n_matches=1000, seed=42); \
    players, performances = generate_player_data(n_players=400, matches_df=matches, seed=42); \
    matches.to_csv('data/raw/matches.csv', index=False); \
    teams.to_csv('data/raw/teams.csv', index=False); \
    players.to_csv('data/raw/players.csv', index=False); \
    performances.to_csv('data/raw/performances.csv', index=False)"

# Run the dashboard
streamlit run dashboard/app.py
```

### Option 2: Docker

```bash
# Build and run with Docker Compose
docker-compose up dashboard

# Or run Jupyter notebooks
docker-compose up jupyter

# Run tests
docker-compose --profile test run test
```

## 📓 Notebooks Guide

### 1. Data Exploration (`01_data_exploration.ipynb`)
- Generate and explore match data
- Analyze distributions and correlations
- Initial insights on home advantage and goal patterns

### 2. Hypothesis Testing (`02_hypothesis_testing.ipynb`)
- Formal statistical tests for sports analytics assumptions
- P-value interpretation and effect size calculations
- Confidence intervals and post-hoc analysis

### 3. Predictive Modeling (`03_predictive_modeling.ipynb`)
- Feature engineering pipeline
- Model comparison (Logistic, Random Forest, Gradient Boosting)
- Cross-validation and hyperparameter tuning
- Feature importance analysis

### 4. Storytelling Analysis (`04_storytelling_analysis.ipynb`)
- Narrative-driven exploration
- Season-long story arcs
- The beauty of unpredictability in sports

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py -v
```

## 📈 Model Performance

### Win Probability Model
| Metric | Score |
|--------|-------|
| Accuracy | ~52% |
| F1-Score | ~0.48 |
| AUC-ROC | ~0.68 |

*Note: Predicting 3-class outcomes in sports is challenging. Random baseline is 33%.*

### Key Predictive Features
1. Team strength differential
2. Home advantage
3. Recent form (last 5 matches)
4. Head-to-head history
5. Market value difference

## 🔬 Statistical Findings

| Hypothesis | Test | Result |
|------------|------|--------|
| Home Advantage Exists | Z-test | ✅ Confirmed (p < 0.001) |
| Home Teams Score More | T-test | ✅ Confirmed (p < 0.001) |
| Team Rating Affects Outcomes | Chi-square | ✅ Confirmed (p < 0.001) |
| Position Affects Performance | ANOVA | ✅ Confirmed (p < 0.001) |

## 🎨 Visualization Examples

The project includes interactive Plotly visualizations:

- **Pie Charts**: Match result distributions
- **Histograms**: Goal scoring patterns
- **Heatmaps**: Head-to-head results matrix
- **Line Charts**: Season progress and form
- **Bar Charts**: Team standings and player rankings

## 🛠️ Technology Stack

- **Python 3.11+**
- **pandas & numpy**: Data manipulation
- **scikit-learn**: Machine learning
- **scipy & statsmodels**: Statistical testing
- **plotly**: Interactive visualizations
- **streamlit**: Dashboard framework
- **pytest**: Testing

## 📋 Requirements

```
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
xgboost>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.18.0
jupyter>=1.0.0
streamlit>=1.28.0
pytest>=7.4.0
statsmodels>=0.14.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Statistical methodologies inspired by academic sports analytics research
- Visualization design influenced by modern sports data journalism
- Built following best practices from the Python data science community

---

*"In sports, as in life, the numbers tell a story - but they don't write the ending."*

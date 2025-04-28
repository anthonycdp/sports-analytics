# Sports Analytics Docker Image
# Python 3.11 with data science stack

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models

# Generate initial data
RUN python -c "from src.data_generation import generate_match_data, generate_player_data; \
    matches, teams = generate_match_data(n_matches=1000, seed=42); \
    players, performances = generate_player_data(n_players=400, matches_df=matches, seed=42); \
    matches.to_csv('data/raw/matches.csv', index=False); \
    teams.to_csv('data/raw/teams.csv', index=False); \
    players.to_csv('data/raw/players.csv', index=False); \
    performances.to_csv('data/raw/performances.csv', index=False)"

# Expose ports
# 8501 for Streamlit
# 8888 for Jupyter
EXPOSE 8501 8888

# Default command
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A clothing recommendation system built using Women's E-Commerce Clothing Reviews dataset. The project processes customer reviews to build recommendation models using collaborative filtering and deep learning approaches.

## Environment Setup

**Python Version:** 3.14

**Virtual Environment:**
```bash
# Activate virtual environment
source venv/bin/activate

# Deactivate when done
deactivate
```

**Dependencies:**
- pandas - data manipulation and cleaning
- matplotlib - data visualization
- seaborn - statistical data visualization
- PyTorch (planned for Phase 4) - deep learning recommendation model

## Data Pipeline

### Dataset Schema
The raw data (`Women_s_E-Commerce_Clothing_Reviews_1594_1.csv`) contains:
- **Clothing.ID**: Product identifier
- **Age**: Customer age
- **Title**: Review title
- **Review.Text**: Full review text (used for NLP/deep learning)
- **Rating**: 1-5 star rating
- **Recommended.IND**: Binary recommendation indicator (0/1)
- **Positive.Feedback.Count**: Number of positive feedback votes
- **Division.Name**: Product division (General, General Petite, Intimates)
- **Department.Name**: Product department (Dresses, Tops, Bottoms, etc.)
- **Class.Name**: Product class

### Data Processing Pipeline

**1. Clean Data:**
```bash
python cleanup.py
```

This script:
- Removes redundant index columns (`Unnamed: 0`, `X`)
- Fixes typo: "Initmates" → "Intimates"
- Drops rows missing critical metadata or review text
- Fills missing titles with "No Title"
- **Cold start filtering**: Keeps only products with 5+ reviews for stable recommendations

**Outputs:**
- `dataset/Cleaned_Recommendation_Data.csv` - Full cleaned dataset for EDA (Phase 3)
- `dataset/RecSys_Training_Data.csv` - Filtered dataset for model training (Phase 4)

**2. Generate EDA Charts:**
```bash
python eda_charts.py
```

Creates visualizations in `charts/`:
- `rating_distribution.png` - Univariate analysis of rating frequency (class imbalance check)
- `ratings_by_division.png` - Bivariate analysis of ratings by product division
- `recommend_probability.png` - Recommendation probability by rating level

## Project Architecture

The project follows a phased framework:

**Phase 3:** Exploratory Data Analysis ✅
- Data cleaning and preprocessing
- Statistical visualizations
- Business context analysis

**Phase 4:** Model Training ✅
- 3 production models: Popularity Baseline, Matrix Factorization, Neural CF
- PyTorch collaborative filtering
- Comprehensive evaluation (RMSE, Hit@K, NDCG)

**Phase 5:** Evaluation & Analysis ✅
- Model comparison and benchmarking
- Business impact analysis
- Error analysis and segment performance

**Phase 6:** Deployment ✅
- REST API with FastAPI
- Docker containerization
- Production-ready serving infrastructure

### Cold Start Strategy
The 5+ review threshold addresses the cold-start problem by ensuring training data has sufficient signal for each product. Products with fewer reviews are excluded from training but retained in the cleaned dataset for analysis.

## Key Design Decisions

**Why Two Datasets?**
- `Cleaned_Recommendation_Data.csv`: Preserves all valid data for comprehensive EDA and business insights
- `RecSys_Training_Data.csv`: Filters for model stability (items with 5+ reviews only)

**Data Quality:**
- Reviews without text are dropped (critical for NLP features)
- Missing division names are dropped (critical for categorical features)
- Missing titles are filled with placeholder (non-critical field)

## Working with Data

**Load cleaned data:**
```python
import pandas as pd

# For EDA and analysis
df = pd.read_csv('dataset/Cleaned_Recommendation_Data.csv')

# For model training
df_training = pd.read_csv('dataset/RecSys_Training_Data.csv')
```

**Check data quality:**
```python
print(f"Total reviews: {len(df)}")
print(f"Training reviews: {len(df_training)}")
print(df['Division.Name'].value_counts())
print(df['Rating'].value_counts())
```

## API Deployment (Phase 6)

**Quick Start:**
```bash
# Install dependencies
pip install -r requirements.txt

# Run API locally
python run_api.py

# Access API docs
open http://localhost:8000/docs
```

**Docker Deployment:**
```bash
# Start all services (API + Redis + PostgreSQL)
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

**API Endpoints:**
- `GET /health` - Health check
- `GET /models` - List available models
- `POST /recommend` - Get recommendations
- `POST /predict` - Predict rating

**Example Request:**
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 100,
    "k": 10,
    "model": "mf"
  }'
```

**Documentation:**
- API Reference: `API_DOCUMENTATION.md`
- Deployment Summary: `PHASE6_DEPLOYMENT_SUMMARY.md`
- Interactive Docs: http://localhost:8000/docs

## Common Issues

**If charts don't generate:** Ensure the `charts/` directory exists (script auto-creates it) and matplotlib backend is configured correctly.

**If cleaning fails:** Verify the raw CSV exists at `dataset/Women_s_E-Commerce_Clothing_Reviews_1594_1.csv`.

**If imports fail:** Activate the virtual environment first with `source venv/bin/activate`.

**If API fails to start:** Check that models exist in `checkpoints/` and data files in `dataset/`.

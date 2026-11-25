# Plume

**Pollutant-Based AQI Modeling and Seasonal Analysis for India**  

Plume is a regression modeling project and Streamlit web application that predicts the Air Quality Index (AQI) of 26 Indian cities from daily pollutant concentrations and temporal features, mapped to the official CPCB AQI scale.

---

## Dataset
- **Source:** Air Quality Data in India (2015–2020) — `city_day.csv` (CPCB / Kaggle)
- **Raw Records:** 29,531 daily observations across 26 Indian cities (`2015-01-01` to `2020-07-01`)
- **Cleaned Records:** 24,850 observations after removing rows with missing `AQI` targets
- **Target Variable:** `AQI`
- **Core Features:** `City`, `Year`, `Month`, `DayOfWeek`, `PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, `O3`

---

## Data Preprocessing & Feature Engineering
1. **Target Cleaning:** Dropped 4,681 rows (15.85%) missing `AQI` to avoid fabricating ground-truth labels.
2. **Date Feature Extraction:** Split `Date` into `Year`, `Month`, and `DayOfWeek` to capture multi-year trends, seasonal cycles, and weekday/weekend traffic patterns.
3. **Feature Selection:** Retained the 6 core pollutants (`PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, `O3`) and removed sparsely recorded columns (`Xylene`, `NH3`, `Toluene`, `Benzene`).
4. **Missing Value Imputation:** Imputed missing pollutant readings using city-wise medians (robust to right-skewed winter smog spikes) with a global median fallback.
5. **Categorical Encoding:** Applied One-Hot Encoding (`pd.get_dummies`) to `City` to avoid introducing artificial ordinal rankings.

---

## Exploratory Data Analysis Highlights
- **Most Polluted Cities (Mean AQI):** Ahmedabad (`452.12`), Delhi (`259.49`), Patna (`240.78`), Gurugram (`225.12`), Lucknow (`217.97`).
- **Seasonal Pattern:** AQI peaks in **November** (`241.68`) due to winter inversions and post-harvest stubble burning, and drops to its lowest in **July** (`111.85`) during peak monsoon.
- **Top AQI Correlations:** `CO` (`0.68`), `PM2.5` (`0.66`), `NO2` (`0.53`), `PM10` (`0.49`), `SO2` (`0.49`).

---

## Model Evaluation (80/20 Train-Test Split)

| Model | MAE | MSE | RMSE | R² Score |
| :--- | :---: | :---: | :---: | :---: |
| Linear Regression | 30.14 | 3264.18 | 57.13 | 0.8217 |
| **Random Forest Regressor** | **21.10** | **1689.29** | **41.10** | **0.9077** |

Random Forest Regressor reduces RMSE by **28.1%** compared to Linear Regression and explains **90.77%** of the variance in AQI, making it the deployed model in the Streamlit app.

---

## Project Structure
```text
plume-aqi/
├── .streamlit/
│   └── config.toml                # Light theme configuration
├── data/
│   └── city_day.csv               # Kaggle CPCB dataset
├── notebooks/
│   └── plume_aqi.ipynb            # EDA, preprocessing, training & evaluation
├── models/
│   ├── best_aqi_model.joblib      # Trained Random Forest Regressor
│   └── metadata.joblib            # Cities, medians, feature list & metrics
├── app.py                         # Streamlit web application
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md
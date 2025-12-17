
# Leveraging World Bank Data to Model and Forecast Infant Mortality in Ethiopia : A Comparative Study with East African Nations

## Project Overview
This project presents a comprehensive time-series analysis of Infant Mortality trends in similar East African countries using World Bank health indicators. The study combines data cleaning, exploratory data analysis(EDA), comparative data analysis, predictive modeling and provides an interactive dashboard for data-driven public health insights.

## Project Structure
InfantMortalityProject/
│
├── screenshot/
│   └── (dashboard screenshots and visual outputs)
│
├── Cleaned_data.csv
│   └── Cleaned World Bank health indicators dataset used for analysis and modeling
│
├── arima_test_results.csv
│   └── Actual vs predicted values for ARIMA model evaluation
│
├── comparative_forecasts_all.csv
│   └── Forecasted infant mortality rates for selected East African countries
│
├── ethiopia_forecast_2024_2030.csv
│   └── ARIMA forecast results for Ethiopia (2024–2030)
│
├── ethiopia_arima_model.pkl
│   └── Trained ARIMA model saved for reuse
│
├── dashboard.py
│   └── Plotly Dash application for interactive visualization and forecasting
│
├── README.md
│   └── Project documentation and instructions
│
├── requirements.txt
│   └── Python dependencies required to run the project

## Features
*   **Exploratory Analysis:** Univariate,Bivariate and Multivariate analysis
*   **Comparative Analysis:** Compare Ethiopia's infant mortality trends with Kenya, Tanzania, Rwanda, and Uganda.
*   **ARIMA Forecasting:** Predictive modeling to forecast infant mortality rates up to 2030.
*   **SDG Tracking:** Visual assessment of progress towards the UN Sustainable Development Goal target.
*   **Interactive Dashboard:** Built with Plotly Dash for real-time data exploration.
*   **Data-Driven Insights:** Generate actionable policy recommendations based on model outputs.

## Methodology
1.  **Data Acquisition:** Sourced from the World Bank's Health Nutrition and Population Statistics database.
2.  **Data Preparation:** Removed metadata roes, handled missing values,reshaped wide year-based columns into long format, engineered additional features including time lags and improvement rates.
3.  **Exploratory Data Analysis (EDA):** Univariate, bivariate, and multivariate analysis to understand trends and correlations.
4.  **Predictive Modeling:** Implemented an ARIMA(1,1,1) model for time-series forecasting of Ethiopia's infant mortalityand evaluated model using MAE, RMSE and MEPE metrics.
5.  **Dashboard Development:** Results compiled into an interactive dashboard for insights.
 

## Tools
**Install all required tools:**
   *python
   *pandas and numpy
   *matplotlib and seaborn
   *statsmodel
   *plotly and dash

## How to Run
   **Install dependencies:** pip intall -r requirements.txt
   **Run the Dashboard:** python dashboard.py
   **Access the Dashboard:** open a web browser and navigate to http://127.0.0.1:8050/
      

## Key Findings & Insights
*   **Steady Decline:** Ethiopia has shown a consistent downward trend in infant mortality since 1993.
*   **Regional Leader:** Rwanda is projected to come closest to achieving the SDG target of 25 deaths per 1000 live births by 2030.
*   **Model Performance:** The ARIMA model demonstrated excellent predictive accuracy with a Mean Absolute Percentage Error (MAPE) of under 10%.
*   **Critical Gap:** Based on current trends, Ethiopia may require accelerated interventions to meet the SDG target by 2030.

##  Author
*   **Author:** Afyana Fekade Desta
*   **Project:** Capstone Project for Data Analysis course
*   **Data Source:** https://databank.worldbank.org/databases
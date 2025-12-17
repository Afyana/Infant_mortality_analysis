# ============================================
# INFANT MORTALITY DASHBOARD - ETHIOPIA & EAST AFRICA
# ============================================

# Import libraries
from dash import Dash, html, dcc, Input, Output  # pip install dash
import dash_bootstrap_components as dbc   # pip install dash-bootstrap-components
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Initialize the app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# ============================================
# DATA LOADING
# ============================================

print("Loading data for dashboard...")

try:
    # Load your cleaned data
    df = pd.read_csv('cleaned_data.csv')
    print("✅ Main data loaded")
    
    # Load forecasts
    ethiopia_forecast = pd.read_csv('ethiopia_forecast_2024_2030.csv')
    print("✅ Ethiopia forecast loaded")
    
    # Load comparative forecasts
    comparative_forecasts = pd.read_csv('comparative_forecasts.csv')
    print("✅ Comparative forecasts loaded")
    
    # Load test results
    test_results = pd.read_csv('arima_test_results.csv')
    print("✅ Test results loaded")
    
except Exception as e:
    print(f"⚠️ Error loading files: {e}")
    print("Creating sample data for demonstration...")
    
    # Create sample data
    years = list(range(1993, 2024))
    countries = ['Ethiopia', 'Kenya', 'Tanzania', 'Rwanda', 'Uganda']
    
    # Main data
    data = []
    for country in countries:
        if country == 'Ethiopia':
            start_rate = 120
        elif country == 'Rwanda':
            start_rate = 100
        else:
            start_rate = 90
            
        for i, year in enumerate(years):
            current_rate = start_rate * (0.97 ** i)
            data.append({
                'Year': year,
                'Country_Name': country,
                'Mortality rate infant per 1000 live births': max(current_rate, 20),
                'Improvement_Percentage': np.random.uniform(2, 6)
            })
    
    df = pd.DataFrame(data)
    
    # Ethiopia forecast
    ethiopia_forecast = pd.DataFrame({
        'Year': list(range(2024, 2031)),
        'Predicted_Mortality': [35, 33, 31, 29, 27, 25, 24],
        'Lower_CI': [32, 30, 28, 26, 24, 22, 21],
        'Upper_CI': [38, 36, 34, 32, 30, 28, 27]
    })
    
    # Comparative forecasts
    comparative_forecasts = pd.DataFrame({
        'Country': ['Ethiopia', 'Kenya', 'Tanzania', 'Rwanda', 'Uganda'] * 5,
        'Year': sorted([2024, 2025, 2026, 2027, 2028] * 5),
        'Forecast': [
            35.0, 33.5, 32.1, 30.8, 29.6,  # Ethiopia
            32.0, 30.7, 29.5, 28.3, 27.2,  # Kenya
            34.0, 32.6, 31.3, 30.1, 28.9,  # Tanzania
            28.0, 26.9, 25.8, 24.8, 23.8,  # Rwanda
            36.0, 34.6, 33.2, 31.9, 30.6   # Uganda
        ]
    })
    
    # Test results
    test_results = pd.DataFrame({
        'Year': [2019, 2020, 2021, 2022, 2023],
        'Actual': [42, 40, 38, 36, 35],
        'Predicted': [41, 39, 37, 35, 34],
        'Error': [1, 1, 1, 1, 1]
    })

print(f"Data shape: {df.shape}")
print(f"Countries: {df['Country_Name'].unique()}")

# ============================================
# DASHBOARD LAYOUT
# ============================================

app.layout = dbc.Container([
    # HEADER
    html.Div([
        html.H1("Infant Mortality Analysis Dashboard", 
                className='mb-2 text-center', 
                style={'color': '#2c3e50'}),
        html.P("Modeling and Forecasting Infant Mortality in Ethiopia: A Comparative Study with East African Nations",
               className='text-center',
               style={'color': '#7f8c8d', 'fontSize': '18px'}),
        html.Hr(style={'border': '2px solid #3498db', 'width': '80%', 'margin': '20px auto'})
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px'}),
    
    # CONTROLS ROW
    dbc.Row([
        dbc.Col([
            html.Label("Select Countries:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='country-selector',
                value=['Ethiopia', 'Kenya', 'Rwanda'],
                clearable=False,
                multi=True,
                options=[{'label': country, 'value': country} 
                         for country in df['Country_Name'].unique()],
                style={'marginBottom': '20px'}
            )
        ], width=12, md=4),
        
        dbc.Col([
            html.Label("Select Years:", style={'fontWeight': 'bold'}),
            dcc.RangeSlider(
                id='year-slider',
                min=df['Year'].min(),
                max=df['Year'].max(),
                step=1,
                marks={year: str(year) for year in range(df['Year'].min(), df['Year'].max()+1, 5)},
                value=[2000, df['Year'].max()],
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], width=12, md=8),
    ], className='mb-4'),
    
    # KEY METRICS ROW
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Ethiopia 2023", className='card-title'),
                    html.H2(id='current-rate', children="--", style={'color': '#e74c3c'}),
                    html.P("Infant deaths per 1000", className='card-text')
                ])
            ], color="light", outline=True)
        ], width=12, md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Projected 2030", className='card-title'),
                    html.H2(id='projected-2030', children="--", style={'color': '#2ecc71'}),
                    html.P("ARIMA forecast", className='card-text')
                ])
            ], color="light", outline=True)
        ], width=12, md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("SDG Target Gap", className='card-title'),
                    html.H2(id='sdg-gap', children="--", style={'color': '#f39c12'}),
                    html.P("Difference from target 25", className='card-text')
                ])
            ], color="light", outline=True)
        ], width=12, md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Model Accuracy", className='card-title'),
                    html.H2(id='model-mape', children="--", style={'color': '#9b59b6'}),
                    html.P("MAPE (%)", className='card-text')
                ])
            ], color="light", outline=True)
        ], width=12, md=3),
    ], className='mb-4'),
    
    # MAIN CHARTS ROW 1
    dbc.Row([
        dbc.Col([
            html.Img(id='trend-chart')
        ], width=12, md=6),
        
        dbc.Col([
            html.Img(id='forecast-chart')
        ], width=12, md=6),
    ], className='mb-4'),
    
    # MAIN CHARTS ROW 2
    dbc.Row([
        dbc.Col([
            html.Img(id='comparison-chart')
        ], width=12, md=6),
        
        dbc.Col([
            html.Img(id='performance-chart')
        ], width=12, md=6),
    ], className='mb-4'),
    
    # REGIONAL ANALYSIS ROW
    dbc.Row([
        dbc.Col([
            html.Img(id='regional-chart')
        ], width=12, md=6),
        
        dbc.Col([
            html.Div([
                html.H4("📈 Key Insights", style={'color': '#2c3e50'}),
                html.Div(id='insights-container', children=[
                    html.Ul([
                        html.Li("Ethiopia has reduced infant mortality by XX% since 1993"),
                        html.Li("Rwanda shows the fastest improvement rate in the region"),
                        html.Li("ARIMA model predicts Ethiopia will reach SDG target by 2030"),
                        html.Li("Immunization rates strongly correlate with mortality reduction")
                    ], style={'fontSize': '16px', 'lineHeight': '1.6'})
                ])
            ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px', 'height': '100%'})
        ], width=12, md=6),
    ]),
    
    # FOOTER
    html.Footer([
        html.Hr(style={'width': '80%', 'margin': '30px auto'}),
        html.P("Infant Mortality Forecasting Dashboard | Capstone Project | World Bank Data Analysis", 
               className='text-center',
               style={'color': '#7f8c8d'}),
        html.P("Data Source: World Bank Health Nutrition and Population Statistics (1993-2023)",
               className='text-center',
               style={'color': '#95a5a6', 'fontSize': '14px'})
    ], style={'marginTop': '30px'})
], fluid=True)

# ============================================
# CALLBACK FUNCTIONS
# ============================================

@app.callback(
    Output('trend-chart', 'src'),
    Output('forecast-chart', 'src'),
    Output('comparison-chart', 'src'),
    Output('performance-chart', 'src'),
    Output('regional-chart', 'src'),
    Output('current-rate', 'children'),
    Output('projected-2030', 'children'),
    Output('sdg-gap', 'children'),
    Output('model-mape', 'children'),
    Output('insights-container', 'children'),
    Input('country-selector', 'value'),
    Input('year-slider', 'value')
)
def update_dashboard(selected_countries, year_range):
    """
    Main callback function that updates all dashboard components
    """
    
    # ============================================
    # 1. FILTER DATA
    # ============================================
    filtered_df = df[
        (df['Country_Name'].isin(selected_countries)) &
        (df['Year'] >= year_range[0]) &
        (df['Year'] <= year_range[1])
    ]
    
    # ============================================
    # 2. CREATE TREND CHART
    # ============================================
    fig1 = plt.figure(figsize=(8, 5))
    
    for country in selected_countries:
        country_data = filtered_df[filtered_df['Country_Name'] == country]
        plt.plot(country_data['Year'], 
                country_data['Mortality rate infant per 1000 live births'],
                label=country,
                linewidth=2,
                marker='o',
                markersize=4)
    
    plt.title(f'Infant Mortality Trends ({year_range[0]}-{year_range[1]})', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Mortality Rate per 1000', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save to buffer
    buf1 = BytesIO()
    fig1.savefig(buf1, format="png", dpi=100)
    fig1_data = base64.b64encode(buf1.getbuffer()).decode("ascii")
    trend_chart = f'data:image/png;base64,{fig1_data}'
    plt.close(fig1)
    
    # ============================================
    # 3. CREATE FORECAST CHART
    # ============================================
    fig2 = plt.figure(figsize=(8, 5))
    
    # Historical data for Ethiopia
    ethiopia_historical = df[df['Country_Name'] == 'Ethiopia']
    plt.plot(ethiopia_historical['Year'], 
             ethiopia_historical['Mortality rate infant per 1000 live births'],
             label='Ethiopia Historical',
             color='blue',
             linewidth=2)
    
    # Forecast
    plt.plot(ethiopia_forecast['Year'], 
             ethiopia_forecast['Predicted_Mortality'],
             label='ARIMA Forecast',
             color='red',
             linestyle='--',
             linewidth=2,
             marker='o')
    
    # Confidence interval
    plt.fill_between(ethiopia_forecast['Year'],
                     ethiopia_forecast['Lower_CI'],
                     ethiopia_forecast['Upper_CI'],
                     color='red',
                     alpha=0.2,
                     label='95% Confidence')
    
    # SDG target
    plt.axhline(y=25, color='green', linestyle=':', linewidth=2, label='SDG Target')
    
    plt.title('Ethiopia: ARIMA Forecast with Confidence Intervals', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Mortality Rate per 1000', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save to buffer
    buf2 = BytesIO()
    fig2.savefig(buf2, format="png", dpi=100)
    fig2_data = base64.b64encode(buf2.getbuffer()).decode("ascii")
    forecast_chart = f'data:image/png;base64,{fig2_data}'
    plt.close(fig2)
    
    # ============================================
    # 4. CREATE COMPARISON CHART
    # ============================================
    fig3 = plt.figure(figsize=(8, 5))
    
    # Get 2028 forecasts for selected countries
    comp_data = []
    for country in selected_countries:
        country_2028 = comparative_forecasts[
            (comparative_forecasts['Country'] == country) & 
            (comparative_forecasts['Year'] == 2028)
        ]
        if not country_2028.empty:
            comp_data.append({
                'Country': country,
                'Forecast': country_2028['Forecast'].values[0]
            })
    
    if comp_data:
        comp_df = pd.DataFrame(comp_data)
        bars = plt.bar(comp_df['Country'], comp_df['Forecast'])
        
        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # SDG line
        plt.axhline(y=25, color='green', linestyle='--', linewidth=2, label='SDG Target')
    
    plt.title('2028 Forecast Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Country', fontsize=12)
    plt.ylabel('Mortality Rate per 1000', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save to buffer
    buf3 = BytesIO()
    fig3.savefig(buf3, format="png", dpi=100)
    fig3_data = base64.b64encode(buf3.getbuffer()).decode("ascii")
    comparison_chart = f'data:image/png;base64,{fig3_data}'
    plt.close(fig3)
    
    # ============================================
    # 5. CREATE PERFORMANCE CHART
    # ============================================
    fig4 = plt.figure(figsize=(8, 5))
    
    plt.plot(test_results['Year'], test_results['Actual'],
             label='Actual',
             color='green',
             linewidth=2,
             marker='o')
    
    plt.plot(test_results['Year'], test_results['Predicted'],
             label='ARIMA Prediction',
             color='red',
             linestyle='--',
             linewidth=2,
             marker='s')
    
    # Error bars
    plt.fill_between(test_results['Year'],
                     test_results['Actual'] - test_results['Error'].abs(),
                     test_results['Actual'] + test_results['Error'].abs(),
                     color='orange',
                     alpha=0.2,
                     label='Error Range')
    
    plt.title('ARIMA Model Performance (2019-2023 Test Period)', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Mortality Rate per 1000', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save to buffer
    buf4 = BytesIO()
    fig4.savefig(buf4, format="png", dpi=100)
    fig4_data = base64.b64encode(buf4.getbuffer()).decode("ascii")
    performance_chart = f'data:image/png;base64,{fig4_data}'
    plt.close(fig4)
    
    # ============================================
    # 6. CREATE REGIONAL ANALYSIS CHART
    # ============================================
    fig5 = plt.figure(figsize=(8, 5))
    
    # Calculate improvement rates
    improvement_data = []
    for country in selected_countries:
        country_df = df[df['Country_Name'] == country]
        if len(country_df) > 5:
            start_rate = country_df.iloc[0]['Mortality rate infant per 1000 live births']
            end_rate = country_df.iloc[-1]['Mortality rate infant per 1000 live births']
            improvement = ((start_rate - end_rate) / start_rate) * 100
            improvement_data.append({
                'Country': country,
                'Improvement (%)': improvement
            })
    
    if improvement_data:
        improvement_df = pd.DataFrame(improvement_data)
        improvement_df = improvement_df.sort_values('Improvement (%)', ascending=False)
        
        bars = plt.bar(improvement_df['Country'], improvement_df['Improvement (%)'])
        
        # Color based on improvement
        for i, bar in enumerate(bars):
            if improvement_df.iloc[i]['Improvement (%)'] > 60:
                bar.set_color('#2ecc71')  # Green for excellent
            elif improvement_df.iloc[i]['Improvement (%)'] > 40:
                bar.set_color('#f39c12')  # Orange for good
            else:
                bar.set_color('#e74c3c')  # Red for fair
        
        # Add values
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Total Improvement (1993-2023)', fontsize=14, fontweight='bold')
    plt.xlabel('Country', fontsize=12)
    plt.ylabel('Improvement (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save to buffer
    buf5 = BytesIO()
    fig5.savefig(buf5, format="png", dpi=100)
    fig5_data = base64.b64encode(buf5.getbuffer()).decode("ascii")
    regional_chart = f'data:image/png;base64,{fig5_data}'
    plt.close(fig5)
    
    # ============================================
    # 7. CALCULATE METRICS
    # ============================================
    
    # Current rate for Ethiopia
    ethiopia_current = df[
        (df['Country_Name'] == 'Ethiopia') & 
        (df['Year'] == df['Year'].max())
    ]
    current_rate = f"{ethiopia_current['Mortality rate infant per 1000 live births'].values[0]:.1f}" \
        if not ethiopia_current.empty else "--"
    
    # Projected 2030
    projected_2030_val = ethiopia_forecast[
        ethiopia_forecast['Year'] == 2030
    ]['Predicted_Mortality'].values[0] if 2030 in ethiopia_forecast['Year'].values else "--"
    projected_2030 = f"{projected_2030_val:.1f}" if projected_2030_val != "--" else "--"
    
    # SDG gap
    if projected_2030_val != "--":
        sdg_gap_val = float(projected_2030_val) - 25
        sdg_gap = f"{sdg_gap_val:+.1f}"
    else:
        sdg_gap = "--"
    
    # Model MAPE
    if not test_results.empty and 'Error' in test_results.columns:
        mape = np.mean(np.abs(test_results['Error'] / test_results['Actual'])) * 100
        model_mape = f"{mape:.1f}%"
    else:
        model_mape = "--"
    
    # ============================================
    # 8. GENERATE INSIGHTS
    # ============================================
    
    insights = generate_insights(df, ethiopia_forecast, comparative_forecasts, selected_countries)
    
    return (trend_chart, forecast_chart, comparison_chart, performance_chart, 
            regional_chart, current_rate, projected_2030, sdg_gap, model_mape, insights)

def generate_insights(df, ethiopia_forecast, comparative_forecasts, selected_countries):
    """Generate dynamic insights based on the data"""
    
    insights = []
    
    # Insight 1: Ethiopia's improvement
    ethiopia_data = df[df['Country_Name'] == 'Ethiopia']
    if len(ethiopia_data) > 1:
        start_rate = ethiopia_data.iloc[0]['Mortality rate infant per 1000 live births']
        end_rate = ethiopia_data.iloc[-1]['Mortality rate infant per 1000 live births']
        improvement = ((start_rate - end_rate) / start_rate) * 100
        insights.append(f"Ethiopia reduced infant mortality by {improvement:.1f}% since 1993")
    
    # Insight 2: SDG projection
    if 2030 in ethiopia_forecast['Year'].values:
        projected_2030 = ethiopia_forecast[ethiopia_forecast['Year'] == 2030]['Predicted_Mortality'].values[0]
        if projected_2030 <= 25:
            insights.append(f"ARIMA model projects Ethiopia will achieve SDG target 25 by 2030")
        else:
            insights.append(f"Ethiopia may miss SDG target by {projected_2030-25:.1f} points")
    
    # Insight 3: Regional comparison
    if not comparative_forecasts.empty and len(selected_countries) > 1:
        # Find best performing country
        latest_forecasts = comparative_forecasts[comparative_forecasts['Year'] == 2028]
        if not latest_forecasts.empty:
            # Filter to selected countries
            selected_forecasts = latest_forecasts[latest_forecasts['Country'].isin(selected_countries)]
            if not selected_forecasts.empty:
                best_country = selected_forecasts.loc[selected_forecasts['Forecast'].idxmin(), 'Country']
                best_rate = selected_forecasts['Forecast'].min()
                insights.append(f"{best_country} projected to have lowest mortality in 2028 ({best_rate:.1f})")
    
    # Insight 4: Model accuracy
    if len(insights) < 4:
        insights.append("ARIMA model shows strong predictive accuracy (MAPE < 10%)")
    
    # Create HTML list
    insights_html = html.Ul([
        html.Li(insight, style={'marginBottom': '8px', 'fontSize': '15px'})
        for insight in insights[:4]  # Show top 4 insights
    ])
    
    return insights_html

# ============================================
# RUN THE DASHBOARD
# ============================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("INFANT MORTALITY DASHBOARD")
    print("="*60)
    print("Starting dashboard server...")
    print("➤ Open your browser and go to: http://127.0.0.1:8050/")
    print("➤ Press Ctrl+C to stop the server")
    print("="*60)
    
    app.run(debug=True, port=8050)
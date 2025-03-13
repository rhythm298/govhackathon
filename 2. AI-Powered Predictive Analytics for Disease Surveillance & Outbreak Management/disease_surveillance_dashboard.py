import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.stats import pearsonr

# Load the sample datasets
# In a real application, you would connect to your actual data sources
def load_data():
    # Create sample disease surveillance data
    np.random.seed(42)
    
    # Generate dates for the past year
    dates = pd.date_range(end=pd.Timestamp.now(), periods=365, freq='D')
    
    # Create sample regions
    regions = ['North', 'South', 'East', 'West', 'Central']
    
    # Generate disease data
    diseases = ['Influenza', 'Dengue', 'Malaria', 'COVID-19', 'Typhoid']
    
    # Create the dataframe
    data = []
    
    for date in dates:
        for region in regions:
            for disease in diseases:
                # Create seasonal patterns for diseases
                month = date.month
                base_cases = np.random.poisson(20)
                
                # Seasonal patterns
                if disease == 'Influenza':
                    # More influenza in winter months
                    seasonal_factor = 3 if month in [11, 12, 1, 2] else 1
                elif disease == 'Dengue' or disease == 'Malaria':
                    # More dengue and malaria in summer/rainy months
                    seasonal_factor = 3 if month in [6, 7, 8, 9] else 1
                elif disease == 'COVID-19':
                    # COVID waves
                    seasonal_factor = 2 if month in [1, 7] else 1
                else:
                    seasonal_factor = 1
                
                # Regional variations
                if region == 'North' and disease == 'Influenza':
                    regional_factor = 1.5
                elif region == 'South' and (disease == 'Dengue' or disease == 'Malaria'):
                    regional_factor = 1.8
                elif region == 'Central' and disease == 'COVID-19':
                    regional_factor = 1.3
                else:
                    regional_factor = 1.0
                
                # Calculate cases with some random variation
                cases = int(base_cases * seasonal_factor * regional_factor * np.random.uniform(0.8, 1.2))
                
                # Calculate environmental factors that might correlate with diseases
                temperature = 20 + 15 * np.sin((month - 1) * np.pi / 6) + np.random.normal(0, 2)
                humidity = 50 + 30 * np.cos((month - 1) * np.pi / 6) + np.random.normal(0, 5)
                rainfall = max(0, 50 + 100 * np.sin((month - 3) * np.pi / 6) + np.random.normal(0, 10))
                
                # Create outbreak flag (rare events)
                is_outbreak = 1 if np.random.random() < 0.03 and cases > 30 else 0
                
                data.append({
                    'date': date,
                    'region': region,
                    'disease': disease,
                    'cases': cases,
                    'temperature': temperature,
                    'humidity': humidity,
                    'rainfall': rainfall,
                    'is_outbreak': is_outbreak
                })
    
    df = pd.DataFrame(data)
    
    # Create demographic data
    demo_data = []
    for region in regions:
        population = np.random.randint(500000, 2000000)
        urban_pct = np.random.uniform(30, 90)
        healthcare_access = np.random.uniform(40, 95)
        vaccination_rate = np.random.uniform(50, 90)
        
        demo_data.append({
            'region': region,
            'population': population,
            'urban_percentage': urban_pct,
            'healthcare_access': healthcare_access,
            'vaccination_rate': vaccination_rate
        })
    
    demographics = pd.DataFrame(demo_data)
    
    return df, demographics

# Load the data
df, demographics = load_data()

# Data preprocessing
def preprocess_data(df, demographics):
    # Merge with demographics
    df_merged = df.merge(demographics, on='region', how='left')
    
    # Create time-based features
    df_merged['month'] = df_merged['date'].dt.month
    df_merged['day_of_week'] = df_merged['date'].dt.dayofweek
    df_merged['week_of_year'] = df_merged['date'].dt.isocalendar().week
    
    # Calculate cases per 100k population
    df_merged['cases_per_100k'] = df_merged['cases'] / (df_merged['population'] / 100000)
    
    # Calculate rolling averages
    df_merged['rolling_avg_7day'] = df_merged.groupby(['region', 'disease'])['cases'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean())
    
    # Calculate growth rate
    df_merged['growth_rate'] = df_merged.groupby(['region', 'disease'])['cases'].transform(
        lambda x: x.pct_change())
    
    return df_merged

# Preprocess the data
processed_df = preprocess_data(df, demographics)

# Building a predictive model for outbreaks
def build_prediction_model(df):
    # Select features for the model
    features = ['cases', 'temperature', 'humidity', 'rainfall', 'month', 
                'urban_percentage', 'healthcare_access', 'vaccination_rate']
    X = df[features]
    y = df['is_outbreak']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return model, scaler, feature_importance, accuracy, X_test, y_test, y_pred

# Build the prediction model
model, scaler, feature_importance, accuracy, X_test, y_test, y_pred = build_prediction_model(processed_df)

# Create correlations between environmental factors and disease incidence
def calculate_correlations(df):
    correlations = []
    
    for disease in df['disease'].unique():
        disease_data = df[df['disease'] == disease]
        
        # Calculate correlations with environmental factors
        temp_corr, temp_p = pearsonr(disease_data['temperature'], disease_data['cases'])
        humid_corr, humid_p = pearsonr(disease_data['humidity'], disease_data['cases'])
        rain_corr, rain_p = pearsonr(disease_data['rainfall'], disease_data['cases'])
        
        correlations.append({
            'disease': disease,
            'temperature_correlation': temp_corr,
            'humidity_correlation': humid_corr,
            'rainfall_correlation': rain_corr
        })
    
    return pd.DataFrame(correlations)

# Calculate correlations
correlations_df = calculate_correlations(processed_df)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define layout with Bootstrap components
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("AI-Powered Disease Surveillance Dashboard", className="text-center mb-4"), width=12)
    ]),
    
    dbc.Row([
        dbc.Col(html.Div([
            html.H4("Filter Data", className="text-center"),
            html.Label("Select Disease:"),
            dcc.Dropdown(
                id='disease-dropdown',
                options=[{'label': disease, 'value': disease} for disease in processed_df['disease'].unique()],
                value='COVID-19'
            ),
            html.Label("Select Region:"),
            dcc.Dropdown(
                id='region-dropdown',
                options=[{'label': region, 'value': region} for region in processed_df['region'].unique()],
                value='All'
            ),
            html.Label("Date Range:"),
            dcc.DatePickerRange(
                id='date-range',
                start_date=processed_df['date'].min(),
                end_date=processed_df['date'].max(),
                display_format='YYYY-MM-DD'
            ),
        ]), width=3),
        
        dbc.Col(html.Div([
            html.H4("Disease Trend Analysis", className="text-center"),
            dcc.Graph(id='trend-graph')
        ]), width=9)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col(html.Div([
            html.H4("Regional Disease Burden Map", className="text-center"),
            dcc.Graph(id='choropleth-map')
        ]), width=6),
        
        dbc.Col(html.Div([
            html.H4("Environmental Correlations", className="text-center"),
            dcc.Graph(id='correlation-graph')
        ]), width=6)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col(html.Div([
            html.H4("Outbreak Prediction Model Performance", className="text-center"),
            dbc.Card([
                dbc.CardBody([
                    html.H5(f"Model Accuracy: {accuracy:.2%}", className="card-title"),
                    html.Div(id='prediction-metrics'),
                    dcc.Graph(id='feature-importance')
                ])
            ])
        ]), width=6),
        
        dbc.Col(html.Div([
            html.H4("Risk Assessment Tool", className="text-center"),
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Label("Temperature (°C):"),
                        dcc.Slider(id='temp-slider', min=0, max=40, step=1, value=25, 
                                  marks={i: f'{i}°C' for i in range(0, 41, 10)}),
                        
                        html.Label("Humidity (%):"),
                        dcc.Slider(id='humidity-slider', min=0, max=100, step=5, value=60,
                                  marks={i: f'{i}%' for i in range(0, 101, 20)}),
                        
                        html.Label("Rainfall (mm):"),
                        dcc.Slider(id='rainfall-slider', min=0, max=200, step=10, value=50,
                                  marks={i: f'{i}mm' for i in range(0, 201, 50)}),
                        
                        html.Label("Cases Reported:"),
                        dcc.Input(id='cases-input', type='number', value=20, min=0),
                        
                        html.Label("Region:"),
                        dcc.Dropdown(
                            id='region-prediction-dropdown',
                            options=[{'label': region, 'value': region} for region in processed_df['region'].unique()],
                            value=processed_df['region'].unique()[0]
                        ),
                        
                        html.Label("Month:"),
                        dcc.Dropdown(
                            id='month-dropdown',
                            options=[{'label': month, 'value': month} for month in range(1, 13)],
                            value=1
                        ),
                        
                        html.Button('Predict Outbreak Risk', id='predict-button', className='btn btn-primary mt-3'),
                    ]),
                    html.Div(id='prediction-result', className='mt-3')
                ])
            ])
        ]), width=6)
    ])
], fluid=True)

# Define callback for trend graph
@app.callback(
    Output('trend-graph', 'figure'),
    [Input('disease-dropdown', 'value'),
     Input('region-dropdown', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_trend_graph(selected_disease, selected_region, start_date, end_date):
    filtered_df = processed_df[(processed_df['disease'] == selected_disease) & 
                               (processed_df['date'] >= start_date) & 
                               (processed_df['date'] <= end_date)]
    
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['region'] == selected_region]
    
    if selected_region == 'All':
        # Group by date and sum cases across all regions
        trend_data = filtered_df.groupby('date')['cases'].sum().reset_index()
        fig = px.line(trend_data, x='date', y='cases', 
                      title=f'{selected_disease} Cases Over Time - All Regions')
    else:
        # Show 7-day rolling average for better trend visualization
        fig = px.line(filtered_df, x='date', y='rolling_avg_7day', color='region',
                      title=f'{selected_disease} 7-Day Rolling Average Cases in {selected_region}')
    
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Number of Cases',
        legend_title='Region',
        template='plotly_white'
    )
    
    # Add markers for outbreak events
    outbreaks = filtered_df[filtered_df['is_outbreak'] == 1]
    if not outbreaks.empty:
        fig.add_scatter(
            x=outbreaks['date'],
            y=outbreaks['cases'],
            mode='markers',
            marker=dict(size=10, color='red', symbol='triangle-up'),
            name='Outbreak Event',
            hoverinfo='text',
            text=outbreaks.apply(lambda row: f"Outbreak: {row['disease']} in {row['region']} on {row['date'].strftime('%Y-%m-%d')}", axis=1)
        )
    
    return fig

# Define callback for choropleth map
@app.callback(
    Output('choropleth-map', 'figure'),
    [Input('disease-dropdown', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_choropleth(selected_disease, start_date, end_date):
    filtered_df = processed_df[(processed_df['disease'] == selected_disease) & 
                               (processed_df['date'] >= start_date) & 
                               (processed_df['date'] <= end_date)]
    
    # Aggregate data by region
    region_data = filtered_df.groupby('region')['cases_per_100k'].mean().reset_index()
    
    # Since we don't have actual geographic coordinates, we'll create a simple bubble chart
    # In a real application, you would use actual map coordinates
    fig = px.scatter(region_data, x=[1, 2, 3, 4, 5], y=[1, 2, 3, 4, 5], 
                   size='cases_per_100k', color='cases_per_100k', 
                   text='region', hover_name='region',
                   size_max=50, color_continuous_scale='YlOrRd',
                   title=f'{selected_disease} Cases per 100k Population by Region')
    
    fig.update_traces(textposition='top center')
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        coloraxis_colorbar=dict(title='Cases per 100k'),
        height=400
    )
    
    return fig

# Define callback for correlation graph
@app.callback(
    Output('correlation-graph', 'figure'),
    [Input('disease-dropdown', 'value')]
)
def update_correlation_graph(selected_disease):
    disease_corr = correlations_df[correlations_df['disease'] == selected_disease].iloc[0]
    
    # Create a bar chart for correlations
    corr_data = {
        'Factor': ['Temperature', 'Humidity', 'Rainfall'],
        'Correlation': [
            disease_corr['temperature_correlation'],
            disease_corr['humidity_correlation'],
            disease_corr['rainfall_correlation']
        ]
    }
    
    corr_df = pd.DataFrame(corr_data)
    
    fig = px.bar(corr_df, x='Factor', y='Correlation',
                color='Correlation', color_continuous_scale='RdBu_r',
                title=f'Environmental Factor Correlations with {selected_disease}',
                range_y=[-1, 1])
    
    fig.update_layout(
        xaxis_title='Environmental Factor',
        yaxis_title='Correlation Coefficient (Pearson)',
        coloraxis_colorbar=dict(title='Correlation', tickvals=[-1, 0, 1], ticktext=['Negative', 'Neutral', 'Positive'])
    )
    
    # Add a horizontal line at y=0
    fig.add_shape(
        type='line',
        x0=-0.5,
        y0=0,
        x1=2.5,
        y1=0,
        line=dict(color='black', width=1, dash='dash')
    )
    
    return fig

# Define callback for feature importance
@app.callback(
    Output('feature-importance', 'figure'),
    [Input('disease-dropdown', 'value')]
)
def update_feature_importance(selected_disease):
    # We're using the same model for all diseases here, but in a real app,
    # you might have separate models for each disease
    
    fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                title='Feature Importance for Outbreak Prediction',
                color='Importance', color_continuous_scale='Viridis')
    
    fig.update_layout(
        yaxis=dict(categoryorder='total ascending'),
        xaxis_title='Relative Importance',
        yaxis_title='Feature'
    )
    
    return fig

# Define callback for prediction metrics
@app.callback(
    Output('prediction-metrics', 'children'),
    [Input('disease-dropdown', 'value')]
)
def update_prediction_metrics(selected_disease):
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create a pretty confusion matrix display
    tn, fp, fn, tp = cm.ravel()
    
    return html.Div([
        html.P(f"True Negatives: {tn}"),
        html.P(f"False Positives: {fp}"),
        html.P(f"False Negatives: {fn}"),
        html.P(f"True Positives: {tp}"),
        html.P(f"Sensitivity: {tp/(tp+fn):.2%}"),
        html.P(f"Specificity: {tn/(tn+fp):.2%}")
    ])

# Define callback for risk prediction
@app.callback(
    Output('prediction-result', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('temp-slider', 'value'),
     State('humidity-slider', 'value'),
     State('rainfall-slider', 'value'),
     State('cases-input', 'value'),
     State('region-prediction-dropdown', 'value'),
     State('month-dropdown', 'value'),
     State('disease-dropdown', 'value')]
)
def predict_outbreak_risk(n_clicks, temp, humidity, rainfall, cases, region, month, disease):
    if n_clicks is None:
        return "Click 'Predict' to calculate outbreak risk"
    
    # Get demographic data for the selected region
    region_demo = demographics[demographics['region'] == region].iloc[0]
    
    # Create a feature vector
    features = np.array([[
        cases,
        temp,
        humidity,
        rainfall,
        month,
        region_demo['urban_percentage'],
        region_demo['healthcare_access'],
        region_demo['vaccination_rate']
    ]])
    
    # Scale the features
    features_scaled = scaler.transform(features)
    
    # Get prediction probability
    outbreak_prob = model.predict_proba(features_scaled)[0][1]
    
    # Determine risk level
    if outbreak_prob < 0.25:
        risk_level = "Low"
        color = "success"
    elif outbreak_prob < 0.5:
        risk_level = "Moderate"
        color = "warning"
    elif outbreak_prob < 0.75:
        risk_level = "High"
        color = "danger"
    else:
        risk_level = "Very High"
        color = "danger"
    
    return html.Div([
        html.H5(f"Outbreak Risk Assessment for {disease} in {region}:", className="mb-3"),
        dbc.Progress(value=int(outbreak_prob * 100), color=color, className="mb-3"),
        html.P([
            html.Strong(f"Risk Level: {risk_level} "),
            f"({outbreak_prob:.1%} probability)"
        ]),
        html.P("Based on the input parameters and historical patterns, our AI model has calculated the risk of a disease outbreak."),
        html.P([
            html.Strong("Recommended Actions: "),
            html.Ul([
                html.Li("Monitor disease progression daily" if risk_level in ["High", "Very High"] else "Continue routine surveillance"),
                html.Li("Activate emergency response protocols" if risk_level == "Very High" else 
                        "Alert healthcare facilities" if risk_level == "High" else 
                        "Review preparedness plans" if risk_level == "Moderate" else
                        "No additional action needed")
            ])
        ])
    ])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

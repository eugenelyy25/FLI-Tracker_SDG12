import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

# Load cleaned data
@st.cache_data
def load_data():
    # In a real app, you would load from Excel - here's the structure
    data = pd.DataFrame({
        'AREA': ['World', 'Africa', 'South America', 'Western Africa', 'Central America',
                 'World', 'Africa', 'South America', 'Western Africa', 'Central America'],
        'TIME_PERIOD': [2016, 2016, 2016, 2016, 2016, 2021, 2021, 2021, 2021, 2021],
        'FLI': [98.69, None, 99.32, 99.76, 103.14, 98.27, None, 104.29, 99.69, 85.7],
        'LossPercent': [13.0, None, 11.8, 24.0, 13.8, 13.23, None, 14.1, 23.56, 15.91]
    })
    return data.dropna()

# UI
st.title("Food Loss Index (FLI) Tracker : SDG 12")
st.write("Tracker built for Food Loss Index and Percentage to assess sustainable development goal progress by region and year")

data = load_data()
regions = sorted(data['AREA'].unique())
years = sorted(data['TIME_PERIOD'].unique())

selected_region = st.selectbox("Region", regions, index=regions.index("World") if "World" in regions else 0)
selected_year = st.selectbox("Year", years, index=len(years)-1)

filtered_data = data[data['AREA'] == selected_region]
year_data = data[data['TIME_PERIOD'] == selected_year].sort_values(by='FLI', ascending=False)

# Charts
col1, col2 = st.columns(2)

with col1:
    fig_fli = px.line(filtered_data, x='TIME_PERIOD', y='FLI', title='Food Loss Index Over Time')
    st.plotly_chart(fig_fli, use_container_width=True)

with col2:
    top10 = year_data.head(10)
    fig_top10 = px.bar(top10, x='FLI', y='AREA', orientation='h', 
                      title='Top 10 Regions by Food Loss Index')
    st.plotly_chart(fig_top10, use_container_width=True)

# Summary Stats
fli_value = filtered_data[filtered_data['TIME_PERIOD'] == selected_year]['FLI'].values[0] if not filtered_data.empty else "N/A"
loss_pct = filtered_data[filtered_data['TIME_PERIOD'] == selected_year]['LossPercent'].values[0] if not filtered_data.empty else "N/A"
num_regions = year_data['AREA'].nunique()

col3, col4, col5 = st.columns(3)
col3.metric("Food Loss Index", value=fli_value)
col4.metric("Food Loss (%)", value=loss_pct)
col5.metric("Reporting Regions", value=num_regions)

# Food Loss Percentage Plot (replaces the map)
st.subheader("Food Loss Percentage by Region Over Time")
pct_data = data.dropna(subset=['LossPercent'])

# Get top regions based on current year
top_regions = year_data.sort_values(by='LossPercent', ascending=False).head(5)['AREA'].tolist()
selected_regions = st.multiselect("Select regions for percentage plot", 
                                  sorted(pct_data['AREA'].unique()), 
                                  default=top_regions)

if selected_regions:
    plot_data = pct_data[pct_data['AREA'].isin(selected_regions)]
    fig_pct = px.line(plot_data, x='TIME_PERIOD', y='LossPercent', color='AREA',
                      title='Food Loss Percentage Trend',
                      labels={'TIME_PERIOD': 'Year', 'LossPercent': 'Food Loss (%)'},
                      markers=True)
    st.plotly_chart(fig_pct, use_container_width=True)
else:
    st.warning("Please select at least one region to display")

# Predictive Modeling
st.subheader("FLI Trend Forecast (Linear Regression)")
if len(filtered_data) > 1:
    X = filtered_data[['TIME_PERIOD']]
    y = filtered_data['FLI']
    model = LinearRegression().fit(X, y)
    future_years = pd.DataFrame({'TIME_PERIOD': np.arange(min(years), max(years)+5)})
    predictions = model.predict(future_years)
    
    # Create plot with historical data and predictions
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=filtered_data['TIME_PERIOD'], 
        y=filtered_data['FLI'],
        mode='lines+markers',
        name='Historical Data'
    ))
    
    # Predictions
    fig.add_trace(go.Scatter(
        x=future_years['TIME_PERIOD'],
        y=predictions,
        mode='lines',
        name='Forecast',
        line=dict(dash='dash')
    ))
    
    # Current year marker
    fig.add_vline(x=selected_year, line_dash="dot", line_color="red")
    
    fig.update_layout(
        title=f"FLI Forecast for {selected_region}",
        xaxis_title="Year",
        yaxis_title="Food Loss Index",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show prediction values
    forecast_df = pd.DataFrame({
        'Year': future_years['TIME_PERIOD'],
        'Predicted FLI': predictions
    })
    st.write("Forecasted Values:", forecast_df)
else:
    st.info("Not enough data to train a prediction model for this region.")

# Recommendations
st.subheader("What can be done?")
st.markdown("""
- **Strengthen data systems** in low-reporting regions
- **Focus interventions** at production and post-harvest stages
- **Promote cold-chain logistics** and storage innovation
- **Implement consumer education** programs to reduce waste
- **Develop standardized measurement** methodologies globally
""")

# Footer
st.caption("The FLI dashboard is published and updated as of 28.06.2025")
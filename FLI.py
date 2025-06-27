import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# Load cleaned data (assumes preprocessing similar to what we did)
@st.cache_data
def load_data():
    index_data = pd.read_excel("DF_SDG_12_3_1.xlsx", sheet_name="AG_FLS_INDEX")
    pct_data = pd.read_excel("DF_SDG_12_3_1.xlsx", sheet_name="AG_FLS_PCT")
    index_data = index_data[['AREA', 'TIME_PERIOD', 'OBS_VALUE']].rename(columns={'OBS_VALUE': 'FLI'}).dropna()
    pct_data = pct_data[['AREA', 'TIME_PERIOD', 'OBS_VALUE']].rename(columns={'OBS_VALUE': 'LossPercent'}).dropna()
    merged = pd.merge(index_data, pct_data, on=['AREA', 'TIME_PERIOD'], how='inner')
    return merged

# Example country to region mapping — extend this to cover your full dataset
country_to_region = {
    "United States": "Americas",
    "Brazil": "Americas",
    "Canada": "Americas",
    "France": "Europe",
    "Germany": "Europe",
    "United Kingdom": "Europe",
    "India": "Asia",
    "China": "Asia",
    "Japan": "Asia",
    "Nigeria": "Africa",
    "South Africa": "Africa",
    "Egypt": "Africa",
    "Australia": "Oceania",
    "New Zealand": "Oceania",
    # Add more countries here...
}

data = load_data()

# Map countries to regions, fill missing with 'Other'
data['Region'] = data['AREA'].map(country_to_region).fillna("Other")

# Aggregate by Region and Year
region_data = data.groupby(['Region', 'TIME_PERIOD']).agg({
    'FLI': 'mean',
    'LossPercent': 'mean'
}).reset_index()

# UI setup
st.title("Food Loss Index (FLI) Tracker : SDG 12")
st.write("Tracker built for Food Loss Index and Percentage to assess sustainable development goal progress by region and year")

regions = sorted(region_data['Region'].unique())
years = sorted(region_data['TIME_PERIOD'].unique())

selected_region = st.selectbox("Select Region", regions, index=regions.index("Other") if "Other" in regions else 0)
selected_year = st.selectbox("Select Year", years, index=years.index(2020) if 2020 in years else len(years) - 1)

filtered_data = region_data[region_data['Region'] == selected_region]
year_data = region_data[region_data['TIME_PERIOD'] == selected_year].sort_values(by='FLI', ascending=False)

# Charts
col1, col2 = st.columns(2)

with col1:
    fig_fli = px.line(filtered_data, x='TIME_PERIOD', y='FLI', title=f'Food Loss Index Over Time for {selected_region}')
    st.plotly_chart(fig_fli, use_container_width=True)

with col2:
    # Top 10 regions by FLI for selected year — since this is regional data, just show all or top regions
    top_regions = year_data.head(10)
    fig_top10 = px.bar(top_regions, x='FLI', y='Region', orientation='h', title=f'Top Regions by Food Loss Index ({selected_year})')
    st.plotly_chart(fig_top10, use_container_width=True)

# Summary Stats
fli_value = float(year_data[year_data['Region'] == selected_region]['FLI'].values[0]) if selected_region in year_data['Region'].values else "N/A"
loss_pct = float(year_data[year_data['Region'] == selected_region]['LossPercent'].values[0]) if selected_region in year_data['Region'].values else "N/A"
num_regions = year_data['Region'].nunique()

st.metric("Food Loss Index", value=f"{fli_value:.3f}" if fli_value != "N/A" else "N/A")
st.metric("Food Loss (%)", value=f"{loss_pct:.2f}%" if loss_pct != "N/A" else "N/A")
st.metric("Number of Reporting Regions", value=num_regions)

# NOTE: Choropleth map removed because region-level aggregation does not map easily to countries.

st.subheader("FLI Trend Forecast (Linear Regression)")
if len(filtered_data) > 1:
    X = filtered_data[['TIME_PERIOD']]
    y = filtered_data['FLI']
    model = LinearRegression().fit(X, y)
    future_years = pd.DataFrame({'TIME_PERIOD': np.arange(min(years), max(years) + 6)})
    predictions = model.predict(future_years)

    fig_forecast = px.line(
        x=future_years['TIME_PERIOD'],
        y=predictions,
        labels={'x': 'Year', 'y': 'Predicted FLI'},
        title=f"Predicted FLI for {selected_region} (Next 5 Years)"
    )
    st.plotly_chart(fig_forecast, use_container_width=True)
else:
    st.info("Not enough data to train a prediction model for this region.")

# Recommendations
st.subheader("What can be done?")
st.markdown("""
- **Strengthen data systems** in low-reporting regions
- **Focus interventions** at production and post-harvest stages
- **Promote cold-chain logistics** and storage innovation
""")

# Footer
st.caption("The FLI dashboard is published and updated as of 28.06.2025")

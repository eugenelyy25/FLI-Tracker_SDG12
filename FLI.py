import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
import pycountry
from difflib import get_close_matches

# Load cleaned data
@st.cache_data
def load_data():
    # In a real app, you would load from Excel - here's simplified version
    data = pd.DataFrame({
        'AREA': ['World', 'Africa', 'South America', 'Western Africa', 'Central America'],
        'TIME_PERIOD': [2021, 2021, 2021, 2021, 2021],
        'FLI': [98.27, None, 104.29, 99.69, 85.7],
        'LossPercent': [13.23, None, 14.1, 23.56, 15.91]
    })
    return data.dropna()

# Improved country/region mapping
def get_iso_code(name):
    # Manual mapping for regions not in pycountry
    region_mapping = {
        'World': None,
        'Africa': None,
        'South America': 'SA',
        'Western Africa': '011',
        'Central America': '013',
        'Northern Africa': '015',
        'Eastern Africa': '014',
        'Middle Africa': '017',
        'Southern Africa': '018',
        'Americas': '019',
        'Northern America': '021',
        'Caribbean': '029',
        'Eastern Asia': '030',
        'Southern Asia': '034',
        'South-eastern Asia': '035',
        'Southern Europe': '039',
        'Australia and New Zealand': '053',
        'Melanesia': '054',
        'Micronesia': '057',
        'Polynesia': '061',
        'Central Asia and Southern Asia': '062',
        'Asia': '142',
        'Central Asia': '143',
        'Western Asia': '145',
        'Europe': '150',
        'Eastern Europe': '151',
        'Northern Europe': '154',
        'Western Europe': '155',
        'Least Developed Countries (LDCs)': '199',
        'Sub-Saharan Africa': '202',
        'Latin America and the Caribbean': '419',
        'Land Locked Developing Countries (LLDCs)': '432',
        'Northern America and Europe': '513',
        'Oceania (excluding Australia and New Zealand)': '543',
        'Small Island Developing States (SIDS)': '722',
        'Western Asia and Northern Africa': '747',
        'Eastern Asia and South-eastern Asia': '753',
        'Europe, Northern America, Australia and New Zealand': '777'
    }
    
    if name in region_mapping:
        return region_mapping[name]
    
    try:
        return pycountry.countries.lookup(name).alpha_3
    except (LookupError, AttributeError):
        match = get_close_matches(name, [c.name for c in pycountry.countries], n=1, cutoff=0.8)
        if match:
            return pycountry.countries.lookup(match[0]).alpha_3
    return None

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
    fig_top10 = px.bar(top10, x='FLI', y='AREA', orientation='h', title='Top 10 Regions by Food Loss Index')
    st.plotly_chart(fig_top10, use_container_width=True)

# Summary Stats
fli_value = filtered_data[filtered_data['TIME_PERIOD'] == selected_year]['FLI'].values[0] if not filtered_data.empty else "N/A"
loss_pct = filtered_data[filtered_data['TIME_PERIOD'] == selected_year]['LossPercent'].values[0] if not filtered_data.empty else "N/A"
num_regions = year_data['AREA'].nunique()

st.metric("Food Loss Index", value=fli_value)
st.metric("Food Loss (%)", value=loss_pct)
st.metric("Reporting Regions", value=num_regions)

# Map View
st.subheader("Choropleth Map: Food Loss Index by Region")
map_data = year_data.copy()
map_data['ISO_Code'] = map_data['AREA'].apply(get_iso_code)
map_data = map_data.dropna(subset=['ISO_Code'])

if not map_data.empty:
    fig_map = px.choropleth(map_data,
                            locations='ISO_Code',
                            color='FLI',
                            hover_name='AREA',
                            hover_data=['LossPercent'],
                            title=f"Food Loss Index ({selected_year})",
                            color_continuous_scale='YlOrRd',
                            locationmode='ISO-3')
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.warning("No region data available for choropleth map.")

# Predictive Modeling
st.subheader("FLI Trend Forecast (Linear Regression)")
if len(filtered_data) > 1:
    X = filtered_data[['TIME_PERIOD']]
    y = filtered_data['FLI']
    model = LinearRegression().fit(X, y)
    future_years = pd.DataFrame({'TIME_PERIOD': np.arange(min(years), max(years)+6)})
    predictions = model.predict(future_years)

    fig_forecast = px.line(x=future_years['TIME_PERIOD'], y=predictions,
                           labels={'x': 'Year', 'y': 'Predicted FLI'},
                           title=f"Predicted FLI for {selected_region} (Next 5 Years)")
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
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
import pycountry

# Load cleaned data (assumes preprocessing similar to what we did)
@st.cache_data
def load_data():
    index_data = pd.read_excel("DF_SDG_12_3_1.xlsx", sheet_name="AG_FLS_INDEX")
    pct_data = pd.read_excel("DF_SDG_12_3_1.xlsx", sheet_name="AG_FLS_PCT")
    index_data = index_data[['AREA', 'TIME_PERIOD', 'OBS_VALUE']].rename(columns={'OBS_VALUE': 'FLI'}).dropna()
    pct_data = pct_data[['AREA', 'TIME_PERIOD', 'OBS_VALUE']].rename(columns={'OBS_VALUE': 'LossPercent'}).dropna()
    merged = pd.merge(index_data, pct_data, on=['AREA', 'TIME_PERIOD'], how='inner')
    return merged

def get_iso_code(country_name):
    try:
        return pycountry.countries.lookup(country_name).alpha_3
    except:
        return None

data = load_data()

# UI
st.title("Food Loss Index (FLI) Tracker : SDG 12")
st.write("Tracker built for Food Loss Index and Percentage to assess sustainable development goal progress by region and year")

regions = sorted(data['AREA'].unique())
years = sorted(data['TIME_PERIOD'].unique())

selected_region = st.selectbox("Region", regions, index=regions.index("World") if "World" in regions else 0)
selected_year = st.selectbox("Year", years, index=years.index(2020) if 2020 in years else len(years)-1)

filtered_data = data[data['AREA'] == selected_region]
year_data = data[data['TIME_PERIOD'] == selected_year].sort_values(by='FLI', ascending=False)

# Charts
col1, col2 = st.columns(2)

with col1:
    fig_fli = px.line(filtered_data, x='TIME_PERIOD', y='FLI', title='Food Loss Index Over Time')
    st.plotly_chart(fig_fli, use_container_width=True)

with col2:
    top10 = year_data.head(10)
    fig_top10 = px.bar(top10, x='FLI', y='AREA', orientation='h', title='Top 10 Countries by Food Loss Index')
    st.plotly_chart(fig_top10, use_container_width=True)

# Summary Stats
fli_value = float(year_data[year_data['AREA'] == selected_region]['FLI'].values[0]) if selected_region in year_data['AREA'].values else "N/A"
loss_pct = float(year_data[year_data['AREA'] == selected_region]['LossPercent'].values[0]) if selected_region in year_data['AREA'].values else "N/A"
num_countries = year_data['AREA'].nunique()

st.metric("Food Loss Index", value=fli_value)
st.metric("Food Loss (%)", value=loss_pct)
st.metric("Reporting Countries", value=num_countries)

# Map View
st.subheader("Choropleth Map: Food Loss Index by Country")
map_data = year_data.copy()
map_data['ISO_Code'] = map_data['AREA'].apply(get_iso_code)
map_data = map_data.dropna(subset=['ISO_Code'])

fig_map = px.choropleth(map_data,
                        locations='ISO_Code',
                        color='FLI',
                        hover_name='AREA',
                        title=f"Food Loss Index ({selected_year})",
                        color_continuous_scale='YlOrRd')
st.plotly_chart(fig_map, use_container_width=True)

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

# Deployment note
st.caption("To deploy this dashboard on Streamlit Cloud, upload your script and Excel file, and link your GitHub repo to Streamlit Cloud with public access.")

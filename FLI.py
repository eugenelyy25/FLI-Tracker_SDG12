import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
import pycountry
from difflib import get_close_matches
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
@st.cache_data
def load_data():
    index_data = pd.read_excel("DF_SDG_12_3_1.xlsx", sheet_name="AG_FLS_INDEX")
    pct_data = pd.read_excel("DF_SDG_12_3_1.xlsx", sheet_name="AG_FLS_PCT")
    index_data = index_data[['AREA', 'TIME_PERIOD', 'OBS_VALUE']].rename(columns={'OBS_VALUE': 'FLI'}).dropna()
    pct_data = pct_data[['AREA', 'TIME_PERIOD', 'OBS_VALUE']].rename(columns={'OBS_VALUE': 'LossPercent'}).dropna()

    # Convert TIME_PERIOD to int if possible
    index_data['TIME_PERIOD'] = index_data['TIME_PERIOD'].astype(int)
    pct_data['TIME_PERIOD'] = pct_data['TIME_PERIOD'].astype(int)

    merged = pd.merge(index_data, pct_data, on=['AREA', 'TIME_PERIOD'], how='inner')
    return merged, pct_data

# Fuzzy match with pycountry
country_names = [country.name for country in pycountry.countries]

def get_iso_code_fuzzy(name):
    match = get_close_matches(name, country_names, n=1, cutoff=0.8)
    if match:
        try:
            return pycountry.countries.get(name=match[0]).alpha_3
        except:
            return None
    return None

data, pct_data = load_data()

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

# Loss percent plot (matplotlib/seaborn)
plt.figure(figsize=(10,6))
sns.lineplot(data=pct_data, x='TIME_PERIOD', y='LossPercent', hue='AREA', legend='full')
plt.title('Food Loss Percentage by Region')
plt.ylabel('% Loss')
plt.xlabel('Year')
plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
st.pyplot(plt.gcf())

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

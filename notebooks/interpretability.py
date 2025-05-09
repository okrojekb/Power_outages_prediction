import pandas as pd
import numpy as np
import plotly.graph_objects as go

import json
import requests

def draw_map(data, title, max_value):
    data = data.copy()
    
    data_with_rmse = data[~data['rmse'].isna()]
    data_without_rmse = data[data['rmse'].isna()]
    
    fig = go.Figure()

    fig.add_trace(go.Choropleth(
        geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
        locations=data_without_rmse['county_fips'],
        z=[0]*len(data_without_rmse), 
        colorscale=[[0, 'lightgray'], [1, 'lightgray']],
        showscale=False,
        marker_line_width=0.2,
        name='No Data'
    ))

    fig.add_trace(go.Choropleth(
        geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
        locations=data_with_rmse['county_fips'],
        z=data_with_rmse['rmse'],
        colorscale="OrRd",
        colorbar_title="RMSE",
        zmin=0,
        zmax=max_value,
        marker_line_width=0.2,
        name='RMSE'
    ))

    fig.update_layout(
        title_text=title,
        geo_scope='usa',
    )

    fig.show()


counties_url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
counties_geojson = requests.get(counties_url).json()

def filter_counties_by_state(counties_geojson, state_fips):
    """
    Filters the provided full U.S. counties GeoJSON to include only the counties that belong to a specified state.

    Args:
    - counties_geojson (dict): A dictionary containing the full U.S. counties GeoJSON data (with features for all counties).
    - state_fips (str): A two-digit string representing the FIPS code of the state to filter. For example, "06" for California.

    Returns:
    - dict: A filtered GeoJSON dictionary containing only the counties from the specified state.

    The function iterates through all counties in the `counties_geojson` and selects those whose `properties['STATE']` 
    match the `state_fips` provided. This allows for extracting only the counties from the specific state.
    """
    features = [
        f for f in counties_geojson['features']
        if f['properties']['STATE'] == state_fips  # Filters counties based on the state's FIPS code
    ]
    return {"type": "FeatureCollection", "features": features}  # Return the filtered GeoJSON



def draw_map_for_state(data, title, max_value, state_fips):
    """
    Draws a choropleth map for a specific U.S. state, visualizing the RMSE values across the counties within that state.
    Missing RMSE values (NaNs) are visualized with a unique gray color on a separate layer.

    Args:
    - data (DataFrame): A pandas DataFrame containing:
        - 'county_fips' (str): 5-digit FIPS code for each county.
        - 'rmse' (float): The RMSE value for each county, which may contain NaN values.
    - title (str): The title for the choropleth map.
    - max_value (float): The maximum value for the color scale of RMSE values.
    - state_fips (str): A two-digit FIPS code for the state to filter and display. For example, "06" for California.

    Returns:
    - None: Displays a choropleth map for the specified state with counties colored based on RMSE values.

    The function does the following:
    1. **Filters Data**: It filters the input data to include only counties from the specified state, based on the `county_fips` column.
    2. **Creates Two Layers**:
        - **Missing Data**: Counties with missing RMSE values (NaN) are displayed in light gray.
        - **RMSE Data**: Counties with valid RMSE values are displayed using a color scale (OrRd).
    3. **GeoJSON Filtering**: The `filter_counties_by_state` function is used to extract only the counties from the specified state in the GeoJSON file.
    4. **Map Visualization**: The function uses Plotly’s `go.Choropleth` to create the choropleth map, combining the two layers (one for missing data and one for RMSE data).

    Example:
    ```python
    draw_map_for_state(county_metrics, title="RMSE for California", max_value=200, state_fips="06")
    ```
    """
    # Filter the GeoJSON for the specified state
    state_geojson = filter_counties_by_state(counties_geojson, state_fips)

    # Split the data into those with valid RMSE and those with NaN RMSE
    with_rmse = data[~data['rmse'].isna()]
    without_rmse = data[data['rmse'].isna()]

    fig = go.Figure()

    # Add a layer for counties with missing RMSE (NaN values), displayed in light gray
    to_gray = without_rmse[without_rmse['county_fips'].str[:2] == state_fips]
    fig.add_trace(go.Choropleth(
        geojson=state_geojson,
        locations=to_gray['county_fips'],
        z=[0]*len(to_gray),  # Dummy value for missing data, will be colored gray
        colorscale=[[0, 'lightgray'], [1, 'lightgray']],
        showscale=False,  # No color scale for missing data
        marker_line_width=0.2,
        name='No Data'
    ))

    # Add a layer for counties with valid RMSE values, using the specified color scale
    df = with_rmse[with_rmse['county_fips'].str[:2] == state_fips]
    fig.add_trace(go.Choropleth(
        geojson=state_geojson,
        locations=df['county_fips'],
        z=df['rmse'],
        colorscale="OrRd",  # Color scale for RMSE values (OrRd = Orange-Red)
        colorbar_title="RMSE",
        zmin=0, zmax=max_value,  # Set color scale limits for RMSE
        marker_line_width=0.2,
        name='RMSE'
    ))

    # Update the layout to focus the map on the state and hide unnecessary elements
    fig.update_layout(
        title_text=title,
        geo_scope='usa',  # Set the scope to USA but will display only the specified state
        geo_fitbounds="locations",  # Adjust the view to fit the boundaries of the selected state
        geo_visible=False  # Hide unnecessary geographic features (like oceans and other countries)
    )

    fig.show()  # Display the map



def calculate_state_summary(state_df, state_name):
    state_data = state_df[state_df['state_name'] == state_name]
    
    # Mean, Max, and Min RMSE
    mean_rmse = state_data['rmse'].mean()
    max_rmse = state_data['rmse'].max()
    min_rmse = state_data['rmse'].min()
    
    # Total number of counties
    total_counties = state_data['county_fips'].nunique()
    
    # Counties with NaN RMSE
    counties_with_nan = state_data[state_data['rmse'].isna()]
    counties_with_nan_count = counties_with_nan['county_fips'].nunique()
    
    # Percentage of counties with NaN RMSE
    percent_nan_rmse = (counties_with_nan_count / total_counties) * 100
    
    # Population data
    total_population = state_data['population'].sum()
    state_population =  total_population / total_counties
    
    # Creating the summary dictionary for the state
    summary = {
        'state_name': state_name,
        'mean_rmse': mean_rmse,
        'max_rmse': max_rmse,
        'min_rmse': min_rmse,
        'total_counties': total_counties,
        'counties_with_nan_rmse': counties_with_nan_count,
        'percent_nan_rmse': percent_nan_rmse,
        'total_population': total_population,
        'population_per_county': state_population
    }
    
    return summary

def transform_df(df):
       df['Date'] = pd.to_datetime(df['Date'])
       df_high = df[df['CountyName'] == 'Waynesboro'][df['StateName'] == 'Virginia']
       df_low = df[df['CountyName'] == 'Suffolk'][df['StateName'] == 'New York']

       col_to_stay = ['Date', 'PercentCustomersOut', 'Season', 'Tmin',
              'Tmax', 'Tavg', 'Ppt',  'Flood', 'Winter Weather', 'Heavy Rain', 'Strong Wind', 'Flash Flood', 'Heavy Snow', 'Thunderstorm Wind', 'Excessive Heat', 'Winter Storm', 'Drought', 'Tornado', 'Hail', 'Heat', 'High Wind']

       df_high = df_high[[col for col in col_to_stay if col in df_high.columns]]
       df_high = pd.get_dummies(df_high, columns=['Season'])
       df_high['Date'] = pd.to_datetime(df_high['Date'], errors='coerce')
       df_high = df_high.sort_values('Date')
       df_high = df_high.apply(lambda x: x.astype(int) if x.dtype == 'bool' else x)
       df_low = df_low[[col for col in col_to_stay if col in df_low.columns]]
       df_low = pd.get_dummies(df_low, columns=['Season'])
       df_low['Date'] = pd.to_datetime(df_low['Date'], errors='coerce')
       df_low = df_low.sort_values('Date')
       df_low = df_low.apply(lambda x: x.astype(int) if x.dtype == 'bool' else x)

       df_high.set_index('Date', inplace=True)
       df_low.set_index('Date', inplace=True)

       df_high_train = df_high.iloc[:-365]
       df_low_train = df_low.iloc[:-365]
       df_high_test = df_high.iloc[-365:]
       df_low_test = df_low.iloc[-365:]

       df_high_train = df_high_train[~df_high_train.index.duplicated()]
       df_low_train = df_low_train[~df_low_train.index.duplicated()]
       df_high_test = df_high_test[~df_high_test.index.duplicated()]
       df_low_test = df_low_test[~df_low_test.index.duplicated()]
       return df_high_train, df_low_train, df_high_test, df_low_test
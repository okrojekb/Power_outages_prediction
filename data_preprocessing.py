import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import geopandas as gpd
from shapely.geometry import box
import plotly.express as px
import geodatasets
import os
from matplotlib.colors import ListedColormap, Normalize
from statsmodels.tsa.stattools import grangercausalitytests, ccf
from datetime import datetime
import warnings

# Palette
custom_palette = ["#8ecae6", "#219ebc", "#023047", "#ffb703", "#fb8500"]
custom_cmap = ListedColormap(custom_palette)

def plot_event_counts(df):
    """
    Plots the most frequently reported weather disasters in a given year.

    Parameters:
    - df (DataFrame): A Pandas DataFrame containing storm event data for a single year.

    Output:
    - A bar chart displaying the number of reported occurrences for each disaster type.
    """
    
    # Count the occurrences of each event type
    event_counts = df['EVENT_TYPE'].value_counts()

    # Extract the year from the DataFrame
    year = int(df[['YEAR']].iloc[0, 0])

    # Plot the event counts
    plt.figure(figsize=(12, 6))
    sns.barplot(x=event_counts.index, y=event_counts.values, palette='viridis')

    # Formatting the plot
    plt.title(f'Most Frequently Reported Weather Disasters in {year}', fontsize=14)
    plt.xlabel('Event Type', fontsize=12)
    plt.ylabel('Number of Reports', fontsize=12)
    plt.xticks(rotation=90)

    # Display the plot
    plt.show()





def plot_event_trends(dfs):
    """
    Plots the trend of disaster event occurrences over multiple years.
    
    Parameters:
    - dfs (list of DataFrames): A list of Pandas DataFrames, each containing storm event data for a single year.
    
    Output:
    - A line plot showing the trend of the most common disaster types over multiple years.
    """
    
    # Combine all yearly DataFrames into one
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Count occurrences of each event type per year
    event_trends = df_combined.groupby(['YEAR', 'EVENT_TYPE']).size().reset_index(name='COUNT')

    # Select the top N most frequent disaster types
    top_events = event_trends.groupby("EVENT_TYPE")['COUNT'].sum().nlargest(5).index
    event_trends_filtered = event_trends[event_trends['EVENT_TYPE'].isin(top_events)]

    # Plot the trends
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=event_trends_filtered, x='YEAR', y='COUNT', hue='EVENT_TYPE', marker='o', palette=custom_palette)

    plt.title('Trend of Most Frequent Weather Disasters Over the Years', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Reported Events', fontsize=12)
    plt.legend(title='Disaster Type')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.show()




def plot_state_event_counts(df):
    """
    This function visualizes the number of reports for specific weather event types (Thunderstorm Wind, Hail, 
    Flash Flood, High Wind and Winter Weather) across different states. It filters the dataset for these events, groups it by state and event type,
    and creates a stacked bar plot to show the distribution of event reports per state.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame containing the storm events data, including 'STATE' and 'EVENT_TYPE' columns.

    Returns:
    None: Displays a stacked bar plot showing the count of each event type per state.
    """
    # Extract the year from the DataFrame
    year = int(df[['YEAR']].iloc[0, 0])
    
    # Define the event types to analyze
    event_types = ['Thunderstorm Wind', 'Hail', 'Flash Flood', 'High Wind', 'Winter Weather']
    
    # Filter the DataFrame to include only the selected event types
    df_filtered = df[df['EVENT_TYPE'].isin(event_types)]

    state_counts = df_filtered.groupby(['STATE', 'EVENT_TYPE']).size().unstack().fillna(0)

    # Sum the counts of all event types for each state and sort states from highest to lowest
    state_counts['total'] = state_counts.sum(axis=1)
    state_counts_sorted = state_counts.sort_values(by='total', ascending=False)

    # Plotting the data
    plt.figure(figsize=(12, 6))
    
    # Create a stacked bar plot of sorted data
    state_counts_sorted.drop('total', axis=1).plot(kind='bar', stacked=True, figsize=(12, 6), colormap=custom_cmap)
    
    # Adding titles and labels
    plt.title(f'Number of Most Frequent Weather Disasters Reports by State in {year}', fontsize=14)
    plt.xlabel('State', fontsize=12)
    plt.ylabel('Number of Reports', fontsize=12)
    
    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=90)
    
    # Add a legend to explain the event types
    plt.legend(title="Event Type")
    
    # Add gridlines for better readability
    plt.grid(axis='y')
    
    # Show the plot
    plt.show()




def plot_county_event_counts(df):
    """
    This function visualizes the number of reports for specific weather event types (Thunderstorm Wind, Hail, 
    Flash Flood, High Wind and Winter Weather) at the county level. It filters the dataset for these events, groups it by county and event type, 
    and creates a stacked bar plot to show the distribution of event reports for the top 20 counties.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame containing the storm events data, including 'CZ_NAME' (county), 
                        'STATE', and 'EVENT_TYPE' columns.

    Returns:
    None: Displays a stacked bar plot showing the count of each event type for the top 20 counties.
    """

    # Extract the year from the DataFrame
    year = int(df[['YEAR']].iloc[0, 0])

    # Define the event types to analyze
    event_types = ['Thunderstorm Wind', 'Hail', 'Flash Flood', 'High Wind', 'Winter Weather']
    
    # Filter the DataFrame to include only the selected event types
    df_filtered = df[df['EVENT_TYPE'].isin(event_types)]

    # Create a new column that combines the county name and state
    df_filtered['COUNTY_STATE'] = df_filtered['CZ_NAME'] + " (" + df_filtered['STATE'] + ")"

    # Group by county-state and event type, then count the occurrences of each combination
    county_counts = df_filtered.groupby(['COUNTY_STATE', 'EVENT_TYPE']).size().unstack().fillna(0)

    # Sum the counts of all event types for each county-state and sort by total counts
    county_counts['total'] = county_counts.sum(axis=1)
    county_counts_sorted = county_counts.sort_values(by='total', ascending=False)

    # Plotting the data for the top 20 counties
    plt.figure(figsize=(12, 6))
    
    # Create a stacked bar plot for the top 20 counties sorted by total event counts
    county_counts_sorted.head(20).drop('total', axis=1).plot(kind='bar', stacked=True, figsize=(12, 6), colormap=custom_cmap)
    
    # Adding titles and labels
    plt.title(f'Number of Most Frequent Weather Disasters by County in {year} (Top 20)', fontsize=14)
    plt.xlabel('County (State)', fontsize=12)
    plt.ylabel('Number of Reports', fontsize=12)
    
    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=90)
    
    # Add a legend to explain the event types
    plt.legend(title="Event Type")
    
    # Add gridlines for better readability
    plt.grid(axis='y')
    
    # Show the plot
    plt.show()




def plot_monthly_event_trends(dfs):
    """
    Plots the trend of disaster event occurrences by month for multiple years.
    
    Parameters:
    - dfs (list of DataFrames): A list of Pandas DataFrames, each containing storm event data for a single year.
    
    Output:
    - A line plot showing the trend of the number of disaster events reported for each month across multiple years.
    """

    # Combine all yearly DataFrames into one
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Extract month and year from the event data
    df_combined['MONTH'] = pd.to_datetime(df_combined['BEGIN_DATE_TIME']).dt.month
    df_combined['YEAR'] = pd.to_datetime(df_combined['BEGIN_DATE_TIME']).dt.year

    # Count occurrences of events for each year and month
    event_trends_monthly = df_combined.groupby(['YEAR', 'MONTH']).size().reset_index(name='COUNT')

    # Plot the trends
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=event_trends_monthly, x='MONTH', y='COUNT', hue='YEAR', marker='o', palette='tab10', linewidth=2)

    plt.title('Monthly Trend of Storms Events Over the Years', fontsize=14)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Number of Reported Events', fontsize=12)
    plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend(title='Year')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.show()




def plot_state_event_seasonality(df):
    """
    Plots the seasonality of specific disaster events based on the month.
    
    Parameters:
    - df (DataFrame): A DataFrame containing storm event data, including the event type and start date.
    
    Output:
    - A box plot showing the distribution of disaster events by month.
    """
    # Extract the year from the DataFrame
    year = int(df[['YEAR']].iloc[0, 0])
    
    # Convert 'BEGIN_DATE_TIME' column to datetime
    df['BEGIN_DATE_TIME'] = pd.to_datetime(df['BEGIN_DATE_TIME'])
    
    # Extract month from the 'BEGIN_DATE_TIME' column
    df['MONTH'] = df['BEGIN_DATE_TIME'].dt.month
    
    # List of event types to analyze
    event_types = ['Thunderstorm Wind', 'Hail', 'Flash Flood', 'High Wind', 'Winter Weather']
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Filter data for the selected event types and create a boxplot
    sns.boxplot(x='MONTH', y='EVENT_TYPE', data=df[df['EVENT_TYPE'].isin(event_types)], palette=custom_palette)
    
    # Set the title and labels
    plt.title(f'Seasonality of Most Frequent Weather Disasters in {year}', fontsize=14)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Event Type', fontsize=12)
    
    # Set x-axis labels for months
    plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    # Enable grid for better readability
    plt.grid(True)
    
    # Show the plot
    plt.show()



def plot_event_heatmap(df):
    """
    Plots a heatmap showing the monthly distribution of disaster events for the top 20 most affected counties.

    Parameters:
    - df (DataFrame): A Pandas DataFrame containing storm event data with 'STATE', 'BEGIN_DATE_TIME', and 'YEAR'.

    Output:
    - A heatmap visualizing the frequency of disaster events by month and county.
    """

    # Konwersja daty i ekstrakcja miesiąca
    df['BEGIN_DATE_TIME'] = pd.to_datetime(df['BEGIN_DATE_TIME'])
    df['MONTH'] = df['BEGIN_DATE_TIME'].dt.month

    # Zliczanie liczby zdarzeń dla każdego hrabstwa
    county_counts = df.groupby("STATE").size().nlargest(20).index
    df_filtered = df[df["STATE"].isin(county_counts)]

    # Grupowanie liczby zgłoszeń na miesiąc dla każdego hrabstwa
    heatmap_data = df_filtered.groupby(['STATE', 'MONTH']).size().unstack(fill_value=0)

    # Tworzenie wykresu heatmapy
    plt.figure(figsize=(14, 8))
    sns.heatmap(heatmap_data, cmap="rocket_r", linewidths=0.5, annot=True, fmt="d")

    # Ustawienie etykiet
    plt.title('Monthly Distribution of Storm Events in Top 20 States', fontsize=14)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('County (State)', fontsize=12)
    plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
    plt.yticks(rotation=0)

    # Wyświetlenie wykresu
    plt.show()



def plot_top_damage_events(df, damage_type):
    """
    Plots the top 10 disaster event types that caused the highest property damage.

    Output:
    - A horizontal bar chart displaying the top 10 disaster categories with the highest total property damage.
    """
        
    # Extract the year from the DataFrame
    year = int(df[['YEAR']].iloc[0, 0])

    # Group by event type and sum the total property damage, then select the top 10
    top_damage = df.groupby('EVENT_TYPE')['DAMAGE_PROPERTY'].sum().sort_values(ascending=False).head(10)
    
    # Create the bar chart
    plt.figure(figsize=(12, 6))
    plt.barh(top_damage.index, top_damage.values, color='219ebc')
    
    # Set labels and title
    plt.xlabel('Property Damage ($)', fontsize=12)
    plt.ylabel('Event Type', fontsize=12)
    plt.title(f'Top 10 Disaster Categories with the Highest Property Damage in {year}', fontsize=14)
    
    # Invert the y-axis for better visualization (highest damage at the top)
    plt.gca().invert_yaxis()
    
    # Show the plot
    plt.show()



def plot_top_damage_events(df, damage_type='DAMAGE_PROPERTY'):
    """
    Plots the top 10 disaster event types that caused the highest financial losses.

    Parameters:
    - df (DataFrame): A Pandas DataFrame containing storm event data.
    - damage_type (str): Column name specifying which type of damage to analyze ('DAMAGE_PROPERTY' or 'DAMAGE_CROPS').

    Output:
    - A horizontal bar chart displaying the top 10 disaster categories with the highest total damage.
    """
    # Extract the year from the DataFrame
    year = int(df[['YEAR']].iloc[0, 0])
    
    # Validate damage_type input
    if damage_type not in df.columns:
        raise ValueError(f"Column '{damage_type}' not found in the DataFrame. Choose 'DAMAGE_PROPERTY' or 'DAMAGE_CROPS'.")

    # Group by event type and sum the total damage, then select the top 10
    top_damage = df.groupby('EVENT_TYPE')[damage_type].sum().sort_values(ascending=False).head(10)
    
    # Define plot title dynamically based on selected damage type
    damage_label = "Property Damage" if damage_type == 'DAMAGE_PROPERTY' else "Crop Damage"
    
    # Create the bar chart
    plt.figure(figsize=(12, 6))
    plt.barh(top_damage.index, top_damage.values, color='#219ebc')
    
    # Set labels and title
    plt.xlabel(f'{damage_label} ($)', fontsize=12)
    plt.ylabel('Event Type', fontsize=12)
    plt.title(f'Top 10 Disaster Categories with the Highest {damage_label} in {year}', fontsize=14)
    
    # Invert the y-axis for better visualization (highest damage at the top)
    plt.gca().invert_yaxis()
    
    # Show the plot
    plt.show()




def convert_damage(value):
    """
    Converts damage values from string format with suffixes ('K', 'M', 'B') to numerical format.
    
    Parameters:
    - value (str, float, or NaN): The damage value that may contain suffixes:
        - 'K' (thousands) → Multiplies by 1,000
        - 'M' (millions) → Multiplies by 1,000,000
        - 'B' (billions) → Multiplies by 1,000,000,000
        - If the value is already a number or does not contain a suffix, it is returned as a float.
    
    Returns:
    - float: The numerical equivalent of the damage value.
    """
    
    if pd.isna(value) or value == '':  # Handle missing or empty values
        return 0.0
    
    value = str(value).upper().strip()  # Convert to uppercase and remove spaces
    
    if value.endswith('K'):  # Thousands
        return float(value.replace('K', '')) * 1_000
    elif value.endswith('M'):  # Millions
        return float(value.replace('M', '')) * 1_000_000
    elif value.endswith('B'):  # Billions
        return float(value.replace('B', '')) * 1_000_000_000
    else:  # Assume it's already a numeric value
        return float(value)
    




def plot_monthly_damage(dfs, damage_type='DAMAGE_PROPERTY'):
    """
    Plots the monthly trend of financial damage caused by weather events over multiple years.
    
    Parameters:
    - dfs (list of DataFrames): A list of Pandas DataFrames, each containing storm event data for a single year.
    - damage_type (str): Specifies the type of damage to analyze ('DAMAGE_PROPERTY' or 'DAMAGE_CROPS').

    Output:
    - A line plot displaying monthly financial damage trends for each year.
    """
    # Combine all yearly DataFrames into one
    df_combined = pd.concat(dfs, ignore_index=True)

    # Convert date column to datetime format
    df_combined['BEGIN_DATE_TIME'] = pd.to_datetime(df_combined['BEGIN_DATE_TIME'])

    # Extract year and month
    df_combined['YEAR'] = df_combined['BEGIN_DATE_TIME'].dt.year
    df_combined['MONTH'] = df_combined['BEGIN_DATE_TIME'].dt.month

    # Ensure is numeric and handle missing values
    df_combined['damage_type'] = pd.to_numeric(df_combined[damage_type], errors='coerce').fillna(0)

    # Sum total damage for each month and year
    monthly_damage = df_combined.groupby(['YEAR', 'MONTH'])['damage_type'].sum().reset_index()

    # Define plot title dynamically based on selected damage type
    damage_label = "Property Damage" if damage_type == 'DAMAGE_PROPERTY' else "Crop Damage"

    # Plot the trends
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=monthly_damage, x='MONTH', y='damage_type', hue='YEAR', marker='o', palette='tab10', linewidth=2)

    plt.title(f'Monthly {damage_label} Trends Over the Years', fontsize=14)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Total Property Damage ($)', fontsize=12)
    plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend(title='Year')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.show()




def calculate_reporting_counties(dfs_outages, all_fips):
    """
    Calculates the number and percentage of counties that reported at least one outage
    in each year relative to the total number of counties (all_fips).

    Parameters:
    dfs_outages (list of DataFrames): List of outage DataFrames for different years.
    all_fips (set): Set of all county FIPS codes.

    Returns:
    list of tuples: Each tuple contains (year, reporting_count, total_counties, percentage).
    """
    results = []
    
    for year, df in zip(range(2014, 2024), dfs_outages):
        reporting_fips = set(df["fips_code"].unique())  # Counties that reported outages
        reporting_count = len(reporting_fips)
        total_counties = len(all_fips)
        percentage = (reporting_count / total_counties) * 100

        results.append((year, reporting_count, total_counties, round(percentage, 2)))

    return results




def make_ts_power(county,
                  start_year,
                  start_month,
                  start_day,
                  end_year,
                  end_month,
                  end_day,
                  data_directory = './data/data/eaglei_data'):
    
    """
    Creates a time series of the number of customers without power in a specified county based on outage data 
    from CSV files located in the `data_directory` folder. The steps include:
    1. Reading CSV files for each year in the range [start_year, end_year].
    2. Filtering data for the specified county.
    3. Grouping data by the time of outage (in 15-minute intervals).
    4. Returning the number of customers without power for the specified time range.
    """

    df_list = []
    for year in range(start_year, end_year + 1):
        file_name = f"eaglei_outages_{year}.csv"
        file_path = os.path.join(data_directory, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_name} does not exist in the directory {data_directory}.")

        df = pd.read_csv(file_path)
        df['run_start_time'] = pd.to_datetime(df['run_start_time'])
        df.dropna(subset=['customers_out'],inplace=True)

        df_state = df[df['county'].str.upper()==county.upper()].copy(deep=True)
        df_state_ts_cus = df_state.groupby('run_start_time')['customers_out'].sum().reset_index()
        df_state_ts_cus.drop(df_state_ts_cus.index[-1], inplace=True)
        df_state_ts_cus.set_index('run_start_time', inplace=True)
        df_state_ts_cus.rename_axis('time', inplace=True)


        df_list.append(df_state_ts_cus)

    concat_df = pd.concat(df_list, ignore_index=False)
    start_date = pd.Timestamp(year=start_year, month=start_month, day=start_day)
    end_date = pd.Timestamp(year=end_year, month=end_month, day=end_day+1)
    df_state_ts_power = concat_df.loc[start_date:end_date].copy(deep=True)
    df_state_ts_power.drop(df_state_ts_power.index[-1], inplace=True)

    return df_state_ts_power




def make_ts_events(county, event_types, start_year, start_month, start_day, end_year, end_month, end_day, df):

    """
    Creates a time series of event counts and sums for specific event types (e.g., injuries, deaths) 
    in a specified county from the provided event data.
    The steps include:
    1. Filtering the event data based on the county and event types.
    2. Creating a time series for the events, with data aggregated in 15-minute intervals.
    3. Calculating the sum values for injuries and deaths during the events.
    4. Returning the time series with event counts and sums for the specified time range.
    """

    start_date = datetime(start_year, start_month, start_day)
    end_date = datetime(end_year, end_month, end_day, 23, 45) 
    time_index = pd.date_range(start=start_date, end=end_date, freq='15min')

    new_df = pd.DataFrame({'time': time_index})

    for event_type in event_types:
        new_df[f'event_count {event_type}'] = 0  

    avg_cols = ['injuries_direct', 'injuries_indirect', 'deaths_direct', 'deaths_indirect']
    for col in avg_cols:
        new_df[col] = 0.0

    df['BEGIN_DATETIME'] = pd.to_datetime(
        df['BEGIN_YEARMONTH'].astype(str) + df['BEGIN_DAY'].astype(str).str.zfill(2) +
        df['BEGIN_TIME'].astype(str).str.zfill(4), format='%Y%m%d%H%M'
    )
    df['END_DATETIME'] = pd.to_datetime(
        df['END_YEARMONTH'].astype(str) + df['END_DAY'].astype(str).str.zfill(2) +
        df['END_TIME'].astype(str).str.zfill(4), format='%Y%m%d%H%M'
    )

    filtered_df = df[
        (df['CZ_NAME'].str.upper() == county.upper()) &
        (df['EVENT_TYPE'].isin(event_types)) &
        (df['END_DATETIME'] >= start_date) &
        (df['BEGIN_DATETIME'] <= end_date)
    ].copy(deep=True)

    for event_type in event_types:
        event_subset = filtered_df[filtered_df['EVENT_TYPE'] == event_type]

        for _, row in event_subset.iterrows():
            event_start = row['BEGIN_DATETIME']
            event_end = row['END_DATETIME']

            event_start_rounded = event_start.round('15min')
            event_end_rounded = event_end.round('15min')

            start_idx = new_df['time'].searchsorted(event_start_rounded)
            end_idx = new_df['time'].searchsorted(event_end_rounded)

            if start_idx < len(new_df) and end_idx <= len(new_df):
                new_df.loc[start_idx:end_idx, f'event_count {event_type}'] += 1
                new_df.loc[start_idx:end_idx, 'injuries_direct'] += row['INJURIES_DIRECT']
                new_df.loc[start_idx:end_idx, 'injuries_indirect'] += row['INJURIES_INDIRECT']
                new_df.loc[start_idx:end_idx, 'deaths_direct'] += row['DEATHS_DIRECT']
                new_df.loc[start_idx:end_idx, 'deaths_indirect'] += row['DEATHS_INDIRECT']

    #total_events = new_df[[f'event_count {event_type}' for event_type in event_types]].sum(axis=1)
    #for col in avg_cols:
    #    new_df[col] = new_df[col] / total_events.replace(0, 1) 


    new_df['YEAR'] = new_df['time'].dt.year
    new_df['MONTH'] = new_df['time'].dt.month
    new_df['DAY'] = new_df['time'].dt.day

    cols_order = ['YEAR', 'MONTH', 'DAY', 'time'] + avg_cols + [col for col in new_df.columns if col not in ['YEAR', 'MONTH', 'DAY', 'time'] + avg_cols]
    new_df = new_df[cols_order]

    return new_df




def aggregate_ts(df, agg_type):
    """
    Aggregates the given time series data by either hour or day.
    The function performs the following:
    1. Groups the data by the specified time interval ('hour' or 'day').
    2. Aggregates the values using the sum for each time period.
    3. Returns the aggregated time series.
    """

    df.index = pd.to_datetime(df.index)

    

    if agg_type == 'hour':
        #df_agg = df.groupby(pd.Grouper(freq='h')).mean()
        df_agg = df.groupby(pd.Grouper(freq='h')).sum()

    elif agg_type == 'day':
        #df_agg = df.groupby(pd.Grouper(freq='D')).mean()
        df_agg = df.groupby(pd.Grouper(freq='D')).sum()

    else:
        raise ValueError("Invalid aggregation type. Use 'hour' or 'day'.")

    df_agg.fillna(0, inplace=True)

    return df_agg



def combine_agg_ts1(county,
                   start_year,
                   start_month,
                   start_day,
                   end_year,
                   end_month,
                   end_day,
                   data_directory_power = './data/data/eaglei_data',
                   data_directory_events = './data/data/NOAA_StormEvents'):
    
    """
    Combines the aggregated power outage data and event data (e.g., storm events) for a specified county
    and time range. The steps include:
    1. Loading power outage data for the given county and time range.
    2. Aggregating the power outage data by hour and day.
    3. Loading event data for the given county and time range.
    4. Aggregating the event data by hour and day.
    5. Merging the aggregated power outage data with the event data.
    6. Returning the combined time series data for both hourly and daily aggregations.
    """


    df_state_ts_power = make_ts_power(county = county,
                                      start_year = start_year,
                                      start_month = start_month,
                                      start_day = start_day,
                                      end_year = end_year,
                                      end_month = end_month,
                                      end_day = end_day,
                                      data_directory = data_directory_power)

    df_state_ts_power_hr = aggregate_ts(df_state_ts_power, 'hour')
    df_state_ts_power_day = aggregate_ts(df_state_ts_power, 'day')



    df_events = pd.read_csv(os.path.join(data_directory_events, "StormEvents_2014_2024.csv"))
    df_state_events=df_events[df_events['CZ_NAME'].str.upper()==county.upper()].copy(deep=True)
    event_types_state = list(df_state_events['EVENT_TYPE'].unique())
    df_state_ts_events = make_ts_events(county = county,
                                        event_types= event_types_state,
                                        start_year = start_year,
                                        start_month = start_month,
                                        start_day = start_day,
                                        end_year = end_year,
                                        end_month = end_month,
                                        end_day = end_day,
                                        df=df_events)
    

    df_state_ts_events['time'] = pd.to_datetime(df_state_ts_events['time'])
    df_state_ts_events.set_index('time', inplace=True)
    df_state_ts_events.drop(columns=['YEAR', 'DAY', 'MONTH'], inplace=True)

    df_state_ts_events_hr = aggregate_ts(df_state_ts_events, 'hour')
    df_state_ts_events_day = aggregate_ts(df_state_ts_events, 'day')

    df_state_ts_comb_hr = pd.merge(df_state_ts_events_hr, df_state_ts_power_hr, left_index=True, right_index=True)
    df_state_ts_comb_day = pd.merge(df_state_ts_events_day, df_state_ts_power_day, left_index=True, right_index=True)

    return df_state_ts_comb_hr, df_state_ts_comb_day



def countyCountYEarly(year_start, year_end, dfs):
    """
    Analyzes power outage data from a list of DataFrames for a given range of years.
    Counts the number of unique counties (FIPS codes) affected each year 
    and visualizes the results using a bar chart.

    Parameters:
    year_start (int): The starting year of the analysis (inclusive).
    year_end (int): The ending year of the analysis (inclusive).
    dfs (list of pd.DataFrame): A list of Pandas DataFrames, where each DataFrame corresponds 
                                to a specific year in sequential order (e.g., dfs[0] is for 2014, dfs[1] is for 2015, etc.).
                                Each DataFrame must contain a column named 'fips_code'.

    Output:
    A bar chart displaying the number of unique FIPS codes per year.
    """


    # Generate a range of years to process
    years = range(year_start, year_end + 1)

    # List to store the count of unique FIPS codes per year
    year_counts = []

    # Iterate through each year in the specified range
    for year in years:

        # Access the DataFrame corresponding to the current year
        df_power = dfs[year - 2014]
        # Count the number of unique FIPS codes in the dataset
        unique_fips_count = df_power['fips_code'].nunique()
        
        # Append the count to the list
        year_counts.append(unique_fips_count)

    # Plot the results using a bar chart
    plt.figure(figsize=(10, 6))  # Set the figure size
    bars = plt.bar(years, year_counts, color='skyblue', edgecolor='black')  # Create bars
    
    # Label the axes and title
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Number of Unique FIPS Codes", fontsize=12)
    plt.title("Unique FIPS Codes per Year", fontsize=16)
    plt.xticks(years, rotation=45)  # Rotate x-axis labels for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add a dashed grid for better readability

    # Add text labels on top of each bar
    for bar, count in zip(bars, year_counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), 
                 ha='center', va='bottom', fontsize=10, color='black')

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()





def countyCountBarPlots(year_start, year_end, dfs):
    """
    Analyzes power outage data from CSV files for a given range of years, 
    counts the number of unique counties (FIPS codes) affected in each state per year, 
    and visualizes the results using horizontal bar charts for each year.

    Parameters:
    year_start (int): The starting year of the analysis.
    year_end (int): The ending year of the analysis.
    dfs (list of pd.DataFrame): A list of Pandas DataFrames, where each DataFrame corresponds 
                                to a specific year in sequential order (e.g., dfs[0] is for 2014, dfs[1] is for 2015, etc.).
                                Each DataFrame must contain a column named 'fips_code'.

    Output:
    A grid of horizontal bar charts displaying the number of unique FIPS codes per state for each year.
    """

    # Generate a range of years to process
    years = range(year_start, year_end + 1)

    # Create an empty DataFrame for potential combined analysis (currently unused)
    combined_data = pd.DataFrame()

    # Define the number of columns in the subplot grid
    cols = 3  # Number of columns in the plot grid
    rows = -(-len(years) // cols)  # Calculate the number of rows (ceiling division)

    # Create a figure and axes for multiple subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 7 * rows))
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    # Iterate through each year in the specified range
    for i, year in enumerate(years):

        # Access the DataFrame corresponding to the current year
        df_power = dfs[year - 2014]

        # Standardize state names (corrects 'US Virgin Islands' to 'United States Virgin Islands')
        df_power['state'] = df_power['state'].replace('US Virgin Islands', 'United States Virgin Islands')

        # Count the number of unique FIPS codes per state, sorting the results
        unique_fips = (
            df_power.groupby('state')['fips_code']
            .nunique()
            .reset_index()
            .sort_values(by='fips_code')
        )

        # Create a horizontal bar chart for the given year
        axes[i].barh(unique_fips['state'], unique_fips['fips_code'], color='skyblue')
        axes[i].set_title(f'Number of Counties in {year}')
        axes[i].set_xlabel('Number of Counties')
        axes[i].set_ylabel('State')

    # Remove any unused subplots in the grid
    for j in range(len(years), len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()





def countyCountBarPlots_v2(year_start, year_end, dfs):
    """
    Analyzes power outage data from CSV files for a given range of years, 
    counts the number of unique counties (FIPS codes) affected in each state per year, 
    and visualizes the results using a grouped horizontal bar chart.

    Parameters:
    year_start (int): The starting year of the analysis.
    year_end (int): The ending year of the analysis.
    dfs (list of pd.DataFrame): A list of Pandas DataFrames, where each DataFrame corresponds 
                                to a specific year in sequential order (e.g., dfs[0] is for 2014, dfs[1] is for 2015, etc.).
                                Each DataFrame must contain a column named 'fips_code'.

    Output:
    A horizontal bar chart displaying the number of unique FIPS codes per state across multiple years.
    """


    # Generate a range of years to process
    years = range(year_start, year_end + 1)

    # Initialize an empty DataFrame to store data for all years
    combined_data = pd.DataFrame()

    # Iterate through each year in the specified range
    for year in years:

        # Access the DataFrame corresponding to the current year
        df_power = dfs[year - 2014]

        # Standardize state names (corrects 'US Virgin Islands' to 'United States Virgin Islands')
        df_power['state'] = df_power['state'].replace('US Virgin Islands', 'United States Virgin Islands')

        # Count the number of unique FIPS codes per state, sorting the results
        unique_fips = (
            df_power.groupby('state')['fips_code']
            .nunique()
            .reset_index()
            .sort_values(by='fips_code')
        )

        # Rename the column with the current year to prepare for merging
        unique_fips = unique_fips.rename(columns={'fips_code': year})

        # Merge data for different years into a single DataFrame
        if combined_data.empty:
            combined_data = unique_fips  # First year's data
        else:
            combined_data = pd.merge(combined_data, unique_fips, on='state', how='outer')

    # Extract state names
    states = combined_data['state']

    # Create y-axis positions for states
    y_indexes = np.arange(len(states))  

    # Determine bar height for each year (ensures bars don't overlap completely)
    height = 0.9 / len(years)  

    # Set figure size dynamically based on the number of years
    plt.figure(figsize=(15, 3 * len(years)))

    # Create horizontal bars for each year
    for i, year in enumerate(years):
        plt.barh(y_indexes + i * height, combined_data[year], height=height, label=str(year))

    # Label axes and title
    plt.ylabel("State", fontsize=12)
    plt.xlabel("Number of Counties", fontsize=12)
    plt.title(f"Number of Counties per State with Power Outages Data ({year_start} - {year_end})", fontsize=16)

    # Adjust y-axis ticks to align with state names
    plt.yticks(y_indexes + (len(years) - 1) * height / 2, states)

    # Add legend for years
    plt.legend(title="Year", fontsize=10)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()





def missingCountyMap(years, dfs):
    """
    Visualizes missing county data for power outages across multiple years using maps.

    Parameters:
    years (list of int): A list of years for which missing county data should be mapped.
    dfs (list of pd.DataFrame): A list of Pandas DataFrames, where each DataFrame corresponds 
                                to a specific year in sequential order (e.g., dfs[0] is for 2014, dfs[1] is for 2015, etc.).
                                Each DataFrame must contain a column named 'fips_code'.

    Output:
    A series of maps (one per year) showing counties with and without recorded outage data.
    """

    # Path to the shapefile containing US county boundaries
    shapefile_path = "../data/US_county_2/cb_2018_us_county_500k.shp"

    # Load county geometries using GeoPandas
    counties = gpd.read_file(shapefile_path)
    
    # Ensure GEOID (FIPS code) is stored as a string for proper merging
    counties['GEOID'] = counties['GEOID'].astype(str)

    # Determine the number of columns (one plot per year, arranged in a row)
    cols = len(years)  
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 3))  # Single-row grid of maps

    # Loop through each specified year
    for idx, year in enumerate(years):

        # Access the DataFrame corresponding to the current year
        df_power = dfs[year - 2014]

        # Ensure FIPS codes are strings and correctly formatted to five digits
        df_power['fips_code'] = df_power['fips_code'].astype(str).str.zfill(5)

        # Create a DataFrame containing unique FIPS codes for counties with data
        fips_codes = pd.DataFrame(df_power['fips_code'].unique(), columns=["FIPS"])

        # Merge the county geometries with the FIPS codes from the outage dataset
        merged = counties.merge(fips_codes, left_on='GEOID', right_on='FIPS', how='left')

        # Create a column indicating whether data is available (1 = data available, 0 = missing)
        merged['Data'] = merged['FIPS'].apply(lambda x: 0 if pd.isna(x) else 1)

        # Define the bounding box for the mainland US (excludes Alaska and Hawaii)
        us_bounds = box(-125, 24, -66, 50)
        mainland_us = merged[merged.geometry.intersects(us_bounds)]

        # Separate counties with and without recorded data
        nan_counties = mainland_us[mainland_us['FIPS'].isna()]  # Missing data
        non_nan_counties = mainland_us[mainland_us['FIPS'].notna()]  # Recorded data

        # Select the correct subplot axis for the current year
        ax = axes[idx]

        # Plot missing counties in black
        nan_counties.plot(ax=ax, color="black", edgecolor="black", label="No data")

        # Plot counties with recorded data in white with grey borders
        non_nan_counties.plot(ax=ax, color="white", edgecolor="grey", label="Recorded data")

        # Set the title for the subplot
        ax.set_title(f"Missing counties {year}")

        # Remove axis labels for a cleaner visualization
        ax.set_axis_off()

    # Adjust layout for better spacing
    plt.tight_layout()

    # Display the plots
    plt.show()



def missingCountyMapManyYears(df):
    """
    Generates a map of the mainland US showing counties with available power outage data 
    across multiple years using a color scale.

    Parameters:
    df (DataFrame): A Pandas DataFrame containing county FIPS codes and corresponding 
                    normalized non-null values indicating data availability.

    Output:
    A choropleth map of the mainland US where counties are shaded based on the amount 
    of available power outage data over multiple years.
    """

    # Load state boundaries for reference (GeoJSON format)
    url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
    usa_states = gpd.read_file(url)

    # Load county shapefile (for accurate geographical representation)
    shapefile_path = "../data/US_county_2/cb_2018_us_county_500k.shp"
    counties = gpd.read_file(shapefile_path)

    # Ensure FIPS codes in the county dataset are stored as strings
    counties['GEOID'] = counties['GEOID'].astype(str)

    # Convert FIPS codes in the input DataFrame to strings and ensure they are 5-digit formatted
    df.loc[:, 'fips_code'] = df['fips_code'].astype(str).str.zfill(5)

    # Extract unique FIPS codes from the DataFrame and reshape them into a column
    fips_codes = df.iloc[:, 0].to_numpy().reshape(-1, 1)
    fips_codes = pd.DataFrame(np.unique(fips_codes))

    # Ensure FIPS codes are treated as strings and rename column for clarity
    fips_codes = fips_codes.astype(str)
    fips_codes = fips_codes.rename(columns={0: "FIPS"})

    # Add the 'Data' column, which represents normalized availability of outage data
    fips_codes['Data'] = df['normalized_non_nan']

    # Merge county geometries with FIPS codes to associate data availability
    merged = counties.merge(fips_codes, left_on='GEOID', right_on='FIPS', how='left')

    # Fill missing values with 0 (indicating no outage data for those counties)
    merged['Data'] = merged['Data'].fillna(0)

    # Define the bounding box for the mainland US (excludes Alaska and Hawaii)
    us_bounds = box(-125, 24, -66, 50)

    # Filter counties that fall within the mainland US
    mainland_us = merged[merged.geometry.intersects(us_bounds)]

    # Filter state boundaries to only include mainland US
    mainland_states = usa_states[usa_states.geometry.intersects(us_bounds)]

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    # Plot counties using a red color scale (darker red = more data available)
    mainland_us.plot(column='Data', ax=ax, cmap="Reds", legend=True, 
                     edgecolor="grey", missing_kwds={"color": "lightgrey"})

    # Overlay state boundaries in red
    mainland_states.boundary.plot(ax=ax, color='red', linewidth=1, label="State Borders")

    # Set title for the map
    ax.set_title("Mainland US Counties Present in eaglei_outages", fontsize=16)

    # Display the map
    plt.show()




def drawMap(df, bounds, ax, name):
    """
    Draws a choropleth map of counties within a specified geographic boundary.

    Parameters:
    df (DataFrame): Pandas DataFrame containing county FIPS codes and 
                    normalized data availability values ('normalized_non_nan').
    bounds (shapely.geometry.Polygon): Bounding box defining the region of interest.
    ax (matplotlib.axes.Axes): Matplotlib Axes object where the map will be drawn.
    name (str): Name of the region being mapped (used in the title).

    Output:
    A choropleth map displayed on the provided Matplotlib Axes object.
    """

    # Load the county shapefile (contains county boundaries)
    shapefile_path = "../data/US_county_2/cb_2018_us_county_500k.shp"
    counties = gpd.read_file(shapefile_path)

    # Convert county GEOID codes to string format for proper merging
    counties['GEOID'] = counties['GEOID'].astype(str)

    # Ensure FIPS codes in the DataFrame are strings and formatted as 5-digit codes
    df.loc[:, 'fips_code'] = df['fips_code'].astype(str).str.zfill(5)

    # Extract unique FIPS codes and convert them to a DataFrame
    fips_codes = df.iloc[:, 0].to_numpy().reshape(-1, 1)
    fips_codes = pd.DataFrame(np.unique(fips_codes), columns=['FIPS'])

    # Ensure FIPS codes are treated as strings
    fips_codes = fips_codes.astype(str)

    # Merge county boundaries with FIPS codes
    merged = counties.merge(fips_codes, left_on='GEOID', right_on='FIPS', how='left')

    # Merge with the original DataFrame to retain additional data (e.g., 'normalized_non_nan')
    merged = merged.merge(df, left_on="FIPS", right_on="fips_code", how="left")

    # Fill missing values in 'normalized_non_nan' with 0 (indicating no data available)
    merged['normalized_non_nan'] = merged['normalized_non_nan'].fillna(0)

    # Filter counties that are within the specified geographical boundary
    map_data = merged[merged.geometry.intersects(bounds)]

    # Plot the counties on the given Matplotlib Axes
    map_data.plot(column='normalized_non_nan', ax=ax, cmap="Reds", legend=True, 
                  vmin=0, vmax=1, edgecolor="grey", missing_kwds={"color": "lightgrey"})

    # Set x and y axis limits based on the bounding box
    ax.set_xlim(bounds.bounds[0], bounds.bounds[2])  # Min x to Max x
    ax.set_ylim(bounds.bounds[1], bounds.bounds[3])  # Min y to Max y

    # Set the title for the plot
    ax.set_title(f"{name} counties present in eaglei_outages", fontsize=16)





def countyCountMAP(year_start, year_end, dfs):
    """
    Generates a choropleth map displaying county-level power outage data availability 
    across multiple years. It calculates the frequency of each county appearing in 
    the dataset and normalizes this count over the given time range.

    Parameters:
    year_start (int): The starting year for the analysis.
    year_end (int): The ending year for the analysis.
    dfs (list of pd.DataFrame): A list of Pandas DataFrames, where each DataFrame corresponds 
                                to a specific year in sequential order (e.g., dfs[0] is for 2014, dfs[1] is for 2015, etc.).
                                Each DataFrame must contain a column named 'fips_code'.

    Output:
    - A US mainland map showing data coverage across years.
    - Separate maps for Hawaii, Alaska, and the Caribbean territories.
    """

    # Create a range of years from start to end
    years = range(year_start, year_end + 1)

    # Initialize an empty DataFrame to store FIPS codes across years
    combined_count = pd.DataFrame()

    # Iterate through each year and process the respective outage file
    for year in years:

        # Access the DataFrame corresponding to the current year
        df_power = dfs[year - 2014]

        # Ensure fips_code is stored as a string for consistency
        df_power['fips_code'] = df_power['fips_code'].astype(str)

        # Extract unique FIPS codes (county identifiers) while ignoring duplicate entries
        unique_fips_2 = (
            df_power[['fips_code', 'state']]
            .drop_duplicates()
            .sort_values(by='fips_code')
            .reset_index()
            .drop(columns=['state', 'index'])  # Remove unnecessary columns
        )

        # Create a new column for the current year
        unique_fips_2['col'] = unique_fips_2['fips_code']
        unique_fips_2 = unique_fips_2.rename(columns={'col': year})

        # Merge the data with the combined DataFrame
        if combined_count.empty:
            combined_count = unique_fips_2
        else:
            combined_count = pd.merge(combined_count, unique_fips_2, on='fips_code', how='outer')

    # Count the number of non-null values for each county across years
    combined_count['non_nan_count'] = combined_count.drop(columns=['fips_code']).notna().sum(axis=1)

    # Get the number of years considered (i.e., number of columns excluding metadata columns)
    num_columns_to_count = combined_count.drop(columns=['fips_code', 'non_nan_count']).shape[1]

    # Normalize the count by the number of years to get a proportion
    combined_count['normalized_non_nan'] = combined_count['non_nan_count'] / num_columns_to_count

    # Keep only relevant columns for mapping
    df = combined_count[['fips_code', 'normalized_non_nan']]

    # Generate a nationwide map displaying data availability
    missingCountyMapManyYears(df)

    # Define bounding boxes for different US regions
    hawaii_bounds = box(-161, 18, -154, 23)  # Hawaii
    alaska_bounds = box(-180, 50, -125, 72)  # Alaska
    caribbean_bounds = box(-68, 17.5, -64.4, 18.7)  # Caribbean (Puerto Rico & US Virgin Islands)

    # Create subplots for each region
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Draw maps for Hawaii, Alaska, and the Caribbean
    drawMap(df, hawaii_bounds, axes[0], "Hawaii")
    drawMap(df, alaska_bounds, axes[1], "Alaska")
    drawMap(df, caribbean_bounds, axes[2], "United States Virgin Islands and Puerto Rico")

    # Adjust layout and display the maps
    plt.tight_layout()
    plt.show()




def load_data(file_path):
    """
    Loads weather data and converts the 'date' column to datetime format.
    Convert fips column to 5-digit.
    """
    df = pd.read_csv(file_path)
    df['fips'] = df['fips'].astype(str).str.zfill(5)

    df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    return df



def plot_precipitation_seasonality(df, year):
    """
    Creates a histogram plot of daily precipitation for a selected year.
    """
    data = df[df['year'] == year]
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='date', weights='ppt', fill=True)
    plt.xlabel('Date')
    plt.ylabel('Precipitation Density')
    plt.title(f'Precipitation Seasonality in {year}')
    plt.show()



def plot_temperature_trend(df, year):
    """
    Creates a line plot of the monthly minimum and maximum temperature for a selected year.
    """
    data = df[df['year'] == year].groupby('month')[['tmin', 'tmax']].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x='month', y='tmin', label='Tmin', marker='o')
    sns.lineplot(data=data, x='month', y='tmax', label='Tmax', marker='o')
    plt.xticks(range(1, 13))
    plt.xlabel('Month')
    plt.ylabel('Temperature (°C)')
    plt.title(f'Minimum and Maximum Temperature in {year}')
    plt.legend()
    plt.show()




def plot_precipitation_by_year(year, df):
    """
    This function generates a choropleth map showing the total precipitation in millimeters
    for the selected year across counties in the United States (excluding Alaska and islands).
    
    Parameters:
    year (int): The year for which the precipitation data will be shown.
    df (pandas DataFrame): The DataFrame containing the columns 'fips', 'ppt', and 'year'.
    """
    # Filter the data for the chosen year
    year_data = df[df['year'] == year]
    
    # Aggregate the data by FIPS (summing the precipitation values)
    year_data_aggregated = year_data.groupby('fips').agg({'ppt': 'sum'}).reset_index()
    max_ppt = year_data_aggregated['ppt'].max()
    
    # Create the choropleth map
    fig = px.choropleth(year_data_aggregated, 
                        geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
                        locations='fips', color='ppt',
                        color_continuous_scale="Blues",
                        title=f"Total Precipitation in {year} (mm)",
                        labels={"ppt": "Total Precipitation (mm)"},
                        range_color=[0, max_ppt])
    
    # Exclude Alaska and islands from the map by adjusting the geographic scope
    fig.update_geos(
        scope="usa",  # Set the scope to "usa", excluding Alaska and islands
        visible=False,  # Set visibility to False to exclude from the map
        fitbounds="locations"
    )
    fig.update_traces(marker_line_width=0.2, selector=dict(type='choropleth'))

    # Display the map
    fig.show()


def plot_precipitation_by_year(year, df):
    """
    This function generates a choropleth map showing the total precipitation in millimeters
    for the selected year across counties in the United States (excluding Alaska and islands).
    
    Parameters:
    year (int): The year for which the precipitation data will be shown.
    df (pandas DataFrame): The DataFrame containing the columns 'fips', 'ppt', and 'year'.
    """
    # Filter the data for the chosen year
    year_data = df[df['year'] == year]
    
    # Aggregate the data by FIPS (summing the precipitation values)
    year_data_aggregated = year_data.groupby('fips').agg({'ppt': 'sum'}).reset_index()
    max_ppt = year_data_aggregated['ppt'].max()
    
    # Create the choropleth map
    fig = px.choropleth(year_data_aggregated, 
                        geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
                        locations='fips', color='ppt',
                        color_continuous_scale="Blues",
                        title=f"Total Precipitation in {year} (mm)",
                        labels={"ppt": "Total Precipitation (mm)"},
                        range_color=[0, max_ppt])
    
    # Exclude Alaska and islands from the map by adjusting the geographic scope
    fig.update_geos(
        scope="usa",  # Set the scope to "usa", excluding Alaska and islands
        visible=False,  # Set visibility to False to exclude from the map
        fitbounds="locations"
    )
    fig.update_traces(marker_line_width=0.2, selector=dict(type='choropleth'))

    # Display the map
    fig.show()



def plot_avg_temp_by_year(year, df):
    """
    This function generates a choropleth map showing the avgerage temperature in Cellcius degree
    for the selected year across counties in the United States (excluding Alaska and islands).
    
    Parameters:
    year (int): The year for which the precipitation data will be shown.
    df (pandas DataFrame): The DataFrame containing the columns 'fips', 'tavg', and 'year'.
    """
    # Filter the data for the chosen year
    year_data = df[df['year'] == year]
    
    # Aggregate the data by FIPS (summing the precipitation values)
    year_data_aggregated = year_data.groupby('fips').agg({'tavg': 'mean'}).reset_index()
    max_t = year_data_aggregated['tavg'].max()
    min_t = year_data_aggregated['tavg'].min()
    
    # Create the choropleth map
    fig = px.choropleth(year_data_aggregated, 
                        geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
                        locations='fips', color='tavg',
                        color_continuous_scale="RdBu_r",
                        title=f"Average Temperature in {year} (°C)",
                        labels={"tavg": "Avg Temp (°C)"},
                        range_color=[min_t, max_t])
    
    # Exclude Alaska and islands from the map by adjusting the geographic scope
    fig.update_geos(
        scope="usa",  # Set the scope to "usa", excluding Alaska and islands
        visible=False,  # Set visibility to False to exclude from the map
        fitbounds="locations"
    )
    fig.update_traces(marker_line_width=0.2, selector=dict(type='choropleth'))

    # Display the map
    fig.show()





def visualize_avg_temp_by_month(year, df):
    """
    This function generates an animated choropleth map showing the average temperature by month
    across counties in the United States (excluding Alaska and islands).
    
    Parameters:
    df (pandas DataFrame): The DataFrame containing the columns 'fips', 'tavg', 'year', and 'month'.
    """
    # Ensure that year and month columns exist in the dataframe
    if 'month' not in df.columns or 'year' not in df.columns:
        raise ValueError("The DataFrame must contain 'year' and 'month' columns.")
    
    # Filter the data for the chosen year
    year_data = df[df['year'] == year]
    # Aggregate the data by FIPS and month (averaging the temperature values)
    df_aggregated = year_data.groupby(['fips','year', 'month']).agg({'tavg': 'mean'}).reset_index()
    
    # Get min and max temperature values for color scale
    max_t = df_aggregated['tavg'].max()
    min_t = df_aggregated['tavg'].min()

    range_color = [-max(abs(min_t), abs(max_t)), max(abs(min_t), abs(max_t))]
    
    # Create the animated choropleth map
    fig = px.choropleth(df_aggregated, 
                        geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
                        locations='fips', color='tavg',
                        color_continuous_scale="RdBu_r",
                        animation_frame='month',  # Animate by month
                        animation_group='year',  # Group by year
                        title=f"Average Temperature by Month in the US in {year} (°C)",
                        labels={"tavg": "Avg Temp (°C)"},
                        range_color=range_color)
    fig.update_traces(marker_line_width=0.2, selector=dict(type='choropleth'))
    
    # Exclude Alaska and islands from the map by adjusting the geographic scope
    fig.update_geos(
        scope="usa",  # Set the scope to "usa", excluding Alaska and islands
        visible=False,  # Set visibility to False to exclude from the map
        fitbounds="locations"
    )
    
    # Display the animated map
    fig.show()


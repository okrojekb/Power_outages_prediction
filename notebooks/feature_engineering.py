import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from itertools import product


def summarize_overall_missing(dfs, names=None):
    summaries = []
    
    for i, df in enumerate(dfs):
        dataset_name = names[i] if names else f"stormEvents_{2014+i}"
        
        missing_count = df['MAGNITUDE'].isna().sum()
        present_count = df['MAGNITUDE'].notna().sum()
        total_count = len(df)
        
        summaries.append({
            'DATASET': dataset_name,
            'MISSING_MAGNITUDE': missing_count,
            'PRESENT_MAGNITUDE': present_count,
            'MISSING_PERCENT': round((missing_count / total_count) * 100, 2)
        })
    
    return pd.DataFrame(summaries)



def filter_counties(outages_dfs, mcc_df):
    """
    Delete counties, that have never apperar in any outages datasets from MCC
    
    :param outages_dfs: DataFrame List containing outage reports
    :param mcc_df: DataFrame containing County_FIPS and Customers
    :return: Filteres DataFrame MCC
    """
    appeared_fips = set()

    for df in outages_dfs:
        if 'fips_code' in df.columns:
            appeared_fips.update(df['fips_code'].dropna().astype(str).unique())

    mcc_df['County_FIPS'] = mcc_df['County_FIPS'].astype(str)

    filtered_mcc_df = mcc_df[mcc_df['County_FIPS'].isin(appeared_fips)]

    return filtered_mcc_df



def plot_reporting_distribution(df, time_column, expected_reports):
    """
    Plots the distribution of reporting completeness across counties.
    
    :param df: DataFrame containing outage reports
    :param expected_reports: Expected number of reports per county
    """
    year = pd.to_datetime(df[time_column].iloc[0]).year


    report_counts = df.groupby('fips_code')[time_column].count()
    reporting_percentage = (report_counts / expected_reports) * 100

    plt.figure(figsize=(10, 4))
    plt.hist(reporting_percentage, bins=50, edgecolor='black')
    plt.xlim(0,100)
    plt.xlabel("Percentage of Expected Reports")
    plt.ylabel("Number of Counties")
    plt.title(f"Distribution of Reporting Completeness in {year}")
    plt.show()



def aggregate_daily_outages(df):
    """
    Aggregates outage data by day and county, computing the daily average of customers_out.
    
    :param df: DataFrame containing outage reports
    :return: Aggregated DataFrame with daily averages per county
    """

    df['state'] = df['state'].replace('US Virgin Islands', 'United States Virgin Islands')

    df['run_start_time'] = pd.to_datetime(df['run_start_time'])
    df['date'] = df['run_start_time'].dt.date
    
    aggregated_df = df.groupby(["date", 'fips_code', 'county', 'state'])['customers_out'].mean().reset_index()
    
    return aggregated_df




def plot_removal_effect(df, time_column, expected_reports, thresholds):
    """
    Plots the percentage of counties removed as a function of the threshold.
    
    :param df: DataFrame containing outage reports
    :param time_column: Column name for timestamps
    :param expected_reports: Expected number of reports per county
    :param thresholds: List of threshold values to test
    """
    year = pd.to_datetime(df[time_column].iloc[0]).year

    report_counts = df.groupby('fips_code')[time_column].count()
    reporting_percentage = (report_counts / expected_reports) * 100
    
    removal_percentages = []
    total_counties = len(reporting_percentage)
    
    for threshold in thresholds:
        removed_counties = (reporting_percentage < threshold).sum()
        removal_percentages.append((removed_counties / total_counties) * 100)
    
    plt.figure(figsize=(10, 4))
    plt.plot(thresholds, removal_percentages, marker='o')
    plt.xlabel("Threshold (%)")
    plt.ylabel("Percentage of Counties Removed")
    plt.title(f"Effect of Threshold on County Removal in {year}")
    plt.grid()
    plt.show()




def filter_low_reporting_counties(df, time_column, expected_reports, threshold):
    """
    Removes counties that have reported below a given threshold of expected reports.
    
    :param df: DataFrame containing outage reports
    :param time_column: Column name for timestamps
    :param expected_reports: Expected number of reports per county
    :param threshold: Minimum percentage of expected reports required to keep a county
    :return: Filtered DataFrame
    """
    report_counts = df.groupby('fips_code')[time_column].count()
    reporting_percentage = (report_counts / expected_reports) * 100
    
    valid_fips = reporting_percentage[reporting_percentage >= threshold].index
    filtered_df = df[df['fips_code'].isin(valid_fips)]
    
    return filtered_df



def plot_reporting_distribution(df, time_column, expected_reports):
    """
    Plots the distribution of reporting completeness across counties.
    
    :param df: DataFrame containing outage reports
    :param expected_reports: Expected number of reports per county
    """
    year = pd.to_datetime(df[time_column].iloc[0]).year


    report_counts = df.groupby('fips_code')[time_column].count()
    reporting_percentage = (report_counts / expected_reports) * 100

    plt.figure(figsize=(10, 4))
    plt.hist(reporting_percentage, bins=50, edgecolor='black')
    plt.xlim(0,100)
    plt.xlabel("Percentage of Expected Reports")
    plt.ylabel("Number of Counties")
    plt.title(f"Distribution of Reporting Completeness in {year}")
    plt.show()



def add_valid_data_flag(df, time_col, expected_reports):
    """
    Adds a binary column `'valid_data_flag'`, which indicates whether a given `fips_code` has ≥90% data completeness.  

    **Parameters:**  
    - `df` (*pd.DataFrame*): The original DataFrame containing the data.  
    - `time_col` (*str*): The name of the column containing report timestamps.  
    - `expected_reports` (*int*): The expected number of reports for each `fips_code` per year.  

    **Returns:**  
    - *pd.DataFrame*: A copy of the DataFrame with the added `'valid_data_flag'` column.  
    """
    
    report_counts = df.groupby('fips_code')[time_col].count()
    reporting_percentage = (report_counts / expected_reports) * 100
    valid_data_flag = (reporting_percentage >= 90).astype(int)

    df = df.copy()
    df['valid_data_flag'] = df['fips_code'].map(valid_data_flag)

    return df



def fill_missing_dates(df):
    """
    Ensures that every 'fips_code' has a complete set of daily records for the given year.
    
    Parameters:
    df (pd.DataFrame): Original dataset with columns ['date', 'fips_code', 'county', 'state', 'customers_out', 'valid_data_flag'].

    Returns:
    pd.DataFrame: Updated DataFrame with missing dates filled and 'customers_out' set to 0 for new records.
    """
    year = pd.to_datetime(df['date'].iloc[0]).year

    all_dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')
    unique_fips = df[['fips_code', 'county', 'state', 'valid_data_flag']].drop_duplicates()
    
    full_date_fips = pd.DataFrame(product(unique_fips['fips_code'], all_dates), columns=['fips_code', 'date'])
    full_date_fips['date'] = pd.to_datetime(full_date_fips['date'])
    df['date'] = pd.to_datetime(df['date'])
    full_data = full_date_fips.merge(unique_fips, on='fips_code', how='left')
    
    df = full_data.merge(df, on=['fips_code', 'date', 'county', 'state', 'valid_data_flag'], how='left')
    df['customers_out'] = df['customers_out'].fillna(0)

    return df




def calculate_percent_customers_out(df, mcc):
    """
    Creates a new column `percent_customers_out` in DataFrame `df`, calculated as the ratio of 
    `customers_out` in `df` divided by `Customers` in DataFrame `mcc`, based on matching `fips_code` and `County_FIPS`.
    Any values in `percent_customers_out` greater than 100 are replaced by 100.
    
    Parameters:
    df (pd.DataFrame): The original DataFrame containing the column `customers_out` and `fips_code`.
    mcc (pd.DataFrame): The DataFrame containing the column `Customers` and `County_FIPS`.
    
    Returns:
    pd.DataFrame: The updated DataFrame `df` with the new column `percent_customers_out` and without unnecessary columns.
    """
    
    df['fips_code'] = df['fips_code'].astype(str)
    mcc['County_FIPS'] = mcc['County_FIPS'].astype(str)

    merged_df = df.merge(mcc[['County_FIPS', 'Customers']], left_on='fips_code', right_on='County_FIPS', how='left')
    merged_df['percent_customers_out'] = (merged_df['customers_out'] / merged_df['Customers']) * 100
    merged_df['percent_customers_out'] = merged_df['percent_customers_out'].clip(upper=100)
    merged_df = merged_df.drop(columns=['County_FIPS', 'Customers'])

    return merged_df




def estimate_customers_out(df, mcc):
    """
    Creates a new column `customers_out_estimate` in DataFrame `df`. If `customers_out` is zero,
    it estimates the value based on the mean `percent_customers_out` for the same `date`, `state`.
    The mean is multiplied by the number of `Customers` for the specific `fips_code` from the `mcc` DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing `customers_out`, `percent_customers_out`, and other relevant columns.
    mcc (pd.DataFrame): The DataFrame containing the column `Customers` and `County_FIPS`.
    
    Returns:
    pd.DataFrame: The updated DataFrame `df` with the new column `customers_out_estimate`.
    """
    
    mean_percent = df[df['customers_out'] != 0].groupby(['date', 'state'])['percent_customers_out'].mean().reset_index()
    mean_percent = mean_percent.rename(columns={'percent_customers_out': 'mean_percent_customers_out'})
    
    df = df.merge(mean_percent, on=['date', 'state'], how='left')
    df = df.merge(mcc[['County_FIPS', 'Customers']], left_on='fips_code', right_on='County_FIPS', how='left')
    df['customers_out_estimate'] = df['customers_out']
    
    df.loc[df['customers_out'] == 0, 'customers_out_estimate'] = (
        df.loc[df['customers_out'] == 0, 'mean_percent_customers_out'] * df.loc[df['customers_out'] == 0, 'Customers'] / 100
    )

    df['customers_out_estimate'] = df['customers_out_estimate'].fillna(0)
    df = df.drop(columns=['mean_percent_customers_out', 'County_FIPS', 'Customers'])
    
    return df



def combine_dfs(dfs):
    """
    Combines a list of DataFrames into one large DataFrame by concatenating them vertically.
    
    Parameters:
    dfs (list of pd.DataFrame): A list containing DataFrames to be combined.
    
    Returns:
    pd.DataFrame: A single DataFrame containing all rows from the input DataFrames.
    """
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df



def addGeo(df, geo):
    """
    Adds latitude (lat) and longitude (lng) columns from the geo DataFrame 
    to the df DataFrame based on matching fips_code in df and county_fips in geo.
    
    :param df: DataFrame containing the main data with 'fips_code'
    :param geo: DataFrame containing 'county_fips', 'lat', and 'lng'
    :return: DataFrame with added 'lat' and 'lng' columns
    """
    # Merging df with geo DataFrame on 'fips_code' from df and 'county_fips' from geo
    df_with_geo = pd.merge(df, geo, left_on='fips_code', right_on='county_fips', how='left')
    
    return df_with_geo


def summarize_missing_by_event_type(dfs, names=None):
    summaries = []
    
    for i, df in enumerate(dfs):
        dataset_name = names[i] if names else f"stormEvents_{2014+i}"
        
        # Grupowanie danych po 'event_type' i obliczanie braków dla każdej grupy
        for event_type, group in df.groupby('event_type'):
            missing_count = group['MAGNITUDE'].isna().sum()
            present_count = group['MAGNITUDE'].notna().sum()
            total_count = len(group)
            
            summaries.append({
                'DATASET': dataset_name,
                'EVENT_TYPE': event_type,
                'MISSING_MAGNITUDE': missing_count,
                'PRESENT_MAGNITUDE': present_count,
                'MISSING_PERCENT': round((missing_count / total_count) * 100, 2)
            })
    

def encode_tornado_scale(df):
    """
    Replaces 'EFU' values in the 'TOR_F_SCALE' column with the median of 
    'TOR_F_SCALE' values for rows with the same 'STATE_FIPS'. Keeps other 
    missing values (NaN) unchanged.

    :param df: DataFrame with 'TOR_F_SCALE', 'STATE_FIPS', and 'EVENT_TYPE' columns
    :return: DataFrame with updated 'TOR_F_SCALE' values
    """
    # Map tornado scales to numerical values, converting 'EFU' to -1 temporarily
    mapping = {
        'EF0': 1,
        'EF1': 2,
        'EF2': 3,
        'EF3': 4,
        'EF4': 5,
        'EF5': 6,
        'EFU': -1  # Use -1 to flag 'EFU' for replacement later
    }
    
    # Apply the mapping to convert 'TOR_F_SCALE' values
    df['TOR_F_SCALE'] = df['TOR_F_SCALE'].map(mapping)
    
    # Calculate the median of 'TOR_F_SCALE' grouped by 'STATE_FIPS',
    # excluding rows where 'TOR_F_SCALE' equals -1 (flagged 'EFU') or is NaN.
    state_medians = df[(df["EVENT_TYPE"] == "Tornado") & (df["TOR_F_SCALE"] != -1)].groupby('STATE_FIPS')['TOR_F_SCALE'].median()
    
    # Function to replace 'EFU' (-1) with the median for the corresponding 'STATE_FIPS'
    def replace_efu(row):
        if row["TOR_F_SCALE"] == -1:
            # Use the median for the 'STATE_FIPS', or retain the current value if no median exists
            return state_medians.get(row["STATE_FIPS"], row["TOR_F_SCALE"])
        return row["TOR_F_SCALE"]
    # Apply the replacement function row by row
    df['TOR_F_SCALE'] = df.apply(replace_efu, axis=1)
    
    return df

def fill_magnitude_using_category_and_tor_f_scale(df):
    """
    Fills empty values in the 'MAGNITUDE' column using 'CATEGORY' first, and if 'CATEGORY' 
    is also empty, it uses 'TOR_F_SCALE' to fill the missing value.

    :param df: DataFrame containing the columns 'MAGNITUDE', 'CATEGORY', and 'TOR_F_SCALE'
    :return: DataFrame with updated values in the 'MAGNITUDE' column
    """
    def fill_magnitude(row):
        # Fill 'MAGNITUDE' first using 'CATEGORY', then 'TOR_F_SCALE' if needed
        if pd.isna(row['MAGNITUDE']):
            if not pd.isna(row['CATEGORY']):
                return row['CATEGORY']
            elif not pd.isna(row['TOR_F_SCALE']):
                return row['TOR_F_SCALE']
        return row['MAGNITUDE']
    df = encode_tornado_scale(df)
    # Apply the logic row-wise to fill 'MAGNITUDE'
    df['MAGNITUDE'] = df.apply(fill_magnitude, axis=1)
    
    return df

def create_event_columns(df):
    """
    Creates columns for each unique value in the 'EVENT_TYPE' column.
    For each row:
    - If 'EVENT_TYPE' belongs to the specified set, the column with the name of 'EVENT_TYPE' is filled with 'MAGNITUDE'.
    - Otherwise, the column with the name of 'EVENT_TYPE' is filled with 1.
    In all other columns, the values are set to 0.

    :param df: DataFrame containing 'EVENT_TYPE' and 'MAGNITUDE' columns
    :return: DataFrame with additional columns for each unique 'EVENT_TYPE'
    """

    df = fill_magnitude_using_category_and_tor_f_scale(df)

    # Define the set of EVENT_TYPE values for which MAGNITUDE will be used
    event_type_with_magnitude = {
        "Hail", "High Wind", "Hurricane", "Hurricane (Typhoon)", 
        "Marine Hail", "Marine High Wind", "Marine Strong Wind", 
        "Marine Thunderstorm Wind", "Strong Wind", "Thunderstorm Wind", "Tornado"
    }

    # Iterate through each unique value in the 'EVENT_TYPE' column
    for event in df['EVENT_TYPE'].unique():
        # Initialize the column with default values (0)
        df[event] = 0.0

        # Update column based on conditions
        df.loc[df['EVENT_TYPE'] == event, event] = df.apply(
            lambda row: row['MAGNITUDE'] if row['EVENT_TYPE'] in event_type_with_magnitude else 1,
            axis=1
        )

        columns_to_remove = [
            "TOR_F_SCALE", "CATEGORY"
        ]
    df = df.drop(columns=columns_to_remove)
    
    return df

def convert_damage_property(df):
    """
    Converts the 'DAMAGE_PROPERTY' column in the DataFrame to integer values.
    - 'K' represents thousands (multiplies by 1,000).
    - 'M' represents millions (multiplies by 1,000,000).
    - NaN values are preserved as NaN.

    :param df: DataFrame with a 'DAMAGE_PROPERTY' column
    :return: DataFrame with 'DAMAGE_PROPERTY' converted to integers
    """
    def convert_value(value):
        if pd.isna(value):  # Handle NaN values
            return np.nan
        elif value.endswith('K'):  # Convert 'K' to thousands
            return int(float(value[:-1]) * 1000)
        elif value.endswith('M'):  # Convert 'M' to millions
            return int(float(value[:-1]) * 1000000)
        elif value.endswith('B'):  # Convert 'B' to billions
            return int(float(value[:-1]) * 1000000000)
        else:
            raise ValueError(f"Unexpected format: {value}")

    # Apply the conversion function to the DAMAGE_PROPERTY column
    df['DAMAGE_PROPERTY'] = df['DAMAGE_PROPERTY'].apply(convert_value)
    df['DAMAGE_CROPS'] = df['DAMAGE_CROPS'].apply(convert_value)


    return df

def process_storm_events(df):
    """
    Processes the storm events DataFrame by creating new columns for total damages, 
    total injuries, and percent of deaths. Removes specified columns from the DataFrame.

    :param df: DataFrame containing storm event data with relevant columns
    :return: Modified DataFrame with new calculated columns and removed unwanted columns
    """
    # Calculate total damages by summing DAMAGE_PROPERTY and DAMAGE_CROPS
    df["total_damages"] = df["DAMAGE_PROPERTY"] + df["DAMAGE_CROPS"]
    
    # Calculate total injuries by summing deaths and injuries (direct and indirect)
    df["total_people_injuries"] = (
        df["DEATHS_DIRECT"] + df["DEATHS_INDIRECT"] + 
        df["INJURIES_DIRECT"] + df["INJURIES_INDIRECT"]
    )
    
    # Calculate percent of deaths (avoiding division by zero using pd.NA)
    df["percent_of_deaths"] = (
        (df["DEATHS_DIRECT"] + df["DEATHS_INDIRECT"]) / 
        df["total_people_injuries"].replace(0, pd.NA)
    )
    
    # Remove specified columns
    columns_to_remove = [
        "DAMAGE_PROPERTY", "DAMAGE_CROPS", 
        "DEATHS_DIRECT", "DEATHS_INDIRECT", 
        "INJURIES_DIRECT", "INJURIES_INDIRECT"
    ]
    df = df.drop(columns=columns_to_remove)
    
    return df

def plot_seasonal_data(list_of_datasets, type, ylabel):
    fig, axes = plt.subplots(2, 5, figsize=(15,10))
    for i in range(10):
        k = i % 5
        j = 0
        df = list_of_datasets[i]

        if i > 4:
            j = 1
        if type == 'season_customer':
            column = df.groupby('season')['customers_out'].mean()
        elif type == 'daytime_customer':
            column = df.groupby('daytime')['customers_out'].mean()
        elif type == 'season_catastrophes':
            column = df.groupby('season')['EVENT_TYPE'].count()
        elif type == 'daytime_catastrophes':
            column = df.groupby('daytime')['EVENT_TYPE'].count()

        axes[j, k].bar(column.index, column.values, color=['green', 'yellow', 'orange', 'blue'])
        axes[j, k].set_ylabel(ylabel)
        axes[j, k].set_title(f'Year {i + 2014}')

    plt.tight_layout()
    plt.show()


def plot(list_of_datasets, type, ylabel):
    fig, axes = plt.subplots(2, 5, figsize=(20,12))
    for i in range(10):
        k = i % 5
        j = 0
        df = list_of_datasets[i]
        df['date'] = pd.to_datetime(df['date'])

        if i > 4:
            j = 1
        if type == 'column':
            column = df.groupby('Division')['customers_out'].mean()
            axes[j, k].bar(column.index, column.values, color=['green', 'yellow', 'orange', 'blue', 'red', 'pink', 'purple', 'grey', 'brown'])
            axes[j, k].set_ylabel(ylabel)
            axes[j, k].set_title(f'Year {i + 2014}')
            axes[j, k].tick_params(axis='x', rotation=90) 

        elif type == 'timedata':
            df_grouped = df.groupby(["Division", "date"])["customers_out"].mean().reset_index()
            divisions = ['Pacific', 'Middle Atlantic', 'South Atlantic']
            for division in divisions:
                subset = df_grouped[df_grouped["Division"] == division]
                axes[j, k].plot(subset["date"], subset["customers_out"], 
                     marker="o", linestyle="-", label=division, 
                     linewidth=1, alpha=0.6)
            axes[j, k].set_ylabel(ylabel)
            axes[j, k].set_title(f'Year {i + 2014}')
            axes[j, k].tick_params(axis='x', rotation=90) 

    plt.tight_layout()
    plt.show()

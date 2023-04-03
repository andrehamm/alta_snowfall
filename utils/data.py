import pandas as pd
import numpy as np

def load_alta_data():
    """Load Alta snowfall data"""
    df = pd.read_csv("data/alta_snowfall_history.csv")
    return df

def load_slc_data():
    """Load SLC temperature data"""
    df = pd.read_csv("data/slc_temperature_history.csv")
    return df

def load_agg_data():
    """Aggregate Alta and SLC data"""
    snow_df = load_alta_data()
    temp_df = load_slc_data()
    agg_df = pd.DataFrame([temp_df['Year'], temp_df['Annual'], snow_df['Total']]).T
    agg_df = agg_df.astype({'Year': int})
    agg_df.rename(columns = {'Year': 'year',
                    'Annual': 'max_temp', 
                    'Total': 'snowfall'}, inplace = True)
    agg_df = agg_df.dropna()
    return agg_df


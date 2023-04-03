import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from utils import plot

def predicted_snowfall(reg, x1, x2):
    """predict snowfall with simple linear regression"""
    return reg.intercept_ + (reg.coef_[0]* x1) + (reg.coef_[1] * x2)

def train(agg_df):
    """Train linear regression model and predict snowfall in 2023-2063"""
    # Split data into training and testing sets
    x = agg_df['year']
    y = agg_df['max_temp']
    x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.3, random_state = 0)
    x_train = x_train.values.reshape(-1,1)
    x_test = x_test.values.reshape(-1,1)

    # Train model
    reg = LinearRegression().fit(x_train,y_train)
    yhat = reg.predict(x_test)

    # Predict max temp in SLC 2023-2063
    predicted_years = np.arange(2023,2063)
    d = {'year': predicted_years,'pred_max_temp': np.nan}
    df_f = pd.DataFrame(data = d)

    xhat = predicted_years.reshape(-1,1)
    yhat = reg.predict(xhat).round(2)

    df_f['pred_max_temp'] = yhat

    # Predict average snowfall at Alta, UT with temp and snowfall data
    x = agg_df[['year', 'max_temp']]
    y = agg_df['snowfall']

    reg = LinearRegression().fit(x,y)

    year_list = df_f['year'].values.flatten().tolist()
    temp_list = df_f['pred_max_temp'].values.flatten().tolist()
    snow_list = []

    for year, temp in zip(year_list, temp_list):
        snowfall = predicted_snowfall(reg, year, temp)
        snow_list.append(snowfall)

    df_f['pred_avg_snowfall_in'] = snow_list

    # Add noise to predicted snowfall
    # Modified from https://github.com/MartyC-137/Geoscience_Work_Samples/blob/main/DenverSnowfall_MultipleRegression.ipynb
    n = len(df_f)
    x1 = np.array(year_list)
    x2 = np.array(temp_list)
    y_fit = predicted_snowfall(reg, x1, x2)
    noise = (np.std(agg_df['snowfall'])*2.5) * (np.random.random(n) - 0.5) #I multiplied this by 2.5 because, visually, it seemed to approximate the spread of the data better
    y_ran = y_fit + noise
    df_f['pred_snowfall_in'] = y_ran

    # Plot prediction
    plot.prediction(agg_df, df_f)
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

def histogram(df):
    """Plots a histogram of a column in a pandas dataframe"""
    num_bins = 5
    fig, axs = plt.subplots(1,1, figsize = (10, 7))
    N, bins, bars = axs.hist(df['Total'], num_bins)
    grads = ((N**(1/4))/N.max())
    norm = colors.Normalize(grads.min(), grads.max())
    for gn, pn in zip(grads, bars):
        color = plt.cm.viridis(norm(gn))
        pn.set_facecolor(color)
    for i in range(num_bins):
        plt.text(bins[i], N[i], str(int(N[i])))
    plt.title('Snowfall Histogram for Alta, UT')
    plt.xlabel('Snowfall (in.)')
    plt.ylabel("Seasons")
    plt.show()
    # fig.savefig('visualizations/snowfall_histogram.png')

def boxplot(df):
    """Plots a boxplot of a column in a pandas dataframe"""
    fig = plt.figure(figsize=(10, 5))
    nov = df['Nov.'].dropna().to_numpy()
    dec = df['Dec.'].dropna().to_numpy()
    jan = df['Jan.'].dropna().to_numpy()
    feb = df['Feb.'].dropna().to_numpy()
    mar = df['Mar.'].dropna().to_numpy()
    apr = df['Apr.'].dropna().to_numpy()
    data = [nov, dec, jan, feb, mar, apr]
    ax = fig.add_subplot(111)
    ax.set_xticklabels(['Nov.', 'Dec.', 'Jan.', 'Feb.', 'Mar.', 'Apr.'])
    bp = ax.boxplot(data)
    plt.xlabel("Month")
    plt.ylabel("Snowfall (in.)")
    plt.title("Distribution of snowfall at Alta, UT by month")
    plt.show()
    # fig.savefig('visualizations/monthly_snowfall_boxplot.png')


def scatterplot_snowfall(agg_df):
    """Plots a scatterplot of historic snowfall in Alta, UT, 1946-2022"""
    x = agg_df['year']
    y = agg_df['snowfall']
    smax = agg_df.iloc[agg_df[['snowfall']].idxmax()].astype('int').values.flatten().tolist()
    smin = agg_df.iloc[agg_df[['snowfall']].idxmin()].astype('int').values.flatten().tolist()
    area = [1*n**1.25 for n in y.to_numpy()]
    xticks = np.linspace(1940,2020, 9, dtype = 'int')
    yticks = np.linspace(0,700,6, dtype = 'int')

    fig, ax = plt.subplots(figsize = (10,5))
    m,b = np.polyfit(x,y,1)
    data = ax.scatter(x,y, s = area, c = agg_df.snowfall, cmap = 'winter', alpha = 0.6)
    ax.plot(x, m*x + b, alpha = 0.4)
    ax.set_title('Annual Snowfall in Alta, UT, 1946-2022', fontsize = 24)
    ax.set_xlabel('Year', fontsize = 18)
    ax.set_xticklabels(xticks, fontsize = 16)
    ax.set_ylabel('Snowfall (in)', fontsize = 18)
    ax.set_yticklabels(yticks, fontsize = 16)
    clb = fig.colorbar(data, fraction = 0.1, shrink = 0.7, aspect = 10, pad = 0.01)
    plt.show()
    # fig.savefig('visualizations/scatter_snowfall.png')

def scatterplot_temp(df):
    """Plots a scatterplot of temperature in SLC"""
    x = agg_df['year']
    y = agg_df['max_temp']
    area = [1*n**1.125 for n in y.to_numpy()]
    m, b = np.polyfit(x,y,1)
    xticks = np.linspace(1940,2020, 9, dtype = 'int')
    yticks = np.linspace(90,110,6, dtype = 'int')

    fig, ax = plt.subplots(figsize = (10,5))
    data = ax.scatter(x,y, s = area, c = agg_df.max_temp, cmap = 'autumn', alpha = 0.6)
    ax.plot(x, m*x + b) # original trendline
    ax.set_title('Mean Annual Max Temperature (\N{DEGREE SIGN}F) in SLC, UT, 1948-2020', fontsize = 24)
    ax.set_xlabel('Year', fontsize = 18)
    ax.set_xticklabels(xticks, fontsize = 16)
    ax.set_ylabel('Mean Annual Max Temperature (\N{DEGREE SIGN}F)', fontsize = 18)
    ax.set_yticklabels(yticks, fontsize = 16)
    clb = fig.colorbar(data, fraction = 0.1, shrink = 0.7, aspect = 10, pad = 0.01)
    clb.set_label('Mean Annual Max Temperature (\N{DEGREE SIGN}F) \n 1946-2022', fontsize = 16)
    ax.legend(loc = 'upper left', fontsize = 14)


def prediction(agg_df, df_f):
    """Plots a scatterplot of snowfall prediction in Alta, UT"""
    x = agg_df['year']
    y = agg_df['snowfall']
    x2 = df_f['year']
    y2 = df_f['pred_avg_snowfall_in']
    y3 = df_f['pred_snowfall_in']
    xticks = np.linspace(1940,2060,7, dtype = 'int')
    yticks = np.linspace(0,700,6, dtype = 'int')

    smax = agg_df.iloc[agg_df[['snowfall']].idxmax()].astype('int').values.flatten().tolist()
    smin = agg_df.iloc[agg_df[['snowfall']].idxmin()].astype('int').values.flatten().tolist()
    s48 = agg_df.loc[agg_df['year'] == 1946].astype('int').values.flatten().tolist()
    s_pred_min = df_f.loc[df_f['year'] == 2060].values.flatten().tolist()

    area = [1*n**1.125 for n in y.to_numpy()]
    area2 = [1*n**1.125 for n in y3.to_numpy()]

    fig, ax = plt.subplots(figsize = (10,5))
    m,b = np.polyfit(x,y,1)
    hdata = ax.scatter(x,y, s = area, c = agg_df.snowfall, cmap = 'winter', alpha = 0.6)
    mdata = ax.scatter(x2,df_f['pred_snowfall_in'], s = area2, c= df_f.pred_snowfall_in, cmap = 'autumn', alpha = 0.6)

    ax.plot(x, m*x + b, alpha = 0.4)
    ax.plot(x2, y2, label = 'Model Output') #not sure what else to call this other than 'model output', I'm sure there is a better term, though...
    ax.set_title('Alta, UT, Historical and Modeled Snowfall, 1968-2060', fontsize = 24)
    ax.set_xlabel('Year', fontsize = 16)
    ax.set_xticklabels(xticks, fontsize = 16)
    ax.set_ylabel('Snowfall (in)', fontsize = 16)
    ax.set_yticklabels(yticks, fontsize = 16)
    clbm = fig.colorbar(mdata, fraction = 0.1, shrink = 0.7, aspect = 10, pad = 0.03)
    clbh = fig.colorbar(hdata, fraction = 0.1, shrink = 0.7, aspect = 10, pad = 0.01)
    clbh.set_label('Snowfall (in) \n 1946-2022', fontsize = 16)
    clbm.set_label('Modeled Snowfall (in) \n 2023-2063', fontsize = 16)
    ax.legend()
    plt.show()
    # fig.savefig('visualizations/snowfall_prediction.png')
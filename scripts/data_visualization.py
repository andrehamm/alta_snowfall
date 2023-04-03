import argparse
from utils import data, model, plot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize-data", action="store_true", help="Visualize data")
    parser.add_argument("--train-predict", action="store_true", help="train linear regression model")
    options = parser.parse_args()
    
    alta_df = data.load_alta_data()
    agg_df = data.load_agg_data()

    if options.visualize_data:
        plot.histogram(alta_df)
        plot.boxplot(alta_df)
        plot.scatterplot_snowfall(agg_df)

    if options.train_predict:
        model.train(agg_df)


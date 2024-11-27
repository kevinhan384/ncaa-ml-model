import pandas as pd
import numpy as np

# school_df = pd.read_csv("23-24-school.csv")
# opp_df = pd.read_csv("23-24-opp.csv")

# concat_df = pd.concat([school_df, opp_df], axis=1).T.drop_duplicates().T

# concat_df.drop(concat_df.columns[concat_df.columns.str.contains(
#     'unnamed', case=False)], axis=1, inplace=True)

# concat_df = concat_df[concat_df['School'].str.endswith("NCAA")]

# concat_df.to_csv("23-24-combined.csv", index=False)

def threshold():
    file_path = "data/23-24-combined.csv"
    data = pd.read_csv(file_path)

    numerical_columns = data.columns[2:len(data.columns) - 1]

    def assign_threshold(value, quantiles):
        if value <= quantiles[0.33]:
            return "low"
        elif value <= quantiles[0.66]:
            return "medium"
        else:
            return "high"

    thresholded_data = data.copy()
    for column in numerical_columns:
        if data[column].dtype in [np.float64, np.int64]:
            quantiles = data[column].quantile([0.33, 0.66])
            thresholded_data[column] = data[column].apply(assign_threshold, args=(quantiles,))

    output_path = "data/thresholded_output.csv"
    thresholded_data.to_csv(output_path, index=False)

threshold()
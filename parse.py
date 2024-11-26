import pandas as pd

school_df = pd.read_csv("14-15-school.csv")
opp_df = pd.read_csv("14-15-opp.csv")

concat_df = pd.concat([school_df, opp_df], axis=1).T.drop_duplicates().T

concat_df.drop(concat_df.columns[concat_df.columns.str.contains(
    'unnamed', case=False)], axis=1, inplace=True)

concat_df = concat_df[concat_df['School'].str.endswith("NCAA")]

concat_df.to_csv("14-15-combined.csv", index=False)


import pandas as pd
df = pd.read_csv("dataset_n.csv")
print(df["dataset"].tolist()[1012])
print(df.columns)

# unique_genres = df["Genre"].unique()



# # Print the unique values
# print(unique_genres[:20])
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    df = pd.read_csv("/data/HAM10000_metadata.csv").sample(frac=1, random_state=27)
    self.df_labels = df[["dx"]]
    self.df = df.drop(columns=["lesion_id", "dx"])

from zipfile import Path

import pandas as pd

def df_to_csv_no_index (df: pd.DataFrame, name: str) -> None:
    df.to_csv(f'training_data/{name}.csv', sep="\t", index=False)

def df_to_csv_index (df: pd.DataFrame, name: str) -> None:
    df.to_csv(f'training_data/{name}.csv', sep="\t", index=True)

def load_file (name: str, separator: str) -> pd.DataFrame:
    df = pd.read_csv(f'training_data/{name}', sep=separator)
    return df


import pandas as pd


def load_splits(train_path, val_path, test_path, concat_train_val=False):
    """
    Load time-split parquet files, ensuring t_dat is datetime.

    If concat_train_val is True, returns a 2-tuple: (train_val, test).
    Otherwise, returns a 3-tuple: (train, val, test).
    """
    train_df, val_df, test_df = [
        pd.read_parquet(p).assign(t_dat=lambda df: pd.to_datetime(df["t_dat"]))
        for p in (train_path, val_path, test_path)
    ]

    if concat_train_val:
        train_val_df = pd.concat([train_df, val_df], ignore_index=True)
        return train_val_df, test_df

    return train_df, val_df, test_df
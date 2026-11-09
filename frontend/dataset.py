"""Đọc dữ liệu dùng chung cho web và lớp nối backend.

Hai nguồn:
  * load_full()    dữ liệu thật của repo (../data/split/*.parquet + ../data/cleaned_art.parquet), chỉ kênh online.
  * load_sample()  bộ nhỏ trong frontend/data/ (24 sản phẩm) để dev giao diện / chạy test nhanh.

Cả hai trả về (transactions, articles) cùng định dạng:
  transactions: customer_id (int), article_id (str 10 số), t_dat (datetime64), price (float)
  articles    : article_id (str 10 số) + các cột mô tả của H&M
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FULL_DATA_DIR = Path(os.environ.get("STYLEREC_DATA", REPO / "data"))
SAMPLE_DATA_DIR = HERE / "data"
ONLINE = 2
SPLITS = ("train", "val", "test")


def load_full(data_dir: Path | str = FULL_DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    tx = pd.concat([pd.read_parquet(data_dir / "split" / f"{s}.parquet") for s in SPLITS], ignore_index=True)
    tx = tx.loc[tx["sales_channel_id"] == ONLINE, ["customer_id", "article_id", "t_dat", "price"]]
    tx["t_dat"] = pd.to_datetime(tx["t_dat"])
    tx = tx.sort_values("t_dat", kind="mergesort").reset_index(drop=True)
    art = pd.read_parquet(data_dir / "cleaned_art.parquet")
    art["article_id"] = art["article_id"].astype(str).str.zfill(10)
    return tx, art


def load_sample(data_dir: Path | str = SAMPLE_DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    tx = pd.read_csv(data_dir / "transactions.csv", dtype={"article_id": str})
    tx["customer_id"] = tx["customer_id"].astype("int64")
    tx["article_id"] = tx["article_id"].str.zfill(10)
    tx["t_dat"] = pd.to_datetime(tx["t_dat"])
    art = pd.read_csv(data_dir / "articles.csv", dtype=str).fillna("")
    art["article_id"] = art["article_id"].str.zfill(10)
    return tx.sort_values("t_dat", kind="mergesort").reset_index(drop=True), art


def files_signature(data_dir: Path | str = FULL_DATA_DIR) -> str:
    """Chuỗi đổi khi file dữ liệu đổi (dùng làm khoá cache model)."""
    data_dir = Path(data_dir)
    files = [data_dir / "split" / f"{s}.parquet" for s in SPLITS] + [data_dir / "cleaned_art.parquet"]
    return "|".join(f"{f.name}:{f.stat().st_size}:{int(f.stat().st_mtime)}" for f in files if f.exists())

"""Cắt một bộ dữ liệu demo nhỏ từ parquet H&M thật -> frontend/data/.

Chạy (một lần, kết quả đã commit sẵn nên bình thường không cần chạy lại):
    python tools/build_sample_data.py --parquet-dir ../../UELML_CK_FLASKPROJ/data/parquet

Cách chọn:
  * Chỉ kênh online (sales_channel_id == 2).
  * Lấy 300 sản phẩm bán chạy nhất 12 tuần cuối, đếm số giỏ (khách, ngày) có cặp sản phẩm KHÁC loại,
    rồi chọn tham lam các cặp mua kèm mạnh nhất cho tới khi đủ N sản phẩm (mỗi loại tối đa 2, mỗi nhóm hàng tối đa 6, mỗi product_code 1;
    phần còn lại lấp bằng hàng bán chạy).
    Nhờ vậy gợi ý mua kèm (thuật toán B) trên bộ nhỏ vẫn có tín hiệu thật.
  * Giữ toàn bộ giao dịch online 52 tuần cuối của các sản phẩm đã chọn.
  * Chọn khách mẫu: mua ít (1–4 lượt mua) và mua nhiều (>= 5 lượt), giống phân khúc trong validation/, ưu tiên khách hoạt động gần đây.

Đầu ra:
    data/articles.csv       danh mục sản phẩm (article_id dạng chuỗi 10 chữ số)
    data/transactions.csv   customer_id, article_id, t_dat, price
    data/demo_users.json    danh sách khách mẫu theo tình huống
"""
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).resolve().parents[1] / "data"
ONLINE = 2
ART_COLS = ["article_id", "product_code", "prod_name", "product_type_name", "product_group_name",
            "colour_group_name", "graphical_appearance_name", "index_group_name", "section_name",
            "garment_group_name", "detail_desc"]


def main(parquet_dir: Path, n_items: int, max_per_type: int, max_per_group: int, weeks: int):
    tx = pd.read_parquet(parquet_dir / "transactions.parquet")
    art = pd.read_parquet(parquet_dir / "articles.parquet")
    tx = tx[tx["sales_channel_id"] == ONLINE].copy()
    tx["t_dat"] = pd.to_datetime(tx["t_dat"])
    tx["article_id"] = tx["article_id"].astype(str).str.zfill(10)
    tx["customer_id"] = tx["customer_id"].astype(str)
    art["article_id"] = art["article_id"].astype(str).str.zfill(10)
    last = tx["t_dat"].max()

    art = art[~art["product_type_name"].isin(["Unknown"])]
    recent = tx[(tx["t_dat"] > last - pd.Timedelta(weeks=12)) & tx["article_id"].isin(art["article_id"])]
    top = recent["article_id"].value_counts().head(500)
    r = recent[recent["article_id"].isin(top.index)][["customer_id", "t_dat", "article_id"]].drop_duplicates()
    r = r.merge(art[["article_id", "product_type_name"]], on="article_id")
    pairs = r.merge(r, on=["customer_id", "t_dat"])
    pairs = pairs[(pairs["article_id_x"] < pairs["article_id_y"])
                  & (pairs["product_type_name_x"] != pairs["product_type_name_y"])]
    cnt = pairs.groupby(["article_id_x", "article_id_y"]).size().sort_values(ascending=False)

    info = art.set_index("article_id")
    chosen: list[str] = []
    per_type: dict[str, int] = {}
    per_group: dict[str, int] = {}
    codes: set[str] = set()

    def ok(a):
        t, g = info.at[a, "product_type_name"], info.at[a, "product_group_name"]
        return a in chosen or (per_type.get(t, 0) < max_per_type and per_group.get(g, 0) < max_per_group
                               and info.at[a, "product_code"] not in codes)

    def add(a):
        chosen.append(a)
        t, g = info.at[a, "product_type_name"], info.at[a, "product_group_name"]
        per_type[t] = per_type.get(t, 0) + 1
        per_group[g] = per_group.get(g, 0) + 1
        codes.add(info.at[a, "product_code"])

    pair_target = n_items - n_items // 4          # ~3/4 từ các cặp mua kèm, phần còn lại là hàng bán chạy
    for (a, b), _ in cnt.items():
        if len(chosen) >= pair_target:
            break
        new = [x for x in (a, b) if x not in chosen]
        if new and all(ok(x) for x in (a, b)) and len(chosen) + len(new) <= pair_target:
            for x in new:
                add(x)
    for a in top.index:                            # lấp cho đủ bằng hàng bán chạy, vẫn giữ đa dạng
        if len(chosen) >= n_items:
            break
        if a not in chosen and ok(a):
            add(a)

    sample = tx[tx["article_id"].isin(chosen) & (tx["t_dat"] > last - pd.Timedelta(weeks=weeks))]
    sample = sample[["customer_id", "article_id", "t_dat", "price"]].sort_values(["t_dat", "customer_id"])

    # khách mẫu: theo số lượt mua như validation (light 1-4, heavy 5+)
    st = sample.groupby("customer_id").agg(n=("article_id", "size"), last=("t_dat", "max"))
    st["multi"] = sample.groupby(["customer_id", "t_dat"])["article_id"].nunique().groupby("customer_id").max()
    active = st[st["last"] > last - pd.Timedelta(days=45)].sort_values(["multi", "last"], ascending=False)
    light = active[(active["n"] >= 2) & (active["n"] <= 4) & (active["multi"] >= 2)].head(3).index.tolist()
    heavy = st[st["n"] >= 5].sort_values(["n", "last"], ascending=False).head(3).index.tolist()

    users = [dict(id=c, name=f"Khách mua ít {chr(65 + i)}", scenario="light", n_purchases=int(st.at[c, "n"]))
             for i, c in enumerate(light)]
    users += [dict(id=c, name=f"Khách mua nhiều {chr(65 + i)}", scenario="heavy", n_purchases=int(st.at[c, "n"]))
              for i, c in enumerate(heavy)]

    OUT.mkdir(exist_ok=True)
    a_out = art[art["article_id"].isin(chosen)][ART_COLS].copy()
    a_out["article_id"] = pd.Categorical(a_out["article_id"], chosen, ordered=True)
    a_out.sort_values("article_id").to_csv(OUT / "articles.csv", index=False)
    sample.assign(t_dat=sample["t_dat"].dt.strftime("%Y-%m-%d")).to_csv(OUT / "transactions.csv", index=False)
    (OUT / "demo_users.json").write_text(json.dumps(users, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{len(chosen)} sản phẩm, {len(sample):,} giao dịch, {sample['customer_id'].nunique():,} khách")
    print(a_out[["article_id", "prod_name", "product_type_name", "colour_group_name"]].to_string(index=False))
    print(json.dumps(users, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parquet-dir", type=Path, required=True)
    p.add_argument("--n-items", type=int, default=24)
    p.add_argument("--max-per-type", type=int, default=2)
    p.add_argument("--max-per-group", type=int, default=6)
    p.add_argument("--weeks", type=int, default=52)
    a = p.parse_args()
    main(a.parquet_dir, a.n_items, a.max_per_type, a.max_per_group, a.weeks)

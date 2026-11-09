"""Chọn khách mẫu cho demo từ dữ liệu THẬT của repo (online, train+val+test) -> data/demo_users.json.

Chạy:  cd frontend && python tools/pick_demo_users.py
(Dùng ProductionRecommender nên lần đầu mất ~1 phút để fit / lần sau đọc cache.)

Phân khúc theo đúng utils/eval_tools.py: số dòng giao dịch trong lịch sử; khách mới = 0, mua ít = 1–4, mua nhiều = 5+.
Chọn để demo đủ 3 bài toán:
  * Khách mua ít       : 2–4 lượt, mua gần đây, có gợi ý "vì bạn đã mua".
  * Khách hay mua lại  : >= 8 lượt, nhiều món mua lặp lại ở các ngày khác nhau.
  * Khách thích khám phá: >= 8 lượt, nhiều loại sản phẩm khác nhau, gần như không mua lặp.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd  # noqa: E402

import dataset  # noqa: E402
from production_recommender import ProductionRecommender  # noqa: E402

OUT = dataset.SAMPLE_DATA_DIR / "demo_users.json"


def main():
    tx, art = dataset.load_full()
    rec = ProductionRecommender(tx, art)
    last = tx["t_dat"].max()
    types = art.set_index("article_id")["product_type_no"]
    t = tx.assign(type=tx["article_id"].map(types))
    days = t.groupby(["customer_id", "article_id"])["t_dat"].nunique()
    st = pd.DataFrame({
        "n": t.groupby("customer_id").size(),
        "last": t.groupby("customer_id")["t_dat"].max(),
        "n_types": t.groupby("customer_id")["type"].nunique(),
        "repeats": (days >= 2).groupby("customer_id").sum(),
    })
    st = st[st["last"] >= last - pd.Timedelta(days=21)]

    def because_count(cid):
        return sum(1 for r in rec.recommend_for_user(str(cid), k=12) if r.because)

    def pick(cands, need_because=1):
        for cid in cands.index[:300]:
            if because_count(cid) >= need_because:
                return int(cid)
        return int(cands.index[0])

    light = pick(st[(st["n"] >= 2) & (st["n"] <= 4)].sort_values(["n_types", "last"], ascending=False))
    repeat = pick(st[(st["n"] >= 8) & (st["n"] <= 60)].sort_values(["repeats", "last"], ascending=False), 0)
    explore = pick(st[(st["n"] >= 8) & (st["n"] <= 60) & (st["repeats"] <= 1)]
                   .sort_values(["n_types", "last"], ascending=False), 3)

    users = [
        dict(id=str(light), name="Khách mua ít", segment="light",
             note=f"{st.at[light, 'n']} lượt mua · gợi ý dựa trên vài món đầu tiên"),
        dict(id=str(repeat), name="Khách hay mua lại", segment="heavy",
             note=f"{st.at[repeat, 'n']} lượt mua · {st.at[repeat, 'repeats']} món mua lặp lại"),
        dict(id=str(explore), name="Khách thích khám phá", segment="heavy",
             note=f"{st.at[explore, 'n']} lượt mua · {st.at[explore, 'n_types']} loại sản phẩm khác nhau"),
    ]
    OUT.write_text(json.dumps(users, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(users, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

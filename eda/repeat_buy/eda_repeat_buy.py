"""EDA bài toán Buy again (Repeat Buy): tái tạo 4 hình của thư mục này.

Chỉ dùng tập train (không chạm val/test):
  * cửa sổ đích  = 14 ngày cuối của train (12/08–25/08/2020)
  * history      = mọi giao dịch trước 12/08/2020
  * chỉ xét user vừa có history vừa có mua trong cửa sổ đích

Hình 1  tỷ lệ user / cặp (user, item) có mua lại theo 3 mức: article_id, product_code, product_type_no
Hình 2  phân phối tích lũy khoảng cách từ lần mua cuối trong history đến lần mua lại đầu tiên trong cửa sổ
Hình 3  repeat rate của các cặp trong history theo số dòng giao dịch (A) và số ngày từ lần mua cuối (B)
Hình 4  tỷ lệ user có mua lại: nhóm "đã từng mua lại" (có item >= 2 dòng giao dịch) và nhóm còn lại

Các hình fig1–fig4 *.png gốc được tính trên cả 2 kênh bán (--channel all). Mô hình chỉ dùng kênh online, nên
mặc định script vẽ cho kênh online (--channel online) vào thư mục figures_online/, kèm số liệu 2 kênh để đối chiếu.

Chạy từ thư mục gốc repo:
    python eda/repeat_buy/eda_repeat_buy.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.load_splits import load_splits  # noqa: E402

HERE = Path(__file__).resolve().parent
KEY = ["customer_id", "article_id"]
LEVELS = ["article_id", "product_code", "product_type_no"]
AGE_BINS = [-1, 7, 30, 90, 180, 1e9]
AGE_LABELS = ["0-7", "8-30", "31-90", "91-180", ">180"]


def compute(train: pd.DataFrame, art: pd.DataFrame) -> dict:
    cut = train.t_dat.max() - pd.Timedelta(days=13)
    hist, tgt = train[train.t_dat < cut], train[train.t_dat >= cut]
    warm = tgt[tgt.customer_id.isin(set(hist.customer_id.unique()))]
    res = {"cut": str(cut.date()), "end": str(train.t_dat.max().date()), "n_users": int(warm.customer_id.nunique())}

    fig1 = {}
    for lvl in LEVELS:
        h, v = hist[KEY].copy(), warm[KEY].copy()
        if lvl != "article_id":
            h[lvl] = h.article_id.map(art[lvl]); v[lvl] = v.article_id.map(art[lvl])
        seen = h[["customer_id", lvl]].drop_duplicates().assign(s=1)
        vu = v[["customer_id", lvl]].drop_duplicates().merge(seen, how="left", on=["customer_id", lvl])
        fig1[lvl] = [round(vu[vu.s.notna()].customer_id.nunique() / v.customer_id.nunique() * 100, 1),
                     round(vu.s.notna().mean() * 100, 1)]
    res["fig1"] = fig1

    last = hist.groupby(KEY).t_dat.max().rename("last").reset_index()
    first = tgt.groupby(KEY).t_dat.min().rename("first").reset_index()
    pairs = first.merge(last, on=KEY)
    res["fig2_gaps"] = (pairs["first"] - pairs["last"]).dt.days.tolist()

    pr = (hist[hist.customer_id.isin(warm.customer_id.unique())]
          .groupby(KEY).agg(last=("t_dat", "max"), n=("t_dat", "count")).reset_index())
    pr = pr.merge(tgt[KEY].drop_duplicates().assign(y=1), how="left", on=KEY)
    pr["y"] = pr.y.fillna(0)
    age = pd.cut((cut - pr["last"]).dt.days, AGE_BINS, labels=AGE_LABELS)
    res["fig3_age"] = (pr.groupby(age, observed=True).y.mean() * 100).round(2).to_dict()
    res["fig3_n"] = {str(k): v for k, v in (pr.groupby(pr.n.clip(upper=3)).y.mean() * 100).round(2).items()}

    tp = warm[KEY].drop_duplicates().merge(hist[KEY].drop_duplicates().assign(s=1), how="left", on=KEY)
    tp["rep"] = tp.s.notna()
    repeaters = set(hist.groupby(KEY).size().loc[lambda s: s >= 2].reset_index().customer_id)
    tp["hr"] = tp.customer_id.isin(repeaters)
    u = tp.groupby("customer_id").agg(rep=("rep", "max"), hr=("hr", "first"))
    rate = (u.groupby("hr").rep.mean() * 100).round(1)
    res["fig4"] = {"non_repeater": float(rate[False]), "repeater": float(rate[True])}
    return res


def plot(main: dict, ref: dict, out: Path, main_label="online", ref_label="cả 2 kênh"):
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    blue, light, orange, lorange = "#2b6cb0", "#a9c4e6", "#dd8a2e", "#f2c99a"
    pct = lambda v: f"{v:.1f}%".replace(".", ",")  # noqa: E731

    fig, ax = plt.subplots(figsize=(7.4, 3.6)); x = np.arange(3); w = .2
    for j, (src, cu, cp, lab) in enumerate([(main, blue, orange, main_label), (ref, light, lorange, ref_label)]):
        u = [src["fig1"][lv][0] for lv in LEVELS]; p = [src["fig1"][lv][1] for lv in LEVELS]
        for vals, col, off, name in ((u, cu, -1.5, "user"), (p, cp, -0.5, "cặp (user, item)")):
            bars = ax.bar(x + (j * 2 + off) * w, vals, w, color=col, label=f"Tỷ lệ {name} ({lab})")
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 1, pct(v), ha="center", fontsize=7)
    ax.set_xticks(x, LEVELS); ax.set_ylabel("%"); ax.set_ylim(0, 92)
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
    fig.tight_layout(); fig.savefig(out / "fig1_repeat_share.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    for src, col, lab, lw in [(main, blue, main_label, 2), (ref, "#9aa5b1", ref_label, 1.3)]:
        g = np.sort(np.array(src["fig2_gaps"])); y = np.arange(1, len(g) + 1) / len(g) * 100
        ax.step(g, y, where="post", color=col, lw=lw, label=lab)
    gaps = np.array(main["fig2_gaps"])
    for x0 in (7, 30, 90, 180):
        v = (gaps <= x0).mean() * 100
        ax.axvline(x0, color="#dd6b6b", ls="--", lw=.8); ax.plot(x0, v, "o", color="#c0392b", ms=4)
        ax.text(x0 + 3, v - 6, pct(v), color="#c0392b", fontsize=8)
    ax.set_xlim(0, 200); ax.set_ylim(0, 100)
    ax.set_xlabel("Số ngày từ lần mua cuối trong history đến lần mua lại đầu tiên trong cửa sổ đích")
    ax.set_ylabel("% cặp được mua lại (tích lũy)"); ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "fig2_repeat_timing.png", dpi=200); plt.close(fig)

    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3.3))
    for ax, key, labs, ticks, title in [
            (a, "fig3_n", ["1", "2", "3"], ["1", "2", "≥3"], "A. Theo số dòng giao dịch đã có"),
            (b, "fig3_age", AGE_LABELS, [s.replace("-", "–") for s in AGE_LABELS], "B. Theo số ngày từ lần mua cuối")]:
        vm = [main[key][k] for k in labs]; vr = [ref[key][k] for k in labs]; xx = np.arange(len(labs))
        ax.bar(xx - .18, vm, .36, color=blue, label=main_label); ax.bar(xx + .18, vr, .36, color=light, label=ref_label)
        for i, v in enumerate(vm):
            ax.text(i - .18, v + max(vm) * .02, f"{v:.2f}".replace(".", ","), ha="center", fontsize=7)
        ax.set_xticks(xx, ticks); ax.set_title(title, fontsize=10); ax.set_ylabel("Repeat rate (%)")
    a.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig3_pair_repeat_rate.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 3.3)); xx = np.arange(2)
    vm = [main["fig4"]["non_repeater"], main["fig4"]["repeater"]]
    vr = [ref["fig4"]["non_repeater"], ref["fig4"]["repeater"]]
    ax.bar(xx - .18, vm, .36, color=blue, label=main_label); ax.bar(xx + .18, vr, .36, color=light, label=ref_label)
    for i in range(2):
        ax.text(i - .18, vm[i] + .3, pct(vm[i]), ha="center", fontsize=8)
        ax.text(i + .18, vr[i] + .3, pct(vr[i]), ha="center", fontsize=8)
    ax.set_xticks(xx, ["Chưa từng mua lại", "Đã từng mua lại"]); ax.set_ylabel("% user có mua lại trong cửa sổ đích")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig4_user_repeat_propensity.png", dpi=200); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(HERE / "figures_online"))
    args = parser.parse_args()

    train, _, _ = load_splits(train_path=ROOT / "data/split/train.parquet", val_path=ROOT / "data/split/val.parquet",
                              test_path=ROOT / "data/split/test.parquet")
    art = pd.read_parquet(ROOT / "data/cleaned_art.parquet").set_index("article_id")
    online = compute(train[train.sales_channel_id == 2], art)
    both = compute(train, art)
    out = Path(args.out)
    plot(online, both, out)
    summary = {name: {k: v for k, v in r.items() if k != "fig2_gaps"} for name, r in (("online", online), ("all", both))}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

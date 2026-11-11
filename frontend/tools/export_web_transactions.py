"""Xuất lượt mua trên web (CSDL runtime/stylerec.db) sang định dạng dữ liệu huấn luyện, để gộp vào train định kỳ.

    cd frontend
    python tools/export_web_transactions.py                 # ghi thêm vào ../data/web_transactions.parquet
    python tools/export_web_transactions.py --dry-run       # chỉ xem, không ghi, không đánh dấu

Mỗi dòng: customer_id, t_dat (ngày mô phỏng nối tiếp sau dữ liệu), article_id, price (thang của dataset),
sales_channel_id=2 — cùng cột với data/split/*.parquet. Dòng đã xuất được đánh dấu để lần sau không xuất lại.
Huấn luyện lại: gộp file này với dữ liệu gốc rồi chạy orch_architecture/build_pickle.py (việc của phía backend).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd  # noqa: E402

import dataset  # noqa: E402
from store import Store  # noqa: E402


def main(runtime: Path, out: Path, dry_run: bool):
    store = Store(runtime)
    new = store.export_web_transactions(mark=not dry_run)
    print(f"{len(new)} lượt mua mới trên web cần xuất")
    if new.empty or dry_run:
        print(new.head(20).to_string(index=False) if not new.empty else "")
        return
    if out.exists():
        new = pd.concat([pd.read_parquet(out), new], ignore_index=True)
    new.to_parquet(out, index=False)
    print(f"Đã ghi {out} (tổng {len(new)} dòng)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", type=Path, default=dataset.HERE / "runtime")
    p.add_argument("--out", type=Path, default=dataset.FULL_DATA_DIR / "web_transactions.parquet")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    main(a.runtime, a.out, a.dry_run)

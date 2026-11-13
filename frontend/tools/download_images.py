"""Tải hình ảnh sản phẩm từ Kaggle H&M Dataset vào frontend/static/img/products/

Chạy:
    cd frontend
    python tools/download_images.py              # Tự động tải ảnh cho demo (khách mẫu, trending, sample) + top 100 bán chạy
    python tools/download_images.py --top 200    # Tải thêm top 200 bán chạy
    python tools/download_images.py --ids 0706016001,0399223001
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd  # noqa: E402
import dataset  # noqa: E402

COMPETITION = "h-and-m-personalized-fashion-recommendations"
OUT_DIR = dataset.HERE / "static" / "img" / "products"


def get_demo_and_popular_ids(top_n: int = 100) -> list[str]:
    """Tổng hợp danh sách các article_id cần thiết nhất cho website."""
    needed = set()

    # 1. Các sản phẩm mẫu trong frontend/data/articles.csv
    sample_art_file = dataset.SAMPLE_DATA_DIR / "articles.csv"
    if sample_art_file.exists():
        df_sample = pd.read_csv(sample_art_file, dtype={"article_id": str})
        for aid in df_sample["article_id"].dropna():
            needed.add(str(aid).zfill(10))

    # 2. Các sản phẩm trong lịch sử của demo users & top bán chạy
    demo_file = dataset.SAMPLE_DATA_DIR / "demo_users.json"
    demo_cids = set()
    if demo_file.exists():
        with open(demo_file, "r", encoding="utf-8") as f:
            demo_data = json.load(f)
        for user in demo_data:
            cid = user.get("id") or user.get("customer_id")
            if cid:
                try:
                    demo_cids.add(int(cid))
                except ValueError:
                    pass

    tx_file = dataset.REPO / "data" / "transactions.parquet"
    if tx_file.exists():
        print("Đang đọc lịch sử demo và top sản phẩm bán chạy từ transactions.parquet...")
        df_tx = pd.read_parquet(tx_file, columns=["customer_id", "article_id", "sales_channel_id"])
        df_tx = df_tx[df_tx["sales_channel_id"] == 2]

        # Lấy sản phẩm của demo users
        demo_tx = df_tx[df_tx["customer_id"].isin(demo_cids)]
        for aid in demo_tx["article_id"].dropna().unique():
            needed.add(str(aid).zfill(10))

        # Lấy top bán chạy
        top_popular = df_tx["article_id"].value_counts().head(top_n).index
        for aid in top_popular:
            needed.add(str(aid).zfill(10))
    elif (dataset.SAMPLE_DATA_DIR / "transactions.csv").exists():
        df_tx = pd.read_csv(dataset.SAMPLE_DATA_DIR / "transactions.csv", dtype={"article_id": str})
        top_popular = df_tx["article_id"].value_counts().head(top_n).index
        for aid in top_popular:
            needed.add(str(aid).zfill(10))

    return sorted(needed)


def download_one(api, aid: str, out_dir: Path) -> tuple[str, bool, str]:
    aid = str(aid).zfill(10)
    target_file = out_dir / f"{aid}.jpg"
    if target_file.exists() and target_file.stat().st_size > 0:
        return aid, True, "already exists"

    sub_dir = aid[:3]
    remote_path = f"images/{sub_dir}/{aid}.jpg"

    try:
        api.competition_download_file(COMPETITION, remote_path, path=str(out_dir), quiet=True)
        # Kaggle tải về theo cấu trúc out_dir/images/<sub_dir>/<aid>.jpg
        nested = out_dir / "images" / sub_dir / f"{aid}.jpg"
        if nested.exists():
            shutil.copy2(nested, target_file)
        return aid, True, "downloaded"
    except Exception as e:
        return aid, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Tải ảnh sản phẩm H&M từ Kaggle")
    parser.add_argument("--top", type=int, default=100, help="Số lượng sản phẩm bán chạy cần tải")
    parser.add_argument("--ids", type=str, default="", help="Danh sách article_id cụ thể, cách nhau dấu phẩy")
    parser.add_argument("--workers", type=int, default=4, help="Số luồng tải song song")
    args = parser.parse_args()

    # Kiểm tra Kaggle API
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"[LỖI] Không thể kết nối Kaggle API: {e}")
        print("Vui lòng kiểm tra file ~/.kaggle/kaggle.json hoặc token xác thực.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.ids:
        aids = [x.strip().zfill(10) for x in args.ids.split(",") if x.strip()]
    else:
        aids = get_demo_and_popular_ids(top_n=args.top)

    print(f"Bắt đầu xử lý {len(aids)} sản phẩm (thư mục lưu: {OUT_DIR})...")
    t0 = time.time()
    success_count = 0
    skip_count = 0
    fail_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_aid = {executor.submit(download_one, api, aid, OUT_DIR): aid for aid in aids}
        for future in concurrent.futures.as_completed(future_to_aid):
            aid, ok, msg = future.result()
            if ok:
                if msg == "already exists":
                    skip_count += 1
                else:
                    success_count += 1
                    print(f"  [+] Đã tải: {aid}.jpg")
            else:
                fail_count += 1

    total_images = len(list(OUT_DIR.glob("*.jpg")))
    print("\n" + "=" * 50)
    print(f"Hoàn thành trong {time.time() - t0:.1f}s!")
    print(f"- Đã tải mới: {success_count}")
    print(f"- Đã có sẵn: {skip_count}")
    print(f"- Không tìm thấy: {fail_count}")
    print(f"- Tổng số ảnh hiện có trong thư mục: {total_images} ảnh")
    print("=" * 50)


if __name__ == "__main__":
    main()

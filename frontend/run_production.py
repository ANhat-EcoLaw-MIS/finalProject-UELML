"""Chạy web với dữ liệu THẬT của repo + model backend (ProductionRecommender).

    cd frontend
    python run_production.py              # http://127.0.0.1:5000

Lần đầu fit model mất ~1–1,5 phút rồi lưu runtime/model_cache.pkl; các lần sau chỉ đọc cache (~20 giây).
Biến môi trường: STYLEREC_DATA (thư mục data của repo), STYLEREC_RUNTIME, STYLEREC_SECRET, PORT.
"""
import logging
import os
import warnings

import dataset
from app import create_app
from catalog import Catalog
from production_recommender import ProductionRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
warnings.filterwarnings("ignore", category=FutureWarning)

tx, art = dataset.load_full()                         # đọc dữ liệu một lần, dùng chung cho web + recommender
catalog = Catalog(tx, art)
app = create_app(ProductionRecommender(tx, art), catalog=catalog)   # tiêm recommender thật vào web

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False, threaded=True)

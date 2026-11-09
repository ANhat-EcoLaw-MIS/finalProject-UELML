"""Lưu dữ liệu chạy của web bằng file (demo): tài khoản, đơn hàng, sự kiện. Thư mục runtime/ nằm ngoài git."""
from __future__ import annotations

import csv
import json
import threading
import time
from pathlib import Path

import pandas as pd

EVENT_FIELDS = ["ts", "session_id", "user_id", "scenario", "event", "surface", "shelf", "article_id",
                "position", "order_id", "value"]
EVENT_TYPES = {"impression", "click", "add_to_cart", "purchase"}


class Store:
    def __init__(self, runtime_dir: Path | str):
        self.dir = Path(runtime_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------- json --
    def _read(self, name, default):
        p = self.dir / name
        if not p.exists():
            return default
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return default

    def _write(self, name, obj):
        tmp = self.dir / f"{name}.tmp"
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.dir / name)

    # ------------------------------------------------------------ users --
    def users(self) -> dict:
        return self._read("users.json", {})

    def get_user(self, uid: str) -> dict | None:
        return self.users().get(uid)

    def save_user(self, u: dict):
        with self._lock:
            d = self.users()
            d[u["id"]] = u
            self._write("users.json", d)

    def find_by_email(self, email: str) -> dict | None:
        email = email.strip().lower()
        return next((u for u in self.users().values() if u.get("email", "").lower() == email), None)

    # ----------------------------------------------------------- orders --
    def orders(self, uid: str | None = None) -> list[dict]:
        o = self._read("orders.json", [])
        return [x for x in o if x["uid"] == uid] if uid else o

    def get_order(self, oid: str) -> dict | None:
        return next((o for o in self._read("orders.json", []) if o["id"] == oid), None)

    def save_order(self, o: dict):
        with self._lock:
            all_ = self._read("orders.json", [])
            all_.append(o)
            self._write("orders.json", all_)

    def delete_orders(self, uids: set[str]) -> int:
        with self._lock:
            all_ = self._read("orders.json", [])
            keep = [o for o in all_ if o["uid"] not in uids]
            self._write("orders.json", keep)
            return len(all_) - len(keep)

    def session_purchases(self, uid: str) -> list[str]:
        """Món đã mua trên web của một khách, cũ -> mới (mỗi đơn vị số lượng là một lượt mua)."""
        out = []
        for o in sorted(self.orders(uid), key=lambda x: x["ts"]):
            for line in o["items"]:
                out += [line["article_id"]] * int(line["qty"])
        return out

    # ----------------------------------------------------------- events --
    def log_events(self, rows: list[dict]):
        p = self.dir / "events.csv"
        with self._lock:
            new = not p.exists() or p.stat().st_size == 0
            with p.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, EVENT_FIELDS, extrasaction="ignore")
                if new:
                    w.writeheader()
                for r in rows:
                    r = dict(r)
                    r.setdefault("ts", round(time.time(), 3))
                    w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in EVENT_FIELDS})

    def events(self) -> pd.DataFrame:
        p = self.dir / "events.csv"
        if not p.exists():
            return pd.DataFrame(columns=EVENT_FIELDS)
        return pd.read_csv(p, dtype={"article_id": str, "user_id": str, "order_id": str}).fillna("")

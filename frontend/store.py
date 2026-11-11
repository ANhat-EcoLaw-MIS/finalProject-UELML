"""Cơ sở dữ liệu của WEB (SQLite, file runtime/stylerec.db): tài khoản, đơn hàng, giao dịch mua trên web, sự kiện.

Tách biệt với dữ liệu huấn luyện ML (../data/*.parquet, chỉ đọc):
  * dữ liệu huấn luyện = lịch sử mua H&M, cố định giữa các lần huấn luyện;
  * web.db              = mọi thứ phát sinh trên web. Bảng `web_transactions` lưu từng lượt mua trên web theo ĐÚNG
                          định dạng dữ liệu huấn luyện (customer_id, article_id, t_dat, price, sales_channel_id=2),
                          để định kỳ xuất ra và gộp vào dữ liệu train (export_web_transactions()).
Gợi ý được tính lại NGAY sau khi mua nhờ đọc `web_transactions` (session_purchases) ở mỗi request.
SQLite đủ cho một server demo; đổi sang Postgres/MySQL chỉ cần đổi lớp này (schema giữ nguyên).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

EVENT_FIELDS = ["ts", "session_id", "user_id", "scenario", "event", "surface", "shelf", "article_id",
                "position", "order_id", "value"]
EVENT_TYPES = {"impression", "click", "add_to_cart", "purchase"}
ORDER_FIELDS = ["id", "uid", "ts", "date", "subtotal", "shipping_fee", "total", "shipping", "payment", "name",
                "phone", "address", "scenario_before", "scenario_after"]
ITEM_FIELDS = ["article_id", "name", "size", "qty", "price_vnd", "price", "surface", "shelf", "position"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT, password TEXT,
    customer_id INTEGER,              -- mã khách dùng cho dữ liệu huấn luyện (NULL = chưa có)
    kind TEXT NOT NULL DEFAULT 'account', created REAL);
CREATE UNIQUE INDEX IF NOT EXISTS ux_users_email ON users(lower(email)) WHERE email <> '';
CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY, uid TEXT NOT NULL, ts REAL NOT NULL, date TEXT, subtotal INTEGER, shipping_fee INTEGER,
    total INTEGER, shipping TEXT, payment TEXT, name TEXT, phone TEXT, address TEXT,
    scenario_before TEXT, scenario_after TEXT);
CREATE INDEX IF NOT EXISTS ix_orders_uid ON orders(uid, ts);
CREATE TABLE IF NOT EXISTS order_items (
    order_id TEXT NOT NULL REFERENCES orders(id) ON DELETE CASCADE, line_no INTEGER NOT NULL,
    article_id TEXT NOT NULL, name TEXT, size TEXT, qty INTEGER NOT NULL, price_vnd INTEGER, price REAL,
    surface TEXT, shelf TEXT, position TEXT, PRIMARY KEY (order_id, line_no));
CREATE TABLE IF NOT EXISTS web_transactions (     -- cùng định dạng dữ liệu huấn luyện, 1 dòng = 1 lượt mua
    id INTEGER PRIMARY KEY AUTOINCREMENT, uid TEXT NOT NULL, customer_id INTEGER,
    article_id TEXT NOT NULL, t_dat TEXT NOT NULL,   -- ngày theo "đồng hồ mô phỏng" nối tiếp dữ liệu
    ts REAL NOT NULL,                                -- thời điểm thật
    price REAL, sales_channel_id INTEGER NOT NULL DEFAULT 2,
    order_id TEXT REFERENCES orders(id) ON DELETE CASCADE, exported_at REAL);
CREATE INDEX IF NOT EXISTS ix_webtx_uid ON web_transactions(uid, ts);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, session_id TEXT, user_id TEXT, scenario TEXT, event TEXT,
    surface TEXT, shelf TEXT, article_id TEXT, position TEXT, order_id TEXT, value TEXT);
CREATE INDEX IF NOT EXISTS ix_events_event ON events(event, ts);
"""


class Store:
    """runtime_dir: thư mục chứa stylerec.db.
    sim_anchor : ngày đầu tiên sau dữ liệu huấn luyện. Lượt mua trên web được ghi ngày
                 t_dat = sim_anchor + (số ngày từ lúc web chạy lần đầu), để nối liền dữ liệu khi gộp vào train."""

    def __init__(self, runtime_dir: Path | str, sim_anchor: pd.Timestamp | str | None = None):
        self.dir = Path(runtime_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "stylerec.db"
        self._lock = threading.Lock()
        with self._db() as con:
            con.executescript(SCHEMA)
            con.execute("INSERT OR IGNORE INTO meta VALUES ('launched', ?)", (time.strftime("%Y-%m-%d"),))
            if sim_anchor is not None:
                con.execute("INSERT OR IGNORE INTO meta VALUES ('sim_anchor', ?)",
                            (pd.Timestamp(sim_anchor).strftime("%Y-%m-%d"),))
        self._migrate_json()
        self._backfill_customer_ids()

    @contextmanager
    def _db(self):
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("PRAGMA journal_mode = WAL")
        try:
            with con:
                yield con
        finally:
            con.close()

    def _meta(self, key: str) -> str | None:
        with self._db() as con:
            row = con.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def sim_date(self, ts: float | None = None) -> str:
        """Ngày mô phỏng cho một lượt mua trên web (ngày thật nếu không có sim_anchor)."""
        ts = time.time() if ts is None else ts
        anchor, launched = self._meta("sim_anchor"), self._meta("launched")
        if not anchor:
            return time.strftime("%Y-%m-%d", time.localtime(ts))
        days = (pd.Timestamp(time.strftime("%Y-%m-%d", time.localtime(ts))) - pd.Timestamp(launched)).days
        return (pd.Timestamp(anchor) + pd.Timedelta(days=max(days, 0))).strftime("%Y-%m-%d")

    # ------------------------------------------------------------ users --
    @staticmethod
    def _user(row) -> dict:
        u = dict(row)
        u["temp"] = u.get("kind") == "temp"
        return u

    def users(self) -> dict:
        with self._db() as con:
            return {r["id"]: self._user(r) for r in con.execute("SELECT * FROM users")}

    def get_user(self, uid: str) -> dict | None:
        with self._db() as con:
            row = con.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
        return self._user(row) if row else None

    def save_user(self, u: dict):
        kind = u.get("kind") or ("temp" if u.get("temp") else "account")
        with self._db() as con:
            con.execute("INSERT OR REPLACE INTO users (id, name, email, password, customer_id, kind, created) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (u["id"], u["name"], u.get("email", ""), u.get("password", ""), u.get("customer_id"),
                         kind, u.get("created", time.time())))

    WEB_CUSTOMER_ID_START = 1_000_000   # mã khách cho tài khoản web, không trùng mã trong dữ liệu (~136k)

    def next_customer_id(self) -> int:
        with self._db() as con:
            top = con.execute("SELECT max(customer_id) FROM users WHERE customer_id >= ?",
                              (self.WEB_CUSTOMER_ID_START,)).fetchone()[0]
        return (top or self.WEB_CUSTOMER_ID_START - 1) + 1

    def find_by_email(self, email: str) -> dict | None:
        email = email.strip()
        if not email:
            return None
        with self._db() as con:
            row = con.execute("SELECT * FROM users WHERE lower(email) = lower(?)", (email,)).fetchone()
        return self._user(row) if row else None

    # ----------------------------------------------------------- orders --
    def _orders(self, where: str = "", args: tuple = ()) -> list[dict]:
        with self._db() as con:
            rows = con.execute(f"SELECT * FROM orders {where} ORDER BY ts", args).fetchall()
            out = []
            for r in rows:
                o = dict(r)
                o["items"] = [dict(i) for i in con.execute(
                    f"SELECT {', '.join(ITEM_FIELDS)} FROM order_items WHERE order_id = ? ORDER BY line_no",
                    (o["id"],))]
                out.append(o)
        return out

    def orders(self, uid: str | None = None) -> list[dict]:
        return self._orders("WHERE uid = ?", (uid,)) if uid else self._orders()

    def get_order(self, oid: str) -> dict | None:
        found = self._orders("WHERE id = ?", (oid,))
        return found[0] if found else None

    def save_order(self, o: dict, customer_id: int | None = None):
        """Lưu đơn + từng lượt mua vào web_transactions (mỗi đơn vị số lượng là 1 lượt, như dữ liệu H&M)."""
        t_dat = self.sim_date(o["ts"])
        customer_id = int(customer_id) if customer_id not in (None, "") else None
        with self._lock, self._db() as con:
            con.execute(f"INSERT INTO orders ({', '.join(ORDER_FIELDS)}) VALUES ({', '.join('?' * len(ORDER_FIELDS))})",
                        tuple(o.get(k) for k in ORDER_FIELDS))
            for n, it in enumerate(o["items"], start=1):
                con.execute(f"INSERT INTO order_items (order_id, line_no, {', '.join(ITEM_FIELDS)}) "
                            f"VALUES (?, ?, {', '.join('?' * len(ITEM_FIELDS))})",
                            (o["id"], n, *(it.get(k) for k in ITEM_FIELDS)))
                con.executemany(
                    "INSERT INTO web_transactions (uid, customer_id, article_id, t_dat, ts, price, order_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [(o["uid"], customer_id, it["article_id"], t_dat, o["ts"], it.get("price"), o["id"])]
                    * int(it["qty"]))

    def delete_orders(self, uids: set[str], keep_exported: bool = True) -> int:
        """Xoá đơn (và lượt mua) của các tài khoản. Mặc định giữ đơn đã xuất sang dữ liệu train để hai bên không lệch."""
        if not uids:
            return 0
        marks = ", ".join("?" * len(uids))
        cond = " AND id NOT IN (SELECT order_id FROM web_transactions WHERE exported_at IS NOT NULL)" if keep_exported else ""
        with self._lock, self._db() as con:
            return con.execute(f"DELETE FROM orders WHERE uid IN ({marks}){cond}", tuple(uids)).rowcount

    def session_purchases(self, uid: str) -> list[str]:
        """Món đã mua trên web của một khách, cũ -> mới (đưa cho recommender để tính lại gợi ý ngay)."""
        with self._db() as con:
            return [r["article_id"] for r in con.execute(
                "SELECT article_id FROM web_transactions WHERE uid = ? ORDER BY ts, id", (uid,))]

    # ------------------------------------------- xuất sang dữ liệu huấn luyện --
    def export_web_transactions(self, mark: bool = True, only_new: bool = True) -> pd.DataFrame:
        """Lượt mua trên web theo đúng định dạng dữ liệu train (customer_id, t_dat, article_id, price, sales_channel_id).
        Chỉ lấy tài khoản đã có customer_id. mark=True đánh dấu đã xuất để lần sau không xuất lại."""
        cond = "customer_id IS NOT NULL" + (" AND exported_at IS NULL" if only_new else "")
        with self._lock, self._db() as con:
            rows = con.execute(f"SELECT id, customer_id, t_dat, article_id, price, sales_channel_id "
                               f"FROM web_transactions WHERE {cond} ORDER BY ts, id").fetchall()
            if mark and rows:
                con.executemany("UPDATE web_transactions SET exported_at = ? WHERE id = ?",
                                [(time.time(), r["id"]) for r in rows])
        df = pd.DataFrame([dict(r) for r in rows],
                          columns=["id", "customer_id", "t_dat", "article_id", "price", "sales_channel_id"])
        return df.drop(columns="id").assign(t_dat=lambda d: pd.to_datetime(d["t_dat"]))

    # ----------------------------------------------------------- events --
    def log_events(self, rows: list[dict]):
        vals = [tuple(("" if r.get(k) is None else r.get(k)) if k != "ts" else r.get("ts", round(time.time(), 3))
                      for k in EVENT_FIELDS) for r in rows]
        with self._db() as con:
            con.executemany(f"INSERT INTO events ({', '.join(EVENT_FIELDS)}) VALUES ({', '.join('?' * len(EVENT_FIELDS))})",
                            vals)

    def events(self) -> pd.DataFrame:
        with self._db() as con:
            rows = con.execute(f"SELECT {', '.join(EVENT_FIELDS)} FROM events ORDER BY id").fetchall()
        return pd.DataFrame([dict(r) for r in rows], columns=EVENT_FIELDS).fillna("")

    def stats(self) -> dict:
        with self._db() as con:
            q = lambda sql: con.execute(sql).fetchone()[0]  # noqa: E731
            return dict(db=str(self.path), users=q("SELECT count(*) FROM users"), orders=q("SELECT count(*) FROM orders"),
                        web_transactions=q("SELECT count(*) FROM web_transactions"),
                        not_exported=q("SELECT count(*) FROM web_transactions WHERE exported_at IS NULL"),
                        events=q("SELECT count(*) FROM events"), sim_anchor=self._meta("sim_anchor"),
                        launched=self._meta("launched"))

    def fill_missing_prices(self, price_of) -> int:
        """Gán giá (thang của dataset) cho lượt mua cũ chưa có giá. price_of: article_id -> float."""
        with self._lock, self._db() as con:
            ids = {r[0] for r in con.execute("SELECT DISTINCT article_id FROM web_transactions WHERE price IS NULL")}
            for aid in ids:
                con.execute("UPDATE web_transactions SET price = ? WHERE article_id = ? AND price IS NULL",
                            (float(price_of(aid)), aid))
                con.execute("UPDATE order_items SET price = ? WHERE article_id = ? AND price IS NULL",
                            (float(price_of(aid)), aid))
        return len(ids)

    def _backfill_customer_ids(self):
        """Tài khoản cũ (chuyển từ JSON) chưa có customer_id -> cấp mã web và gắn vào các lượt mua của họ."""
        with self._lock, self._db() as con:
            missing = [r["id"] for r in con.execute(
                "SELECT id FROM users WHERE customer_id IS NULL ORDER BY created, id")]
            if not missing:
                return
            top = con.execute("SELECT max(customer_id) FROM users WHERE customer_id >= ?",
                              (self.WEB_CUSTOMER_ID_START,)).fetchone()[0] or self.WEB_CUSTOMER_ID_START - 1
            for n, uid in enumerate(missing, start=1):
                con.execute("UPDATE users SET customer_id = ? WHERE id = ?", (top + n, uid))
                con.execute("UPDATE web_transactions SET customer_id = ? WHERE uid = ? AND customer_id IS NULL",
                            (top + n, uid))

    # ------------------------------------------------ chuyển từ bản JSON cũ --
    def _migrate_json(self):
        users_f, orders_f = self.dir / "users.json", self.dir / "orders.json"
        if users_f.exists():
            try:
                for u in json.loads(users_f.read_text(encoding="utf-8")).values():
                    if not self.get_user(u["id"]):
                        self.save_user(u)
                users_f.rename(users_f.with_suffix(".json.migrated"))
            except (ValueError, KeyError, OSError, sqlite3.Error):
                pass
        if orders_f.exists():
            try:
                for o in json.loads(orders_f.read_text(encoding="utf-8")):
                    if not self.get_order(o["id"]):
                        self.save_order(o)
                orders_f.rename(orders_f.with_suffix(".json.migrated"))
            except (ValueError, KeyError, OSError, sqlite3.Error):
                pass

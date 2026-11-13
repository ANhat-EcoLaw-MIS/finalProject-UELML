"""Dữ liệu của WEB (không tính gợi ý): danh mục sản phẩm, lịch sử mua trong dataset, khách mẫu.

Catalog nhận (transactions, articles) từ dataset.py, nên dùng được cả với dữ liệu thật của repo (~105k sản phẩm)
lẫn bộ mẫu nhỏ trong frontend/data/. Sản phẩm được dựng lười (khi cần) để khởi động nhanh.
Ảnh: static/img/products/<article_id>.jpg nếu có, không thì giao diện vẽ ảnh minh hoạ theo màu + nhóm hàng.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

import dataset

IMG_DIR = dataset.HERE / "static" / "img" / "products"
DEMO_USERS_FILE = dataset.SAMPLE_DATA_DIR / "demo_users.json"
BROWSE_LIMIT = 3000          # số sản phẩm cho trang danh mục (bán chạy gần đây nhất)

GROUP_VI = {
    "Garment Upper body": "Áo", "Garment Lower body": "Quần & váy", "Garment Full body": "Đầm & jumpsuit",
    "Swimwear": "Đồ bơi", "Underwear": "Đồ lót", "Accessories": "Phụ kiện", "Shoes": "Giày dép",
    "Socks & Tights": "Tất & quần tất", "Nightwear": "Đồ ngủ", "Bags": "Túi", "Items": "Khác",
    "Cosmetic": "Mỹ phẩm", "Underwear/nightwear": "Đồ lót", "Unknown": "Khác",
}
TYPE_VI = {
    "Trousers": "Quần dài", "Dress": "Đầm", "Sweater": "Áo len", "T-shirt": "Áo thun", "Top": "Áo croptop",
    "Blouse": "Áo kiểu", "Jacket": "Áo khoác", "Bra": "Áo ngực", "Underwear bottom": "Quần lót", "Shirt": "Sơ mi",
    "Skirt": "Chân váy", "Vest top": "Áo hai dây", "Socks": "Tất", "Hoodie": "Áo hoodie",
    "Leggings/Tights": "Quần legging", "Cardigan": "Áo cardigan", "Shorts": "Quần short", "Blazer": "Áo blazer",
    "Bikini top": "Áo bikini", "Earring": "Bông tai", "Swimwear bottom": "Quần bơi", "Boots": "Bốt", "Bag": "Túi",
    "Belt": "Thắt lưng", "Necklace": "Vòng cổ", "Hair/alice band": "Băng đô", "Underwear Tights": "Quần tất",
    "Other accessories": "Phụ kiện khác", "Bodysuit": "Bodysuit", "Scarf": "Khăn", "Pyjama set": "Bộ đồ ngủ",
    "Jumpsuit/Playsuit": "Jumpsuit", "Hat/beanie": "Mũ", "Garment Set": "Set đồ", "Swimsuit": "Đồ bơi liền",
    "Sneakers": "Giày sneaker", "Coat": "Áo măng tô", "Hair clip": "Kẹp tóc", "Hair string": "Dây buộc tóc",
    "Ring": "Nhẫn", "Underwear body": "Body lót", "Sunglasses": "Kính râm", "Pyjama bottom": "Quần ngủ",
    "Polo shirt": "Áo polo", "Sandals": "Xăng đan", "Heeled sandals": "Xăng đan cao gót", "Hair ties": "Chun buộc tóc",
    "Ballerinas": "Giày búp bê", "Flat shoe": "Giày bệt", "Gloves": "Găng tay", "Outdoor Waistcoat": "Áo gile",
    "Pumps": "Giày cao gót", "Cap/peaked": "Mũ lưỡi trai", "Beanie": "Mũ len", "Dungarees": "Quần yếm",
    "Bracelet": "Vòng tay", "Slippers": "Dép", "Sarong": "Váy đi biển", "Pyjama jumpsuit/playsuit": "Đồ ngủ liền",
}
COLOUR_HEX = {
    "black": "#26262b", "white": "#efeeea", "off white": "#e8e1d2", "grey": "#8f9197", "dark grey": "#55585f",
    "light grey": "#c9cacd", "dark blue": "#25365a", "blue": "#3c64a8", "light blue": "#a9c3de",
    "pink": "#e58aa0", "light pink": "#f1c6cf", "dark pink": "#b94a6a", "red": "#b3262e", "dark red": "#7a1d26",
    "beige": "#d0bc9f", "light beige": "#e3d8c5", "dark beige": "#a9906f", "greyish beige": "#b9ae9c",
    "dark green": "#2f5240", "green": "#4f8a63", "light green": "#a8cba0", "khaki green": "#7c7a52",
    "greenish khaki": "#8a865a", "yellow": "#e3c24a", "light yellow": "#f1e2a0", "dark yellow": "#b8952c",
    "orange": "#e0873e", "light orange": "#f0b98a", "dark orange": "#c8612b", "brown": "#7a5a43",
    "yellowish brown": "#9a6d3a", "gold": "#c9a74a", "silver": "#b9bcc2", "light purple": "#c7b5dc",
    "purple": "#76508f", "dark purple": "#4b2f61", "turquoise": "#3aa7a3", "light turquoise": "#9bd5d1",
    "dark turquoise": "#1f6f6c",
}
# Từ khoá tiếng Việt để tìm kiếm (dữ liệu gốc bằng tiếng Anh)
COLOUR_VI = {"black": "đen", "white": "trắng", "off white": "trắng ngà kem", "grey": "xám ghi", "blue": "xanh dương",
             "dark blue": "xanh navy xanh đậm", "light blue": "xanh nhạt", "pink": "hồng", "red": "đỏ",
             "beige": "be kem", "green": "xanh lá", "khaki": "kaki rêu", "yellow": "vàng", "orange": "cam",
             "brown": "nâu", "purple": "tím", "turquoise": "xanh ngọc", "gold": "vàng kim", "silver": "bạc",
             "dark": "đậm tối", "light": "nhạt sáng"}
TYPE_SYNONYMS = {"Trousers": "quần dài", "Dress": "váy liền đầm váy", "Skirt": "váy chân váy", "T-shirt": "áo phông áo thun",
                 "Top": "áo", "Blouse": "áo kiểu áo sơ mi nữ", "Sweater": "áo len áo nỉ", "Hoodie": "áo nỉ có mũ",
                 "Shorts": "quần đùi quần ngắn", "Leggings/Tights": "quần bó", "Bra": "áo lót", "Underwear bottom": "quần lót",
                 "Jacket": "áo khoác", "Coat": "áo khoác dài", "Sneakers": "giày thể thao", "Boots": "giày bốt",
                 "Bag": "túi xách", "Socks": "vớ", "Swimsuit": "đồ bơi", "Bikini top": "đồ bơi bikini"}
NAME_SYNONYMS = {"denim": "jean jeans bò", "dnm": "jean jeans bò denim", "trs": "quần", "tee": "áo thun"}


def fold(text: str) -> str:
    """Chữ thường, bỏ dấu tiếng Việt (đ -> d) để so khớp không phân biệt dấu."""
    text = str(text).lower().replace("đ", "d")
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")


CLOTHES = {"Garment Upper body", "Garment Lower body", "Garment Full body", "Swimwear", "Underwear", "Nightwear"}


def colour_hex(name: str) -> str:
    key = (name or "").strip().lower()
    return COLOUR_HEX.get(key) or f"hsl({sum(map(ord, key)) % 360} 28% 62%)"


def is_light(hexcol: str) -> bool:
    if not (hexcol.startswith("#") and len(hexcol) == 7):
        return False
    r, g, b = (int(hexcol[i:i + 2], 16) for i in (1, 3, 5))
    return 0.299 * r + 0.587 * g + 0.114 * b > 190


def to_vnd(price: float) -> int:
    """Giá trong dataset H&M đã chuẩn hoá (~0.01–0.1). Quy đổi minh hoạ: x10 triệu, tròn chục nghìn, đuôi 9.000đ."""
    v = int(round(float(price) * 10_000_000 / 10_000)) * 10_000 - 1_000
    return max(v, 49_000)


def clean_name(name: str) -> str:
    name = re.sub(r"\s*\((?:\d+|[A-Z])\)\s*", " ", str(name))
    name = re.sub(r"\s+", " ", name).strip()
    return name[:1].upper() + name[1:]


@dataclass
class Product:
    article_id: str
    name: str
    type: str
    type_vi: str
    group: str
    group_vi: str
    index_group: str
    colour: str
    swatch: str
    light_bg: bool
    pattern: str
    desc: str
    price_vnd: int
    sizes: list[str]
    image: str | None
    sold7: int = 0
    badges: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)


class Catalog:
    """Tra cứu sản phẩm + lịch sử mua. Không gợi ý."""

    def __init__(self, transactions: pd.DataFrame, articles: pd.DataFrame,
                 demo_users_file: Path | str | None = DEMO_USERS_FILE, browse_limit: int = BROWSE_LIMIT):
        tx = transactions
        self.transactions = tx
        self.last_date: pd.Timestamp = tx["t_dat"].max()
        self._art = articles.drop_duplicates("article_id").set_index("article_id")
        self._rows = tx.groupby("customer_id").indices
        self._price = tx.groupby("article_id")["price"].median()
        recent = tx[tx["t_dat"] > self.last_date - pd.Timedelta(days=28)]
        self._sold7 = recent[recent["t_dat"] > self.last_date - pd.Timedelta(days=7)]["article_id"].value_counts()
        sold28 = recent["article_id"].value_counts()
        total = tx["article_id"].value_counts()
        rank = pd.DataFrame({"s7": self._sold7, "s28": sold28, "all": total}).fillna(0)
        rank = rank[rank.index.isin(self._art.index)]
        rank = rank.sort_values(["s7", "s28", "all"], ascending=False, kind="mergesort")
        self.order: list[str] = rank.index[:browse_limit].tolist()
        self._search_raw = self._build_search_text(self.order)       # chữ thường, giữ dấu
        self._search_fold = self._search_raw.map(fold)                 # đã bỏ dấu
        self._best = set(self._sold7.index[:12])
        self.demo_users = self._load_demo_users(demo_users_file)

    @classmethod
    def sample(cls) -> "Catalog":
        tx, art = dataset.load_sample()
        return cls(tx, art)

    # ------------------------------------------------------------ sản phẩm --
    def has(self, aid: str) -> bool:
        return aid in self._art.index

    @lru_cache(maxsize=20000)
    def get(self, aid: str) -> Product | None:
        if aid not in self._art.index:
            return None
        r = self._art.loc[aid]
        g = lambda c: "" if pd.isna(r.get(c, "")) else str(r.get(c, ""))  # noqa: E731
        grp, typ = g("product_group_name"), g("product_type_name")
        hexcol = colour_hex(g("colour_group_name"))
        img = None
        for rel in (aid, f"{aid[:3]}/{aid}", f"images/{aid[:3]}/{aid}"):
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                if (IMG_DIR / f"{rel}{ext}").exists():
                    img = f"img/products/{rel}{ext}".replace("\\", "/")
                    break
            if img:
                break
        return Product(
            article_id=aid, name=clean_name(g("prod_name")), type=typ, type_vi=TYPE_VI.get(typ, typ), group=grp,
            group_vi=GROUP_VI.get(grp, grp), index_group=g("index_group_name"), colour=g("colour_group_name"),
            swatch=hexcol, light_bg=is_light(hexcol), pattern=g("graphical_appearance_name"), desc=g("detail_desc"),
            price_vnd=to_vnd(self._price.get(aid, 0.03)), sizes=["XS", "S", "M", "L", "XL"] if grp in CLOTHES else [],
            image=img, sold7=int(self._sold7.get(aid, 0)), badges=["Bán chạy"] if aid in self._best else [])

    def raw_price(self, aid: str) -> float:
        """Giá theo thang của dataset (dùng khi ghi lượt mua web theo định dạng dữ liệu huấn luyện)."""
        return float(self._price.get(aid, 0.03))

    def name_of(self, aid: str) -> str:
        p = self.get(aid)
        return p.name if p else aid

    def groups(self) -> list[tuple[str, str, int]]:
        cnt = self._art.loc[self.order, "product_group_name"].value_counts()
        return [(gname, GROUP_VI.get(gname, gname), int(n)) for gname, n in cnt.items() if gname != "Unknown"]

    def _build_search_text(self, ids: list[str]) -> pd.Series:
        """Chuỗi tìm kiếm cho từng sản phẩm: tên, loại (Anh + Việt), màu (Anh + Việt), nhóm, từ đồng nghĩa."""
        sub = self._art.loc[ids]
        def row_text(r):
            name, typ = str(r["prod_name"]), str(r["product_type_name"])
            colour, grp = str(r["colour_group_name"]).lower(), str(r["product_group_name"])
            parts = [name, typ, TYPE_VI.get(typ, ""), TYPE_SYNONYMS.get(typ, ""), colour,
                     " ".join(v for k, v in COLOUR_VI.items() if k in colour), str(r.get("garment_group_name", ""))]
            if grp != "Garment Lower body":     # "Quần & váy" quá rộng: để loại sản phẩm quyết định quần hay váy
                parts.append(GROUP_VI.get(grp, ""))
            parts += [v for k, v in NAME_SYNONYMS.items() if k in name.lower()]
            return " ".join(parts).lower()
        return pd.Series([row_text(r) for _, r in sub.iterrows()], index=sub.index)

    def search(self, group: str = "", q: str = "", sort: str = "popular", page: int = 1, per_page: int = 24):
        """Tìm trong danh mục: mọi từ trong q đều phải khớp, giữ thứ tự bán chạy. Từ gõ có dấu thì khớp có dấu
        (để "đầm" không trùng "đậm"); từ gõ không dấu thì khớp không dấu ("ao den" = "áo đen")."""
        ids = self.order
        sub = self._art.loc[ids]
        mask = pd.Series(True, index=sub.index)
        if group:
            mask &= sub["product_group_name"] == group
        for token in q.lower().split():
            folded = fold(token)
            pool = self._search_fold if folded == token else self._search_raw
            mask &= pool.str.contains(re.escape(token if folded != token else folded), regex=True)
        ids = sub.index[mask.to_numpy()].tolist()
        if sort in ("price_asc", "price_desc"):
            ids.sort(key=lambda a: self._price.get(a, 0.03), reverse=sort == "price_desc")
        total = len(ids)
        start = (max(page, 1) - 1) * per_page
        return [self.get(a) for a in ids[start:start + per_page]], total

    # ------------------------------------------------------- khách / lịch sử --
    @staticmethod
    def _cid(customer_id):
        try:
            return int(str(customer_id).strip())
        except (TypeError, ValueError):
            return None

    def n_purchases(self, customer_id) -> int:
        """Số lượt mua online trong dataset (số dòng giao dịch, đúng định nghĩa phân khúc ở utils/eval_tools.py)."""
        rows = self._rows.get(self._cid(customer_id))
        return 0 if rows is None else len(rows)

    def history(self, customer_id, limit: int | None = None) -> list[dict]:
        """Lịch sử mua trong dataset, MỚI -> CŨ: [{date, article_id, price_vnd}]."""
        rows = self._rows.get(self._cid(customer_id))
        if rows is None:
            return []
        h = self.transactions.iloc[rows[::-1]]
        if limit:
            h = h.head(limit)
        return [dict(date=d.strftime("%Y-%m-%d"), article_id=a, price_vnd=to_vnd(p))
                for d, a, p in zip(h["t_dat"], h["article_id"], h["price"])]

    def demo_user(self, uid: str) -> dict | None:
        return next((u for u in self.demo_users if u["id"] == str(uid)), None)

    def _load_demo_users(self, path) -> list[dict]:
        users = []
        if path and Path(path).exists():
            users = [u for u in json.loads(Path(path).read_text(encoding="utf-8"))
                     if self.n_purchases(u["id"]) > 0]
        if users:
            return users
        # dữ liệu không chứa khách mẫu (ví dụ bộ mẫu nhỏ) -> tự chọn 1 khách mua ít + 2 khách mua nhiều
        n = pd.Series({c: len(r) for c, r in self._rows.items()})
        light = n[(n >= 2) & (n <= 4)].sort_index().index[:1].tolist()
        heavy = n[n >= 5].sort_values(ascending=False).index[:2].tolist()
        return ([dict(id=str(c), name="Khách mua ít", segment="light", note="1–4 lượt mua") for c in light]
                + [dict(id=str(c), name=f"Khách mua nhiều {i + 1}", segment="heavy", note="5+ lượt mua")
                   for i, c in enumerate(heavy)])

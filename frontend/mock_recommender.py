"""Recommender GIẢ, đúng hợp đồng recommender_port.py, chạy tức thì trên Catalog.transactions.

Dùng khi làm giao diện / chạy test (không cần fit model backend ~1 phút). Logic cố ý đơn giản,
chỉ để có dữ liệu hợp lý trên màn hình; gợi ý thật là production_recommender.py.
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence

import pandas as pd

from recommender_port import SHELF_CART, SHELF_DISCOVERY, SHELF_POPULAR, SHELF_REPEAT, Rec


class MockRecommender:
    def __init__(self, catalog):
        tx = catalog.transactions
        self.catalog = catalog
        self.tx = tx
        last = tx["t_dat"].max()
        sold7 = tx[tx["t_dat"] > last - pd.Timedelta(days=7)]["article_id"].value_counts()
        total = tx["article_id"].value_counts()
        self.popular = list(dict.fromkeys(list(sold7.index) + list(total.index)))
        self.sold7 = sold7
        # đồng mua theo giỏ (khách, ngày)
        b = tx[["customer_id", "t_dat", "article_id"]].drop_duplicates()
        b = b[b.groupby(["customer_id", "t_dat"])["article_id"].transform("size").between(2, 20)]
        pairs = b.merge(b, on=["customer_id", "t_dat"])
        pairs = pairs[pairs["article_id_x"] != pairs["article_id_y"]]
        self.co = pairs.groupby(["article_id_x", "article_id_y"]).size()

    def _type(self, aid):
        p = self.catalog.get(aid)
        return p.type if p else None

    def _pop(self, k, exclude=(), shelf=SHELF_POPULAR, skip_types=()):
        out = []
        for a in self.popular:
            if a in exclude or self._type(a) in skip_types:
                continue
            out.append(Rec(a, f"Đã bán {int(self.sold7.get(a, 0))} sản phẩm trong 7 ngày qua", shelf))
            if len(out) >= k:
                break
        return out

    def _co(self, sources, exclude, k, shelf, skip_types=()):
        score: Counter = Counter()
        because = {}
        for s in sources:
            if s in self.co.index.get_level_values(0):
                for nb, v in self.co.loc[s].items():
                    if nb in exclude or self._type(nb) in skip_types:
                        continue
                    score[nb] += v
                    because.setdefault(nb, s)
        return [Rec(a, "Hay được mua cùng", shelf, because[a]) for a, _ in score.most_common(k)]

    def recommend_for_user(self, user_id, k: int = 12, session_purchases: Sequence[str] = ()) -> list[Rec]:
        sp = list(session_purchases)
        hist = self.catalog.history(user_id) if user_id else []
        owned = [h["article_id"] for h in reversed(hist)] + sp
        recs = self._pop(k, exclude=set(sp))
        if not owned:
            return recs
        cnt = Counter(owned)
        last_seen = {a: i for i, a in enumerate(owned)}
        for a in sorted(cnt, key=lambda a: (-cnt[a], -last_seen[a]))[:k]:
            web = a in sp
            recs.append(Rec(a, ("Bạn vừa mua trên web · " if web else "") + f"Bạn đã mua {cnt[a]} lần", SHELF_REPEAT))
        disc = self._co(list(dict.fromkeys(reversed(owned)))[:5], set(owned), k, SHELF_DISCOVERY)
        for r in disc:
            recs.append(Rec(r.article_id, "Khách mua món bạn đã mua cũng mua món này", SHELF_DISCOVERY, r.because))
        if len(disc) < k:
            have = set(owned) | {r.article_id for r in disc}
            recs += [Rec(r.article_id, "Bán chạy, bạn chưa mua", SHELF_DISCOVERY)
                     for r in self._pop(k - len(disc), exclude=have)]
        return recs

    def recommend_for_cart(self, article_ids: Sequence[str], k: int = 6) -> list[Rec]:
        cart = list(dict.fromkeys(article_ids))
        if not cart:
            return []
        types = {self._type(a) for a in cart}
        recs = self._co(cart, set(cart), k, SHELF_CART, skip_types=types)
        if len(recs) < k:
            recs += self._pop(k - len(recs), exclude=set(cart) | {r.article_id for r in recs},
                              shelf=SHELF_CART, skip_types=types)
        return recs

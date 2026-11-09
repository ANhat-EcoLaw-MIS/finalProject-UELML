"""Lớp nối web <-> backend: bọc các model ĐÃ CHỐT trong ../models theo hợp đồng recommender_port.py.

KHÔNG sửa code backend. File này chỉ:
  1. load dữ liệu online (train + val + test), fit 3 bài toán đúng cấu hình ở final_test/:
       cold-start  PopularityModel(window_days=7)
       mua lại     CategoryRepurchaseModel(product_type_no, horizon_days=7)
       khám phá    BlendModel{cat_pop: 0.65, cf_lastday_nocolour: 0.35}, rrf_c=10
  2. cache model đã fit ra runtime/model_cache.pkl (lần sau bật web nhanh);
  3. đổi DataFrame của model -> list[Rec] có nhãn mục + lý do;
  4. phần backend chưa có, viết ở đây bằng chính trạng thái đã fit của model:
       - khách mới / mã lạ  -> mục popular;
       - giỏ hàng (mua kèm) -> cộng các hàng của ma trận tương đồng ItemCF (S_);
       - tính lại sau thanh toán (session_purchases) -> chấm lại theo đúng công thức của từng model
         với các món vừa mua coi như lần mua mới nhất (không phải fit lại 60 giây).
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
import sys
import time
import warnings
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import dataset
from recommender_port import SHELF_CART, SHELF_DISCOVERY, SHELF_POPULAR, SHELF_REPEAT, Rec

if str(dataset.REPO) not in sys.path:
    sys.path.insert(0, str(dataset.REPO))

from models.blend_user_discovery import BlendModel  # noqa: E402
from models.cat_pop import CategoryPopularityModel  # noqa: E402
from models.cat_repurchase import CategoryRepurchaseModel  # noqa: E402
from models.itemCFv2 import ItemCFModel  # noqa: E402
from models.popularity import PopularityModel  # noqa: E402

log = logging.getLogger("stylerec.recommender")

# Cấu hình lấy đúng từ final_test/*.ipynb
CONFIG = dict(
    version=1,
    pop_window_days=7,
    rep_horizon_days=7,
    blend_weight_cat_pop=0.65,
    blend_rrf_c=10,
    blend_pool_k=20,
    cart_exclude_cart_types=True,   # giỏ hàng: không gợi ý cùng loại sản phẩm đã có trong giỏ (bộ đồ "hoàn chỉnh")
)
CF, CP = "cf_lastday_nocolour", "cat_pop"


class ProductionRecommender:
    """Recommender thật. `history`/`articles` có thể truyền vào để dùng chung DataFrame với Catalog."""

    def __init__(self, history: pd.DataFrame | None = None, articles: pd.DataFrame | None = None,
                 cache_path: Path | str | None = None, use_cache: bool = True):
        if history is None or articles is None:
            history, articles = dataset.load_full()
        self.history = history
        self.anchor = history["t_dat"].max() + pd.Timedelta(days=1)
        art = articles.drop_duplicates("article_id").set_index("article_id")
        self.types = art["product_type_no"]
        self.codes = art["product_code"]
        self.cache_path = Path(cache_path) if cache_path else dataset.HERE / "runtime" / "model_cache.pkl"

        t0 = time.time()
        models = self._load_cache() if use_cache else None
        if models is None:
            models = self._fit()
            if use_cache:
                self._save_cache(models)
        self.pop, self.rep, self.cf, self.cp = models["pop"], models["rep"], models["cf"], models["cp"]
        self.blend = BlendModel({CF: self.cf, CP: self.cp},
                                weights={CP: CONFIG["blend_weight_cat_pop"], CF: 1 - CONFIG["blend_weight_cat_pop"]},
                                rrf_c=CONFIG["blend_rrf_c"], pool_k=CONFIG["blend_pool_k"],
                                fit_components=False, k=12).fit(history, self.anchor)
        self._index()
        self.ready_seconds = round(time.time() - t0, 1)
        self._cache: OrderedDict = OrderedDict()
        log.info("ProductionRecommender sẵn sàng sau %.1fs (anchor %s)", self.ready_seconds, self.anchor.date())

    # ------------------------------------------------------------------ fit / cache
    def _fit(self) -> dict:
        h = self.history
        log.info("Đang fit model trên %s giao dịch online (lần đầu ~1 phút)...", f"{len(h):,}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pop = PopularityModel(window_days=CONFIG["pop_window_days"]).fit(h, self.anchor)
            rep = CategoryRepurchaseModel(self.types, horizon_days=CONFIG["rep_horizon_days"]).fit(h, self.anchor)
            cf = ItemCFModel(self.types, allow_same_category=True, source="last_day",
                             exclude_owned_product_code=True, article_product=self.codes).fit(h, self.anchor)
            cp = CategoryPopularityModel(self.types).fit(h, self.anchor)
        return dict(pop=pop, rep=rep, cf=cf, cp=cp)

    def _cache_key(self) -> str:
        raw = json.dumps(CONFIG, sort_keys=True) + f"|{len(self.history)}|{self.anchor}|" + dataset.files_signature()
        return hashlib.sha1(raw.encode()).hexdigest()

    def _load_cache(self):
        try:
            with self.cache_path.open("rb") as f:
                blob = pickle.load(f)
            if blob.get("key") == self._cache_key():
                log.info("Dùng model đã cache: %s", self.cache_path)
                return blob["models"]
        except FileNotFoundError:
            pass
        except Exception as exc:  # cache hỏng / khác phiên bản thư viện -> fit lại
            log.warning("Bỏ cache model (%s)", exc)
        return None

    def _save_cache(self, models: dict):
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".tmp")
            with tmp.open("wb") as f:
                pickle.dump({"key": self._cache_key(), "models": models}, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(self.cache_path)
        except Exception as exc:
            log.warning("Không lưu được cache model (%s)", exc)

    def _index(self):
        """Tra nhanh theo khách (tránh lọc cả bảng mỗi request)."""
        self.known = set(self.history["customer_id"].unique().tolist())
        self._hist_rows = self.history.groupby("customer_id").indices
        self._rep_rows = self.rep.scores_.groupby("customer_id").indices
        rates = self.rep.rate_table_
        self._rate0 = dict(zip(rates.loc[rates["bucket"] == 0, "category"], rates.loc[rates["bucket"] == 0, "rate"]))
        self._rate0_default = float(self.rep.bucket_rate_.iloc[0])
        self._pop_rank = self.pop.ranking_.reset_index(drop=True)
        cat_items = self.cp.cat_items_
        self._cat_items = {c: g for c, g in cat_items.groupby("cat")}

    # ------------------------------------------------------------------ helpers
    def _cid(self, user_id) -> int | None:
        try:
            cid = int(str(user_id).strip())
        except (TypeError, ValueError):
            return None
        return cid if cid in self.known else None

    def _owned(self, cid: int | None) -> list[str]:
        if cid is None:
            return []
        return self.history["article_id"].to_numpy()[self._hist_rows[cid]].tolist()

    def _popular(self, k: int, exclude=frozenset(), skip_types=frozenset(), skip_codes=frozenset(),
                 shelf=SHELF_POPULAR) -> list[Rec]:
        out = []
        for aid, n in zip(self._pop_rank["article_id"], self._pop_rank["score"]):
            if aid in exclude or self.types.get(aid) in skip_types or self.codes.get(aid) in skip_codes:
                continue
            out.append(Rec(aid, f"Đã bán {int(n)} sản phẩm trong 7 ngày qua", shelf))
            if len(out) >= k:
                break
        return out

    # ------------------------------------------------------------------ hợp đồng
    def recommend_for_user(self, user_id, k: int = 12, session_purchases: Sequence[str] = ()) -> list[Rec]:
        cid = self._cid(user_id)
        sp = tuple(str(a).zfill(10) for a in session_purchases if str(a).strip())
        key = ("user", cid if cid is not None else str(user_id), sp, k)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        recs = self._popular(k, exclude=frozenset(sp))
        if cid is not None or sp:
            if sp:
                recs += self._repeat_live(cid, sp, k) + self._discovery_live(cid, sp, k)
            else:
                recs += self._repeat(cid, k) + self._discovery(cid, k)
        self._cache[key] = recs
        if len(self._cache) > 512:
            self._cache.popitem(last=False)
        return recs

    def recommend_for_cart(self, article_ids: Sequence[str], k: int = 6) -> list[Rec]:
        cart = list(dict.fromkeys(str(a).zfill(10) for a in article_ids if str(a).strip()))
        if not cart:
            return []
        cart_set = set(cart)
        cart_types = {self.types.get(a) for a in cart} if CONFIG["cart_exclude_cart_types"] else set()
        cart_codes = {self.codes.get(a) for a in cart}
        recs: list[Rec] = []
        src = [int(self.cf.item_pos_[a]) for a in cart if a in self.cf.item_pos_.index]
        if src:
            scores = np.asarray(self.cf.S_[src].sum(axis=0)).ravel()
            order = np.argsort(-scores, kind="stable")
            picked = []
            for j in order:
                if scores[j] <= 0 or len(picked) >= k:
                    break
                aid = self.cf.items_[j]
                if aid in cart_set or self.codes.get(aid) in cart_codes or self.types.get(aid) in cart_types:
                    continue
                picked.append(j)
            if picked:
                contrib = self.cf.S_[src][:, picked].toarray()
                for col, j in enumerate(picked):
                    because = self.cf.items_[src[int(contrib[:, col].argmax())]]
                    recs.append(Rec(self.cf.items_[j], "Hay được mua cùng", SHELF_CART, because))
        if len(recs) < k:  # lấp bằng bán chạy 7 ngày thuộc loại khác
            have = cart_set | {r.article_id for r in recs}
            recs += self._popular(k - len(recs), exclude=have, skip_types=cart_types, skip_codes=cart_codes,
                                  shelf=SHELF_CART)
        return recs

    # ------------------------------------------------------------------ mua lại
    def _repeat(self, cid: int, k: int) -> list[Rec]:
        rows = self._rep_rows.get(cid)
        if rows is None:
            return []
        table = self.rep.scores_.iloc[rows[:k]]
        return [Rec(a, f"Bạn đã mua {int(n)} lần · lần cuối {d:%d/%m/%Y}", SHELF_REPEAT)
                for a, n, d in zip(table["article_id"], table["n_purchase"], table["last_date"])]

    def _repeat_live(self, cid: int | None, sp: tuple[str, ...], k: int) -> list[Rec]:
        """CategoryRepurchase với các món vừa mua: lần mua mới nhất = anchor (tuổi 0 ngày, bucket 0)."""
        rows = self._rep_rows.get(cid) if cid is not None else None
        base = (self.rep.scores_.iloc[rows][["article_id", "last_date", "n_purchase", "score"]].copy()
                if rows is not None else pd.DataFrame(columns=["article_id", "last_date", "n_purchase", "score"]))
        base["web"] = False
        base = base.set_index("article_id")
        for aid, cnt in Counter(sp).items():
            rate = self._rate0.get(self.types.get(aid), self._rate0_default)
            prev = int(base.at[aid, "n_purchase"]) if aid in base.index else 0
            base.loc[aid, ["last_date", "n_purchase", "score", "web"]] = [self.anchor, prev + cnt, rate, True]
        base = base.reset_index().sort_values(["score", "last_date", "article_id"],
                                              ascending=[False, False, False], kind="mergesort").head(k)
        out = []
        for a, n, d, web in zip(base["article_id"], base["n_purchase"], base["last_date"], base["web"]):
            reason = (f"Bạn vừa mua trên web · đã mua {int(n)} lần" if web
                      else f"Bạn đã mua {int(n)} lần · lần cuối {pd.Timestamp(d):%d/%m/%Y}")
            out.append(Rec(a, reason, SHELF_REPEAT))
        return out

    # ------------------------------------------------------------------ khám phá
    def _discovery(self, cid: int, k: int) -> list[Rec]:
        fused = self.blend.score_items(cid, k=k)
        if fused.empty:
            owned = set(self._owned(cid))
            return [Rec(r.article_id, "Bán chạy, bạn chưa mua", SHELF_DISCOVERY)
                    for r in self._popular(k, exclude=owned)]
        because = {}
        if fused[f"rank_{CF}"].notna().any():
            cfi = self.cf.score_items(cid, k=CONFIG["blend_pool_k"])
            because = dict(zip(cfi["article_id"], cfi["because_article_id"]))
        return [self._disc_rec(a, because.get(a) if pd.notna(rcf) else None)
                for a, rcf in zip(fused["article_id"], fused[f"rank_{CF}"])]

    def _disc_rec(self, aid: str, because: str | None) -> Rec:
        if because:
            return Rec(aid, "Khách mua món bạn đã mua cũng mua món này", SHELF_DISCOVERY, because)
        return Rec(aid, "Bán chạy trong loại hàng bạn hay mua", SHELF_DISCOVERY)

    def _discovery_live(self, cid: int | None, sp: tuple[str, ...], k: int) -> list[Rec]:
        """BlendModel chấm lại với các món vừa mua: chúng là 'ngày mua cuối' cho ItemCF và
        được cộng vào hồ sơ loại hàng của CategoryPopularity. Cùng công thức RRF như BlendModel."""
        owned = self._owned(cid)
        owned_set = set(owned) | set(sp)
        pool = CONFIG["blend_pool_k"]

        # --- ItemCF (source=last_day, binary, exclude_seen, exclude_owned_product_code)
        cf_rank, because = {}, {}
        src = [int(self.cf.item_pos_[a]) for a in dict.fromkeys(sp) if a in self.cf.item_pos_.index]
        if not src and cid is not None and cid in self.cf.user_pos_.index:
            src = self.cf.XL_[int(self.cf.user_pos_[cid])].indices.tolist()
        if src:
            scores = np.asarray(self.cf.S_[src].sum(axis=0)).ravel()
            owned_codes = {self.codes.get(a) for a in owned_set}
            picked = []
            for j in np.lexsort((-np.arange(len(scores)), -scores)):
                if scores[j] <= 0 or len(picked) >= pool:
                    break
                aid = self.cf.items_[j]
                if aid in owned_set or self.codes.get(aid) in owned_codes:
                    continue
                picked.append(j)
            if picked:
                contrib = self.cf.S_[src][:, picked].toarray()
                for r, j in enumerate(picked, start=1):
                    aid = self.cf.items_[j]
                    cf_rank[aid] = r
                    because[aid] = self.cf.items_[src[int(contrib[:, r - 1].argmax())]]

        # --- CategoryPopularity: share_u(c) * p(j|c), top max_categories loại
        cats = Counter(self.types.get(a) for a in owned + list(sp))
        cats.pop(None, None)
        total = sum(cats.values())
        cand = []
        for c, n in sorted(cats.items(), key=lambda x: (-x[1], x[0]))[: self.cp.max_categories]:
            g = self._cat_items.get(c)
            if g is not None:
                cand += [(a, n / total * p) for a, p in zip(g["article_id"], g["p_item"]) if a not in owned_set]
        cand.sort(key=lambda x: x[0], reverse=True)   # cùng thứ tự với model: score giảm, article_id giảm
        cand.sort(key=lambda x: -x[1])
        cp_rank = {a: r for r, (a, _) in enumerate(cand[:pool], start=1)}

        # --- RRF
        w_cp, c = CONFIG["blend_weight_cat_pop"], CONFIG["blend_rrf_c"]
        fused = Counter()
        for a, r in cp_rank.items():
            fused[a] += w_cp / (c + r)
        for a, r in cf_rank.items():
            fused[a] += (1 - w_cp) / (c + r)
        top = sorted(sorted(fused.items(), key=lambda x: x[0], reverse=True), key=lambda x: -x[1])[:k]
        recs = [self._disc_rec(a, because.get(a)) for a, _ in top]
        if not recs:
            recs = [Rec(r.article_id, "Bán chạy, bạn chưa mua", SHELF_DISCOVERY)
                    for r in self._popular(k, exclude=owned_set)]
        return recs

    # ------------------------------------------------------------------ thông tin
    def info(self) -> dict:
        return dict(name="ProductionRecommender", anchor=str(self.anchor.date()), ready_seconds=self.ready_seconds,
                    n_transactions=len(self.history), n_customers=len(self.known), config=CONFIG)

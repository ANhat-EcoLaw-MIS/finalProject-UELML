"""Lớp nối web <-> backend, dựa trên Orchestrator của team (orch_architecture/orchestrator.py).

KHÔNG sửa code backend. File này:
  1. nạp Orchestrator đã fit từ orch_architecture/orchestrator.pkl (file team chia sẻ, xem orchestrator_guide.md).
     Chưa có file thì tự dựng bằng đúng build_default() của team, fit rồi lưu vào cùng chỗ (như build_pickle.py);
  2. gọi orc.page(user) và đổi các kệ của team sang nhãn mục của web:
       "Trending now"       -> popular    (chỉ khách mới / dự phòng, đúng thiết kế Orchestrator)
       "Buy it again"       -> repeat
       "Because you bought" -> discovery  (cân bằng để có vài món "Vì bạn đã mua X", như guide khuyên)
     lý do được viết lại bằng tiếng Việt từ chính các model trong Orchestrator;
  3. phần Orchestrator chưa có, viết ở đây bằng các model đã fit lấy từ Orchestrator:
       - giỏ hàng (mua kèm) -> cộng các hàng của ma trận tương đồng ItemCF (S_);
       - tính lại sau thanh toán (session_purchases) -> chấm lại theo đúng công thức của từng model,
         coi các món vừa mua là lần mua mới nhất (không phải fit lại).
"""
from __future__ import annotations

import hashlib
import logging
import os
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

from orch_architecture.orchestrator import Orchestrator, build_default  # noqa: E402

log = logging.getLogger("stylerec.recommender")

ORCH_PKL = Path(os.environ.get("STYLEREC_ORCH", dataset.REPO / "orch_architecture" / "orchestrator.pkl"))
TEAM_SHELVES = {"Trending now": SHELF_POPULAR, "Buy it again": SHELF_REPEAT, "Because you bought": SHELF_DISCOVERY}
SETTINGS = dict(
    trending_for_returning=False,   # Orchestrator chỉ cho khách mới xem Trending; True = thêm kệ bán chạy cho khách quen
    discovery_min_because=4,        # kệ khám phá: cố giữ tối thiểu N món "Vì bạn đã mua X" (nếu model có)
    cart_exclude_cart_types=True,   # giỏ hàng: không gợi ý cùng loại sản phẩm đã có trong giỏ
)


def load_orchestrator(history: pd.DataFrame, articles: pd.DataFrame, path: Path = ORCH_PKL) -> Orchestrator:
    """Nạp orchestrator.pkl; chưa có thì dựng + fit bằng build_default() rồi lưu (giống build_pickle.py)."""
    if path.exists():
        log.info("Nạp Orchestrator: %s", path)
        return Orchestrator.load(path)
    log.warning("Chưa có %s -> tự fit bằng build_default() của team (~1–1,5 phút) rồi lưu lại.", path)
    art = articles.drop_duplicates("article_id").set_index("article_id")
    anchor = history["t_dat"].max() + pd.Timedelta(days=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orc = build_default(art["product_type_no"], article_product=art["product_code"]).fit(history, anchor)
    if path.exists():   # có người đặt file vào trong lúc đang fit -> KHÔNG ghi đè, dùng file đó
        log.warning("%s vừa xuất hiện trong lúc fit -> dùng file đó, không ghi đè.", path)
        return Orchestrator.load(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".building")
    orc.save(tmp)
    try:
        tmp.rename(path)          # rename không ghi đè file đã có (Windows báo lỗi nếu path đã tồn tại)
    except FileExistsError:
        tmp.unlink(missing_ok=True)
        log.warning("%s đã có sẵn -> giữ file đó, bỏ bản vừa fit.", path)
        return Orchestrator.load(path)
    log.info("Đã lưu Orchestrator tự fit vào %s", path)
    return orc


class ProductionRecommender:
    """Recommender thật theo hợp đồng recommender_port.py, đứng trên Orchestrator của team."""

    def __init__(self, history: pd.DataFrame | None = None, articles: pd.DataFrame | None = None,
                 orchestrator: Orchestrator | None = None, orch_path: Path | str = ORCH_PKL):
        if history is None or articles is None:
            history, articles = dataset.load_full()
        t0 = time.time()
        self.history = history
        self.orch_path = Path(orch_path)
        self.orc = orchestrator or load_orchestrator(history, articles, self.orch_path)
        self.orch_sha256 = ""
        if orchestrator is None and self.orch_path.exists():   # để /healthz cho biết đang dùng đúng file nào
            with self.orch_path.open("rb") as f:
                self.orch_sha256 = hashlib.file_digest(f, "sha256").hexdigest()
            log.info("orchestrator.pkl sha256=%s (%s bytes)", self.orch_sha256[:16], self.orch_path.stat().st_size)
        art = articles.drop_duplicates("article_id").set_index("article_id")
        self.types, self.codes = art["product_type_no"], art["product_code"]

        by_name = {s.name: s for s in self.orc.sections}
        missing = set(TEAM_SHELVES) - set(by_name)
        if missing:
            raise ValueError(f"Orchestrator thiếu kệ {sorted(missing)}; có: {sorted(by_name)}")
        self.trending = self.orc.fallback or by_name["Trending now"]
        self.pop = by_name["Trending now"].model
        self.rep = by_name["Buy it again"].model
        self.blend = by_name["Because you bought"].model
        self.cf, self.cp = self.blend.models["cf"], self.blend.models["cat_pop"]
        self.anchor = pd.Timestamp(self.orc.anchor_date_)
        data_anchor = history["t_dat"].max() + pd.Timedelta(days=1)
        if data_anchor != self.anchor:
            log.warning("Mốc của Orchestrator (%s) khác dữ liệu web (%s)", self.anchor.date(), data_anchor.date())
        self._index()
        self._cache: OrderedDict = OrderedDict()
        self.ready_seconds = round(time.time() - t0, 1)
        log.info("ProductionRecommender (Orchestrator) sẵn sàng sau %.1fs, mốc %s",
                 self.ready_seconds, self.anchor.date())

    def _index(self):
        self.known = self.orc.known_users_
        self._hist_rows = self.history.groupby("customer_id").indices
        self._rep_rows = self.rep.scores_.groupby("customer_id").indices
        rates = self.rep.rate_table_
        b0 = rates[rates["bucket"] == 0]
        self._rate0 = dict(zip(b0["category"], b0["rate"]))
        self._rate0_default = float(self.rep.bucket_rate_.iloc[0])
        self._pop_rank = self.pop.ranking_.reset_index(drop=True)
        self._pop_count = dict(zip(self._pop_rank["article_id"], self._pop_rank["score"]))
        self._cat_items = {c: g for c, g in self.cp.cat_items_.groupby("cat")}

    # ------------------------------------------------------------------ helpers
    def _cid(self, user_id) -> int | None:
        try:
            cid = int(str(user_id).strip())
        except (TypeError, ValueError):
            return None
        return cid if cid in self.known else None

    def _owned(self, cid: int | None) -> list[str]:
        rows = self._hist_rows.get(cid) if cid is not None else None
        return [] if rows is None else self.history["article_id"].to_numpy()[rows].tolist()

    def _pop_rec(self, aid: str, shelf=SHELF_POPULAR) -> Rec:
        return Rec(aid, f"Đã bán {int(self._pop_count.get(aid, 0))} sản phẩm trong 7 ngày qua", shelf)

    def _popular(self, k: int, exclude=frozenset(), skip_types=frozenset(), skip_codes=frozenset(),
                 shelf=SHELF_POPULAR) -> list[Rec]:
        out = []
        for aid in self._pop_rank["article_id"]:
            if aid in exclude or self.types.get(aid) in skip_types or self.codes.get(aid) in skip_codes:
                continue
            out.append(self._pop_rec(aid, shelf))
            if len(out) >= k:
                break
        return out

    def _disc_rec(self, aid: str, because: str | None) -> Rec:
        if because:
            return Rec(aid, "Khách mua món bạn đã mua cũng mua món này", SHELF_DISCOVERY, because)
        return Rec(aid, "", SHELF_DISCOVERY)

    @staticmethod
    def _balance(recs: list[Rec], k: int, min_because: int) -> list[Rec]:
        """Giữ thứ tự của model nhưng bảo đảm tối thiểu min_because món có 'because' trong top k
        (thay các món cat_pop cuối danh sách). Đây là bước lọc phía server mà orchestrator_guide.md khuyên."""
        top, rest = recs[:k], recs[k:]
        need = min(min_because, sum(1 for r in recs if r.because)) - sum(1 for r in top if r.because)
        extra = [r for r in rest if r.because][:max(need, 0)]
        if extra:
            drop = set(id(r) for r in [r for r in top if not r.because][-len(extra):])
            top = [r for r in top if id(r) not in drop]
            order = {id(r): i for i, r in enumerate(recs)}
            top = sorted(top + extra, key=lambda r: order[id(r)])
        return top

    # ------------------------------------------------------------------ hợp đồng
    def recommend_for_user(self, user_id, k: int = 12, session_purchases: Sequence[str] = ()) -> list[Rec]:
        cid = self._cid(user_id)
        sp = tuple(str(a).zfill(10) for a in session_purchases if str(a).strip())
        key = (cid if cid is not None else "new", sp, k)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        if sp:      # vừa mua trên web -> khách quen, chấm lại ngay bằng model của Orchestrator
            recs = self._repeat_live(cid, sp, k) + self._discovery_live(cid, sp, k)
        else:
            recs = self._from_page(cid, k)
        returning = cid is not None or bool(sp)
        if returning and SETTINGS["trending_for_returning"]:
            shown = {r.article_id for r in recs} | set(sp) | set(self._owned(cid))
            recs += self._popular(k, exclude=shown)

        self._cache[key] = recs
        if len(self._cache) > 512:
            self._cache.popitem(last=False)
        return recs

    def _from_page(self, cid: int | None, k: int) -> list[Rec]:
        """Kết quả của orc.page() -> list[Rec] với lý do tiếng Việt."""
        page = self.orc.page(cid if cid is not None else -1)
        recs: list[Rec] = []
        for name, items in page.items():
            shelf = TEAM_SHELVES.get(name)
            ids = [i["article_id"] for i in items]
            if shelf == SHELF_POPULAR or name == self.trending.name:
                recs += [self._pop_rec(a) for a in ids[:k]]
            elif shelf == SHELF_REPEAT:
                recs += self._repeat_reasons(cid, ids[:k])
            elif shelf == SHELF_DISCOVERY:
                cfi = self.cf.score_items(cid, k=self.blend.pool_k)   # cùng pool CF mà Blend dùng
                because = dict(zip(cfi["article_id"], cfi["because_article_id"]))
                disc = [self._disc_rec(a, because.get(a)) for a in ids]
                recs += self._balance(disc, k, SETTINGS["discovery_min_because"])
        return recs

    def _repeat_reasons(self, cid: int, ids: list[str]) -> list[Rec]:
        rows = self._rep_rows.get(cid)
        info = {}
        if rows is not None:
            t = self.rep.scores_.iloc[rows]
            info = dict(zip(t["article_id"], zip(t["n_purchase"], t["last_date"])))
        out = []
        for a in ids:
            n, d = info.get(a, (None, None))
            out.append(Rec(a, f"Bạn đã mua {int(n)} lần · lần cuối {pd.Timestamp(d):%d/%m/%Y}" if n else
                           "Bạn đã từng mua món này", SHELF_REPEAT))
        return out

    def recommend_for_cart(self, article_ids: Sequence[str], k: int = 6) -> list[Rec]:
        cart = list(dict.fromkeys(str(a).zfill(10) for a in article_ids if str(a).strip()))
        if not cart:
            return []
        cart_set = set(cart)
        cart_types = {self.types.get(a) for a in cart} if SETTINGS["cart_exclude_cart_types"] else set()
        cart_codes = {self.codes.get(a) for a in cart}
        recs: list[Rec] = []
        src = [int(self.cf.item_pos_[a]) for a in cart if a in self.cf.item_pos_.index]
        if src:
            scores = np.asarray(self.cf.S_[src].sum(axis=0)).ravel()
            picked = []
            for j in np.argsort(-scores, kind="stable"):
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

    # ------------------------------------------------------------------ tính lại sau thanh toán
    def _repeat_live(self, cid: int | None, sp: tuple[str, ...], k: int) -> list[Rec]:
        """CategoryRepurchase với các món vừa mua: lần mua mới nhất = mốc (tuổi 0 ngày, bucket 0)."""
        rows = self._rep_rows.get(cid) if cid is not None else None
        cols = ["article_id", "last_date", "n_purchase", "score"]
        base = self.rep.scores_.iloc[rows][cols].copy() if rows is not None else pd.DataFrame(columns=cols)
        base["web"] = False
        base = base.set_index("article_id")
        for aid, cnt in Counter(sp).items():
            rate = self._rate0.get(self.types.get(aid), self._rate0_default)
            prev = int(base.at[aid, "n_purchase"]) if aid in base.index else 0
            base.loc[aid, ["last_date", "n_purchase", "score", "web"]] = [self.anchor, prev + cnt, rate, True]
        base = base.reset_index().sort_values(["score", "last_date", "article_id"],
                                              ascending=[False, False, False], kind="mergesort").head(k)
        return [Rec(a, f"Bạn vừa mua trên web · đã mua {int(n)} lần" if web
                    else f"Bạn đã mua {int(n)} lần · lần cuối {pd.Timestamp(d):%d/%m/%Y}", SHELF_REPEAT)
                for a, n, d, web in zip(base["article_id"], base["n_purchase"], base["last_date"], base["web"])]

    def _discovery_live(self, cid: int | None, sp: tuple[str, ...], k: int) -> list[Rec]:
        """Blend chấm lại: các món vừa mua là 'ngày mua cuối' cho ItemCF và được cộng vào hồ sơ loại hàng
        của CategoryPopularity; gộp bằng RRF với đúng trọng số / rrf_c / pool_k của Blend trong Orchestrator."""
        owned = self._owned(cid)
        owned_set = set(owned) | set(sp)
        pool = self.blend.pool_k

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

        w, c = self.blend.weights, self.blend.rrf_c
        fused = Counter()
        for a, r in cp_rank.items():
            fused[a] += w["cat_pop"] / (c + r)
        for a, r in cf_rank.items():
            fused[a] += w["cf"] / (c + r)
        ranked = sorted(sorted(fused.items(), key=lambda x: x[0], reverse=True), key=lambda x: -x[1])
        recs = [self._disc_rec(a, because.get(a)) for a, _ in ranked]
        recs = self._balance(recs, k, SETTINGS["discovery_min_because"])
        if not recs:
            recs = [Rec(r.article_id, "Bán chạy, bạn chưa mua", SHELF_DISCOVERY)
                    for r in self._popular(k, exclude=owned_set)]
        return recs

    # ------------------------------------------------------------------ thông tin
    def info(self) -> dict:
        return dict(name="ProductionRecommender (Orchestrator)", orchestrator=str(self.orch_path),
                    orchestrator_sha256=self.orch_sha256,
                    orchestrator_bytes=self.orch_path.stat().st_size if self.orch_path.exists() else None,
                    shelves=[s.name for s in self.orc.sections], anchor=str(self.anchor.date()),
                    ready_seconds=self.ready_seconds, n_customers=len(self.known), settings=SETTINGS)

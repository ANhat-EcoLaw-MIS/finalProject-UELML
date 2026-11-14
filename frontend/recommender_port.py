"""HỢP ĐỒNG giữa web (Flask) và recommender.

Web chỉ biết hai hàm dưới đây và coi recommender là hộp đen. Recommender thật (production_recommender.py,
bọc Orchestrator của team) và các recommender giả trong tests/ đều tuân theo hợp đồng này.

    recommend_for_user(user_id, k, session_purchases)  -> trang chủ (các mục gợi ý của một khách)
    recommend_for_cart(article_ids, k)                 -> giỏ hàng (mua kèm)

Các mục ("shelf") theo 3 bài toán của team:
    popular    cold-start: bán chạy 7 ngày           (PopularityModel, window_days=7)
    repeat     mua lại món quen                      (CategoryRepurchaseModel)
    discovery  khám phá món mới, "vì bạn đã mua X"   (BlendModel = CategoryPopularity + ItemCF last-day)
    cart       mua kèm theo giỏ hàng                 (ma trận tương đồng của ItemCF)
Item không có nhãn mục (recommender khác trả về) được gom vào mục "for_you".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol, Sequence, runtime_checkable

SHELF_POPULAR = "popular"
SHELF_REPEAT = "repeat"
SHELF_DISCOVERY = "discovery"
SHELF_CART = "cart"
SHELF_FOR_YOU = "for_you"
SHELVES = (SHELF_POPULAR, SHELF_REPEAT, SHELF_DISCOVERY, SHELF_CART, SHELF_FOR_YOU)


@dataclass(frozen=True)
class Rec:
    """Một sản phẩm được gợi ý: chỉ có MÃ và lý do; tên, giá, ảnh web tự tra trong danh mục."""
    article_id: str             # mã 10 chữ số dạng chuỗi, giữ số 0 đầu, ví dụ "0108775015"
    reason: str = ""            # câu giải thích ngắn hiển thị dưới thẻ sản phẩm
    shelf: str = SHELF_FOR_YOU  # popular | repeat | discovery | cart | for_you
    because: str = ""           # (tuỳ chọn) mã món đã kích hoạt gợi ý -> web hiện "Vì bạn đã mua <tên>"


@runtime_checkable
class Recommender(Protocol):
    def recommend_for_user(self, user_id: str | None, k: int = 12,
                           session_purchases: Sequence[str] = ()) -> list[Rec]:
        """user_id           mã khách trong dữ liệu. None / mã lạ (tài khoản mới) = khách mới -> chỉ mục popular.
        k                 số sản phẩm tối đa MỖI mục.
        session_purchases món khách đã mua trên web (cũ -> mới), chưa có trong dữ liệu huấn luyện;
                          dùng để tính lại gợi ý ngay sau thanh toán."""
        ...

    def recommend_for_cart(self, article_ids: Sequence[str], k: int = 6) -> list[Rec]:
        """Mua kèm cho giỏ hàng. Giỏ rỗng -> []. Không trả lại món đã có trong giỏ."""
        ...


def ensure_recommender(obj) -> Recommender:
    """Báo lỗi sớm, rõ ràng khi tiêm nhầm object vào web."""
    if not isinstance(obj, Recommender):
        raise TypeError(f"{type(obj).__name__} chưa đúng hợp đồng: cần recommend_for_user() và recommend_for_cart().")
    return obj


def _aid(x: Any) -> str:
    s = str(x).strip()
    return s.zfill(10) if s.isdigit() else s


def normalize(items: Any, default_shelf: str = SHELF_FOR_YOU) -> list[Rec]:
    """Chấp nhận Rec / dict / tuple / str / DataFrame có cột article_id -> list[Rec] (bỏ trùng, giữ thứ tự).
    Dùng khi nối một recommender không trả đúng Rec."""
    if items is None:
        return []
    if hasattr(items, "to_dict") and hasattr(items, "columns"):          # DataFrame
        items = items.to_dict("records")
    out: list[Rec] = []
    seen: set[tuple[str, str]] = set()
    for it in items if isinstance(items, Iterable) and not isinstance(items, (str, bytes)) else [items]:
        if isinstance(it, Rec):
            rec = it
        elif isinstance(it, dict):
            if "article_id" not in it:
                continue
            rec = Rec(_aid(it["article_id"]), str(it.get("reason") or ""), str(it.get("shelf") or default_shelf),
                      _aid(it["because"]) if it.get("because") else "")
        elif isinstance(it, (tuple, list)) and it:
            rec = Rec(_aid(it[0]), str(it[1]) if len(it) > 1 else "", str(it[2]) if len(it) > 2 else default_shelf)
        elif isinstance(it, (str, int)):
            rec = Rec(_aid(it), "", default_shelf)
        else:
            continue
        if rec.shelf not in SHELVES:
            rec = Rec(rec.article_id, rec.reason, SHELF_FOR_YOU, rec.because)
        key = (rec.shelf, rec.article_id)
        if key not in seen:
            seen.add(key)
            out.append(rec)
    return out

"""Test web (Flask test client) trên bộ dữ liệu mẫu, không cần model backend.

    cd frontend && python -m pytest -q          (hoặc: python tests/test_app.py)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from app import create_app  # noqa: E402
from catalog import Catalog  # noqa: E402
from mock_recommender import MockRecommender  # noqa: E402
from recommender_port import (SHELF_CART, SHELF_DISCOVERY, SHELF_POPULAR, SHELF_REPEAT, Rec,  # noqa: E402
                              normalize)

CATALOG = Catalog.sample()
IDS = CATALOG.order


class FakeRecommender:
    """Ghi lại input, trả danh sách cố định (có cả mã lạ và món trùng giỏ để thử độ bền)."""
    def __init__(self):
        self.calls = []

    def recommend_for_user(self, user_id, k=12, session_purchases=()):
        self.calls.append(("user", user_id, tuple(session_purchases)))
        recs = [Rec(IDS[0], "pop", SHELF_POPULAR), Rec("9999999999", "mã lạ", SHELF_POPULAR)]
        try:
            int(str(user_id))
            warm = True
        except (TypeError, ValueError):
            warm = False
        if warm or session_purchases:
            recs += [Rec(IDS[1], "rep", SHELF_REPEAT), Rec(IDS[2], "disc", SHELF_DISCOVERY, because=IDS[1])]
        return recs

    def recommend_for_cart(self, article_ids, k=6):
        self.calls.append(("cart", tuple(article_ids)))
        return [Rec(IDS[3], "cart"), Rec(article_ids[0], "trùng giỏ"), "0000000000"]


class Broken:
    def recommend_for_user(self, *a, **kw):
        raise RuntimeError("backend down")

    def recommend_for_cart(self, *a, **kw):
        raise RuntimeError("backend down")


def make(rec=None):
    rec = rec or FakeRecommender()
    app = create_app(rec, catalog=CATALOG, runtime_dir=tempfile.mkdtemp(), secret_key="test")
    app.config["TESTING"] = True
    return app, app.test_client(), rec


def shelf_keys(client, **params):
    data = client.get("/api/recommendations", query_string=params).get_json()
    return data, [s["key"] for s in data["shelves"]]


DEMO = CATALOG.demo_users


def test_cold_guest_sees_one_shelf():
    _, c, _ = make()
    data, keys = shelf_keys(c)
    assert data["scenario"] == "cold" and keys == [SHELF_POPULAR]
    assert c.get("/").status_code == 200


def test_demo_users_see_three_shelves():
    _, c, _ = make()
    for d in DEMO:
        assert c.post(f"/auth/demo/{d['id']}").status_code == 302
        data, keys = shelf_keys(c)
        assert keys == [SHELF_REPEAT, SHELF_DISCOVERY, SHELF_POPULAR]
        assert data["scenario"] in ("light", "heavy")
        html = c.get("/").get_data(as_text=True)
        assert "Mua lại món quen" in html and "Có thể bạn sẽ thích" in html


def test_unknown_article_ids_are_dropped_and_because_rendered():
    _, c, _ = make()
    c.post(f"/auth/demo/{DEMO[0]['id']}")
    data, _ = shelf_keys(c)
    ids = [i["article_id"] for s in data["shelves"] for i in s["items"]]
    assert "9999999999" not in ids
    disc = next(s for s in data["shelves"] if s["key"] == SHELF_DISCOVERY)["items"][0]
    assert disc["reason"].startswith("Vì bạn đã mua ") and disc["because"] == IDS[1]
    for key in ("article_id", "name", "price_vnd", "image_url", "url", "reason", "because"):
        assert key in disc


def test_signup_is_cold_then_checkout_makes_warm():
    _, c, rec = make()
    r = c.post("/auth/signup", data=dict(name="An", email="an@example.test", password="secret1"))
    assert r.status_code == 302
    assert shelf_keys(c)[0]["scenario"] == "cold"
    c.post("/cart/add", data=dict(article_id=IDS[5], size="M", qty=2, surface="home", shelf="popular", position=3))
    r = c.post("/checkout", data=dict(name="An", phone="0901234567", address="12 Lê Lợi, Quận 1",
                                      shipping="standard", payment="cod"))
    assert r.status_code == 302 and "/order/" in r.headers["Location"]
    page = c.get(r.headers["Location"]).get_data(as_text=True)
    assert "Đặt hàng thành công" in page and "Khách mới" in page
    data, keys = shelf_keys(c)
    assert data["scenario"] == "light" and SHELF_REPEAT in keys
    assert rec.calls[-1][2] == (IDS[5], IDS[5])          # session_purchases truyền đúng cho recommender
    assert "Đơn hàng trên web" in c.get("/profile").get_data(as_text=True)


def test_checkout_requires_login_and_validates():
    _, c, _ = make()
    c.post("/cart/add", data=dict(article_id=IDS[0]))
    assert "/auth/signin" in c.get("/checkout").headers["Location"]
    c.post("/auth/demo/cold")
    r = c.post("/checkout", data=dict(name="", phone="12", address="x", shipping="?", payment="cod"))
    assert r.status_code == 200 and "Số điện thoại" in r.get_data(as_text=True)


def test_signin_with_password():
    _, c, _ = make()
    c.post("/auth/signup", data=dict(name="Bình", email="b@example.test", password="secret1"))
    c.post("/auth/signout")
    assert c.post("/auth/signin", data=dict(email="b@example.test", password="wrong")).status_code == 200
    assert c.post("/auth/signin", data=dict(email="B@example.test", password="secret1")).status_code == 302
    assert "Bình" in c.get("/").get_data(as_text=True)


def test_cart_cross_sell_excludes_basket_and_unknown():
    _, c, rec = make()
    c.post("/cart/add", data=dict(article_id=IDS[0]))
    html = c.get("/cart").get_data(as_text=True)
    assert "Thường được mua kèm" in html
    data = c.get("/api/recommendations?surface=cart").get_json()
    items = [i["article_id"] for i in data["shelves"][0]["items"]]
    assert items == [IDS[3]]
    assert rec.calls[-1] == ("cart", (IDS[0],))
    data = c.get(f"/api/recommendations?surface=cart&items={IDS[7]},{IDS[8]}").get_json()
    assert rec.calls[-1] == ("cart", (IDS[7], IDS[8]))


def test_events_logged_all_types():
    app, c, _ = make()
    c.post(f"/auth/demo/{DEMO[0]['id']}")
    r = c.post("/api/events", json={"events": [
        dict(event="impression", article_id=IDS[0], surface="home", shelf="popular", position=1),
        dict(event="click", article_id=IDS[0], surface="home", shelf="popular", position=1),
        dict(event="hack", article_id=IDS[0]), dict(event="click", article_id="nope")]})
    assert r.get_json() == dict(ok=True, logged=2, rejected=2)
    assert c.post("/api/events", json=dict(event="impression", article_id=IDS[1])).get_json()["logged"] == 1
    c.post("/cart/add", data=dict(article_id=IDS[2], surface="home", shelf="repeat", position=2),
           headers={"X-Requested-With": "fetch"})
    c.post("/checkout", data=dict(name="Demo", phone="0901234567", address="1 Nguyễn Huệ, Q1",
                                  shipping="express", payment="card"))
    ev = app.config["STORE"].events()
    assert {"impression", "click", "add_to_cart", "purchase"} <= set(ev["event"])
    purchase = ev[ev["event"] == "purchase"].iloc[0]
    assert purchase["shelf"] == "repeat" and purchase["surface"] == "home" and purchase["order_id"]
    assert list(ev.columns) == ["ts", "session_id", "user_id", "scenario", "event", "surface", "shelf",
                                "article_id", "position", "order_id", "value"]


def test_broken_recommender_does_not_crash():
    _, c, _ = make(Broken())
    assert c.get("/").status_code == 200
    c.post("/cart/add", data=dict(article_id=IDS[0]))
    assert c.get("/cart").status_code == 200
    assert c.get("/api/recommendations").get_json()["shelves"] == []


def test_pages_render_and_404():
    _, c, _ = make()
    assert c.get("/products?q=a&sort=price_desc").status_code == 200
    assert c.get(f"/product/{IDS[0]}").status_code == 200
    assert c.get("/product/0000000000").status_code == 404
    assert c.get("/auth/signin").status_code == 200 and c.get("/auth/signup").status_code == 200
    assert c.get("/healthz").get_json()["ok"]


def test_info_pages_and_newsletter():
    from app import INFO_PAGES
    _, c, _ = make()
    for slug in INFO_PAGES:
        assert c.get(f"/info/{slug}").status_code == 200
    assert c.get("/info/khong-co").status_code == 404
    html = c.get("/").get_data(as_text=True)
    assert all(f"/info/{slug}" in html for slug in INFO_PAGES)       # footer không có link chết
    r = c.post("/newsletter", data=dict(email="a@example.test"), follow_redirects=True)
    assert "Cảm ơn bạn đã đăng ký" in r.get_data(as_text=True)


def test_demo_reset_clears_demo_orders():
    app, c, _ = make()
    c.post(f"/auth/demo/{DEMO[0]['id']}")
    c.post("/cart/add", data=dict(article_id=IDS[0]))
    c.post("/checkout", data=dict(name="Demo", phone="0901234567", address="1 Nguyễn Huệ, Q1",
                                  shipping="standard", payment="cod"))
    assert app.config["STORE"].orders(DEMO[0]["id"])
    c.post("/demo/reset")
    assert not app.config["STORE"].orders(DEMO[0]["id"])


def test_wrong_recommender_rejected_and_normalize():
    with pytest.raises(TypeError):
        create_app(object(), catalog=CATALOG, runtime_dir=tempfile.mkdtemp())
    import pandas as pd
    recs = normalize(pd.DataFrame({"article_id": ["108775015", "108775015"], "score": [1, 2]}))
    assert recs == [Rec("0108775015", "", "for_you")]
    assert normalize([("1", "r", "weird"), {"article_id": 2, "shelf": "repeat"}, 3])[0].shelf == "for_you"


def test_mock_recommender_contract():
    m = MockRecommender(CATALOG)
    cold = m.recommend_for_user(None)
    assert cold and all(r.shelf == SHELF_POPULAR for r in cold)
    warm = m.recommend_for_user(DEMO[0]["id"])
    assert {SHELF_REPEAT, SHELF_DISCOVERY} <= {r.shelf for r in warm}
    cart = m.recommend_for_cart([IDS[0]])
    assert cart and IDS[0] not in [r.article_id for r in cart]
    assert m.recommend_for_cart([]) == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

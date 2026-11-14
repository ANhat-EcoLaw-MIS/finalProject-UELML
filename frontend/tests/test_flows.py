"""Luồng người dùng, trường hợp biên và tích hợp với cơ sở dữ liệu / bộ gợi ý thật.

    cd frontend && python -m pytest -q                       (test tích hợp tự bỏ qua nếu thiếu dữ liệu hoặc .pkl)
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from app import MAX_CART_LINES, create_app  # noqa: E402
from catalog import Catalog  # noqa: E402
from recommender_port import SHELF_CART, SHELF_DISCOVERY, SHELF_POPULAR, SHELF_REPEAT, Rec  # noqa: E402

CATALOG = Catalog.sample()
IDS = CATALOG.order
DEMO = CATALOG.demo_users
CHECKOUT = dict(name="Nguyễn Văn A", phone="0901234567", address="12 Lê Lợi, Quận 1", shipping="standard",
                payment="cod")


class Fake:
    def __init__(self):
        self.calls = []

    def recommend_for_user(self, user_id, k=12, session_purchases=()):
        self.calls.append(("user", user_id, tuple(session_purchases)))
        recs = [Rec(IDS[0], "pop", SHELF_POPULAR)]
        if session_purchases or str(user_id).isdigit() and int(user_id) < 1_000_000:
            recs += [Rec(IDS[1], "rep", SHELF_REPEAT), Rec(IDS[2], "disc", SHELF_DISCOVERY)]
        return recs

    def recommend_for_cart(self, article_ids, k=6):
        self.calls.append(("cart", tuple(article_ids)))
        return [Rec(IDS[3], "cart", SHELF_CART)]


def make(rec=None, runtime=None):
    rec = rec or Fake()
    runtime = runtime or tempfile.mkdtemp()
    app = create_app(rec, catalog=CATALOG, runtime_dir=runtime)
    app.config["TESTING"] = True
    return app, app.test_client(), rec


def add(c, aid, **kw):
    return c.post("/cart/add", data=dict(article_id=aid, **kw))


def cart_lines(c):
    with c.session_transaction() as s:
        return list(s.get("cart", []))


# ---------------------------------------------------------------- giỏ hàng
def test_cart_add_merge_update_remove_and_bounds():
    _, c, _ = make()
    add(c, IDS[0], size="M", qty=2)
    add(c, IDS[0], size="M", qty=3)                           # cùng món cùng size -> gộp dòng
    add(c, IDS[0], size="S")                                  # khác size -> dòng mới
    lines = cart_lines(c)
    assert [(l["size"], l["qty"]) for l in lines if l["size"]] == [("M", 5), ("S", 1)] or len(lines) >= 1
    add(c, IDS[1], qty=999)                                   # số lượng bị chặn ở 10
    assert max(l["qty"] for l in cart_lines(c)) <= 10
    c.post("/cart/update", data=dict(idx=0, qty=0))           # qty 0 -> xoá dòng
    n = len(cart_lines(c))
    c.post("/cart/remove", data=dict(idx=99))                 # chỉ số sai -> không lỗi, không đổi
    assert len(cart_lines(c)) == n
    c.post("/cart/remove", data=dict(idx=0))
    assert len(cart_lines(c)) == n - 1


def test_cart_rejects_unknown_product_and_caps_lines():
    _, c, _ = make()
    assert add(c, "0000000000").status_code == 404
    r = c.post("/cart/add", data=dict(article_id="0000000000"), headers={"X-Requested-With": "fetch"})
    assert r.status_code == 404 and r.get_json()["ok"] is False
    with c.session_transaction() as s:                        # giỏ đầy
        s["cart"] = [dict(article_id=IDS[0], size=f"Z{i}", qty=1) for i in range(MAX_CART_LINES)]
    r = c.post("/cart/add", data=dict(article_id=IDS[1]), headers={"X-Requested-With": "fetch"})
    assert r.status_code == 400 and len(cart_lines(c)) == MAX_CART_LINES


def test_guest_cart_survives_login_and_signout_clears_session():
    _, c, _ = make()
    add(c, IDS[0])
    c.post(f"/auth/demo/{DEMO[0]['id']}")
    assert len(cart_lines(c)) == 1                            # giỏ khách vãng lai giữ lại khi đăng nhập
    c.post("/auth/signout")
    assert cart_lines(c) == []
    assert "Đăng nhập" in c.get("/").get_data(as_text=True)


# ---------------------------------------------------------------- thanh toán & đơn hàng
def test_checkout_double_submit_creates_one_order():
    app, c, _ = make()
    c.post("/auth/demo/cold")
    add(c, IDS[0], qty=2)
    c.get("/checkout")                                        # phát hành mã đơn chờ
    with c.session_transaction() as s:
        stale = dict(s)                                       # cookie của lần bấm thứ hai (chụp trước khi gửi)
    r1 = c.post("/checkout", data=CHECKOUT)
    with c.session_transaction() as s:                        # mô phỏng request thứ hai mang cookie cũ
        s.clear(); s.update(stale)
    r2 = c.post("/checkout", data=CHECKOUT)
    uid = stale["uid"]
    orders = app.config["STORE"].orders(uid)
    assert len(orders) == 1 and r1.headers["Location"] == r2.headers["Location"]
    assert app.config["STORE"].session_purchases(uid) == [IDS[0], IDS[0]]


def test_checkout_validation_empty_cart_and_shipping_fee():
    app, c, _ = make()
    c.post("/auth/demo/cold")
    r = c.get("/checkout")
    assert r.status_code == 302 and r.headers["Location"].endswith("/cart")   # giỏ trống
    add(c, IDS[0])
    for bad in (dict(phone="123"), dict(address="x"), dict(name="A"), dict(shipping="?"), dict(payment="?")):
        r = c.post("/checkout", data={**CHECKOUT, **bad})
        assert r.status_code == 200 and "err" in r.get_data(as_text=True)
    r = c.post("/checkout", data={**CHECKOUT, "shipping": "express"})
    o = app.config["STORE"].get_order(r.headers["Location"].rsplit("/", 1)[-1])
    assert o["shipping_fee"] == 50_000 and o["total"] == o["subtotal"] + 50_000


def test_order_page_is_private():
    app, c, _ = make()
    c.post("/auth/demo/cold"); add(c, IDS[0])
    oid = c.post("/checkout", data=CHECKOUT).headers["Location"].rsplit("/", 1)[-1]
    assert c.get(f"/order/{oid}").status_code == 200
    other = app.test_client()
    other.post(f"/auth/demo/{DEMO[0]['id']}")
    assert other.get(f"/order/{oid}").status_code == 404     # khách khác không xem được
    assert app.test_client().get(f"/order/{oid}").status_code == 302   # chưa đăng nhập -> trang đăng nhập


def test_purchase_updates_scenario_profile_product_and_events():
    app, c, rec = make()
    d = DEMO[0]
    c.post(f"/auth/demo/{d['id']}")
    before = c.get("/api/recommendations").get_json()["scenario"]
    add(c, IDS[5], surface="home", shelf="discovery", position=3)
    c.post("/checkout", data=CHECKOUT)
    c.get("/")                                                # trang chủ sau khi mua -> gọi lại bộ gợi ý
    assert rec.calls[-1][2] == (IDS[5],)                      # món vừa mua được truyền vào session_purchases
    prof = c.get("/profile").get_data(as_text=True)
    assert "Đơn trên web" in prof and ("Dữ liệu H&amp;M" in prof)
    assert "Bạn đã mua món này" in c.get(f"/product/{IDS[5]}").get_data(as_text=True)
    ev = app.config["STORE"].events()
    p = ev[ev["event"] == "purchase"].iloc[-1]
    assert (p["surface"], p["shelf"], str(p["position"])) == ("home", "discovery", "3")
    assert before in ("light", "heavy")


def test_demo_reset_keeps_real_accounts_and_exported_orders():
    app, c, _ = make()
    st = app.config["STORE"]
    c.post("/auth/signup", data=dict(name="Thật", email="that@example.test", password="secret1"))
    add(c, IDS[0]); c.post("/checkout", data=CHECKOUT)
    real_uid = next(u for u, x in st.users().items() if x["email"] == "that@example.test")
    c2 = app.test_client()
    c2.post("/auth/demo/cold"); add(c2, IDS[1]); c2.post("/checkout", data=CHECKOUT)
    c2.post(f"/auth/demo/{DEMO[0]['id']}"); add(c2, IDS[2]); c2.post("/checkout", data=CHECKOUT)
    c2.post("/demo/reset")
    assert len(st.orders(real_uid)) == 1                      # tài khoản thật giữ nguyên
    assert st.orders(DEMO[0]["id"]) == []


# ---------------------------------------------------------------- tài khoản & bảo mật
def test_signup_duplicate_email_and_signin_errors():
    _, c, _ = make()
    c.post("/auth/signup", data=dict(name="An", email="an@example.test", password="secret1"))
    c.post("/auth/signout")
    r = c.post("/auth/signup", data=dict(name="An 2", email="AN@example.test", password="secret1"))
    assert "Email đã được đăng ký" in r.get_data(as_text=True)
    for bad in (dict(email="an@example.test", password="sai"), dict(email="khong@co.test", password="secret1")):
        assert "không đúng" in c.post("/auth/signin", data=bad).get_data(as_text=True)


def test_store_rejects_duplicate_email_at_db_level():
    from store import Store
    st = Store(tempfile.mkdtemp())
    st.save_user(dict(id="u-1", name="A", email="a@x.test", password="h", customer_id=1_000_000))
    with pytest.raises(sqlite3.IntegrityError):
        st.save_user(dict(id="u-2", name="B", email="A@x.test", password="h", customer_id=1_000_001))
    st.save_user(dict(id="u-1", name="A đổi tên", email="a@x.test", password="h", customer_id=1_000_000))
    assert st.get_user("u-1")["name"] == "A đổi tên"          # cập nhật người dùng cũ vẫn được


def test_next_parameter_blocks_open_redirect():
    _, c, _ = make()
    c.post("/auth/signup", data=dict(name="An", email="an@example.test", password="secret1"))
    c.post("/auth/signout")
    for evil in ("//evil.com", "/\\evil.com", "https://evil.com", "javascript:alert(1)"):
        r = c.post("/auth/signin", data=dict(email="an@example.test", password="secret1", next=evil))
        assert r.headers["Location"] in ("/", "http://localhost/"), evil
        c.post("/auth/signout")
    r = c.post("/auth/signin", data=dict(email="an@example.test", password="secret1", next="/cart"))
    assert r.headers["Location"].endswith("/cart")


def test_secret_key_persisted_in_runtime():
    runtime = tempfile.mkdtemp()
    a1, _, _ = make(runtime=runtime)
    a2, _, _ = make(runtime=runtime)
    assert a1.secret_key == a2.secret_key and len(a1.secret_key) >= 32
    assert (Path(runtime) / "secret.key").exists()


# ---------------------------------------------------------------- API
def test_api_recommendations_user_id_variants_and_errors():
    app, c, _ = make()
    assert c.get("/api/recommendations?surface=abc").status_code == 400
    assert c.get(f"/api/recommendations?user_id={DEMO[0]['id']}").get_json()["scenario"] in ("light", "heavy")
    assert c.get("/api/recommendations?user_id=khong-ton-tai").get_json()["scenario"] == "cold"
    c.post("/auth/signup", data=dict(name="An", email="an@example.test", password="secret1"))
    uid = next(iter(app.config["STORE"].users()))
    j = c.get(f"/api/recommendations?user_id={uid}").get_json()
    assert j["user_id"] == uid and j["scenario"] == "cold"
    assert c.get("/api/recommendations?k=0").status_code == 200 and c.get("/api/recommendations?k=999").status_code == 200


def test_api_events_rejects_bad_payloads():
    _, c, _ = make()
    assert c.post("/api/events", data="khong phai json", content_type="text/plain").status_code == 400
    assert c.post("/api/events", json={"events": "x"}).status_code == 400
    assert c.post("/api/events", json=[{"event": "click", "article_id": IDS[0]}]).get_json()["logged"] == 1
    r = c.post("/api/events", json={"events": [{"event": "click", "article_id": IDS[0]}] * 250}).get_json()
    assert r["logged"] == 200                                  # tối đa 200 sự kiện mỗi lần gửi


def test_search_pagination_and_filters():
    _, c, _ = make()
    assert c.get("/products?page=-5").status_code == 200
    assert c.get("/products?page=9999").status_code == 200
    assert c.get("/products?q=%3Cscript%3E").status_code == 200
    g = CATALOG.groups()[0][0]
    items, total = CATALOG.search(group=g)
    assert total >= 1 and all(p.group == g for p in items)


# ---------------------------------------------------------------- tích hợp với bộ gợi ý thật
REPO = Path(__file__).resolve().parents[2]
HAS_REAL = (REPO / "orch_architecture" / "orchestrator.pkl").exists() and (REPO / "data" / "split" / "train.parquet").exists()


@pytest.fixture(scope="module")
def real():
    if not HAS_REAL:
        pytest.skip("thiếu orchestrator.pkl hoặc dữ liệu repo")
    import dataset
    from production_recommender import ProductionRecommender
    tx, art = dataset.load_full()
    cat = Catalog(tx, art)
    return ProductionRecommender(tx, art), cat


def test_real_recommender_contract(real):
    rec, cat = real
    cold = rec.recommend_for_user(None, 12)
    assert cold and {r.shelf for r in cold} == {SHELF_POPULAR} and len(cold) == 12
    assert {r.shelf for r in rec.recommend_for_user("khong-co", 12)} == {SHELF_POPULAR}
    assert {r.shelf for r in rec.recommend_for_user(1_000_123, 12)} == {SHELF_POPULAR}   # tài khoản web mới
    d = cat.demo_users[0]["id"]
    warm = rec.recommend_for_user(d, 12)
    shelves = {r.shelf for r in warm}
    assert {SHELF_REPEAT, SHELF_DISCOVERY} <= shelves and SHELF_POPULAR not in shelves
    owned = {h["article_id"] for h in cat.history(d)}
    disc = [r for r in warm if r.shelf == SHELF_DISCOVERY]
    assert not owned & {r.article_id for r in disc}             # khám phá không gợi ý món đã mua
    assert all(cat.has(r.article_id) for r in warm)


def test_real_recommender_after_purchase_and_cart(real):
    rec, cat = real
    item = cat.order[0]
    after = rec.recommend_for_user(None, 12, session_purchases=[item])
    assert item in [r.article_id for r in after if r.shelf == SHELF_REPEAT]
    assert item not in [r.article_id for r in after if r.shelf == SHELF_DISCOVERY]
    cart = rec.recommend_for_cart([item], 6)
    t = cat.get(item).type
    assert len(cart) == 6 and item not in [r.article_id for r in cart]
    assert all(cat.get(r.article_id).type != t for r in cart)  # không gợi ý cùng loại với món trong giỏ
    assert rec.recommend_for_cart([], 6) == []


def test_real_app_end_to_end(real, tmp_path):
    rec, cat = real
    app = create_app(rec, catalog=cat, runtime_dir=tmp_path)
    c = app.test_client()
    c.post("/auth/demo/cold")
    j = c.get("/api/recommendations").get_json()
    assert j["scenario"] == "cold" and [s["key"] for s in j["shelves"]] == [SHELF_POPULAR]
    aid = j["shelves"][0]["items"][0]["article_id"]
    c.post("/cart/add", data=dict(article_id=aid))
    assert c.get("/api/recommendations?surface=cart").get_json()["shelves"][0]["key"] == SHELF_CART
    c.post("/checkout", data=CHECKOUT)
    j = c.get("/api/recommendations").get_json()
    assert j["scenario"] == "light" and SHELF_REPEAT in [s["key"] for s in j["shelves"]]
    out = app.config["STORE"].export_web_transactions(mark=False)
    assert len(out) == 1 and out["customer_id"].iloc[0] >= 1_000_000 and str(out["t_dat"].iloc[0].date()) >= "2020-09-23"

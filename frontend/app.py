"""StyleRec — web demo hệ khuyến nghị (Flask + Jinja).

Web KHÔNG tự tính gợi ý: recommender được tiêm vào create_app() (hợp đồng ở recommender_port.py).

    flask --app app run --debug          # Flask tự gọi create_app(): dev nhanh: bộ dữ liệu mẫu + MockRecommender
    python run_production.py             # dữ liệu thật của repo + ProductionRecommender (model backend)
"""
from __future__ import annotations

import logging
import os
import re
import time
import uuid
from dataclasses import replace
from functools import wraps
from pathlib import Path

from flask import (Flask, abort, flash, g, jsonify, redirect, render_template, request, session,
                   url_for)
from markupsafe import Markup
from werkzeug.security import check_password_hash, generate_password_hash

import pandas as pd

from catalog import Catalog
from recommender_port import (SHELF_CART, SHELF_DISCOVERY, SHELF_FOR_YOU, SHELF_POPULAR, SHELF_REPEAT,
                              ensure_recommender, normalize)
from store import EVENT_TYPES, Store

HERE = Path(__file__).resolve().parent
K_HOME, K_CART = 12, 6
CLIENT_EVENTS = {"impression", "click", "add_to_cart"}

SHELF_META = {
    SHELF_REPEAT: ("Mua lại món quen", "Những món bạn hay mua lại, xếp theo khả năng bạn sẽ mua tiếp"),
    SHELF_DISCOVERY: ("Có thể bạn sẽ thích", "Món mới bạn chưa mua, dựa trên những gì bạn đã mua"),
    SHELF_POPULAR: ("Bán chạy 7 ngày qua", "Sản phẩm được mua nhiều nhất tuần này"),
    SHELF_FOR_YOU: ("Dành cho bạn", ""),
    SHELF_CART: ("Thường được mua kèm", "Hoàn thiện bộ đồ với những món khách khác hay mua cùng"),
}
HOME_ORDER_WARM = [SHELF_REPEAT, SHELF_DISCOVERY, SHELF_POPULAR, SHELF_FOR_YOU]
SCENARIO_LABEL = {"cold": "Khách mới", "light": "Khách quen · mua ít", "heavy": "Khách quen · mua nhiều"}
SHIPPING = {"standard": ("Tiêu chuẩn (3–5 ngày)", 30_000), "express": ("Hoả tốc (trong 24h)", 50_000)}
PAYMENT = {"cod": "Thanh toán khi nhận hàng", "card": "Thẻ (giả lập)", "wallet": "Ví điện tử (giả lập)"}
FREE_SHIP_FROM = 499_000

# Trang thông tin ở footer: {slug: {title, intro, sections: [(heading, [đoạn / gạch đầu dòng])]}}
INFO_PAGES = {
    "gioi-thieu": dict(title="Giới thiệu StyleRec", intro=(
        "StyleRec là web demo của đồ án môn học: một cửa hàng thời trang trực tuyến có hệ khuyến nghị cá nhân hoá, "
        "chạy trên dữ liệu giao dịch online thật của H&M (bộ Kaggle H&M Personalized Fashion Recommendations)."),
        sections=[("Bạn có thể làm gì", ["Đăng ký, đăng nhập hoặc chọn khách mẫu để xem gợi ý theo từng tình huống.",
                                          "Thêm vào giỏ, thanh toán giả lập và xem gợi ý được tính lại ngay sau đơn hàng.",
                                          "Xem lịch sử mua trong dữ liệu và các đơn đã đặt trên web ở trang Hồ sơ."]),
                  ("Lưu ý", ["Giá được quy đổi minh hoạ từ giá chuẩn hoá trong dữ liệu.",
                             "Chưa có ảnh sản phẩm: thẻ hiển thị hình minh hoạ theo màu và nhóm hàng.",
                             "Không thu tiền, không giao hàng thật."])]),
    "cach-goi-y": dict(title="Cách StyleRec gợi ý", intro=(
        "Gợi ý dựa trên lịch sử mua (chỉ kênh online), không dùng thông tin cá nhân như tuổi hay nơi ở. "
        "Mỗi mục trên trang chủ ứng với một bài toán và một model đã được đánh giá trên tập kiểm tra."),
        sections=[("Bán chạy 7 ngày qua", ["Mục của khách mới (chưa có lịch sử mua), và là mục dự phòng khi không có gợi ý khác.",
                                           "Xếp theo số lượt bán trong 7 ngày gần nhất."]),
                  ("Mua lại món quen", ["Những món bạn từng mua, xếp theo khả năng mua lại của từng loại hàng "
                                        "và số ngày từ lần mua gần nhất."]),
                  ("Có thể bạn sẽ thích", ["Món mới bạn chưa mua, kết hợp hai nguồn:",
                                           "Khách mua món bạn đã mua cũng mua món này (\"Vì bạn đã mua …\").",
                                           "Hàng bán chạy trong những loại sản phẩm bạn hay mua."]),
                  ("Thường được mua kèm (giỏ hàng)", [
                      "Dùng mô hình Item-to-Item CF (ItemCF) nằm trong Orchestrator: độ tương đồng cosine giữa các sản "
                      "phẩm, tính từ việc cùng được những khách hàng giống nhau mua.",
                      "Cộng độ tương đồng của mọi món trong giỏ, bỏ món đã có, bỏ món cùng loại hoặc cùng mẫu (khác màu) "
                      "với món trong giỏ; thiếu thì bù bằng hàng bán chạy thuộc loại khác.",
                      "Orchestrator chỉ có hàm gợi ý theo khách (page), nên phần theo giỏ hàng được dựng trên chính mô hình "
                      "ItemCF đã huấn luyện của Orchestrator."]),
                  ("Sau khi thanh toán", [
                      "Đơn hàng được lưu vào cơ sở dữ liệu của web (bảng orders, order_items, web_transactions).",
                      "Mỗi lần mở trang, các lượt mua trên web được đọc lại và coi như lần mua mới nhất, nên gợi ý thay đổi "
                      "ngay; khách mới chuyển sang khách quen.",
                      "Định kỳ, các lượt mua trên web được xuất theo đúng định dạng dữ liệu huấn luyện để gộp vào và huấn "
                      "luyện lại mô hình (tools/export_web_transactions.py)."])]),
    "van-chuyen": dict(title="Chính sách vận chuyển", intro="Áp dụng cho bản demo, không giao hàng thật.",
                       sections=[("Phí vận chuyển", ["Tiêu chuẩn (3–5 ngày): 30.000đ, miễn phí cho đơn từ 499.000đ.",
                                                     "Hoả tốc (trong 24h): 50.000đ."]),
                                 ("Theo dõi đơn", ["Xem các đơn đã đặt trong trang Hồ sơ."])]),
    "doi-tra": dict(title="Đổi trả & hoàn tiền", intro="Đổi trả miễn phí trong 30 ngày kể từ khi nhận hàng.",
                    sections=[("Điều kiện", ["Sản phẩm còn nguyên tem mác, chưa qua sử dụng.",
                                             "Đồ lót, đồ bơi và phụ kiện xỏ khuyên không áp dụng đổi trả."]),
                              ("Hoàn tiền", ["Hoàn về phương thức thanh toán ban đầu trong 5–7 ngày làm việc."])]),
    "chon-size": dict(title="Hướng dẫn chọn size", intro="Bảng size tham khảo cho áo và quần nữ (cm).",
                      sections=[("Áo", ["XS: ngực 78–82 · S: 82–86 · M: 86–90 · L: 90–96 · XL: 96–102"]),
                                ("Quần", ["XS: eo 60–64 · S: 64–68 · M: 68–72 · L: 72–78 · XL: 78–84"]),
                                ("Mẹo", ["Nếu số đo nằm giữa hai size, chọn size lớn hơn để thoải mái."])]),
    "thanh-toan": dict(title="Phương thức thanh toán", intro="Mọi hình thức thanh toán trên web đều là giả lập.",
                       sections=[("Hỗ trợ", ["Thanh toán khi nhận hàng (COD).", "Thẻ (giả lập).",
                                             "Ví điện tử (giả lập)."])]),
    "cau-hoi": dict(title="Câu hỏi thường gặp", intro="",
                    sections=[("Vì sao tôi chỉ thấy mục Bán chạy?", ["Bạn đang là khách mới (chưa có lượt mua). "
                                                                      "Mua một đơn để xem gợi ý cá nhân hoá."]),
                              ("Vì sao gợi ý đổi sau khi tôi mua?", ["Món vừa mua được tính ngay vào lịch sử của bạn."]),
                              ("Làm sao xem lại demo từ đầu?", ["Bấm \"Đặt lại demo\" trên thanh demo để xoá các đơn "
                                                                "của khách mẫu."])]),
}


def vnd(v) -> str:
    return f"{int(v):,}".replace(",", ".") + "đ"


def create_app(recommender=None, catalog: Catalog | None = None, runtime_dir: Path | str | None = None,
               secret_key: str | None = None) -> Flask:
    catalog = catalog or Catalog.sample()
    if recommender is None:
        from mock_recommender import MockRecommender
        recommender = MockRecommender(catalog)
    recommender = ensure_recommender(recommender)
    store = Store(runtime_dir or os.environ.get("STYLEREC_RUNTIME", HERE / "runtime"),
                  sim_anchor=catalog.last_date + pd.Timedelta(days=1))
    store.fill_missing_prices(catalog.raw_price)   # đơn cũ (trước khi có cột price) -> giá theo thang dataset

    app = Flask(__name__)
    app.secret_key = secret_key or os.environ.get("STYLEREC_SECRET", "stylerec-dev-secret")
    app.config.update(CATALOG=catalog, RECOMMENDER=recommender, STORE=store)
    app.jinja_env.filters["vnd"] = vnd
    app.jinja_env.filters["sw"] = lambda hexcol: Markup(f"--sw:{hexcol}")
    app.logger.setLevel(logging.INFO)

    # ================================================================ người dùng / tình huống
    def current_user() -> dict | None:
        if "user" in g:
            return g.user
        uid, kind = session.get("uid"), session.get("kind")
        u = None
        if kind == "demo" and catalog.demo_user(uid):
            d = catalog.demo_user(uid)
            u = dict(id=uid, name=d["name"], kind="demo", customer_id=uid, email="", note=d.get("note", ""))
        elif kind in ("account", "temp"):
            rec = store.get_user(uid)
            if rec:
                u = dict(rec, kind=kind)
        if u is None and uid:
            session.pop("uid", None)
            session.pop("kind", None)
        g.user = u
        return u

    def session_id() -> str:
        if "sid" not in session:
            session["sid"] = uuid.uuid4().hex[:16]
        return session["sid"]

    def web_purchases(u) -> list[str]:
        if not u:
            return []
        memo = g.setdefault("web_purchases", {})
        if u["id"] not in memo:
            memo[u["id"]] = store.session_purchases(u["id"])
        return memo[u["id"]]

    def scenario_of(u, extra: int = 0) -> dict:
        n_data = catalog.n_purchases(u.get("customer_id")) if u and u.get("customer_id") else 0
        n_web = len(web_purchases(u)) + extra
        n = n_data + n_web
        key = "cold" if n == 0 else "light" if n <= 4 else "heavy"
        return dict(key=key, label=SCENARIO_LABEL[key], n=n, n_data=n_data, n_web=n_web, warm=n > 0)

    def rec_user_id(u):
        """Mã gửi cho recommender: mã khách trong dataset, hoặc mã web (recommender coi là khách mới)."""
        if not u:
            return None
        return u.get("customer_id") or u["id"]

    # ================================================================ gợi ý -> các mục
    def safe_call(fn, *args, **kw):
        try:
            return normalize(fn(*args, **kw))
        except Exception:  # recommender lỗi không được làm sập trang
            app.logger.exception("Recommender lỗi")
            g.rec_error = True
            return []

    def to_card(rec, pos: int) -> dict | None:
        p = catalog.get(rec.article_id)
        if p is None:
            return None
        because_name = catalog.name_of(rec.because) if rec.because and catalog.has(rec.because) else ""
        reason = f"Vì bạn đã mua {because_name}" if because_name and rec.shelf != SHELF_CART else rec.reason
        if because_name and rec.shelf == SHELF_CART:
            reason = f"Hay được mua cùng {because_name}"
        return dict(p.to_json(), reason=reason, because=rec.because, because_name=because_name,
                    shelf=rec.shelf, position=pos)

    def shelves_for(recs, order, exclude=frozenset(), limit=None) -> list[dict]:
        by: dict[str, list] = {}
        for r in recs:
            if r.article_id not in exclude:
                by.setdefault(r.shelf, []).append(r)
        out = []
        for key in order:
            cards = [c for c in (to_card(r, i) for i, r in enumerate(by.get(key, []), start=1)) if c]
            if limit:
                cards = cards[:limit]
            if cards:
                title, sub = SHELF_META[key]
                out.append(dict(key=key, title=title, subtitle=sub, cards=cards))
        return out

    def home_shelves(u, k=K_HOME, limit=None):
        scen = scenario_of(u)
        recs = safe_call(recommender.recommend_for_user, rec_user_id(u), k=k, session_purchases=web_purchases(u))
        order = HOME_ORDER_WARM if scen["warm"] else [SHELF_POPULAR]
        return shelves_for(recs, order, limit=limit), scen

    def cart_shelves(items, k=K_CART):
        if not items:
            return []
        recs = safe_call(recommender.recommend_for_cart, list(items), k=k)
        recs = [replace(r, shelf=SHELF_CART) for r in recs]  # mọi item ở giỏ hàng đều vào mục mua kèm
        return shelves_for(recs, [SHELF_CART], exclude=frozenset(items))

    # ================================================================ giỏ hàng (session)
    def cart() -> list[dict]:
        return session.setdefault("cart", [])

    def cart_view():
        lines, subtotal = [], 0
        for i, line in enumerate(cart()):
            p = catalog.get(line["article_id"])
            if not p:
                continue
            total = p.price_vnd * line["qty"]
            subtotal += total
            lines.append(dict(line, idx=i, product=p, total=total))
        ship_free = subtotal >= FREE_SHIP_FROM
        return lines, subtotal, ship_free

    def log(event, **kw):
        u = current_user()
        store.log_events([dict(kw, event=event, session_id=session_id(), user_id=u["id"] if u else "",
                               scenario=scenario_of(u)["key"])])

    def wants_json() -> bool:
        return request.headers.get("X-Requested-With") == "fetch" or request.accept_mimetypes.best == "application/json"

    def login_required(fn):
        @wraps(fn)
        def inner(*a, **kw):
            if not current_user():
                flash("Vui lòng đăng nhập để tiếp tục.", "info")
                return redirect(url_for("signin", next=request.full_path))
            return fn(*a, **kw)
        return inner

    def safe_next(default="index"):
        nxt = request.values.get("next", "")
        return nxt if nxt.startswith("/") and not nxt.startswith("//") else url_for(default)

    @app.context_processor
    def inject():
        u = current_user()
        return dict(user=u, scenario=scenario_of(u), cart_count=sum(x["qty"] for x in cart()),
                    demo_users=catalog.demo_users, groups=catalog.groups()[:8], rec_error=g.get("rec_error", False),
                    info_pages=INFO_PAGES)

    # ================================================================ trang
    @app.get("/")
    def index():
        u = current_user()
        shelves, scen = home_shelves(u)
        return render_template("index.html", shelves=shelves, surface="home")

    @app.get("/products")
    def products():
        group, q, sort = request.args.get("group", ""), request.args.get("q", ""), request.args.get("sort", "popular")
        page = max(request.args.get("page", 1, type=int) or 1, 1)
        items, total = catalog.search(group, q, sort, page=page, per_page=24)
        pages = max((total + 23) // 24, 1)
        return render_template("products.html", items=items, total=total, group=group, q=q, sort=sort,
                               page=min(page, pages), pages=pages, all_groups=catalog.groups())

    @app.get("/product/<article_id>")
    def product(article_id):
        p = catalog.get(article_id)
        if not p:
            abort(404)
        u = current_user()
        bought = 0
        if u:
            bought = (sum(1 for h in catalog.history(u.get("customer_id")) if h["article_id"] == article_id)
                      + web_purchases(u).count(article_id))
        return render_template("product.html", p=p, bought=bought, surface=request.args.get("surface", ""),
                               shelf=request.args.get("shelf", ""), position=request.args.get("position", ""))

    @app.get("/info/<slug>")
    def info(slug):
        page = INFO_PAGES.get(slug)
        if not page:
            abort(404)
        return render_template("info.html", page=page, slug=slug)

    @app.post("/newsletter")
    def newsletter():
        # Bản demo: không lưu email, chỉ xác nhận.
        if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", request.form.get("email", "").strip()):
            flash("Cảm ơn bạn đã đăng ký nhận tin (bản demo, email không được lưu).", "ok")
        else:
            flash("Email không hợp lệ.", "error")
        return redirect(request.referrer if (request.referrer or "").startswith(request.host_url) else url_for("index"))

    # ---------------------------------------------------------------- giỏ hàng
    @app.get("/cart")
    def cart_page():
        lines, subtotal, ship_free = cart_view()
        shelves = cart_shelves([l["article_id"] for l in lines])
        return render_template("cart.html", lines=lines, subtotal=subtotal, ship_free=ship_free,
                               free_from=FREE_SHIP_FROM, shelves=shelves, surface="cart")

    @app.post("/cart/add")
    def cart_add():
        aid = request.form.get("article_id", "")
        p = catalog.get(aid)
        if not p:
            if wants_json():
                return jsonify(ok=False, error="Sản phẩm không tồn tại"), 404
            abort(404)
        size = request.form.get("size", "") if p.sizes else ""
        if p.sizes and size not in p.sizes:
            size = p.sizes[len(p.sizes) // 2]
        qty = min(max(request.form.get("qty", 1, type=int) or 1, 1), 10)
        surface, shelf = request.form.get("surface", ""), request.form.get("shelf", "")
        position = request.form.get("position", "")
        items = cart()
        for line in items:
            if line["article_id"] == aid and line["size"] == size:
                line["qty"] = min(line["qty"] + qty, 10)
                break
        else:
            items.append(dict(article_id=aid, size=size, qty=qty, surface=surface, shelf=shelf, position=position))
        session.modified = True
        log("add_to_cart", article_id=aid, surface=surface or "product", shelf=shelf, position=position, value=qty)
        if wants_json():
            return jsonify(ok=True, name=p.name, size=size, cart_count=sum(x["qty"] for x in items))
        flash(f"Đã thêm {p.name}{' · size ' + size if size else ''} vào giỏ.", "ok")
        return redirect(safe_next("cart_page"))

    @app.post("/cart/update")
    def cart_update():
        idx, qty = request.form.get("idx", -1, type=int), request.form.get("qty", 1, type=int)
        items = cart()
        if 0 <= idx < len(items):
            if qty <= 0:
                items.pop(idx)
            else:
                items[idx]["qty"] = min(qty, 10)
            session.modified = True
        return redirect(url_for("cart_page"))

    @app.post("/cart/remove")
    def cart_remove():
        idx = request.form.get("idx", -1, type=int)
        items = cart()
        if 0 <= idx < len(items):
            items.pop(idx)
            session.modified = True
        return redirect(url_for("cart_page"))

    # ---------------------------------------------------------------- thanh toán
    @app.route("/checkout", methods=["GET", "POST"])
    @login_required
    def checkout():
        lines, subtotal, ship_free = cart_view()
        if not lines:
            flash("Giỏ hàng đang trống.", "info")
            return redirect(url_for("cart_page"))
        u = current_user()
        form = dict(name=u.get("name", "") if u["kind"] != "demo" else "", phone="", address="",
                    shipping="standard", payment="cod")
        errors = {}
        if request.method == "POST":
            form.update({k: request.form.get(k, "").strip() for k in form})
            if len(form["name"]) < 2:
                errors["name"] = "Nhập họ tên người nhận."
            if not re.fullmatch(r"0\d{9,10}", form["phone"].replace(" ", "")):
                errors["phone"] = "Số điện thoại gồm 10–11 chữ số, bắt đầu bằng 0."
            if len(form["address"]) < 8:
                errors["address"] = "Nhập địa chỉ nhận hàng đầy đủ."
            if form["shipping"] not in SHIPPING:
                errors["shipping"] = "Chọn phương thức vận chuyển."
            if form["payment"] not in PAYMENT:
                errors["payment"] = "Chọn phương thức thanh toán."
            if not errors:
                before = scenario_of(u)
                fee = 0 if ship_free and form["shipping"] == "standard" else SHIPPING[form["shipping"]][1]
                oid = "SR" + time.strftime("%y%m%d") + uuid.uuid4().hex[:6].upper()
                items = [dict(article_id=l["article_id"], name=l["product"].name, size=l["size"], qty=l["qty"],
                              price_vnd=l["product"].price_vnd, price=catalog.raw_price(l["article_id"]),
                              surface=l.get("surface", ""),
                              shelf=l.get("shelf", ""), position=l.get("position", "")) for l in lines]
                n_items = sum(i["qty"] for i in items)
                after = scenario_of(u, extra=n_items)
                order = dict(id=oid, uid=u["id"], ts=time.time(), date=time.strftime("%d/%m/%Y %H:%M"),
                             items=items, subtotal=subtotal, shipping_fee=fee, total=subtotal + fee,
                             shipping=form["shipping"], payment=form["payment"], name=form["name"],
                             phone=form["phone"], address=form["address"],
                             scenario_before=before["key"], scenario_after=after["key"])
                store.save_order(order, customer_id=u.get("customer_id"))
                store.log_events([dict(event="purchase", session_id=session_id(), user_id=u["id"],
                                       scenario=after["key"], surface=i["surface"] or "direct", shelf=i["shelf"],
                                       article_id=i["article_id"], position=i["position"], order_id=oid,
                                       value=i["qty"]) for i in items])
                session["cart"] = []
                return redirect(url_for("order_page", oid=oid))
        return render_template("checkout.html", lines=lines, subtotal=subtotal, ship_free=ship_free, form=form,
                               errors=errors, shipping=SHIPPING, payment=PAYMENT)

    @app.get("/order/<oid>")
    @login_required
    def order_page(oid):
        o = store.get_order(oid)
        if not o or o["uid"] != current_user()["id"]:
            abort(404)
        shelves, _ = home_shelves(current_user(), limit=6)
        return render_template("order.html", o=o, shelves=shelves, surface="order", label=SCENARIO_LABEL,
                               shipping=SHIPPING, payment=PAYMENT)

    @app.get("/profile")
    @login_required
    def profile():
        """Một dòng thời gian lịch sử mua, gộp 2 nguồn:
        - đơn trên web (CSDL của web, tính ngay vào gợi ý), mới nhất trước;
        - lịch sử trong dữ liệu H&M (dữ liệu huấn luyện, chỉ đọc), theo ngày."""
        u = current_user()
        timeline = []
        for o in sorted(store.orders(u["id"]), key=lambda o: -o["ts"]):
            items = [dict(i, product=catalog.get(i["article_id"])) for i in o["items"]]
            timeline.append(dict(source="web", date=o["date"], order=o, items=[i for i in items if i["product"]]))
        by_day: dict[str, list] = {}
        for h in catalog.history(u.get("customer_id"), limit=80):
            p = catalog.get(h["article_id"])
            if p:
                by_day.setdefault(h["date"], []).append(dict(h, product=p, qty=1))
        for day, items in by_day.items():
            timeline.append(dict(source="data", date=pd.Timestamp(day).strftime("%d/%m/%Y"), items=items))
        return render_template("profile.html", timeline=timeline, n_history=catalog.n_purchases(u.get("customer_id")),
                               n_orders=len(store.orders(u["id"])), shipping=SHIPPING, payment=PAYMENT)

    # ---------------------------------------------------------------- tài khoản
    def login(uid, kind):
        cart_keep = session.get("cart", [])
        session.clear()
        session.update(uid=uid, kind=kind, cart=cart_keep)
        g.pop("user", None)

    @app.route("/auth/signin", methods=["GET", "POST"])
    def signin():
        error = ""
        if request.method == "POST":
            u = store.find_by_email(request.form.get("email", ""))
            if u and u.get("password") and check_password_hash(u["password"], request.form.get("password", "")):
                login(u["id"], "account")
                flash(f"Xin chào {u['name']}!", "ok")
                return redirect(safe_next())
            error = "Email hoặc mật khẩu không đúng."
        return render_template("signin.html", error=error, next=request.values.get("next", ""))

    @app.route("/auth/signup", methods=["GET", "POST"])
    def signup():
        form, errors = dict(name="", email=""), {}
        if request.method == "POST":
            form = dict(name=request.form.get("name", "").strip(), email=request.form.get("email", "").strip())
            pw = request.form.get("password", "")
            if len(form["name"]) < 2:
                errors["name"] = "Nhập họ tên."
            if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", form["email"]):
                errors["email"] = "Email không hợp lệ."
            elif store.find_by_email(form["email"]):
                errors["email"] = "Email đã được đăng ký."
            if len(pw) < 6:
                errors["password"] = "Mật khẩu tối thiểu 6 ký tự."
            if not errors:
                uid = "u-" + uuid.uuid4().hex[:10]
                store.save_user(dict(id=uid, name=form["name"], email=form["email"],
                                     password=generate_password_hash(pw), customer_id=store.next_customer_id(),
                                     created=time.time()))
                login(uid, "account")
                flash("Tạo tài khoản thành công. Bạn là khách mới: gợi ý đang dựa trên hàng bán chạy.", "ok")
                return redirect(url_for("index"))
        return render_template("signup.html", form=form, errors=errors)

    @app.post("/auth/signout")
    def signout():
        session.clear()
        flash("Đã đăng xuất.", "info")
        return redirect(url_for("index"))

    @app.post("/auth/demo/<who>")
    def demo_login(who):
        if who == "cold":
            uid = "t-" + uuid.uuid4().hex[:10]
            store.save_user(dict(id=uid, name="Khách mới (demo)", email="", password="",
                                 customer_id=store.next_customer_id(), created=time.time(), temp=True))
            login(uid, "temp")
        elif catalog.demo_user(who):
            login(who, "demo")
        else:
            abort(404)
        return redirect(safe_next())

    @app.post("/demo/reset")
    def demo_reset():
        ids = {d["id"] for d in catalog.demo_users} | {uid for uid, x in store.users().items() if x.get("temp")}
        n = store.delete_orders(ids)
        session["cart"] = []
        flash(f"Đã xoá {n} đơn của khách mẫu. Demo trở về trạng thái ban đầu.", "ok")
        g.pop("web_purchases", None)
        return redirect(url_for("index"))

    # ================================================================ API
    @app.get("/api/recommendations")
    def api_recommendations():
        surface = request.args.get("surface", "home")
        k = min(max(request.args.get("k", K_HOME, type=int) or K_HOME, 1), 50)
        uid = request.args.get("user_id")
        u = current_user()
        if uid:  # tra theo mã truyền vào (khách mẫu / tài khoản web / mã dataset bất kỳ)
            if catalog.demo_user(uid) or catalog.n_purchases(uid):
                u = dict(id=uid, customer_id=uid)
            else:
                rec = store.get_user(uid)
                u = dict(rec) if rec else dict(id=uid, customer_id=None)
        if surface == "home":
            shelves, scen = home_shelves(u, k=k)
        elif surface == "cart":
            items = [a for a in request.args.get("items", "").split(",") if a]
            if not items:
                items = [l["article_id"] for l in cart()]
            shelves, scen = cart_shelves(items, k=k), scenario_of(u)
        else:
            return jsonify(error="surface phải là home hoặc cart"), 400
        keep = ("article_id", "name", "price_vnd", "reason", "because", "because_name", "position")
        return jsonify(
            user_id=u["id"] if u else None, scenario=scen["key"], surface=surface,
            shelves=[dict(key=s["key"], title=s["title"], items=[
                dict({k_: c[k_] for k_ in keep}, url=url_for("product", article_id=c["article_id"]),
                     image_url=url_for("static", filename=c["image"]) if c["image"] else None)
                for c in s["cards"]]) for s in shelves])

    @app.post("/api/events")
    def api_events():
        data = request.get_json(force=True, silent=True)
        if data is None:
            return jsonify(ok=False, error="Cần JSON"), 400
        rows = data.get("events", [data]) if isinstance(data, dict) else data
        if not isinstance(rows, list):
            return jsonify(ok=False, error="Sai định dạng"), 400
        u = current_user()
        base = dict(session_id=session_id(), user_id=u["id"] if u else "", scenario=scenario_of(u)["key"])
        good = []
        for r in rows[:200]:
            if not isinstance(r, dict) or r.get("event") not in CLIENT_EVENTS or not catalog.has(str(r.get("article_id", ""))):
                continue
            good.append(dict(base, event=r["event"], surface=str(r.get("surface", ""))[:20],
                             shelf=str(r.get("shelf", ""))[:20], article_id=str(r["article_id"]),
                             position=r.get("position", ""), value=r.get("value", "")))
        if good:
            store.log_events(good)
        return jsonify(ok=True, logged=len(good), rejected=len(rows[:200]) - len(good))

    @app.get("/healthz")
    def healthz():
        info = getattr(recommender, "info", None)
        return jsonify(ok=True, recommender=info() if callable(info) else type(recommender).__name__,
                       storage=store.stats(),
                       n_products_browse=len(catalog.order), event_types=sorted(EVENT_TYPES))

    @app.errorhandler(404)
    def not_found(_):
        return render_template("404.html"), 404

    return app


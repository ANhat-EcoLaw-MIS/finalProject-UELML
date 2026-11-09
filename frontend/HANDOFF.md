# HANDOFF — hoàn thiện web StyleRec (Flask)

> **Đã cũ.** Tài liệu này mô tả bản mockup ban đầu. Bản hiện hành xem `README.md` (3 bài toán cold-start / mua lại / khám phá theo model đã chốt của team).

Tài liệu bàn giao để làm tiếp ở repo mới (repo backend). Đọc hết file này trước khi code.

## 1. Mục tiêu (spec của team)

Web demo hệ khuyến nghị, Flask + Jinja. Một người làm hết (không chia team nữa).

- **Tài khoản**: đăng nhập, đăng ký, đăng xuất. Đăng nhập có thể **chọn khách mẫu** để vào đúng tình huống.
  Đăng ký mới → lưu file (`runtime/users.json`) → chạy theo cold-start.
- **3 tình huống**: khách mới (cold), mua ít (light), mua nhiều (heavy).
  Phân khúc theo **số lượt mua** giống `validation/validate.py`: cold = 0, light = 1–4, heavy ≥ 5.
- **Trang chủ**: khách mới chỉ thấy **1 mục** (bán chạy 7 ngày). Khách cũ thấy **3 mục**: phổ biến, mua lại,
  mua kèm ("Vì bạn đã mua …").
- **Giỏ hàng**: chỉ có mục **mua kèm (cross-sell)** theo các món đang trong giỏ.
- **Thanh toán giả**: đơn vào lịch sử mua, gợi ý **tính lại ngay** (như Shopee).
- **Hồ sơ**: lịch sử mua hàng.
- **`GET /api/recommendations`**: JSON gồm sản phẩm + lý do gợi ý.
- **`POST /api/events`**: ghi impression, click, add_to_cart vào CSV.
- Ràng buộc: chỉ thuật toán **B**, chỉ **kênh online**, **không** dùng biến nhân khẩu học, **chưa** làm trang báo cáo nhân viên.

## 2. Giao tiếp web ↔ backend

Web **không tự tính gợi ý**. Backend cung cấp class `ProductionRecommender`, web import thẳng và tiêm vào
`create_app(recommender)` (dependency injection). Web coi nó là hộp đen.

| Khi nào | Hàm | Input | Output |
| --- | --- | --- | --- |
| Đăng nhập / trang chủ | `recommend_for_user(user_id, k, session_purchases)` | mã khách (None/mã lạ = cold) | list sản phẩm + lý do + nhãn mục |
| Giỏ hàng | `recommend_for_cart(article_ids, k)` | các mã sản phẩm trong giỏ | list sản phẩm mua kèm |

Hợp đồng chi tiết: `recommender_port.py` (dataclass `Rec(article_id, reason, shelf, because)`, nhãn mục
`popular | repeat | cross`). **Việc cần làm khi nối backend thật**:
1. Đọc class backend: hàm khởi tạo cần gì, tên/IO hai hàm, định dạng `customer_id`, có trả lý do/nhãn mục không.
2. Nếu lệch hợp đồng → viết **adapter mỏng phía web** (`backend_adapter.py`), KHÔNG sửa code backend.
   Nên có hàm `normalize()` chấp nhận `Rec` / dict / tuple / str / DataFrame có cột `article_id`;
   item không có nhãn mục thì gom vào mục "Dành cho bạn".
3. `run_production.py` import class backend và gọi `create_app(...)`.
4. Backend có thể trả mã sản phẩm **ngoài** 24 món của web → cho `Catalog` đọc thêm `articles.parquet`
   đầy đủ (biến môi trường, ví dụ `STYLEREC_ARTICLES`) để tra tên/giá; món không có ảnh thì dùng ảnh minh hoạ.
5. `customer_id` trong parquet hiện tại là **int32 đã đánh lại số** (ví dụ `"49753"`), khách mẫu ở
   `data/demo_users.json` dùng đúng mã này → phải khớp với dữ liệu backend dùng để huấn luyện.

Trước khi có backend: dùng `MockRecommender` (chưa viết, xem mục 4).

## 3. Hiện trạng (đã xong)

| File | Trạng thái |
| --- | --- |
| `tools/build_sample_data.py` | Xong. Cắt từ parquet thật: online, 24 sản phẩm chọn theo cặp mua kèm khác loại mạnh nhất (≤2/loại, ≤6/nhóm), lấp bằng hàng bán chạy; giao dịch 52 tuần cuối; 3 khách mua ít + 3 khách mua nhiều. Chạy: `python tools/build_sample_data.py --parquet-dir <thư mục parquet>` (parquet gốc đang ở `E:/PRJ_ML/UELML_CK_FLASKPROJ/data/parquet`). |
| `data/articles.csv`, `data/transactions.csv`, `data/demo_users.json` | Xong (24 sản phẩm, 5.920 giao dịch, 6 khách mẫu). Nhỏ, commit được. |
| `catalog.py` | Xong. `Catalog`: tra sản phẩm (`Product` có tên đã làm sạch, loại/nhóm tiếng Việt, màu → hex, giá VND quy đổi, size, ảnh), `search(group, q, sort)`, `groups()`, `demo_user(uid)`, `history(customer_id)`. Ảnh lấy ở `static/img/products/<article_id>.jpg` nếu có. |
| `recommender_port.py` | Giữ từ mockup, dùng được; nên thêm `normalize()` + nhãn `for_you`. |
| `store.py` | Từ mockup, dùng được (users.json, orders.json, events.csv trong `runtime/`). Đổi cột events thành: `ts, session_id, user_id, scenario, event, surface, shelf, article_id, position, order_id, value`. |
| `app.py`, `templates/`, `static/`, `mock_recommender.py`, `recsys.py`, `mock_data.py`, `tests/` | **Mockup cũ, nhiều bug → viết lại.** `recsys.py` + `mock_data.py` + `wishlist.html` + `_drawer.html` sẽ xoá. Có thể tái dùng macro `garment()` (SVG bóng dáng theo nhóm hàng) và bảng màu CSS trong `static/style.css`. |

## 4. Kế hoạch còn lại

1. **`mock_recommender.py`** (viết mới trên `Catalog.transactions`):
   - popular: trend-weighted 7 ngày cuối: `0.7*R/Rmax + 0.3*clip((R-P)/(P+5),-1,3)/3`
     (R = lượt bán 7 ngày cuối, P = trung bình tuần của 21 ngày trước đó). Lý do: "Đã bán N cái trong 7 ngày".
   - repeat: món đã mua, xếp theo số lần + độ mới. Lý do: "Bạn đã mua N lần · lần cuối …".
   - cross (**thuật toán B**): Item-to-Item CF, cosine trên giỏ (khách, ngày) = `co/sqrt(n_i*n_j)`, loại món đã có
     và **mọi loại sản phẩm đã có trong giỏ**; thiếu thì lấp bằng bán chạy của loại khác. `because` = món nguồn.
   - `session_purchases` (món vừa mua trên web) coi là giỏ mới nhất → gợi ý đổi ngay sau thanh toán.
2. **`app.py`** — `create_app(recommender=None, data_dir=None)`:
   - auth: `/auth/signin` (email+mật khẩu, hoặc chọn khách mẫu), `/auth/signup`, `/auth/signout`,
     `/auth/demo/<cold|uid>`; "Khách mới (demo)" tạo tài khoản tạm. Thêm `/demo/reset` xoá đơn của khách mẫu để demo lặp lại.
   - trang: `/` (1 hoặc 3 mục), `/products` (lọc nhóm, tìm, sắp xếp), `/product/<id>` (không có mục gợi ý),
     `/cart` (+ mục cross-sell), `/cart/add|update|remove`, `/checkout` (bắt đăng nhập; tên, SĐT, địa chỉ, vận chuyển,
     thanh toán giả), `/order/<id>` (xác nhận + chuyển tình huống, ví dụ Khách mới → Mua ít + gợi ý mới), `/profile`.
   - scenario = lượt mua trong dataset + số món trong đơn web.
   - API: `/api/recommendations?surface=home|cart&user_id=&items=a,b&k=` →
     `{user_id, scenario, surface, shelves:[{key,title,items:[{article_id,name,price_vnd,image_url,url,reason,because}]}]}`;
     `/api/events` nhận một object hoặc `{events:[...]}`, event ∈ impression | click | add_to_cart.
     Server tự ghi add_to_cart (khi `/cart/add`) và purchase (khi thanh toán) kèm surface/shelf/position.
3. **Giao diện**: base (header, thanh demo đổi nhanh 3 tình huống, badge tình huống), thẻ sản phẩm có lý do gợi ý,
   ảnh thật hoặc ảnh minh hoạ màu + SVG. `app.js`: IntersectionObserver → impression (một lần/thẻ, gửi gộp),
   click thẻ gợi ý → click (sendBeacon), thêm giỏ bằng fetch + toast (form vẫn chạy khi tắt JS).
4. **Test** (`tests/test_app.py`, Flask test client + FakeRecommender): 3 tình huống ra đúng số mục; đăng ký → cold;
   thanh toán → lịch sử + gợi ý đổi; events.csv có đủ 3 loại; API đúng JSON; recommender trả mã lạ không làm sập trang.
5. Chạy thử trên trình duyệt cả 3 tình huống; cập nhật README; xoá file mockup cũ.

## 5. Còn chờ chủ repo xác nhận

- **Ảnh sản phẩm**: trong máy chưa có ảnh H&M. Ảnh Kaggle (`images/0xx/<article_id>.jpg`) cần tải bằng tài khoản Kaggle;
  web H&M không tra ảnh được theo article_id. Cần hỏi: có sẵn thư mục ảnh Kaggle không, hay dùng ảnh minh hoạ.
- Định dạng thật của class backend (mục 2).

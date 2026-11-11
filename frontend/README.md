# StyleRec — web demo hệ khuyến nghị (Flask + Jinja)

Web demo cho 3 bài toán gợi ý của team, chỉ dùng **kênh online** và **không** dùng biến nhân khẩu học.

| Tình huống | Điều kiện (giống `utils/eval_tools.py`) | Trang chủ thấy |
| --- | --- | --- |
| Khách mới | 0 lượt mua | 1 mục: **Bán chạy 7 ngày qua** |
| Khách quen · mua ít | 1–4 lượt mua (dataset + đơn trên web) | 2 mục: **Mua lại món quen**, **Có thể bạn sẽ thích** (bán chạy chỉ dành cho khách mới, đúng thiết kế Orchestrator) |
| Khách quen · mua nhiều | 5+ lượt mua | như trên |

Giỏ hàng có mục **Thường được mua kèm**. Sau khi thanh toán (giả lập), gợi ý được **tính lại ngay** và tình huống
chuyển theo, ví dụ từ khách mới sang khách mua ít.

## Chạy

```bash
cd frontend
pip install -r requirements.txt
python run_production.py          # dữ liệu thật + model backend  -> http://127.0.0.1:5000
flask --app app run --debug       # dev giao diện: 24 sản phẩm mẫu + MockRecommender (chạy ngay)
python -m pytest -q               # test (dữ liệu mẫu, không cần fit model)
```

- `run_production.py` dùng **Orchestrator của team**: nạp `orch_architecture/orchestrator.pkl` (tải theo `orchestrator_guide.md` ở gốc repo).
  Chưa có file thì web tự fit bằng `build_default()` của team (khoảng 1,5–2 phút) rồi lưu vào đúng chỗ đó. Từ lần sau chỉ nạp file (khoảng 30 giây).
  Cần chạy với thư viện cùng phiên bản lúc tạo pickle (xem `orchestrator_guide.md`).
- Thanh **Demo** dưới header cho phép đổi nhanh tình huống: Khách mới (tạo tài khoản tạm), Khách mua ít, Khách hay mua lại,
  Khách thích khám phá. Nút **Đặt lại demo** xoá các đơn đã đặt của khách mẫu để demo lại từ đầu.
- Muốn chọn lại khách mẫu: `python tools/pick_demo_users.py` (ghi đè `data/demo_users.json`).

## Kiến trúc

```
models/ utils/ data/          backend (KHÔNG sửa từ phía web)
frontend/
  recommender_port.py         hợp đồng web <-> recommender: Rec(article_id, reason, shelf, because), normalize()
  production_recommender.py   lớp nối: nạp Orchestrator (orch_architecture/), gọi orc.page(), đổi sang Rec + lý do tiếng Việt
  mock_recommender.py         recommender giả cho dev/test (chỉ dùng với bộ dữ liệu mẫu)
  dataset.py                  đọc dữ liệu: load_full() (parquet của repo) / load_sample() (frontend/data)
  catalog.py                  danh mục sản phẩm, lịch sử mua, khách mẫu (web chỉ tra cứu, không gợi ý)
  store.py                    runtime/: users.json, orders.json, events.csv
  app.py                      create_app(recommender, catalog): chỉ có web, recommender được tiêm vào
  run_production.py           ghép dữ liệu thật + ProductionRecommender
```

Web chỉ gọi 2 hàm:

| Khi nào | Hàm | Trả về |
| --- | --- | --- |
| Trang chủ, trang đơn hàng | `recommend_for_user(user_id, k, session_purchases)` | `Rec` có nhãn `popular` / `repeat` / `discovery` |
| Giỏ hàng | `recommend_for_cart(article_ids, k)` | `Rec` mua kèm |

Kệ của Orchestrator và model đứng sau từng mục (`build_default()` trong `orch_architecture/orchestrator.py`):

| Mục | Model | Lý do hiển thị |
| --- | --- | --- |
| popular ← "Trending now" | `PopularityModel(window_days=7)`, chỉ khách mới / dự phòng | "Đã bán N sản phẩm trong 7 ngày qua" |
| repeat ← "Buy it again" | `CategoryRepurchaseModel(product_type_no)` | "Bạn đã mua N lần · lần cuối …" |
| discovery ← "Because you bought" | `BlendModel{cat_pop 0.65, cf 0.35}, rrf_c=10` | "Vì bạn đã mua X" (từ ItemCF) / "Bán chạy trong loại hàng bạn hay mua"; giữ tối thiểu 4 món "Vì bạn đã mua" như guide khuyên |
| cart | ma trận tương đồng `ItemCFModel.S_`, loại món cùng loại hoặc cùng product_code với giỏ | "Hay được mua cùng X" |

Orchestrator chưa có hai phần sau, nên chúng được viết trong `production_recommender.py` bằng chính các model đã fit lấy từ Orchestrator:
- **Mua kèm theo giỏ:** dựng từ ma trận tương đồng của ItemCF.
- **Tính lại sau thanh toán:** chấm lại bằng đúng công thức của từng model, coi các món vừa mua trên web là lần mua mới nhất.
  Đã kiểm tra: khi chưa có món mới, kết quả trùng với `BlendModel` ở 39/40 khách. Khách còn lại lệch vì model bỏ qua
  những ngày mua sỉ trên 20 món.

Tuỳ chọn trong `SETTINGS` của `production_recommender.py`: `trending_for_returning` (thêm kệ bán chạy cho khách quen),
`discovery_min_because`, `cart_exclude_cart_types`.

## API

- `GET /api/recommendations?surface=home|cart&user_id=&items=a,b&k=`
  trả về `{user_id, scenario, surface, shelves:[{key, title, items:[{article_id, name, price_vnd, image_url, url, reason, because, because_name, position}]}]}`.
- `POST /api/events`: nhận một object hoặc `{events:[...]}`, với `event` ∈ `impression | click | add_to_cart`.
  Ghi vào `runtime/events.csv` với các cột `ts, session_id, user_id, scenario, event, surface, shelf, article_id, position, order_id, value`.
  Server tự ghi `add_to_cart` (khi `/cart/add`) và `purchase` (khi thanh toán), kèm surface/shelf/position của thẻ gợi ý gốc.
- `GET /healthz`: thông tin recommender đang chạy.

## Ghi chú

- Giá trong dataset đã chuẩn hoá. Web quy đổi minh hoạ ra VND (`catalog.to_vnd`).
- Chưa có ảnh H&M: thẻ sản phẩm vẽ ảnh minh hoạ theo màu và nhóm hàng. Nếu có ảnh, đặt vào `static/img/products/<article_id>.jpg`.
- Trang danh mục chỉ liệt kê 3.000 sản phẩm bán chạy gần nhất. Trang chi tiết mở được mọi sản phẩm trong `cleaned_art.parquet`.
- `HANDOFF.md` là tài liệu của bản mockup cũ, chỉ để tham khảo. README này là bản hiện hành.

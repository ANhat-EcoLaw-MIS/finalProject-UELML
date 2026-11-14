# StyleRec: hệ khuyến nghị thời trang (dữ liệu H&M, kênh online)

Đồ án môn Machine Learning: xây dựng và đánh giá mô hình gợi ý sản phẩm cho 3 bài toán, gom lại bằng một bộ điều phối
(Orchestrator), rồi triển khai lên một web thương mại điện tử đơn giản.

| Bài toán | Ai | Mô hình | Kệ trên web |
| --- | --- | --- | --- |
| Cold-start | khách chưa có lịch sử mua | Bán chạy 7 ngày (`PopularityModel`) | Bán chạy 7 ngày qua |
| Buy again (mua lại) | khách đã có lịch sử | `CategoryRepurchaseModel` | Mua lại món quen |
| Because you bought (khám phá) | khách đã có lịch sử | Blend `CategoryPopularity` + `ItemCF` | Có thể bạn sẽ thích |
| Mua kèm | giỏ hàng | `ItemCF` (lấy từ Orchestrator) | Thường được mua kèm |

## Cấu trúc thư mục

```
finalProject-UELML/
├── README.md                   tổng quan, cấu trúc, cách chạy (file này)
├── requirements.txt            thư viện cho phần ML
│
├── data/                       (1) dữ liệu
│   ├── transactions.parquet    giao dịch mẫu 10% khách hàng
│   ├── cleaned_art.parquet     thông tin sản phẩm đã làm sạch
│   ├── cleaned_cust.parquet    thông tin khách hàng đã làm sạch
│   └── split/                  train / val / test chia theo thời gian
│
├── eda/                        (2) EDA theo bài toán, chỉ trên tập train
│   ├── cold_start/
│   └── repeat_buy/             kèm eda_repeat_buy.py để chạy lại
│
├── models/                     (3) các candidate, cùng kế thừa models/base.py (BaseRecommender)
│   ├── popularity.py, trend_popularity.py                      cold-start
│   ├── r1_recent.py, recency_frequency.py, cat_repurchase.py   buy again
│   └── cat_pop.py, itemCF.py, itemCFv2.py, blend_user_discovery.py   because you bought
├── utils/                      công cụ dùng chung: load_splits.py, eval_tools.py (build_truth, metric, bootstrap)
│
├── evaluation_design/          (4) thẩm định trên tập val, mỗi bài toán một notebook + figures/ + results/
│   ├── coldstart/
│   ├── repurchase/
│   └── discovery/
├── final_test/                 (5) chấm một lần trên tập test, kết quả đóng băng trong frozen_results/
│
├── orch_architecture/          (6) Orchestrator: gom 3 mô hình thành các kệ gợi ý
│   ├── orchestrator.py         lớp Orchestrator, build_default()
│   ├── build_pickle.py         fit và lưu orchestrator.pkl
│   └── demo.py                 ví dụ: python -m orch_architecture.demo
│
├── frontend/                   (7) web demo StyleRec (Flask), xem frontend/README.md
│
└── docs/
    └── orchestrator_guide.md   hướng dẫn dùng Orchestrator, link tải orchestrator.pkl
```

Thứ tự (1) → (7) đúng với quy trình của đồ án: dữ liệu → EDA → candidate → thẩm định trên val → test một lần →
Orchestrator → triển khai web.

> Các package `models/`, `utils/`, `orch_architecture/` phải giữ nguyên tên và vị trí ở gốc repo: `orchestrator.pkl`
> lưu đường dẫn module theo đúng tên này, và các notebook dùng đường dẫn tương đối `../../data`.

## Chạy từng phần

Mọi lệnh chạy từ thư mục gốc repo, sau khi `pip install -r requirements.txt`.

| Phần | Lệnh |
| --- | --- |
| EDA Buy again | `python eda/repeat_buy/eda_repeat_buy.py` |
| Thẩm định trên val | mở notebook trong `evaluation_design/<bài toán>/` bằng Jupyter |
| Test cuối | mở `final_test/<bài toán>.ipynb` |
| Fit lại Orchestrator | `python -m orch_architecture.build_pickle` |
| Thử Orchestrator | `python -m orch_architecture.demo` (cần `orchestrator.pkl`) |
| Web demo | xem hướng dẫn ngay dưới đây |

---

## Hướng dẫn mở web (frontend)

### 1. Chuẩn bị

- **Python 3.13**, chạy được lệnh `python` và `pip` trong terminal.
- **Code đúng nhánh:** web hiện nằm ở nhánh `frontend-stylerec` (chưa merge vào `main`).

  ```bash
  git clone https://github.com/ANhat-EcoLaw-MIS/finalProject-UELML.git
  cd finalProject-UELML
  git checkout frontend-stylerec
  ```

  Nếu đã clone từ trước thì chỉ cần `git fetch` rồi `git checkout frontend-stylerec` và `git pull`.

### 2. Cài thư viện

```bash
pip install -r frontend/requirements.txt
```

Nếu sau này nạp mô hình báo lỗi `ImportError` hoặc `AttributeError`, hãy cài đúng phiên bản lúc tạo file mô hình
(xem `docs/orchestrator_guide.md`):

```bash
pip install numpy==2.5.3 pandas==3.0.6 pyarrow==24.0.0 scipy==1.18.1
```

### 3. Tải file mô hình `orchestrator.pkl`

File khoảng 250 MB nên không để trên git. Tải tại link trong [`docs/orchestrator_guide.md`](docs/orchestrator_guide.md)
(mục "Cài đặt và Phiên bản thư viện"), rồi đặt đúng chỗ:

```
finalProject-UELML/
└── orch_architecture/
    └── orchestrator.pkl     <- đặt file tải về ở đây
```

Chưa tải cũng chạy được: web sẽ tự huấn luyện lại Orchestrator bằng đúng cấu hình của team (khoảng 1,5–2 phút ở lần đầu)
rồi lưu vào chỗ trên. Nếu đã có file thì web **không bao giờ ghi đè** lên nó.

### 4. Chạy web

```bash
cd frontend
python run_production.py
```

Đợi đến khi terminal hiện dòng `Running on http://127.0.0.1:5000` (khoảng 20–30 giây để đọc dữ liệu và mô hình),
rồi mở trình duyệt vào **http://127.0.0.1:5000**. Tắt web: bấm `Ctrl + C` trong terminal.

Muốn kiểm tra web đang dùng mô hình nào thì mở http://127.0.0.1:5000/healthz. Trang này hiện tên các kệ, mã băm (sha256)
của file `orchestrator.pkl` đang dùng và số liệu trong cơ sở dữ liệu của web.

> Cần chế độ debug của Flask (tự tải lại khi sửa code)? Chạy `flask --app app run --debug` trong thư mục `frontend/`.
> Bản này dùng cùng dữ liệu thật và Orchestrator như `run_production.py`.

---

## Hướng dẫn sử dụng

### Thanh "Demo" dưới header

Dùng để đổi nhanh giữa các tình huống mà không cần đăng ký:

| Nút | Tình huống | Trang chủ thấy |
| --- | --- | --- |
| **Khách mới** | tạo một tài khoản tạm, chưa mua gì | 1 kệ: Bán chạy 7 ngày qua |
| **Khách mua ít** | khách thật trong dữ liệu, 4 lượt mua | Mua lại món quen, Có thể bạn sẽ thích |
| **Khách hay mua lại** | khách mua nhiều, có nhiều món mua lặp lại | như trên, kệ "Mua lại" dày hơn |
| **Khách thích khám phá** | khách mua nhiều loại sản phẩm khác nhau | như trên, nhiều món "Vì bạn đã mua …" |

Góc phải thanh Demo cho biết bạn đang ở tình huống nào và đã có bao nhiêu lượt mua.
Nút **Đặt lại demo** xoá các đơn đã đặt của khách mẫu và tài khoản tạm, để demo lại từ đầu.

### Kịch bản thử nhanh (khoảng 3 phút)

1. Bấm **Khách mới**. Trang chủ chỉ có kệ "Bán chạy 7 ngày qua", vì khách chưa có lịch sử mua.
2. Bấm **Thêm** dưới một sản phẩm (chọn size trước nếu muốn). Số trên biểu tượng túi ở góc phải tăng lên.
3. Bấm **biểu tượng túi** để vào giỏ hàng. Bên dưới có kệ **"Thường được mua kèm"**. Bấm **Thêm** ở kệ này thì trang
   giỏ tự cập nhật.
4. Bấm **Thanh toán**, điền tên, số điện thoại (10–11 số, bắt đầu bằng 0), địa chỉ, rồi bấm **Đặt hàng**.
   Thanh toán là giả lập, không thu tiền.
5. Trang xác nhận cho thấy tình huống chuyển từ **Khách mới** sang **Khách quen · mua ít**, kèm gợi ý mới.
   Quay về trang chủ sẽ thấy kệ "Mua lại món quen" và "Có thể bạn sẽ thích" tính theo món vừa mua.
6. Bấm tên tài khoản ở góc phải để mở **Hồ sơ** và xem lịch sử mua (đơn trên web + lịch sử trong dữ liệu H&M).
7. Bấm **Đặt lại demo** khi xong.

### Các chức năng khác

- **Tìm kiếm** (ô trên header): gõ tiếng Việt có dấu hoặc không dấu, nhiều từ cùng lúc, ví dụ `áo thun trắng`,
  `quan jean`, `chân váy đen`.
- **Danh mục**: menu trên header hoặc trang "Tất cả", lọc theo nhóm hàng, sắp xếp theo bán chạy hoặc giá.
- **Đăng ký / Đăng nhập**: tài khoản mới bắt đầu ở tình huống khách mới. Đăng nhập lại sau đó vẫn giữ lịch sử mua trên web.
- **Lý do gợi ý**: mỗi thẻ sản phẩm được gợi ý có một dòng giải thích, ví dụ "Vì bạn đã mua …",
  "Bạn đã mua 2 lần · lần cuối …", "Đã bán 59 sản phẩm trong 7 ngày qua".
- **Trang "Cách StyleRec gợi ý"** (link ở footer): giải thích từng kệ dùng mô hình nào.

---

## Dữ liệu thao tác trên web được lưu ở đâu?

Web **không ghi** vào dữ liệu huấn luyện. Mọi thứ phát sinh trên web nằm trong một cơ sở dữ liệu SQLite riêng:
**`frontend/runtime/stylerec.db`**. File này được tạo tự động ở lần chạy đầu và không đưa lên git, nên mỗi máy có một bản riêng.

| Thao tác trên web | Lưu vào | Ghi chú |
| --- | --- | --- |
| Đăng ký, tài khoản tạm "Khách mới" | bảng `users` | mật khẩu được băm; mỗi tài khoản có `customer_id` từ 1.000.000 trở lên |
| Đặt hàng | bảng `orders` + `order_items` | thông tin giao hàng, phương thức, tổng tiền, từng món |
| Từng lượt mua (mỗi đơn vị số lượng là 1 lượt) | bảng `web_transactions` | cùng cột với dữ liệu train: `customer_id, article_id, t_dat, price, sales_channel_id=2` |
| Xem, bấm, thêm giỏ, mua | bảng `events` | impression, click, add_to_cart, purchase, kèm kệ và vị trí của thẻ gợi ý |
| Giỏ hàng đang chọn | cookie phiên của trình duyệt | mất khi đổi trình duyệt; chỉ thành đơn khi thanh toán |

Dữ liệu huấn luyện (`data/*.parquet`) và file mô hình (`orch_architecture/orchestrator.pkl`) chỉ được đọc.

**Vì sao gợi ý đổi ngay sau khi mua?** Mỗi lần mở trang, web đọc các lượt mua của bạn trong `web_transactions` và đưa
cho bộ gợi ý. Bộ gợi ý coi đó là lần mua mới nhất và chấm lại bằng các mô hình của Orchestrator, không cần huấn luyện lại.

**Đưa dữ liệu web vào huấn luyện (cập nhật định kỳ):**

```bash
cd frontend
python tools/export_web_transactions.py --dry-run   # xem trước
python tools/export_web_transactions.py             # xuất ra data/web_transactions.parquet
```

File xuất ra có cùng định dạng với dữ liệu gốc. Ngày mua được ghi nối tiếp sau 22/09/2020, theo số ngày kể từ khi web
chạy lần đầu. Gộp file này với dữ liệu gốc rồi chạy lại `python -m orch_architecture.build_pickle` để có mô hình mới.
Mỗi lượt mua chỉ được xuất một lần.

**Xoá sạch dữ liệu web để bắt đầu lại:** tắt web, xoá `frontend/runtime/stylerec.db` (và hai file đi kèm `stylerec.db-wal`, `stylerec.db-shm` nếu có), rồi chạy lại.
Nếu chỉ muốn xoá đơn của khách mẫu thì bấm **Đặt lại demo** trên web.

Xem nhanh nội dung cơ sở dữ liệu: mở http://127.0.0.1:5000/healthz (mục `storage`), hoặc mở file `stylerec.db`
bằng một công cụ xem SQLite như *DB Browser for SQLite*.

---

## Lỗi thường gặp

| Hiện tượng | Cách xử lý |
| --- | --- |
| `ModuleNotFoundError: No module named 'flask'` (hoặc pandas, scipy…) | chạy lại bước 2: `pip install -r frontend/requirements.txt` |
| `ImportError` / `AttributeError` khi nạp `orchestrator.pkl` | cài đúng phiên bản thư viện ở bước 2, hoặc xoá file `.pkl` để web tự huấn luyện lại |
| `Address already in use` / cổng 5000 đang bận | tắt chương trình đang dùng cổng 5000, hoặc chạy cổng khác: `PORT=5001 python run_production.py` (PowerShell: `$env:PORT=5001; python run_production.py`) |
| Lần đầu chạy rất lâu (1–2 phút) | chưa có `orchestrator.pkl` nên web đang tự huấn luyện; lần sau sẽ nhanh |
| Trang báo "Hệ thống gợi ý đang bận" | xem lỗi trong terminal; trang vẫn dùng được, chỉ thiếu kệ gợi ý |
| Báo không tìm thấy `data/split/...parquet` hoặc `data/cleaned_art.parquet` | dữ liệu nằm sẵn trong repo: chạy `git pull` và giữ nguyên cấu trúc thư mục (không di chuyển `frontend/` ra ngoài repo) |

File này sẽ hướng dẫn mọi người cách sử dụng và truy vấn orchestrator.
_Lưu ý_: Model đã được lưu bằng pickle file - nên nó sẽ có tỷ lệ gặp các lỗi về ImportError, AttributeError -> quy chung là lỗi về import không chính xác và cấu hình thư mục không chuẩn

### **TIÊN QUYẾT**
#### 1. Layout thư mục
```aiignore
project_root/
├── models/ # the models listed below are prerequisites
│   ├── __init__.py # must include
│   ├── base.py
│   ├── blend_user_discovery.py   # (the file names you actually have)
│   ├── category_popularity.py
│   ├── category_repurchase.py
│   ├── itemCFv2.py
│   └── popularity.py
├── orch_architecture/
│   ├── __init__.py               # can be empty
│   ├── orchestrator.py
│   ├── build_pickle.py
│   └── orchestrator.pkl
├── requirements.txt
└── app.py                        # the web entry point, run from project_root
```

#### 2. Cài đặt và Phiên bản thư viện
Cần đảm bảo môi trường có đủ các thư viện với đúng phiên bản và mô hình `.pkl` được down về local theo đúng cái cấu trúc thư mục trên

Link download mô hình: [Link](https://drive.google.com/file/d/1bKdTqaGch2JnYUDPSZQQv9ZU7KN9wy78/view?usp=drive_link)
```aiignore
numpy == 2.5.3
pandas == 3.0.6
pyarrow  == 24.0.0
scipy  == 1.18.1
```

### **HƯỚNG DẪN SỬ DỤNG**
#### 1. Tổng quan kiến trúc
File `orchestrator.py` đóng vai trò là tầng điều phối (**Orchestrator**) cho hệ thống gợi ý sản phẩm. Hệ thống này chịu trách nhiệm phân phối các mô hình thành các **shelf** (kệ sản phẩm) trên giao diện web, phân loại người dùng, lọc trùng lặp, xử lý cold-start và gán nhãn giải thích (**reason**) cho từng sản phẩm.

#### 2. Các thành phần cấu thành
##### **Section** (Định nghĩa một kệ sản phẩm)
Mục đích: để "chỗ chứa" các sản phẩm gợi ý theo format dictionary và các logic nhất định
- **name**: Tiêu đề hiển thị trên website.
- **model**: Mô hình gợi ý cơ sở (**BaseRecommender**).
- **k**: Kích thước cố định cho khách mới hoặc kệ dự phòng.
- **audience**: Đối tượng (`new`, `returning`, hoặc `all`).
- **reason**: Nhãn giải thích cấp độ kệ.
- **explain**: Bộ sinh nhãn giải thích chi tiết từng sản phẩm (`item-level explainer`).

##### **Orchestrator** (Bộ điều phối chính)
Mục đích: là "đầu não" để huấn luyện và đưa ra khuyến nghị đầu cuối - sử dụng `.page()` và `.trim()` để định hướng lọc
- **fit()**: Huấn luyện toàn bộ các mô hình liên quan.
- **segment()**: Phân loại người dùng (`returning` có lịch sử mua hàng, ngược lại là `new`).
- **recommend()**: Tổng hợp gợi ý, thực hiện **overfetch** và loại bỏ các sản phẩm trùng lặp đã hiển thị ở các kệ trước đó.
- **page()**: Trả về cấu trúc dữ liệu theo định dạng cho một user cụ thể.
- **trim()**: Lọc phía server theo ngưỡng điểm (`min_score`) và số lượng (`max_items`, `min_items`).

##### **BlendExplainer**
- Xử lý giải thích chi tiết cho kệ blend (kết hợp item-CF và phổ biến danh mục):
  - Nếu từ CF: `"Because you bought <article>"`
  - Ngược lại: `"Popular in categories you buy"`

### 3. Cách sử dụng cơ bản với `.page()` và `.trim()`
**MENTAL MODEL**: Input-Output cơ bản của method `.page()`:
* Input: user_id và article_names (tên của article dựa vào article_id)
* Output: Khuyến nghị theo dạng dictionary ứng với mỗi problem: `{"Trending now": [...], "Buy again": [...], "Because you bought": [...]}` kèm với nhãn explaination

_Lưu ý_: đối với cái Because you buy -> nó là mô hình lai nên nó sẽ có hai trạng thái một là "Because you bought" (thuần ItemCF) hoặc "Popular in categories" (thuần về Item Discovery) -> và thông thường CatPop sẽ được xuất hiện nhiều hơn => nên khi lọc kết quả thì để ý, làm sao đó nên có **ít nhất 1-2 sản phẩm với nhãn "Because you bought..."**

------
**BƯỚC SỬ DỤNG**
* B0: Load mô hình orchestrator vào môi trường
```python
from orch_architecture.orchestrator import Orchestrator
orc=Orchestrator.load("orch_architecture/orchestrator.pkl") #nhớ đúng địa chỉ load đến pickle file
```
* B1: Cung cấp mô hình dữ liệu về `articles_names`
```python
import pandas as pd

articles = pd.read_parquet("data/cleaned_art.parquet")
names = dict(zip(articles["article_id"], articles["prod_name"]))
```
* B2: Sử dụng phương thức `.page(user_id, articles_names)` để khuyến nghị
```pycon
page=orc.page(27682, article_names=names)
```
* B3: Sử dụng phương thức `.trim(page, rules)` để cắt bớt kết quả thừa. **Mục đích:** phục vụ cho server tự cấu hình những gì muốn hiển thị thay về là để mô hình tự quyết
```pycon
# 1. Lấy trang gợi ý cho một khách hàng cụ thể
page = orch.page(user="customer_a", article_names=names)

# 2. Định nghĩa các quy tắc lọc (rules) cho từng kệ
rules = {
    "Buy it again": {
        "min_score": 0.01, 
        "max_items": 12
    },
    "Because you bought": {
        "max_items": 8, 
        "min_items": 3  # Kệ này sẽ bị ẩn nếu có ít hơn 3 item đạt chuẩn
    }
}

# 3. Thực hiện cắt tỉa dữ liệu trang thông qua method trim
trimmed_page = Orchestrator.trim(page, rules)
```
* B4: Thêm bộ lọc cần thiết (optional nhưng mà khuyên dùng). Bởi vì phương thức `trim()` chỉ nhằm mục tiêu "cắt tỉa" chứ không thể "lọc" chuyên sâu - và ta cũng không nên nhúng cái logic này vào trong bản thân mô hình, mô hình này chỉ nên dùng để truy vấn kết quả
```pycon
# một ví dụ lọc chuyên sâu là - cân bằng giữa các khuyến nghị trong "Because you buy"
# thông thường các khuyến nghị của mô hình CatPop sẽ có điểm số cao hơn nên được đẩy lên đầu
# Và ta sẽ muốn các khuyến nghị "Because you bought X, so buy Y" xuất hiện nên ta cần cân bằng

# Ở đây tui chỉ can thiệp ở mức trước khi trim() nha -> tức là trực tiếp vào page được sản sinh ra
# Thì cái này nó sẽ bị chưa TỐI ƯU do lúc nào nó cũng gọi cái này - mặc cho là cái "Because you buy" nó có gợi ý được không, và gợi ý 
# có trúng cái mình cần cân bằng không "Because you bought..."
def balance_because_you_bought(page, max_items=12, min_prioritized_items=4):
    """
    Hàm tùy chỉnh để cân bằng kệ 'Because you bought':
    Ưu tiên giữ lại tối thiểu một số lượng item có reason 'Because you bought ...' 
    rồi mới fill đầy phần còn lại bằng 'Popular in categories...'.
    """
    shelf_name = "Because you bought"
    if shelf_name not in page:
        return page
        
    items = page[shelf_name]
    
    # 1. Phân tách thành 2 nhóm dựa vào reason
    prioritized = [i for i in items if "Because you bought" in i.get("reason", "")]
    fallback = [i for i in items if "Because you bought" not in i.get("reason", "")]
    
    # 2. Cân bằng số lượng (Ví dụ: ưu tiên tối đa lấy hết prioritized, phần thiếu bù bằng fallback sao cho tổng <= max_items)
    # Hoặc ép cứng: lấy tối thiểu min_prioritized_items nếu có, còn lại điền fallback
    selected_prioritized = prioritized[:max_items] # Lấy hết mức có thể hoặc giới hạn
    remaining_slots = max_items - len(selected_prioritized)
    
    selected_fallback = fallback[:max_items if remaining_slots < 0 else remaining_slots]
    
    # Gộp lại và sắp xếp lại theo điểm số (score) giảm dần nếu cần
    balanced_items = selected_prioritized + selected_fallback
    balanced_items.sort(key=lambda x: x["score"], reverse=True)
    
    # Cập nhật lại vào page
    page[shelf_name] = balanced_items[:max_items]
    return page

# --- Cách sử dụng trong luồng code của bạn ---

# Bước 1: Lấy page từ orchestrator
page = orch.page(user="customer_a", article_names=names)

# Bước 2: Gọi hàm cân chỉnh phức tạp do bạn tự định nghĩa cho kệ cần thiết
page = balance_because_you_bought(page, max_items=12)

# Bước 3: Sau đó tiếp tục áp dụng Orchestrator.trim() bình thường cho các rule khác (nếu có)
rules = {
    "Buy it again": {"max_items": 8, "min_items": 2},
}

final_page = Orchestrator.trim(page, rules)
```
* OUTPUT coi thử - trong file `demo.py` ở root nha:
```
.page(27682, article_names=names) -> {"Trending now":...},
.trim(page(27682, article_names=names), rules = {"Buy it again": {"max_items": 8, "min_items": 2}}
-> 
```
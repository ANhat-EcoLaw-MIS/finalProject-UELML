# EDA theo từng bài toán

EDA chỉ chạy trên tập train (`data/split/train.parquet`), không chạm vào val/test. Mỗi thư mục con trả lời các câu hỏi
dẫn tới việc chọn candidate cho một bài toán; kết quả được kiểm chứng ở `evaluation_design/`.

| Thư mục | Bài toán | Nội dung |
| --- | --- | --- |
| `cold_start/` | Cold-start (khách mới) | Q1 top-12 gần đây so với toàn thời gian · Q2 độ ổn định top-12 qua các giai đoạn 28 ngày · Q3–Q4 mức tập trung lượt mua (Pareto) |
| `repeat_buy/` | Buy again (mua lại) | Hình 1 mua lại có tồn tại không · Hình 2 khi nào mua lại · Hình 3 repeat rate theo frequency và recency · Hình 4 ai mua lại; `conclusion.docx`, `candidate_models.docx` |

EDA của bài toán Because you bought X (E1–E10, chọn ItemCF) được trình bày trong báo cáo Chương III.

## Chạy lại EDA Buy again

```bash
python eda/repeat_buy/eda_repeat_buy.py
```

Script tái tạo đúng số liệu của 4 hình `fig1`–`fig4` (cả 2 kênh bán) và vẽ thêm bản cho **kênh online**, khớp phạm vi
của mô hình, vào `repeat_buy/figures_online/` (kèm `summary.json`).

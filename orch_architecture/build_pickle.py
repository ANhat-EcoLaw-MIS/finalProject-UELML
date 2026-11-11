from orch_architecture.orchestrator import build_default
import pandas as pd

all_data = pd.read_parquet("data/transactions.parquet")
all_data['t_dat']=pd.to_datetime(all_data['t_dat'])
all_data=all_data[all_data['sales_channel_id']==2]
articles = pd.read_parquet("data/cleaned_art.parquet")
pc = articles.set_index("article_id")["product_code"]  # product code
types = articles.set_index("article_id")["product_type_no"]  # product type
anchor_date = all_data['t_dat'].max() + pd.Timedelta(days=1)

orch = build_default(types, article_product=pc).fit(all_data, anchor_date)
orch.save("orchestrator.pkl")


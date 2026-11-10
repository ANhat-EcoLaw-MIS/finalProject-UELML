from orch_architecture.orchestrator import Orchestrator
import pandas as pd

articles = pd.read_parquet("data/cleaned_art.parquet")
names = dict(zip(articles["article_id"], articles["prod_name"]))

orc=Orchestrator.load("orch_architecture/orchestrator.pkl")

page=orc.page(27682, article_names=names)
trimmed_page=orc.trim(page,{"Buy it again": {"max_items": 8, "min_items": 2}})
print(trimmed_page)
print (trimmed_page['Buy it again'])
print (len(trimmed_page['Buy it again']))
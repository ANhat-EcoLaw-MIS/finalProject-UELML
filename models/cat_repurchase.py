import numpy as np
import pandas as pd

from models.base import BaseRecommender


class CategoryRepurchaseModel(BaseRecommender):
    """Ranks each user's past articles by how likely that KIND of article is bought again.

    Recency alone treats a pair of socks and a party dress the same. This model learns, from the
    training history itself, how often an article is re-bought given
        (its category, days since the user last bought it)
    and ranks the user's articles by that repeat rate.

    How the rate table is learned (no validation data is touched):
        1. Step back to `cut = anchor - horizon_days`.
        2. Take every (user, article) pair bought before `cut` and measure its age at `cut`.
        3. Label it 1 if the user bought the same article again between `cut` and `anchor`.
        4. rate(category, age bucket) = share of labelled pairs that repeated, shrunk towards the
           overall rate of that age bucket when a category has few pairs (smoothing).
    Then every pair is scored at `anchor` with the learned rate. Ties keep the R1 order
    (most recent first), so the model differs from R1 only where categories differ.

    article_type  Series, index = article_id, value = category (e.g. product_type_no)
    horizon_days  length of the label window; use the length of the period you predict (7)
    age_edges     upper bounds (days) of the age buckets; () = a single bucket (recency ignored)
    smoothing     strength of the shrinkage towards the age-bucket rate (0 = no shrinkage)
    Users without history get no recommendations.
    """

    def __init__(self, article_type, horizon_days=7, age_edges=(3, 7, 14, 30, 60, 90, 180),
                 smoothing=20, k=12):
        edges = tuple(age_edges)
        self._require(horizon_days >= 1, "horizon_days must be at least 1")
        self._require(all(e >= 0 for e in edges) and list(edges) == sorted(set(edges)),
                      "age_edges must be non-negative and strictly increasing")
        self._require(smoothing >= 0, "smoothing must not be negative")
        super().__init__(k=k)
        self.article_type = article_type
        self.horizon_days = horizon_days
        self.age_edges = edges
        self.smoothing = smoothing

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _last_purchase(history):
        return history.groupby(["customer_id", "article_id"], as_index=False, sort=False).agg(
            last_date=("t_dat", "max"),
            n_purchase=("t_dat", "count"),
        )

    def _bucket(self, age_days):
        """Bucket id 0..len(age_edges): bucket i holds ages in (edges[i-1], edges[i]]."""
        return np.searchsorted(np.asarray(self.age_edges), np.asarray(age_days), side="left")

    def bucket_labels(self):
        edges = list(self.age_edges)
        if not edges:
            return ["all"]
        middle = [f"{lo + 1}-{hi}d" for lo, hi in zip(edges[:-1], edges[1:])]
        return [f"<={edges[0]}d"] + middle + [f">{edges[-1]}d"]

    # ---------------------------------------------------------------------- fit
    def fit(self, history_df, anchor_date):
        self._require(self.article_type is not None, "article_type is required to fit")
        self._check_history(history_df, anchor_date)

        history = history_df.loc[:, ["t_dat", "customer_id", "article_id"]]
        cut = anchor_date - pd.Timedelta(days=self.horizon_days)
        past = history.loc[history["t_dat"] < cut]
        future = history.loc[history["t_dat"] >= cut]
        self._require(len(past) > 0, "history is not longer than horizon_days: nothing to learn from")
        self._require(len(future) > 0, "no purchases in the last horizon_days to learn from")

        # 1) learn: as of `cut`, was the pair bought again before `anchor`?
        pairs = self._last_purchase(past)
        again = future[["customer_id", "article_id"]].drop_duplicates().assign(repeated=1)
        pairs = pairs.merge(again, how="left", on=["customer_id", "article_id"])
        pairs["repeated"] = pairs["repeated"].fillna(0)
        pairs["bucket"] = self._bucket((cut - pairs["last_date"]).dt.days)
        pairs["category"] = pairs["article_id"].map(self.article_type)
        self._learn_rates(pairs)

        # 2) score every (user, article) as of `anchor`
        scored = self._last_purchase(history)
        scored["bucket"] = self._bucket((anchor_date - scored["last_date"]).dt.days)
        scored["category"] = scored["article_id"].map(self.article_type)
        rate = scored.merge(self.rate_table_[["category", "bucket", "rate"]],
                            how="left", on=["category", "bucket"])["rate"].to_numpy()
        # articles without a category fall back to the overall rate of their age bucket
        scored["score"] = np.where(np.isnan(rate), scored["bucket"].map(self.bucket_rate_), rate)

        self.scores_ = (
            scored[["customer_id", "article_id", "last_date", "n_purchase", "score"]]
            .sort_values(["customer_id", "score", "last_date", "article_id"],
                         ascending=[True, False, False, False], kind="mergesort")
            .reset_index(drop=True)
        )
        self.anchor_date_ = anchor_date
        return self

    def _learn_rates(self, pairs):
        n_buckets = len(self.age_edges) + 1
        by_bucket = pairs.groupby("bucket")["repeated"].agg(["sum", "count"])
        by_bucket = by_bucket.reindex(range(n_buckets), fill_value=0)
        overall = pairs["repeated"].mean()
        bucket_rate = (by_bucket["sum"] / by_bucket["count"].replace(0, np.nan)).fillna(overall)

        table = (
            pairs.dropna(subset=["category"])
            .groupby(["category", "bucket"])["repeated"]
            .agg(repeats="sum", pairs="count")
        )
        grid = pd.MultiIndex.from_product(
            [table.index.get_level_values("category").unique(), range(n_buckets)],
            names=["category", "bucket"],
        )
        table = table.reindex(grid, fill_value=0).reset_index()
        table["prior"] = table["bucket"].map(bucket_rate)
        table["rate"] = (
                (table["repeats"] + self.smoothing * table["prior"])
                / (table["pairs"] + self.smoothing)
        ).fillna(table["prior"])  # smoothing = 0 and no pairs -> use the bucket rate

        self.rate_table_ = table[["category", "bucket", "rate", "repeats", "pairs"]]
        self.bucket_rate_ = bucket_rate

    # --------------------------------------------------------------- inspection
    def category_rates(self, what="rate"):
        """Category x age-bucket table of the learned repeat rate ("rate"), or the raw counts
        ("pairs", "repeats"). The last row is the overall rate of each age bucket."""
        self._require(hasattr(self, "rate_table_"), "fit must be called before category_rates.csv")
        self._require(what in ("rate", "pairs", "repeats"), "what must be rate, pairs or repeats")
        wide = self.rate_table_.pivot(index="category", columns="bucket", values=what)
        wide.columns = self.bucket_labels()
        if what == "rate":
            wide.loc["ALL"] = self.bucket_rate_.to_numpy()
        return wide

    # ---------------------------------------------------------------- recommend
    def recommend(self, users=None, k=None):
        return self._recommend_from_scores(users=users, k=k)

    def score_items(self, customer_id):
        return self._score_items(customer_id)

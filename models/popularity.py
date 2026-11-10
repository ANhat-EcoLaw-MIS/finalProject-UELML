import json
from pathlib import Path

import pandas as pd

from models.base import BaseRecommender


class PopularityModel(BaseRecommender):
    """Recommends the same top-k most purchased articles to every user.

    window_days=None -> count purchases over the whole history (classic popularity baseline).
    window_days=7    -> count only the last 7 days before anchor_date (recent popularity).
    Works for coldstart users, because it never looks at the individual's history.

    exclude_seen=True -> skip articles the user already bought in history, so each user gets
    the k most popular articles that are new to them (the "discovery" baseline).
    """

    def __init__(self, window_days=None, exclude_seen=False, k=12):
        self._require(
            window_days is None or window_days > 0, "window_days must be greater than 0"
        )
        super().__init__(k=k)
        self.window_days = window_days
        self.exclude_seen = exclude_seen

    def fit(self, history_df, anchor_date):
        self._check_history(history_df, anchor_date)

        history = history_df.loc[:, ["t_dat", "customer_id", "article_id"]]
        if self.window_days is not None:
            start = anchor_date - pd.Timedelta(days=self.window_days)
            history = history.loc[history["t_dat"] >= start]
        self._require(len(history) > 0, "no purchases inside the popularity window")

        self.ranking_ = (
            history.groupby("article_id", as_index=False)
            .size()
            .rename(columns={"size": "score"})
            .sort_values(["score", "article_id"], ascending=[False, False], kind="mergesort")
            .reset_index(drop=True)
        )
        self.users_ = history_df["customer_id"].unique()  # used when recommend(users=None)
        if self.exclude_seen:  # uses the full history, not only the popularity window
            self.seen_ = history_df[["customer_id", "article_id"]].drop_duplicates()
        self.anchor_date_ = anchor_date
        return self

    def recommend(self, users=None, k=None):
        self._require(hasattr(self, "ranking_"), "fit must be called before recommend")
        users = self.users_ if users is None else users
        seen = self.seen_ if self.exclude_seen else None
        return self._recommend_from_ranking(self.ranking_, users=users, k=k, seen=seen)

    def score_items(self, customer_id=None):
        """Popularity is not personalised, so the ranking is the same for everyone."""
        self._require(hasattr(self, "ranking_"), "fit must be called before score_items")
        return self.ranking_.copy()
import json
from pathlib import Path

import pandas as pd

from models.base import BaseRecommender


class R1RecentModel(BaseRecommender):
    """Ranks each user's historical articles by their latest purchase date, newest first."""

    def __init__(self, k=12):
        super().__init__(k=k)

    def fit(self, history_df, anchor_date):
        self._check_history(history_df, anchor_date)

        history = history_df.loc[:, ["t_dat", "customer_id", "article_id"]]
        self.scores_ = (
            history.groupby(["customer_id", "article_id"], as_index=False, sort=False)
            .agg(
                last_date=("t_dat", "max"),
                n_purchase=("t_dat", "count"),
            )
            .assign(
                score=lambda frame: -(anchor_date - frame["last_date"]).dt.days
            )
            .sort_values(
                ["customer_id", "last_date", "article_id"],
                ascending=[True, False, False],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )
        self.anchor_date_ = anchor_date
        return self

    def recommend(self, users=None, k=None):
        return self._recommend_from_scores(users=users, k=k)

    def score_items(self, customer_id):
        return self._score_items(customer_id)

import json
from pathlib import Path

import numpy as np
import pandas as pd

from models.base import BaseRecommender


class RecencyFrequencyModel(BaseRecommender):
    """Ranks each user's historical articles by summed half-life-decayed purchase weights."""

    def __init__(self, half_life_days=30, history_cap_days=365, k=12):
        self._require(half_life_days > 0, "half_life_days must be greater than 0")
        self._require(history_cap_days > 0, "history_cap_days must be greater than 0")
        super().__init__(k=k)
        self.half_life_days = half_life_days
        self.history_cap_days = history_cap_days

    def fit(self, history_df, anchor_date):
        self._check_history(history_df, anchor_date)

        history = history_df.loc[:, ["t_dat", "customer_id", "article_id"]].copy()
        history["age_days"] = (anchor_date - history["t_dat"]).dt.days
        history = history.loc[history["age_days"] <= self.history_cap_days].copy()
        history["weight"] = np.power(0.5, history["age_days"] / self.half_life_days)

        self.scores_ = (
            history.groupby(["customer_id", "article_id"], as_index=False, sort=False)
            .agg(
                score=("weight", "sum"),
                n_purchase=("t_dat", "count"),
                last_date=("t_dat", "max"),
            )
            .sort_values(
                ["customer_id", "score", "last_date", "article_id"],
                ascending=[True, False, False, False],
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

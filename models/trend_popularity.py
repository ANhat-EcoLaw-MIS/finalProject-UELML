import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from models.base import BaseRecommender


class TrendPopularityModel(BaseRecommender):
    """Popularity + momentum: ranks articles by recent volume and by how fast they are growing.

    Two consecutive windows before anchor_date, each `window_days` long:
        recent   : [anchor - w,  anchor)        -> R_i, purchases in the recent window
        previous : [anchor - 2w, anchor - w)    -> P_i, purchases in the window before it
    Trend    T_i = (R_i - P_i) / (P_i + epsilon)
    Score    S_i = alpha * norm(R_i) + beta * norm(T_i)

    Like PopularityModel, every user gets the same list, so coldstart users are served.

    window_days  : length of each window; use multiples of 7 to avoid weekly-seasonality bias
    epsilon      : smoothing; a larger value damps the trend of articles with a tiny P_i
    alpha, beta  : weight of popularity (volume) and trend (momentum)
    normalize    : "minmax" (default), "zscore" or None, applied to R and T before weighting
    min_sales    : the trend only counts for articles with R_i >= min_sales (others get T = 0);
                   the default 0 means no threshold
    exclude_seen : skip articles the user already bought in history (discovery baseline)
    """

    def __init__(self, window_days=7, epsilon=50, alpha=1.0, beta=1.0,
                 normalize="minmax", min_sales=0, exclude_seen=False, k=12):
        self._require(window_days > 0, "window_days must be greater than 0")
        self._require(epsilon > 0, "epsilon must be greater than 0")
        self._require(alpha >= 0 and beta >= 0, "alpha and beta must not be negative")
        self._require(alpha + beta > 0, "alpha and beta cannot both be 0")
        self._require(normalize in ("minmax", "zscore", None),
                      "normalize must be 'minmax', 'zscore' or None")
        self._require(min_sales >= 0, "min_sales must not be negative")
        super().__init__(k=k)
        if window_days % 7 != 0:
            warnings.warn("window_days is not a multiple of 7: weekly seasonality may bias the trend")
        self.window_days = window_days
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.normalize = normalize
        self.min_sales = min_sales
        self.exclude_seen = exclude_seen

    def _norm(self, values):
        values = np.asarray(values, dtype=float)
        if self.normalize == "minmax":
            span = values.max() - values.min()
            return (values - values.min()) / span if span > 0 else np.zeros_like(values)
        if self.normalize == "zscore":
            std = values.std()
            return (values - values.mean()) / std if std > 0 else np.zeros_like(values)
        return values

    def fit(self, history_df, anchor_date):
        self._check_history(history_df, anchor_date)

        window = pd.Timedelta(days=self.window_days)
        recent_start = anchor_date - window
        previous_start = anchor_date - 2 * window
        if history_df["t_dat"].min() > previous_start:
            warnings.warn("history does not cover two full windows: the previous window is "
                          "partial, so trend scores will be inflated")

        history = history_df.loc[history_df["t_dat"] >= previous_start, ["t_dat", "article_id"]]
        is_recent = history["t_dat"] >= recent_start
        self._require(len(history) > 0, "no purchases inside the two popularity windows")

        recent = history.loc[is_recent].groupby("article_id").size().rename("recent_volume")
        previous = history.loc[~is_recent].groupby("article_id").size().rename("previous_volume")
        table = pd.concat([recent, previous], axis=1).fillna(0).astype(int).reset_index()

        raw_trend = (table["recent_volume"] - table["previous_volume"]) / (
            table["previous_volume"] + self.epsilon
        )
        table["trend_score"] = np.where(table["recent_volume"] >= self.min_sales, raw_trend, 0.0)
        table["score"] = (
            self.alpha * self._norm(table["recent_volume"])
            + self.beta * self._norm(table["trend_score"])
        )

        self.ranking_ = (
            table.sort_values(
                ["score", "recent_volume", "article_id"],
                ascending=[False, False, False],
                kind="mergesort",
            )
            .reset_index(drop=True)
            .assign(rank=lambda frame: np.arange(1, len(frame) + 1))
        )
        self.users_ = history_df["customer_id"].unique()  # used when recommend(users=None)
        if self.exclude_seen:  # uses the full history, not only the two windows
            self.seen_ = history_df[["customer_id", "article_id"]].drop_duplicates()
        self.anchor_date_ = anchor_date
        return self

    def recommend(self, users=None, k=None):
        self._require(hasattr(self, "ranking_"), "fit must be called before recommend")
        users = self.users_ if users is None else users
        seen = self.seen_ if self.exclude_seen else None
        return self._recommend_from_ranking(self.ranking_, users=users, k=k, seen=seen)

    def score_items(self, customer_id=None):
        """Candidate table (same for every user): article_id, recent_volume, trend_score,
        final_score, rank."""
        self._require(hasattr(self, "ranking_"), "fit must be called before score_items")
        return (
            self.ranking_.rename(columns={"score": "final_score"})
            [["article_id", "recent_volume", "trend_score", "final_score", "rank"]]
            .copy()
        )

    def to_parquet(self, path):
        self._require(hasattr(self, "ranking_"), "fit must be called before to_parquet")
        path = Path(path)
        self.ranking_.to_parquet(path, index=False)
        if self.exclude_seen:
            self.seen_.to_parquet(path.with_name(path.stem + "_seen.parquet"), index=False)
        settings = {
            "window_days": self.window_days,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "beta": self.beta,
            "normalize": self.normalize,
            "min_sales": self.min_sales,
            "exclude_seen": self.exclude_seen,
            "k": self.k,
            "anchor_date": self.anchor_date_.isoformat(),
        }
        path.with_suffix(".json").write_text(json.dumps(settings), encoding="utf-8")

    @classmethod
    def from_parquet(cls, path):
        path = Path(path)
        settings = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        anchor = settings.pop("anchor_date")
        model = cls(**settings)
        model.ranking_ = pd.read_parquet(path)
        if model.exclude_seen:
            model.seen_ = pd.read_parquet(path.with_name(path.stem + "_seen.parquet"))
        model.anchor_date_ = pd.Timestamp(anchor)
        return model
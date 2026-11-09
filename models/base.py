from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseRecommender(ABC):
    """Common interface for all recommenders.

    Contract
    --------
    fit(history_df, anchor_date) -> self
        history_df needs t_dat (datetime), customer_id, article_id; every t_dat < anchor_date.
    recommend(users=None, k=None) -> DataFrame[customer_id, article_id, rank, score]
        rank starts at 1; at most k rows per user; users the model cannot serve get no rows.

    Two kinds of fitted state (a model sets exactly one of them):
    - scores_  : personalised table (customer_id, article_id, score), pre-sorted per customer
                 -> use _recommend_from_scores   (e.g. recent, recency-frequency)
    - ranking_ : one non-personalised list (article_id, score), sorted best first
                 -> use _recommend_from_ranking  (e.g. popularity; also serves cold-start users)
    """

    scores_: pd.DataFrame
    ranking_: pd.DataFrame
    anchor_date_: pd.Timestamp

    def __init__(self, k=12):
        self._require(k >= 1, "k must be at least 1")
        self.k = k

    @abstractmethod
    def fit(self, history_df, anchor_date):
        raise NotImplementedError

    @abstractmethod
    def recommend(self, users=None, k=None):
        raise NotImplementedError

    @staticmethod
    def _require(condition, message):
        if not condition:
            print(message)
            raise AssertionError(message)

    @classmethod
    def _check_history(cls, history_df, anchor_date):
        required_columns = {"t_dat", "customer_id", "article_id"}
        missing_columns = required_columns.difference(history_df.columns)
        cls._require(
            not missing_columns,
            f"history_df is missing required columns: {sorted(missing_columns)}",
        )
        cls._require(
            pd.api.types.is_datetime64_any_dtype(history_df["t_dat"]),
            "history_df.t_dat must have a datetime64 dtype",
        )
        cls._require(
            not history_df[["t_dat", "customer_id", "article_id"]]
            .isna()
            .any()
            .any(),
            "history_df must not contain nulls in t_dat, customer_id, or article_id",
        )
        cls._require(
            isinstance(anchor_date, pd.Timestamp),
            "anchor_date must be a pandas Timestamp",
        )
        cls._require(
            history_df["t_dat"].max() < anchor_date,
            "history_df.t_dat.max() must be earlier than anchor_date",
        )

    def _recommend_from_scores(self, users=None, k=None):
        self._require(hasattr(self, "scores_"), "fit must be called before recommend")
        limit = self.k if k is None else k
        self._require(limit >= 1, "k must be at least 1")

        scores = self.scores_
        if users is not None:
            scores = scores.loc[scores["customer_id"].isin(users)]

        recommendations = scores.groupby("customer_id", sort=False).head(limit).copy()
        recommendations["rank"] = (
            recommendations.groupby("customer_id", sort=False).cumcount() + 1
        )
        return recommendations[["customer_id", "article_id", "rank", "score"]]

    def _recommend_from_ranking(self, ranking, users, k=None, seen=None):
        """Give every requested user the top-k of `ranking` (works for unseen users).

        seen : optional DataFrame[customer_id, article_id] of items to skip per user
               (e.g. what they already bought). Each user still gets k items if enough remain.
        """
        limit = self.k if k is None else k
        self._require(limit >= 1, "k must be at least 1")
        self._require(users is not None, "users must be provided for a non-personalised model")

        users = pd.Series(users).drop_duplicates().to_numpy()
        if seen is None:
            return self._top_for_users(ranking, users, limit)

        # Skip seen items: take a pool larger than k, drop each user's seen items, and
        # re-try with a bigger pool only for users who still have fewer than k items.
        pool_size = min(limit * 4, len(ranking))
        pending, done = users, []
        while True:
            pool = ranking.head(pool_size)
            cand = self._top_for_users(pool, pending, pool_size)
            seen_sub = seen.loc[
                seen["customer_id"].isin(pending) & seen["article_id"].isin(pool["article_id"]),
                ["customer_id", "article_id"],
            ].assign(_seen=1)
            cand = cand.merge(seen_sub, how="left", on=["customer_id", "article_id"])
            cand = cand.loc[cand["_seen"].isna()].drop(columns="_seen")
            cand["rank"] = cand.groupby("customer_id", sort=False).cumcount() + 1
            cand = cand.loc[cand["rank"] <= limit]

            counts = cand.groupby("customer_id").size()
            full = counts.index[counts >= limit]
            exhausted = pool_size >= len(ranking)
            if exhausted:
                done.append(cand)  # nothing more to try: keep whatever is left
                break
            done.append(cand.loc[cand["customer_id"].isin(full)])
            pending = pending[~pd.Series(pending).isin(full).to_numpy()]
            if len(pending) == 0:
                break
            pool_size = min(pool_size * 2, len(ranking))
        return pd.concat(done, ignore_index=True)[["customer_id", "article_id", "rank", "score"]]

    @staticmethod
    def _top_for_users(ranking, users, limit):
        top = ranking.head(limit)
        n_items = len(top)
        return pd.DataFrame(
            {
                "customer_id": np.repeat(users, n_items),
                "article_id": np.tile(top["article_id"].to_numpy(), len(users)),
                "rank": np.tile(np.arange(1, n_items + 1), len(users)),
                "score": np.tile(top["score"].to_numpy(), len(users)),
            }
        )

    def _score_items(self, customer_id):
        self._require(hasattr(self, "scores_"), "fit must be called before score_items")
        return self.scores_.loc[self.scores_["customer_id"] == customer_id].copy()
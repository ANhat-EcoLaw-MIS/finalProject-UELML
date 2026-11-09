import pandas as pd

from models.base import BaseRecommender


class CategoryPopularityModel(BaseRecommender):
    """Personalised popularity: the best sellers inside the categories each user likes.

    Two-step score for user u and article j in category c:
        score(j | u) = share_u(c) * p(j | c)
    - share_u(c) : fraction of u's history (optionally recency-weighted) in category c
    - p(j | c)   : fraction of category c's recent sales (last `window_days`) that were article j

    It uses the same "because you bought X" link as item-item CF (category), but has no
    co-purchase signal, so it is the baseline CF must beat to justify its complexity.
    Users with no history get no recommendations (pair it with PopularityModel for them).

    article_type   : Series, index = article_id, value = category (e.g. product_type_no)
    window_days    : window used to measure article popularity inside a category
    half_life_days : if set, recent purchases weigh more in the user's category profile
    max_categories : only the user's top categories are used (keeps the candidate table small)
    per_category   : candidates kept per category (must exceed k so seen items can be skipped)
    exclude_seen   : skip articles the user already bought (discovery)
    """

    def __init__(self, article_type, window_days=28, half_life_days=None,
                 max_categories=5, per_category=50, exclude_seen=True, k=12):
        self._require(window_days > 0, "window_days must be greater than 0")
        self._require(half_life_days is None or half_life_days > 0,
                      "half_life_days must be greater than 0")
        self._require(max_categories >= 1, "max_categories must be at least 1")
        self._require(per_category >= k, "per_category must be at least k")
        super().__init__(k=k)
        self.article_type = article_type
        self.window_days = window_days
        self.half_life_days = half_life_days
        self.max_categories = max_categories
        self.per_category = per_category
        self.exclude_seen = exclude_seen

    def fit(self, history_df, anchor_date):
        self._check_history(history_df, anchor_date)

        history = history_df.loc[:, ["t_dat", "customer_id", "article_id"]].copy()
        history["cat"] = history["article_id"].map(self.article_type)
        history = history.dropna(subset=["cat"])
        self._require(len(history) > 0, "no purchased article could be mapped to a category")

        # p(j | c): popularity of each article inside its category, recent window only
        start = anchor_date - pd.Timedelta(days=self.window_days)
        recent = history.loc[history["t_dat"] >= start]
        self._require(len(recent) > 0, "no purchases inside the popularity window")
        items = recent.groupby(["cat", "article_id"], as_index=False).size().rename(
            columns={"size": "n"}
        )
        items["p_item"] = items["n"] / items.groupby("cat")["n"].transform("sum")
        self.cat_items_ = (
            items.sort_values(["cat", "p_item", "article_id"], ascending=[True, False, False],
                              kind="mergesort")
            .groupby("cat")
            .head(self.per_category)[["cat", "article_id", "p_item"]]
            .reset_index(drop=True)
        )

        # share_u(c): the user's category profile, optionally recency-weighted
        if self.half_life_days is None:
            history["w"] = 1.0
        else:
            age_days = (anchor_date - history["t_dat"]).dt.days
            history["w"] = 0.5 ** (age_days / self.half_life_days)
        profile = history.groupby(["customer_id", "cat"], as_index=False)["w"].sum()
        profile["share"] = profile["w"] / profile.groupby("customer_id")["w"].transform("sum")
        self.profile_ = (
            profile.sort_values(["customer_id", "share", "cat"], ascending=[True, False, True],
                                kind="mergesort")
            .groupby("customer_id")
            .head(self.max_categories)[["customer_id", "cat", "share"]]
            .reset_index(drop=True)
        )

        if self.exclude_seen:  # full history, not only the popularity window
            self.seen_ = history_df[["customer_id", "article_id"]].drop_duplicates()
        self.anchor_date_ = anchor_date
        return self

    def recommend(self, users=None, k=None):
        self._require(hasattr(self, "profile_"), "fit must be called before recommend")
        limit = self.k if k is None else k
        self._require(limit >= 1, "k must be at least 1")
        self._require(limit <= self.per_category, "k cannot exceed per_category")

        profile = self.profile_
        if users is not None:
            profile = profile.loc[profile["customer_id"].isin(pd.Series(users).unique())]

        cand = profile.merge(self.cat_items_, on="cat")
        cand["score"] = cand["share"] * cand["p_item"]
        if self.exclude_seen:
            seen = self.seen_.loc[
                self.seen_["customer_id"].isin(profile["customer_id"].unique())
                & self.seen_["article_id"].isin(cand["article_id"].unique())
            ].assign(_seen=1)
            cand = cand.merge(seen, how="left", on=["customer_id", "article_id"])
            cand = cand.loc[cand["_seen"].isna()]

        cand = cand.sort_values(["customer_id", "score", "article_id"],
                                ascending=[True, False, False], kind="mergesort")
        recs = cand.groupby("customer_id", sort=False).head(limit).copy()
        recs["rank"] = recs.groupby("customer_id", sort=False).cumcount() + 1
        return recs[["customer_id", "article_id", "rank", "score"]].reset_index(drop=True)

    def score_items(self, customer_id):
        """The user's candidate articles with their scores (before the k cut)."""
        self._require(hasattr(self, "profile_"), "fit must be called before score_items")
        return self.recommend(users=[customer_id], k=self.per_category)
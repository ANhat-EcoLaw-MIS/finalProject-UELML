import pandas as pd

from models.base import BaseRecommender


class BlendModel(BaseRecommender):
    """Combine several recommenders into one list with Reciprocal Rank Fusion (RRF).

    Every component recommends a pool of `pool_k` articles per user. An article's fused score is
        score(a) = sum over components m of  weight_m / (rrf_c + rank_m(a))
    so an article found by several components, or ranked high by a heavy component, rises to
    the top. The best `k` fused articles are returned.

    Useful properties
    - Equal weights and few overlaps behave like slot allocation (2 + 2 for k=4).
    - A user that only one component can serve simply gets that component's list, so the
      blend doubles as a fallback chain (e.g. no history -> popularity, if it is a component).
    - rrf_c controls how much the top ranks matter: a small c rewards rank 1 strongly, a large
      c (the literature default is 60) flattens the ranks so agreement between components wins.

    models          dict {name: BaseRecommender}, e.g. {"cat_pop": ..., "cf_lastday": ...}
    weights         dict {name: weight}; missing names get 1.0
    rrf_c           RRF constant
    pool_k          candidates requested from each component (must be >= k)
    fit_components  True: fit() also fits every component.
                    False: the components are assumed to be already fitted on the same history
                    (cheap re-weighting; list the components BEFORE the blend in compare_models).
    """

    def __init__(self, models, weights=None, rrf_c=60, pool_k=20, fit_components=True, k=12):
        self._require(len(models) >= 1, "at least one component model is required")
        weights = {} if weights is None else dict(weights)
        self._require(set(weights) <= set(models), "weights has names that are not in models")
        self._require(rrf_c >= 0, "rrf_c must not be negative")
        self._require(pool_k >= k, "pool_k must be at least k")
        super().__init__(k=k)
        self.models = dict(models)
        self.weights = {name: float(weights.get(name, 1.0)) for name in self.models}
        self._require(all(w >= 0 for w in self.weights.values()), "weights must not be negative")
        self._require(sum(self.weights.values()) > 0, "at least one weight must be positive")
        self.rrf_c = rrf_c
        self.pool_k = pool_k
        self.fit_components = fit_components

    def fit(self, history_df, anchor_date):
        self._check_history(history_df, anchor_date)
        if self.fit_components:
            for model in self.models.values():
                model.fit(history_df, anchor_date)
        self.anchor_date_ = anchor_date
        return self

    def reweighted(self, weights=None, rrf_c=None, pool_k=None):
        """A copy with other weights / constants that shares the SAME fitted components.

        Lets you tune the blend without refitting anything (fit_components is False).
        Call it after this blend has been fitted."""
        self._require(hasattr(self, "anchor_date_"), "fit must be called before reweighted")
        other = BlendModel(
            self.models,
            weights=self.weights if weights is None else weights,
            rrf_c=self.rrf_c if rrf_c is None else rrf_c,
            pool_k=self.pool_k if pool_k is None else pool_k,
            fit_components=False,
            k=self.k,
        )
        other.anchor_date_ = self.anchor_date_
        return other

    def _pool(self, users):
        """Components' recommendations, with each row's RRF contribution."""
        parts = []
        for name, model in self.models.items():
            weight = self.weights[name]
            if weight == 0:
                continue
            recs = model.recommend(users=users, k=self.pool_k)
            parts.append(
                recs[["customer_id", "article_id", "rank"]].assign(
                    rrf=weight / (self.rrf_c + recs["rank"])
                )
            )
        return pd.concat(parts, ignore_index=True)

    def recommend(self, users=None, k=None):
        self._require(hasattr(self, "anchor_date_"), "fit must be called before recommend")
        limit = self.k if k is None else k
        self._require(limit >= 1, "k must be at least 1")

        fused = (
            self._pool(users)
            .groupby(["customer_id", "article_id"], as_index=False)["rrf"]
            .sum()
            .rename(columns={"rrf": "score"})
            .sort_values(["customer_id", "score", "article_id"],
                         ascending=[True, False, False], kind="mergesort")
        )
        top = fused.groupby("customer_id", sort=False).head(limit).copy()
        top["rank"] = top.groupby("customer_id", sort=False).cumcount() + 1
        return top[["customer_id", "article_id", "rank", "score"]].reset_index(drop=True)

    def score_items(self, customer_id, k=None):
        """The fused list for one user plus each component's rank for it (NaN = not in its pool).
        Use it to say where a recommendation came from."""
        self._require(hasattr(self, "anchor_date_"), "fit must be called before score_items")
        fused = self.recommend(users=[customer_id], k=k).set_index("article_id")
        ranks = {
            f"rank_{name}": model.recommend(users=[customer_id], k=self.pool_k)
            .set_index("article_id")["rank"]
            for name, model in self.models.items()
        }
        return fused[["rank", "score"]].join(pd.DataFrame(ranks)).reset_index()
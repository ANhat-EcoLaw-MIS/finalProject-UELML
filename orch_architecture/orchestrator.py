"""Orchestrator: routes each user to the right models and assembles the website shelves.

It does NOT contain any recommendation logic. It only
  1. decides who is a new visitor and who is a returning customer,
  2. calls fit() / recommend() of the models it was given,
  3. removes articles already shown in an earlier shelf,
  4. gives users who ended up with nothing the fallback shelf (cold start),
  5. attaches a reason to each article (shelf-level, plus item-level where a Section has an explainer).
Any BaseRecommender can be plugged in, so models can be swapped without touching this file.

Sizes
  - New visitors / fallback shelf: fixed size (Section.k). The web never needs to pass k for them.
  - Returning-user shelves: up to `max_k` scored candidates are returned and the SERVER decides
    where to cut them (see `trim`).

    orch = build_default(types, article_product=pc).fit(train, anchor_date)
    orch.save("orch_architecture.pkl")                       # later: Orchestrator.load(...)
    page = orch.page("customer_a", article_names=names) # {shelf: [{article_id, name, score, reason}]}
    page = Orchestrator.trim(page, {"Buy it again": {"min_score": 0.01, "max_items": 12},
                                    "Because you bought": {"max_items": 8, "min_items": 3}})
"""
import pickle
from dataclasses import dataclass

import pandas as pd

COLUMNS = ["customer_id", "section", "rank", "article_id", "score"]
AUDIENCES = ("new", "returning", "all")


@dataclass
class Section:
    """One shelf on the page."""
    name: str                    # shelf title shown on the website
    model: object                # any BaseRecommender
    k: int = 12                  # fixed size, used for "new" audience / fallback shelves only
    audience: str = "returning"  # "new" (no history), "returning" (has history) or "all"
    reason: str = ""             # shelf-level label for every article (defaults to the shelf name)
    explain: object = None       # optional item-level labeler: explain(user, recs, names) -> {article: text}


class Orchestrator:
    """sections  : shelves in display order. A shelf is skipped for a user the model cannot serve.
    fallback  : Section used for users who end up with no shelf at all (usually the cold-start one).
    overfetch : extra articles requested from each model so that removing duplicates
                shown in earlier shelves still leaves enough articles.
    max_k     : number of scored candidates returned per returning-user shelf. Models hold a limited
                pool (the blend has pool_k=20), so a larger value would silently return fewer.
    """

    def __init__(self, sections, fallback=None, overfetch=8, max_k=20):
        names = [s.name for s in sections]
        assert len(set(names)) == len(names), "section names must be unique"
        assert all(s.audience in AUDIENCES for s in sections), f"audience must be one of {AUDIENCES}"
        assert all(s.k >= 1 for s in sections), "k must be at least 1"
        self.sections = list(sections)
        self.fallback = fallback
        self.overfetch = overfetch
        self.max_k = max_k

    def fit(self, history_df, anchor_date, fit_models=True):
        """Fit every distinct model once. fit_models=False reuses models that are already fitted."""
        if fit_models:
            used = self.sections + ([self.fallback] if self.fallback is not None else [])
            for model in {id(s.model): s.model for s in used}.values():
                model.fit(history_df, anchor_date)
        self.known_users_ = set(history_df["customer_id"].unique())
        self.anchor_date_ = anchor_date
        return self

    def segment(self, users):
        """'returning' if the user has purchase history, otherwise 'new'."""
        users = self._as_series(users)
        return pd.Series(["returning" if u in self.known_users_ else "new" for u in users],
                         index=users.to_numpy(), name="segment")

    def recommend(self, users, k=None):
        """Tidy table. Returning-user shelves hold up to k (default max_k) scored candidates;
        fixed shelves (new audience / fallback) always keep their own size."""
        assert hasattr(self, "known_users_"), "fit must be called before recommend"
        assert k is None or k >= 1, "k must be at least 1"
        k = self.max_k if k is None else min(int(k), self.max_k)
        users = self._as_series(users)
        known = users.isin(self.known_users_)
        audience = {"new": users[~known], "returning": users[known], "all": users}

        parts, shown = [], pd.DataFrame(columns=["customer_id", "article_id"])
        for section in self.sections:
            recs = self._run(section, audience[section.audience], shown, self._limit(section, k))
            if not recs.empty:
                parts.append(recs)
                shown = pd.concat([shown, recs[["customer_id", "article_id"]]], ignore_index=True)

        if self.fallback is not None:
            served = pd.concat(parts)["customer_id"].unique() if parts else []
            recs = self._run(self.fallback, users[~users.isin(served)], shown, self.fallback.k)
            if not recs.empty:
                parts.append(recs)

        if not parts:
            return pd.DataFrame(columns=COLUMNS)
        order = [s.name for s in self.sections]
        if self.fallback is not None and self.fallback.name not in order:
            order.append(self.fallback.name)
        out = pd.concat(parts, ignore_index=True)
        out["_u"] = out["customer_id"].map({u: i for i, u in enumerate(users)})
        out["_s"] = out["section"].map({n: i for i, n in enumerate(order)})
        return (out.sort_values(["_u", "_s", "rank"], kind="mergesort")
                .drop(columns=["_u", "_s"]).reset_index(drop=True))

    def page(self, user, article_names=None, k=None):
        """What the web calls, for ONE user:
        {shelf: [{"article_id", "name", "score", "reason"}, ...]}  (empty dict if nothing).
        Cut the lists with `trim` (or in the server). Reasons: item-level when the shelf has an
        explainer, otherwise the shelf's `reason` text, otherwise the shelf name."""
        names = article_names or {}
        by_name = {s.name: s for s in self.sections + ([self.fallback] if self.fallback else [])}
        recs = self.recommend(user, k=k)
        page = {}
        for section_name, group in recs.groupby("section", sort=False):
            section = by_name[section_name]
            item_reason = section.explain(user, group, names) if section.explain else {}
            default = section.reason or section.name
            page[section_name] = [
                {"article_id": a, "name": names.get(a, a), "score": float(s),
                 "reason": item_reason.get(a, default)}
                for a, s in zip(group["article_id"], group["score"])
            ]
        return page

    @staticmethod
    def trim(page, rules):
        """Server-side cut. rules = {shelf: {"min_score": x, "max_items": n, "min_items": m}} (all
        optional). Items are cut by score and size; a shelf with fewer than min_items is hidden.
        Shelves without a rule are kept as they are."""
        out = {}
        for shelf, items in page.items():
            rule = rules.get(shelf, {})
            kept = [i for i in items if i["score"] >= rule.get("min_score", float("-inf"))]
            kept = kept[:rule.get("max_items", len(kept))]
            if len(kept) >= rule.get("min_items", 1):
                out[shelf] = kept
        return out

    def save(self, path):
        """Pickle the fitted orch_architecture so the web app can load it instead of refitting."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def shelves(recs, article_names=None):
        """{customer_id: {shelf name: [article_id or name, ...]}} from a recommend() table."""
        page = {}
        for (user, section), group in recs.groupby(["customer_id", "section"], sort=False):
            ids = group["article_id"].tolist()
            page.setdefault(user, {})[section] = (
                ids if article_names is None else [article_names.get(a, a) for a in ids])
        return page

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _as_series(users):
        if isinstance(users, str):
            users = [users]
        return pd.Series(users).drop_duplicates().reset_index(drop=True)

    @staticmethod
    def _limit(section, k):
        return section.k if section.audience == "new" else k

    def _run(self, section, target, shown, limit):
        """Ask the model for limit + overfetch articles, drop those already shown, keep the top limit."""
        if len(target) == 0:
            return pd.DataFrame(columns=COLUMNS)
        recs = section.model.recommend(users=target.to_numpy(), k=limit + self.overfetch)
        if recs.empty:
            return pd.DataFrame(columns=COLUMNS)
        recs = recs[["customer_id", "article_id", "rank", "score"]]
        if len(shown):
            recs = recs.merge(shown.assign(_dup=1), how="left", on=["customer_id", "article_id"])
            recs = recs[recs["_dup"].isna()].drop(columns="_dup")
        recs = (recs.sort_values(["customer_id", "rank"], kind="mergesort")
                .groupby("customer_id", sort=False).head(limit).copy())
        recs["rank"] = recs.groupby("customer_id", sort=False).cumcount() + 1
        recs["section"] = section.name
        return recs[COLUMNS]


class BlendExplainer:
    """Item-level label for the blend shelf. DEMO-LEVEL: it knows the blend has an item-CF component
    (default name "cf") and asks it which owned article drove each recommendation.
        from CF       -> "Because you bought <article>"
        otherwise     -> "Popular in categories you buy"
    A class (not a lambda) so the orch_architecture stays picklable."""

    def __init__(self, blend, cf_name="cf", fallback_text="Popular in categories you buy"):
        self.blend, self.cf_name, self.fallback_text = blend, cf_name, fallback_text

    def __call__(self, user, recs, names=None):
        names = names or {}
        try:
            fused = self.blend.score_items(user, k=1000).set_index("article_id")
            cf_rank = fused.get(f"rank_{self.cf_name}")
            cf = self.blend.models[self.cf_name].score_items(user, k=self.blend.pool_k)
            because = dict(zip(cf["article_id"], cf["because_article_id"]))
        except Exception:      # an explanation must never break the page
            return {}
        out = {}
        for article in recs["article_id"]:
            from_cf = cf_rank is not None and article in cf_rank.index and pd.notna(cf_rank[article])
            if from_cf and article in because:
                out[article] = f"Because you bought {names.get(because[article], because[article])}"
            else:
                out[article] = self.fallback_text
        return out


def build_default(article_type, article_product=None, trending_k=12, because_k=4,
                  cat_pop_weight=0.65, rrf_c=10, max_k=20):
    """The three deployed models:
         new visitors     -> recent popularity (last 7 days)            [fixed size, also the fallback]
         returning users  -> "Buy it again"  : category repurchase
                          -> "Because you bought": blend of category popularity and item CF
    cat_pop_weight is the blend weight of category popularity (item CF gets 1 - weight).
    article_type    Series article_id -> category (e.g. product_type_no)
    article_product Series article_id -> product_code (blocks other colours of owned articles in CF)
    """
    from models.blend_user_discovery import BlendModel
    from models.cat_pop import CategoryPopularityModel
    from models.cat_repurchase import CategoryRepurchaseModel
    from models.itemCFv2 import ItemCFModel
    from models.popularity import PopularityModel

    cat_pop = CategoryPopularityModel(article_type)
    cf = ItemCFModel(article_type, allow_same_category=True, source="last_day",
                     exclude_owned_product_code=article_product is not None,
                     article_product=article_product)
    blend = BlendModel({"cat_pop": cat_pop, "cf": cf},
                       weights={"cat_pop": cat_pop_weight, "cf": 1 - cat_pop_weight},
                       rrf_c=rrf_c, pool_k=20, k=because_k)
    trending = Section("Trending now", PopularityModel(window_days=7), k=trending_k,
                       audience="new", reason="Popular this week")
    return Orchestrator(
        [trending,
         Section("Buy it again", CategoryRepurchaseModel(article_type),
                 reason="You've bought this before"),
         Section("Because you bought", blend, explain=BlendExplainer(blend))],
        fallback=trending, max_k=max_k,
    )



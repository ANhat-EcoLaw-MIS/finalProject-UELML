import numpy as np
import pandas as pd
from scipy import sparse

from models.base import BaseRecommender


class ItemCFModel(BaseRecommender):
    """Item-to-item collaborative filtering for "Because you bought X" (discovery).

    Similarity between articles i and j is the cosine of their buyer vectors (customers x
    articles matrix, binary or time-decayed). Every user is scored by summing the similarity
    between each article they own and each candidate:
        score(j | u) = sum_i  X[u, i] * S[i, j]
    so each recommendation can be traced back to the owned article that drove it (score_items).

    Category switches (the ablation to run for problem 3):
      allow_same_category=False   S[i, j] = 0 when i and j share a category (complements only)
      allow_same_category=True    no category mask in S (plain item-item CF)
      exclude_owned_categories    also drop candidates from ANY category the user already owns
                                  (the stricter cart-style rule)

    article_type        Series, index = article_id, value = category (e.g. product_type_no)
    k_neighbors         neighbours kept per article
    min_cooc            minimum number of shared buyers for a link to count
    min_buyers          articles with fewer buyers are ignored (K-core filter)
    cap_basket_size     (customer, day) baskets with more items are dropped as wholesale noise
    half_life_days      None = binary matrix; otherwise recent purchases weigh more
    exclude_owned_product_code
                        drop candidates that share a product_code with ANY article the user
                        owns (other colours of an owned product). Needs article_product.
    article_product     Series, index = article_id, value = product_code
    source              "history" = all owned articles are sources; "last_day" = only the
                        articles from the user's latest purchase day (the "anchor" variant)
    exclude_seen        skip articles the user already bought
    Users without history, or whose articles all fall outside the catalog, get no rows.
    """

    def __init__(self, article_type, k_neighbors=50, min_cooc=2, min_buyers=5,
                 cap_basket_size=20, half_life_days=None, allow_same_category=False,
                 exclude_owned_categories=False, exclude_owned_product_code=False,
                 article_product=None, source="history", exclude_seen=True,
                 block_size=1000, batch_users=5000, k=12):
        self._require(k_neighbors >= 1, "k_neighbors must be at least 1")
        self._require(min_cooc >= 1, "min_cooc must be at least 1")
        self._require(min_buyers >= 1, "min_buyers must be at least 1")
        self._require(cap_basket_size is None or cap_basket_size >= 1,
                      "cap_basket_size must be None or at least 1")
        self._require(half_life_days is None or half_life_days > 0,
                      "half_life_days must be None or greater than 0")
        self._require(source in ("history", "last_day"), "source must be 'history' or 'last_day'")
        self._require(block_size >= 1 and batch_users >= 1, "block sizes must be at least 1")
        self._require(not exclude_owned_product_code or article_product is not None,
                      "article_product is required when exclude_owned_product_code=True")
        super().__init__(k=k)
        self.article_type = article_type
        self.k_neighbors = k_neighbors
        self.min_cooc = min_cooc
        self.min_buyers = min_buyers
        self.cap_basket_size = cap_basket_size
        self.half_life_days = half_life_days
        self.allow_same_category = allow_same_category
        self.exclude_owned_categories = exclude_owned_categories
        self.exclude_owned_product_code = exclude_owned_product_code
        self.article_product = article_product
        self.source = source
        self.exclude_seen = exclude_seen
        self.block_size = block_size
        self.batch_users = batch_users

    # ------------------------------------------------------------------ fit
    def fit(self, history_df, anchor_date):
        self._check_history(history_df, anchor_date)

        # one row per (customer, day, article); drop wholesale baskets and unmapped articles
        tx = history_df.loc[:, ["t_dat", "customer_id", "article_id"]].drop_duplicates()
        if self.cap_basket_size is not None:
            size = tx.groupby(["customer_id", "t_dat"])["article_id"].transform("size")
            tx = tx.loc[size <= self.cap_basket_size]
        tx = tx.loc[tx["article_id"].isin(self.article_type.index)]

        # customer-article pairs (latest purchase date) and the K-core filter on buyers
        pairs = tx.groupby(["customer_id", "article_id"], as_index=False)["t_dat"].max()
        buyers = pairs["article_id"].value_counts()
        pairs = pairs.loc[pairs["article_id"].isin(buyers.index[buyers >= self.min_buyers])]
        self._require(len(pairs) > 0, "no article passes the min_buyers filter")

        users, u = np.unique(pairs["customer_id"], return_inverse=True)
        items, i = np.unique(pairs["article_id"], return_inverse=True)
        shape = (len(users), len(items))
        if self.half_life_days is None:
            weight = np.ones(len(pairs))
        else:
            age_days = (anchor_date - pairs["t_dat"]).dt.days.to_numpy()
            weight = 0.5 ** (age_days / self.half_life_days)
        X = sparse.csr_matrix((weight, (u, i)), shape=shape)
        Xb = X.copy()
        Xb.data[:] = 1

        # articles from each user's latest purchase day (the "anchor" sources)
        user_pos = pd.Series(np.arange(len(users)), index=users)
        item_pos = pd.Series(np.arange(len(items)), index=items)
        last = tx.loc[tx["t_dat"] == tx.groupby("customer_id")["t_dat"].transform("max")]
        lu, li = last["customer_id"].map(user_pos), last["article_id"].map(item_pos)
        ok = lu.notna() & li.notna()
        XL = sparse.csr_matrix(
            (np.ones(ok.sum()), (lu[ok].astype(int), li[ok].astype(int))), shape=shape
        )
        XL.data[:] = 1

        type_id, _ = pd.factorize(self.article_type.reindex(items))
        T = sparse.csr_matrix(
            (np.ones(len(items)), (np.arange(len(items)), type_id)),
            shape=(len(items), type_id.max() + 1),
        )

        if self.exclude_owned_product_code:
            # P: article -> its product_code; UP: user -> product codes of EVERYTHING they own
            # (taken before the K-core filter, so variants of rare owned articles are blocked too)
            codes = pd.Index(self.article_product.dropna().unique())
            item_code = codes.get_indexer(self.article_product.reindex(items).to_numpy())
            has_code = item_code >= 0
            self.P_ = sparse.csr_matrix(
                (np.ones(has_code.sum()), (np.flatnonzero(has_code), item_code[has_code])),
                shape=(len(items), len(codes)),
            )
            owned_pairs = tx[["customer_id", "article_id"]].drop_duplicates()
            ou = owned_pairs["customer_id"].map(user_pos)
            oc = codes.get_indexer(self.article_product.reindex(owned_pairs["article_id"]).to_numpy())
            ok = ou.notna().to_numpy() & (oc >= 0)
            UP = sparse.csr_matrix(
                (np.ones(ok.sum()), (ou[ok].astype(int).to_numpy(), oc[ok])),
                shape=(len(users), len(codes)),
            )
            UP.data[:] = 1
            self.UP_ = UP

        self.S_ = self._similarity(X, Xb, type_id)
        self.X_, self.Xb_, self.XL_, self.T_ = X, Xb, XL, T
        self.users_, self.items_ = users, items
        self.user_pos_, self.item_pos_ = user_pos, item_pos
        self.anchor_date_ = anchor_date
        return self

    def _similarity(self, X, Xb, type_id):
        """Top-k_neighbors cosine neighbours per article, computed block by block."""
        n_items = X.shape[1]
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=0)).ravel())
        Xt, Xc = X.T.tocsr(), X.tocsc()
        if self.half_life_days is not None:
            Xbt, Xbc = Xb.T.tocsr(), Xb.tocsc()

        kept = []
        for start in range(0, n_items, self.block_size):
            stop = min(start + self.block_size, n_items)
            weighted = Xt @ Xc[:, start:stop]                 # all articles x block
            shared = weighted if self.half_life_days is None else Xbt @ Xbc[:, start:stop]
            weighted = weighted.multiply((shared >= self.min_cooc).astype(float)).tocoo()

            nb, src = weighted.row, weighted.col + start      # neighbour, source article
            value = weighted.data / (norms[nb] * norms[src])
            keep = nb != src
            if not self.allow_same_category:
                keep &= type_id[nb] != type_id[src]
            block = pd.DataFrame({"src": src[keep], "nb": nb[keep], "v": value[keep]})
            block = (
                block.sort_values(["src", "v", "nb"], ascending=[True, False, False])
                .groupby("src")
                .head(self.k_neighbors)
            )
            kept.append(block)

        kept = pd.concat(kept, ignore_index=True)
        return sparse.csr_matrix(
            (kept["v"].to_numpy(), (kept["src"].to_numpy(), kept["nb"].to_numpy())),
            shape=(n_items, n_items),
        )

    # ------------------------------------------------------------ recommend
    def _sources(self, rows):
        return (self.XL_ if self.source == "last_day" else self.X_)[rows]

    def recommend(self, users=None, k=None):
        self._require(hasattr(self, "S_"), "fit must be called before recommend")
        limit = self.k if k is None else k
        self._require(limit >= 1, "k must be at least 1")

        wanted = self.users_ if users is None else pd.Series(users).drop_duplicates().to_numpy()
        rows = self.user_pos_.reindex(wanted).dropna().astype(int)

        parts = []
        for start in range(0, len(rows), self.batch_users):
            batch = rows.iloc[start:start + self.batch_users]
            idx = batch.to_numpy()
            owned = self.Xb_[idx]
            scores = (self._sources(idx) @ self.S_).tocsr()
            if self.exclude_seen:
                scores = scores - scores.multiply(owned)
            if self.exclude_owned_categories:
                blocked = ((owned @ self.T_) @ self.T_.T).tocsr()
                blocked.data[:] = 1
                scores = scores - scores.multiply(blocked)
            if self.exclude_owned_product_code:
                blocked = (self.UP_[idx] @ self.P_.T).tocsr()
                blocked.data[:] = 1
                scores = scores - scores.multiply(blocked)
            scores = scores.tocsr()
            scores.eliminate_zeros()

            for r, customer in enumerate(batch.index):
                lo, hi = scores.indptr[r], scores.indptr[r + 1]
                if lo == hi:
                    continue
                cols, vals = scores.indices[lo:hi], scores.data[lo:hi]
                order = np.lexsort((-cols, -vals))[:limit]
                parts.append(pd.DataFrame({
                    "customer_id": customer,
                    "article_id": self.items_[cols[order]],
                    "rank": np.arange(1, len(order) + 1),
                    "score": vals[order],
                }))

        if not parts:
            return pd.DataFrame(columns=["customer_id", "article_id", "rank", "score"])
        return pd.concat(parts, ignore_index=True)

    def score_items(self, customer_id, k=None):
        """The user's recommendations plus the owned article that drove each one
        ("because you bought ...")."""
        recs = self.recommend(users=[customer_id], k=k)
        if recs.empty:
            return recs.assign(because_article_id=pd.Series(dtype=object))
        row = self.user_pos_[customer_id]
        source = self._sources([row])
        owned, weight = source.indices, source.data
        cols = self.item_pos_.reindex(recs["article_id"]).to_numpy()
        contribution = self.S_[owned][:, cols].toarray() * weight[:, None]
        return recs.assign(because_article_id=self.items_[owned[contribution.argmax(axis=0)]])
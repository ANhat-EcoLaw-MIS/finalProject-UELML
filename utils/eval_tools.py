"""Lightweight validation for BaseRecommender models (H&M personalization).

Usage
-----
    from validation import load_splits, compare_models, plot_comparison

    train, val, test = load_splits("data/train.parquet", "data/val.parquet", "data/test.parquet")
    models = {"recent": RecentRecommender(), "recency_frequency": RecencyFrequencyRecommender()}

    summary, per_user = compare_models(models, history_df=train, target_df=val, baseline="recent")
    plot_comparison(summary, baseline="recent", save_path="val_comparison.png")

    # One call per business problem (who is scored x what counts as a hit)
    s1, _ = compare_models(models_p1, train, val, baseline="popularity", problem="cold_start")
    s2, _ = compare_models(models_p2, train, val, baseline="recent", problem="buy_again")
    s3, _ = compare_models(models_p3, train, val, baseline="popularity", problem="discovery")

    # Segment breakdown (new vs light vs heavy buyers)
    seg = compare_by_segment(per_user, history_df=train, baseline="recent")
    plot_segments(seg, baseline="recent", metric="HIT")

    # Bootstrap CIs + paired significance vs baseline(s); history_df adds per-segment results
    ci_df, paired_df = bootstrap_compare(per_user, baseline="recent", history_df=train)
    plot_bootstrap(ci_df, paired_df, baseline="recent", metric="HIT@12", segment="all")

Adding a model = adding one entry to the `models` dict.
Final test run: compare_models(models, history_df=pd.concat([train, val]), target_df=test).
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import bootstrap

K = 12
METRIC_KEYS = ["HIT", "Precision", "Recall"]            # reported, bootstrap-tested and plotted by default
ALL_METRICS = ["MAP", "HIT", "Precision", "Recall"]    # always computed per user; MAP is a reference only


# Each business problem = which users are scored x which purchases count as a hit.
PROBLEMS = {
    "cold_start": dict(user_group="cold", truth_mode="all"),     # no history: any purchase counts
    "buy_again": dict(user_group="warm", truth_mode="repeat"),   # has history: re-bought items only
    "discovery": dict(user_group="warm", truth_mode="novel"),    # has history: items never bought
}
USER_GROUPS = ("all", "cold", "warm")
TRUTH_MODES = ("all", "repeat", "novel", "new_item")


def build_truth(target_df, history_df=None, user_group="all", truth_mode="all"):
    """Unique (customer_id, article_id) purchases in the target window that count as hits.

    user_group : "all" | "cold" (no purchases in history) | "warm" (>= 1 purchase in history)
    truth_mode : "all"      every purchase
                 "repeat"   only articles the user already bought in history
                 "novel"    only articles the user never bought in history
                 "new_item" only articles nobody bought in history (item cold start)
    Users left with no relevant purchase are dropped: they cannot be scored.
    """
    if user_group not in USER_GROUPS:
        raise ValueError(f"user_group must be one of {USER_GROUPS}")
    if truth_mode not in TRUTH_MODES:
        raise ValueError(f"truth_mode must be one of {TRUTH_MODES}")

    truth = target_df[["customer_id", "article_id"]].drop_duplicates()
    if history_df is None:
        if user_group != "all" or truth_mode != "all":
            raise ValueError("history_df is required for user_group / truth_mode filters")
        return truth.reset_index(drop=True)

    if user_group != "all":
        is_warm = truth["customer_id"].isin(history_df["customer_id"].unique())
        truth = truth[is_warm] if user_group == "warm" else truth[~is_warm]

    if truth_mode != "all":
        if truth_mode == "new_item":
            seen = truth["article_id"].isin(history_df["article_id"].unique()).to_numpy()
        else:
            pairs = history_df[["customer_id", "article_id"]].drop_duplicates().assign(_seen=1)
            seen = truth.merge(pairs, how="left", on=["customer_id", "article_id"])[
                "_seen"
            ].notna().to_numpy()
        truth = truth[seen] if truth_mode == "repeat" else truth[~seen]

    if truth.empty:
        raise ValueError(f"No relevant purchases for user_group={user_group!r}, "
                         f"truth_mode={truth_mode!r}")
    return truth.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Metrics (vectorised, per user)
# ----------------------------------------------------------------------------
def score_recommendations(recs, truth, k=K):
    """Per-user MAP@k, HIT@k, Precision@k, Recall@k.

    Every user in `truth` is scored; users with no recommendations get 0.
    - AP@k        = sum_i(precision@i * rel_i) / min(n_relevant, k)   (Kaggle H&M definition)
    - HIT@k       = 1 if at least one recommended item was bought
    - Precision@k = hits / k
    - Recall@k    = hits / n_relevant
    """
    n_rel = truth.groupby("customer_id").size().rename("n_relevant")

    recs = recs.loc[recs["rank"] <= k, ["customer_id", "article_id", "rank"]]
    recs = recs.merge(truth.assign(hit=1), on=["customer_id", "article_id"], how="left")
    recs["hit"] = recs["hit"].fillna(0).astype(int)
    recs = recs.sort_values(["customer_id", "rank"])
    recs["cum_hits"] = recs.groupby("customer_id")["hit"].cumsum()
    recs["ap_term"] = recs["hit"] * recs["cum_hits"] / recs["rank"]

    agg = recs.groupby("customer_id").agg(hits=("hit", "sum"), ap_sum=("ap_term", "sum"))

    out = n_rel.to_frame().join(agg)
    out["has_recs"] = out["hits"].notna()
    out = out.fillna({"hits": 0, "ap_sum": 0})

    out[f"MAP@{k}"] = out["ap_sum"] / np.minimum(out["n_relevant"], k)
    out[f"HIT@{k}"] = (out["hits"] > 0).astype(float)
    out[f"Precision@{k}"] = out["hits"] / k
    out[f"Recall@{k}"] = out["hits"] / out["n_relevant"]
    return out.drop(columns=["ap_sum"])


# ----------------------------------------------------------------------------
# Evaluation / comparison
# ----------------------------------------------------------------------------
def beyond_accuracy(recs, truth, history_df, k=K, head_frac=0.01):
    """Metrics that accuracy alone hides (all computed on the top-k lists):
    - catalog_cov : distinct articles recommended / articles bought in history
    - reach       : share of relevant purchases whose article was recommended to ANY user
                    (an article the model never surfaces can never be a hit)
    - head_share  : share of recommendation slots filled by the top `head_frac` best sellers
                    (popularity bias: 1.0 means the list is only best sellers)
    """
    recs = recs.loc[recs["rank"] <= k]
    if recs.empty:
        return dict(catalog_cov=0.0, reach=0.0, head_share=np.nan)
    popularity = history_df["article_id"].value_counts()
    head = set(popularity.index[: max(1, int(len(popularity) * head_frac))])
    recommended = recs["article_id"].unique()
    return dict(
        catalog_cov=len(recommended) / len(popularity),
        reach=truth["article_id"].isin(recommended).mean(),
        head_share=recs["article_id"].isin(head).mean(),
    )


def evaluate_model(model, history_df, truth, anchor_date, k=K):
    """Fit on history, recommend for the users in `truth`, score against `truth`.

    truth comes from build_truth, so every model is scored on identical users and hits.
    Returns (summary: dict, per_user: DataFrame).
    """
    eval_users = truth["customer_id"].unique()

    t0 = time.perf_counter()
    model.fit(history_df, anchor_date)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    recs = model.recommend(users=eval_users, k=k)
    rec_s = time.perf_counter() - t0

    per_user = score_recommendations(recs, truth, k)
    summary = {f"{m}@{k}": per_user[f"{m}@{k}"].mean() for m in METRIC_KEYS}
    summary.update(
        coverage=per_user["has_recs"].mean(),  # share of eval users that got recs
        n_users=len(per_user),
        fit_s=round(fit_s, 1),
        recommend_s=round(rec_s, 1),
    )
    summary.update(beyond_accuracy(recs, truth, history_df, k))
    return summary, per_user


def compare_models(models, history_df, target_df, baseline, k=K, sample_users=None, seed=42,
                   problem=None, user_group="all", truth_mode="all"):
    """Evaluate every model on the same users and hits, and add lift % vs `baseline`.

    models       : dict {name: BaseRecommender instance}
    problem      : shortcut for a business problem, see PROBLEMS:
                   "cold_start" | "buy_again" | "discovery"   (overrides the two below)
    user_group   : "all" | "cold" | "warm"                     (who is scored)
    truth_mode   : "all" | "repeat" | "novel" | "new_item"     (what counts as a hit)
    sample_users : optional int -> random subset of the scored users (fast iteration)
    Returns (summary DataFrame indexed by model, per-user long DataFrame).
    """
    if baseline not in models:
        raise ValueError(f"baseline '{baseline}' not in models: {list(models)}")
    if problem is not None:
        if problem not in PROBLEMS:
            raise ValueError(f"problem must be one of {list(PROBLEMS)}")
        user_group, truth_mode = PROBLEMS[problem]["user_group"], PROBLEMS[problem]["truth_mode"]

    truth = build_truth(target_df, history_df, user_group, truth_mode)
    if sample_users:
        all_users = truth["customer_id"].unique()
        rng = np.random.default_rng(seed)
        picked = rng.choice(all_users, size=min(sample_users, len(all_users)), replace=False)
        truth = truth[truth["customer_id"].isin(picked)]
    print(f"Setup: user_group={user_group}, truth_mode={truth_mode} -> "
          f"{truth['customer_id'].nunique():,} users, {len(truth):,} relevant purchases")
    known = truth["article_id"].isin(history_df["article_id"].unique()).mean()
    print(f"        {1 - known:.1%} of relevant purchases are articles never seen in history "
          f"(no history-based model can reach them)")

    anchor_date = target_df["t_dat"].min()
    rows, per_user_frames = {}, []
    for name, model in models.items():
        print(f"Evaluating {name} ...")
        rows[name], per_user = evaluate_model(model, history_df, truth, anchor_date, k)
        per_user_frames.append(per_user.assign(model=name).reset_index())

    summary = pd.DataFrame(rows).T
    metric_cols = [f"{m}@{k}" for m in METRIC_KEYS]
    summary[metric_cols] = summary[metric_cols].astype(float)
    for col in metric_cols:
        base = summary.loc[baseline, col]
        summary[f"{col} lift %"] = (summary[col] / base - 1) * 100 if base else np.nan
    summary.attrs.update(problem=problem, user_group=user_group, truth_mode=truth_mode)

    return summary, pd.concat(per_user_frames, ignore_index=True)


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def plot_comparison(summary, baseline, k=None, save_path=None, metrics=None):
    """One panel per metric; baseline in grey, lift vs baseline annotated on each bar."""
    metrics = _resolve_metrics(metrics, _infer_k(summary, k))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))

    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = ["#9e9e9e" if m == baseline else palette[i % len(palette)]
              for i, m in enumerate(summary.index)]

    for ax, metric in zip(axes, metrics):
        vals = summary[metric]
        bars = ax.bar(summary.index, vals, color=colors)
        for bar, name in zip(bars, summary.index):
            label = f"{bar.get_height():.4f}"
            if name != baseline:
                label += f"\n({summary.loc[name, f'{metric} lift %']:+.0f}%)"
            ax.annotate(label, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=8)
        ax.set_title(metric)
        ax.set_ylim(0, vals.max() * 1.25)
        ax.tick_params(axis="x", rotation=30)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Model comparison (baseline: {baseline})", y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------------
# Segment breakdown
# ----------------------------------------------------------------------------
# Segments are defined by how many purchases the customer has in the history window.
DEFAULT_BINS = [-1, 0, 4, np.inf]
DEFAULT_LABELS = ["new (0)", "light (1-4)", "heavy (5+)"]


def _add_segment(per_user, history_df, bins, labels):
    counts = history_df.groupby("customer_id").size()
    df = per_user.copy()
    df["n_hist"] = df["customer_id"].map(counts).fillna(0)
    df["segment"] = pd.cut(df["n_hist"], bins=bins, labels=labels)
    return df


def _infer_k(df, k=None):
    """Read k from column names such as 'HIT@4' or 'HIT@4 lift %' when it is not given."""
    if k is not None:
        return k
    for col in df.columns:
        if "@" in str(col):
            return int(str(col).split("@")[1].split()[0])
    raise ValueError("Cannot infer k: no metric columns like 'HIT@12' found")


def _resolve_metrics(metrics, k=K):
    """None -> all four metrics; otherwise a name or list such as "MAP", "recall@12", ["MAP", "HIT"]."""
    if metrics is None:
        return [f"{m}@{k}" for m in METRIC_KEYS]
    names = [metrics] if isinstance(metrics, str) else list(metrics)
    lookup = {m.lower(): m for m in ALL_METRICS}
    resolved = []
    for name in names:
        key = str(name).lower().split("@")[0]
        if key not in lookup:
            raise ValueError(f"Unknown metric {name!r}; choose from {ALL_METRICS}")
        resolved.append(f"{lookup[key]}@{k}")
    return resolved


def compare_by_segment(per_user, history_df, baseline, k=None,
                       bins=DEFAULT_BINS, labels=DEFAULT_LABELS, metrics=None):
    """Average metrics per (segment, model) plus lift % vs baseline inside each segment.

    per_user   : second output of compare_models
    history_df : the history the models were fitted on (defines the segments)
    bins/labels: edit to change segment definitions, e.g. bins=[-1, 0, np.inf],
                 labels=["new", "returning"] for a plain new-vs-returning split.
    metrics    : which metrics to keep; None = all four, or e.g. "MAP", ["MAP", "Recall"]
    """
    df = _add_segment(per_user, history_df, bins, labels)

    metric_cols = _resolve_metrics(metrics, _infer_k(per_user, k))
    grouped = df.groupby(["segment", "model"], observed=True)
    out = grouped[metric_cols].mean()
    out.insert(0, "n_users", grouped.size())

    base = out.xs(baseline, level="model")[metric_cols]
    base = base.reindex(out.index.get_level_values("segment"))
    for col in metric_cols:
        out[f"{col} lift %"] = (out[col].to_numpy() / base[col].replace(0, np.nan).to_numpy() - 1) * 100
    return out


def _draw_segment_bars(ax, seg_summary, baseline, metric):
    vals = seg_summary[metric].unstack("model")
    lift = seg_summary[f"{metric} lift %"].unstack("model")
    n_users = seg_summary["n_users"].unstack("model").iloc[:, 0]
    models = [baseline] + [m for m in vals.columns if m != baseline]

    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    width = 0.8 / len(models)
    x = np.arange(len(vals))
    for i, m in enumerate(models):
        color = "#9e9e9e" if m == baseline else palette[i % len(palette)]
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals[m], width, label=m, color=color)
        for bar, seg in zip(bars, vals.index):
            label = f"{bar.get_height():.3f}"
            if m != baseline and np.isfinite(lift.loc[seg, m]):
                label += f"\n({lift.loc[seg, m]:+.0f}%)"
            ax.annotate(label, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\nn={n:,}" for s, n in zip(vals.index, n_users)])
    ax.set_ylabel(metric)
    ax.set_ylim(0, max(vals.to_numpy().max(), 1e-12) * 1.25)
    ax.set_title(f"{metric} by customer segment")
    ax.spines[["top", "right"]].set_visible(False)


def plot_segments(seg_summary, baseline, metric=None, save_path=None):
    """Grouped bars: segments on the x-axis, one bar per model, lift vs baseline annotated.

    metric : one metric ("HIT", "HIT@4") or a list (["HIT", "Recall"]) -> one panel each.
             None (default) = every metric kept in the table by compare_by_segment(metrics=...).
    """
    if list(seg_summary.index.names) != ["segment", "model"]:
        raise ValueError(
            "plot_segments needs the table returned by compare_by_segment "
            "(index = segment, model). Got index "
            f"{list(seg_summary.index.names)}; pass compare_by_segment(per_user, ...) "
            "rather than the summary from compare_models."
        )
    if metric is None:  # default: whatever metrics the table holds
        wanted = [c for c in seg_summary.columns if "@" in c and "lift" not in c]
    else:
        wanted = _resolve_metrics(metric, _infer_k(seg_summary))
    missing = [m for m in wanted if m not in seg_summary.columns]
    if missing:
        raise ValueError(f"{missing} not in the table; it has "
                         f"{[c for c in seg_summary.columns if '@' in c and 'lift' not in c]}")

    n_seg = seg_summary.index.get_level_values("segment").nunique()
    fig, axes = plt.subplots(1, len(wanted), figsize=(len(wanted) * (1.5 + 2.2 * n_seg), 4.5),
                             squeeze=False)
    for ax, name in zip(axes[0], wanted):
        _draw_segment_bars(ax, seg_summary, baseline, name)
    axes[0][0].legend(frameon=False)
    fig.suptitle(f"Baseline: {baseline}", y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------------
# Bootstrap confidence intervals and paired significance tests
# ----------------------------------------------------------------------------
def _paired_boot(cand, base, stat, n_boot, ci, seed):
    """scipy paired bootstrap of stat(cand, base) -> (CI low, CI high, resampled distribution).

    Same `seed` + same n gives the same resampled users on every call, so all
    models/metrics are compared on identical resamples.
    """
    res = bootstrap((cand, base), stat, paired=True, vectorized=True, method="percentile",
                    n_resamples=n_boot, confidence_level=ci / 100, random_state=seed)
    return res.confidence_interval.low, res.confidence_interval.high, res.bootstrap_distribution


def _diff(a, b, axis):
    return a.mean(axis=axis) - b.mean(axis=axis)


def _lift_pct(a, b, axis):
    return (a.mean(axis=axis) / b.mean(axis=axis) - 1) * 100


def bootstrap_compare(per_user, baseline, k=None, n_boot=1000, ci=95, seed=42,
                      history_df=None, bins=DEFAULT_BINS, labels=DEFAULT_LABELS, metrics=None):
    """Bootstrap CIs for each model's metrics and paired tests against the baseline(s).

    per_user   : second output of compare_models
    baseline   : a model name, or a list of names (e.g. ["popularity", "recent"])
    history_df : optional; if given, results are also computed for each customer segment
    metrics    : which metrics to test; None = all. Fewer metrics = fewer chance "wins"
    Returns (ci_df, paired_df):
      ci_df     : segment, metric, model, mean, ci_low, ci_high
      paired_df : segment, metric, model, baseline, diff, diff_low, diff_high,
                  lift_pct, lift_low, lift_high, p_value, significant
    """
    baselines = [baseline] if isinstance(baseline, str) else list(baseline)
    metrics = _resolve_metrics(metrics, _infer_k(per_user, k))

    groups = [("all", per_user)]
    if history_df is not None:
        seg_df = _add_segment(per_user, history_df, bins, labels)
        groups += [(str(seg), g) for seg, g in seg_df.groupby("segment", observed=True)]

    ci_rows, paired_rows = [], []
    for seg, df in groups:
        for metric in metrics:
            wide = df.pivot(index="customer_id", columns="model", values=metric)
            if len(wide) < 2:
                continue
            cols = {m: wide[m].to_numpy() for m in wide.columns}

            # CI of each model's mean
            for m, x in cols.items():
                res = bootstrap((x,), np.mean, vectorized=True, method="percentile",
                                n_resamples=n_boot, confidence_level=ci / 100, random_state=seed)
                ci_rows.append(dict(segment=seg, metric=metric, model=m, mean=x.mean(),
                                    ci_low=res.confidence_interval.low,
                                    ci_high=res.confidence_interval.high))

            # Paired difference / lift of each model vs each baseline
            for b in baselines:
                for m, x in cols.items():
                    if m == b:
                        continue
                    d_lo, d_hi, d_dist = _paired_boot(x, cols[b], _diff, n_boot, ci, seed)
                    base_mean = cols[b].mean()
                    l_lo = l_hi = lift = np.nan
                    if base_mean:  # lift undefined when the baseline scores 0
                        l_lo, l_hi, _ = _paired_boot(x, cols[b], _lift_pct, n_boot, ci, seed)
                        lift = (x.mean() / base_mean - 1) * 100
                    paired_rows.append(dict(
                        segment=seg, metric=metric, model=m, baseline=b,
                        diff=x.mean() - base_mean, diff_low=d_lo, diff_high=d_hi,
                        lift_pct=lift, lift_low=l_lo, lift_high=l_hi,
                        p_value=min(1.0, 2 * min((d_dist <= 0).mean(), (d_dist >= 0).mean())),
                        significant=bool(d_lo > 0 or d_hi < 0),
                    ))
    return pd.DataFrame(ci_rows), pd.DataFrame(paired_rows)


def plot_bootstrap(ci_df, paired_df, baseline, metric=None, segment="all", save_path=None):
    """Left: metric with CI per model. Right: lift % vs baseline with CI (filled = significant)."""
    k_used = int(ci_df["metric"].iloc[0].split("@")[1])
    metric = _resolve_metrics(metric or ci_df["metric"].iloc[0], k_used)[0]  # "HIT" or "HIT@4"
    c = ci_df[(ci_df["metric"] == metric) & (ci_df["segment"] == segment)].set_index("model")
    p = paired_df[(paired_df["metric"] == metric) & (paired_df["segment"] == segment)
                  & (paired_df["baseline"] == baseline)].set_index("model")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 0.9 * len(c) + 1.8), sharey=True)
    y = np.arange(len(c))

    # Add horizontal dotted reference lines for each model row across both axes
    for yi in y:
        ax1.axhline(yi, color="#e0e0e0", linestyle=":", linewidth=0.8, zorder=0)
        ax2.axhline(yi, color="#e0e0e0", linestyle=":", linewidth=0.8, zorder=0)

    colors = ["#9e9e9e" if m == baseline else "#1f77b4" for m in c.index]
    ax1.errorbar(c["mean"], y, xerr=[c["mean"] - c["ci_low"], c["ci_high"] - c["mean"]],
                 fmt="none", ecolor="black", capsize=3, zorder=1)
    ax1.scatter(c["mean"], y, c=colors, zorder=2, s=40)
    ax1.set_yticks(y)
    ax1.set_yticklabels(c.index)
    ax1.set_title(f"{metric} (95% CI) - {segment}")

    for i, m in enumerate(c.index):
        if m not in p.index:
            continue
        r = p.loc[m]
        ax2.errorbar(r["lift_pct"], i, xerr=[[r["lift_pct"] - r["lift_low"]],
                                             [r["lift_high"] - r["lift_pct"]]],
                     fmt="o", color="#d62728" if r["significant"] else "#9e9e9e",
                     mfc="#d62728" if r["significant"] else "white", capsize=3, zorder=3)
        ax2.annotate(f"p={r['p_value']:.3f}", (r["lift_high"], i), xytext=(5, 0),
                     textcoords="offset points", va="center", fontsize=8)

    # Reference dotted line at 0% lift against the baseline
    ax2.axvline(0, color="black", lw=1, ls=":", zorder=1)
    ax2.set_title(f"Paired lift vs {baseline} (%)")

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_tradeoff(summary, x="catalog_cov", y=None, save_path=None):
    """Accuracy vs. a beyond-accuracy metric, one labelled point per model."""
    y = y or next(c for c in summary.columns if "@" in c and "lift" not in c)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(summary[x], summary[y], s=50)
    for name, row in summary.iterrows():
        ax.annotate(name, (row[x], row[y]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
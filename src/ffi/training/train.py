"""Offline training and walk-forward backtest of the supervised misstatement model (D.06; BLUEPRINT §10.3, §15)."""

from __future__ import annotations

import random
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ffi import __version__
from ffi.detectors.d_stats import RATIOS, ratio_vector
from ffi.edgar import endpoints
from ffi.edgar.client import EdgarClient, EdgarError
from ffi.models_store import save_model
from ffi.peers.crosssection import build as build_xsec
from ffi.xbrl.facts import FactStore

FEATURES = list(RATIOS)


def features_for(client: EdgarClient, cik: int, fy: int) -> dict[str, float] | None:
    """Ratios for fiscal year `fy`, as first reported and known by April 30 of fy+1 (point in time)."""
    try:
        cf = endpoints.companyfacts(client, cik)
    except EdgarError:
        return None
    fs = FactStore.from_companyfacts(cik, cf, as_of=date(fy + 1, 4, 30))
    a = fs.annual("first")
    per = [p for p in a.values.index if p.year == fy or (p.year == fy + 1 and p.month <= 3)]
    if not per:
        return None
    v = ratio_vector(a, per[-1])
    return v if sum(np.isfinite(x) for x in v.values()) >= 8 else None


def build_dataset(client: EdgarClient, labels: pd.DataFrame, controls_per_positive: int, console=None, seed: int = 7) -> pd.DataFrame:
    rng = random.Random(seed)
    pos = labels[labels["label"] == 1][["cik", "fiscal_year"]].drop_duplicates()
    years = sorted(pos["fiscal_year"].unique())
    xsec = build_xsec(client, None, [int(y) for y in years])
    pool = {int(y): sorted(set(xsec.loc[xsec["year"] == y, "cik"]) - set(pos["cik"])) for y in years}
    from ffi.concurrency import map_concurrent

    jobs: list[tuple[int, int, int]] = []
    for r in pos.itertuples(index=False):
        jobs.append((int(r.cik), int(r.fiscal_year), 1))
        cands = pool.get(int(r.fiscal_year), [])
        jobs += [(int(c), int(r.fiscal_year), 0) for c in rng.sample(cands, min(controls_per_positive, len(cands)))]
    rows = []
    for start in range(0, len(jobs), 100):
        batch = jobs[start: start + 100]
        feats = map_concurrent(lambda j: features_for(client, j[0], j[1]), batch, workers=8)
        rows += [{"cik": c, "fiscal_year": fy, "label": lab, **f} for (c, fy, lab), f in zip(batch, feats, strict=True)
                 if isinstance(f, dict)]
        if console:
            console.print(f"[dim]features: {min(start + 100, len(jobs))}/{len(jobs)} firm-years, {len(rows)} usable[/dim]")
    return pd.DataFrame(rows)


def _fit(X: pd.DataFrame, y: pd.Series):
    import lightgbm as lgb

    pos = max(int(y.sum()), 1)
    params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 20,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
              "scale_pos_weight": (len(y) - pos) / pos, "verbose": -1, "seed": 7}
    return lgb.train(params, lgb.Dataset(X.to_numpy(float), label=y.to_numpy(), feature_name=FEATURES), num_boost_round=300)


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    p = model.predict(X.to_numpy(float))
    k = max(1, int(len(p) * 0.10))
    top = np.argsort(-p)[:k]
    return {"n": int(len(y)), "positives": int(y.sum()), "roc_auc": float(roc_auc_score(y, p)) if y.nunique() > 1 else float("nan"),
            "pr_auc": float(average_precision_score(y, p)) if y.nunique() > 1 else float("nan"),
            "recall_top_decile": float(y.iloc[top].sum() / max(y.sum(), 1)), "base_rate": float(y.mean())}


def train(labels_csv: Path, controls_per_positive: int = 3, backtest: bool = False, console=None, gap_years: int = 2) -> str:
    client = EdgarClient()
    labels = pd.read_csv(labels_csv)
    data_path = labels_csv.with_suffix(".features.parquet")
    if data_path.exists():
        df = pd.read_parquet(data_path)
    else:
        df = build_dataset(client, labels, controls_per_positive, console)
        df.to_parquet(data_path, index=False)
    df = df.dropna(subset=["label"])
    X = df[FEATURES].replace([np.inf, -np.inf], np.nan).clip(-1e3, 1e3)
    y = df["label"].astype(int)
    lines = [f"Dataset: {len(df)} firm-years, {int(y.sum())} positives ({labels_csv.name})"]
    if backtest:
        years = sorted(df["fiscal_year"].unique())
        res = []
        for test_year in years:
            tr = df["fiscal_year"] <= test_year - gap_years - 1
            te = df["fiscal_year"] == test_year
            if y[tr].sum() < 20 or y[te].sum() < 3:
                continue
            m = _fit(X[tr], y[tr])
            r = evaluate(m, X[te], y[te])
            r["test_year"] = int(test_year)
            res.append(r)
            lines.append(f"  test {test_year}: ROC-AUC {r['roc_auc']:.3f}  PR-AUC {r['pr_auc']:.3f}  recall@top10% {r['recall_top_decile']:.2f}  (n={r['n']}, pos={r['positives']})")
        if res:
            lines.append(f"Mean ROC-AUC {np.nanmean([r['roc_auc'] for r in res]):.3f}, mean recall@top10% {np.mean([r['recall_top_decile'] for r in res]):.2f}")
        else:
            lines.append("Not enough labelled years for a walk-forward backtest.")
        return "\n".join(lines)
    model = _fit(X, y)
    meta = {"version": f"{__version__}-{date.today().isoformat()}", "trained_on": f"{labels_csv.name}, {len(df)} firm-years",
            "features": FEATURES, "base_rate": float(y.mean()), "label_source": "see labels CSV 'source' column",
            "in_sample": evaluate(model, X, y),
            "caveat": "8-K 4.02 labels are weak (restated year approximated); case-control sampling inflates the base rate."}
    path = save_model("misstatement", model, meta)
    lines.append(f"Saved model to {path}")
    return "\n".join(lines)

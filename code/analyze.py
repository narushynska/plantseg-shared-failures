"""Analyse per-image results of the multi-seed PlantSeg runs.
python analyze.py <results_dir> <Metadata.csv of v2> <txt list of test images kept in 2025 release> <out_dir>
Writes out_dir/results.json (every number quoted in the paper) and CSV tables incl. hard_subset.csv.
"""
import os, json, glob, itertools
import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

NAMES = {"unet": "U-Net", "att_unet": "Attention U-Net", "fpn": "FPN", "deeplabv3p": "DeepLabV3+"}
ORDER = ["unet", "att_unet", "fpn", "deeplabv3p"]
CATS = [(0, 0.01, "very small (<1%)"), (0.01, 0.05, "small (1-5%)"),
        (0.05, 0.15, "medium (5-15%)"), (0.15, 1.01, "large (>=15%)")]
FAIL = 0.5
rng = np.random.default_rng(2026)


def load(RES):
    rows = []
    for f in glob.glob(os.path.join(RES, "per_image_*_s*.csv")):
        tag = os.path.basename(f)[10:-4]; m, s = tag.rsplit("_s", 1)
        df = pd.read_csv(f); df["model"] = m; df["seed"] = int(s); rows.append(df)
    pi = pd.concat(rows, ignore_index=True)
    summ = [json.load(open(f)) for f in glob.glob(os.path.join(RES, "summary_*.json"))]
    return pi, pd.DataFrame([{k: v for k, v in s.items() if k != "history"} for s in summ]), summ


def pooled(df):
    TP, FP, FN = df.tp.sum(), df.fp.sum(), df.fn.sum()
    return dict(precision=TP / (TP + FP), recall=TP / (TP + FN), f1=2 * TP / (2 * TP + FP + FN), iou=TP / (TP + FP + FN))


def boot_diff(a, b, B=10000):
    n = len(a); idx = rng.integers(0, n, (B, n)); d = a[idx].mean(1) - b[idx].mean(1)
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def f1_of(x):
    return 2 * x[..., 0] / (2 * x[..., 0] + x[..., 1] + x[..., 2])


def boot_f1_diff(A, Bm, B=4000):
    n = len(A); i = rng.integers(0, n, (B, n))
    d = f1_of(A[i].sum(1)) - f1_of(Bm[i].sum(1))
    return float(f1_of(A.sum(0)) - f1_of(Bm.sum(0))), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main(RES, META_CSV, KEEP_TXT, OUT):
    os.makedirs(OUT, exist_ok=True)
    keep = set(open(KEEP_TXT, encoding="utf-8").read().splitlines())
    pi, sm_df, summ = load(RES)
    meta = pd.read_csv(os.path.join(RES, "meta_test.csv"))
    md = pd.read_csv(META_CSV); md["image"] = md["Name"].str.strip()
    meta = meta.merge(md[["image", "Plant", "Disease"]], on="image", how="left")
    meta["Disease"] = meta["Disease"].str.strip()
    meta["cat"] = pd.cut(meta.lesion_frac, [c[0] for c in CATS] + [1.01], right=False, labels=[c[2] for c in CATS])
    meta["in_2025"] = meta.image.isin(keep)
    models = [m for m in ORDER if m in pi.model.unique()]
    seeds = sorted(pi.seed.unique())
    R = {"n_test": int(len(meta)), "seeds": [int(s) for s in seeds], "models": models,
         "cat_counts": meta.cat.value_counts().reindex([c[2] for c in CATS]).astype(int).tolist()}

    # 1. accuracy per seed and mean +- sd
    acc = []
    for m in models:
        for s in seeds:
            d = pi[(pi.model == m) & (pi.seed == s)]
            if d.empty: continue
            row = dict(model=m, seed=s, dice_img=d.dice.mean(), **pooled(d))
            srow = sm_df[(sm_df.model == m) & (sm_df.seed == s)].iloc[0]
            row.update(dice_batch=srow.test_dice_batch, train_min=srow.train_minutes, best_val=srow.best_val_dice)
            for sub, flag in [("2025", True), ("removed", False)]:
                dd = d[d.image.isin(meta.image[meta.in_2025 == flag])]
                row[f"dice_img_{sub}"] = dd.dice.mean(); row[f"f1_{sub}"] = pooled(dd)["f1"]
            acc.append(row)
    acc = pd.DataFrame(acc); acc.to_csv(os.path.join(OUT, "accuracy_per_seed.csv"), index=False)
    keys = ["dice_batch", "dice_img", "iou", "precision", "recall", "f1", "train_min", "best_val",
            "dice_img_2025", "f1_2025", "dice_img_removed", "f1_removed"]
    agg = acc.groupby("model")[keys].agg(["mean", "std"]).reindex(models)
    agg.to_csv(os.path.join(OUT, "accuracy_mean_sd.csv"))
    R["accuracy"] = {m: {k: [float(agg.loc[m, (k, "mean")]), float(np.nan_to_num(agg.loc[m, (k, "std")]))] for k in keys} for m in models}
    R["seed_range_dice_img"] = {m: float(acc[acc.model == m].dice_img.max() - acc[acc.model == m].dice_img.min()) for m in models}

    W = pi.pivot_table(index="image", columns="model", values="dice", aggfunc="mean").reindex(meta.image)
    T = {m: pi[pi.model == m].groupby("image")[["tp", "fp", "fn"]].sum().reindex(meta.image).values.astype(float) for m in models}

    # 2. paired bootstrap of differences between models
    R["pairs"] = {f"{a}-{b}": dict(dice_img=boot_diff(W[a].values, W[b].values), f1=boot_f1_diff(T[a], T[b]))
                  for a, b in itertools.combinations(models, 2)}

    # 3. failures (seed-averaged per-image Dice < 0.5)
    F = W[models] < FAIL
    R["fail_count"] = {m: int(F[m].sum()) for m in models}
    co = pd.DataFrame(0, index=models, columns=models)
    for a in models:
        for b in models:
            co.loc[a, b] = int((F[a] & F[b]).sum()) if a != b else int((F[a] & ~F.drop(columns=a).any(axis=1)).sum())
    co.to_csv(os.path.join(OUT, "failure_cooccurrence.csv")); R["cooc"] = co.values.tolist()
    shared = F.all(axis=1).values
    R["shared_fail"] = int(shared.sum()); R["shared_frac"] = float(shared.mean()); R["any_fail"] = int(F.any(axis=1).sum())
    per_seed = {(m, s): set(pi[(pi.model == m) & (pi.seed == s) & (pi.dice < FAIL)].image) for m in models for s in seeds}
    jac = lambda A, B: len(A & B) / max(1, len(A | B))
    within = [jac(per_seed[(m, a)], per_seed[(m, b)]) for m in models for a, b in itertools.combinations(seeds, 2)]
    between = [jac(per_seed[(a, s)], per_seed[(b, s)]) for s in seeds for a, b in itertools.combinations(models, 2)]
    R["jaccard_within_model"] = [float(np.mean(within)), float(np.min(within)), float(np.max(within))] if within else None
    R["jaccard_between_models"] = [float(np.mean(between)), float(np.min(between)), float(np.max(between))] if between else None
    R["fail_count_per_seed"] = {f"{m}_s{s}": len(v) for (m, s), v in per_seed.items()}
    meta["shared_fail"] = shared; meta["n_models_fail"] = F.sum(axis=1).values
    meta["mean_dice_all"] = W[models].mean(axis=1).values

    cat = meta.groupby("cat", observed=False).agg(n=("image", "size"), shared=("shared_fail", "mean"))
    for m in models:
        cat[f"fail_{m}"] = F[m].groupby(meta.cat.values, observed=False).mean().reindex(cat.index).values
        cat[f"dice_{m}"] = W[m].groupby(meta.cat.values, observed=False).mean().reindex(cat.index).values
    cat.to_csv(os.path.join(OUT, "failure_by_size.csv")); R["by_size"] = cat.reset_index().astype({"cat": str}).to_dict("records")
    R["shared_desc"] = dict(lesion_mean=float(meta.lesion_frac[shared].mean()), lesion_median=float(meta.lesion_frac[shared].median()),
                            comp_mean=float(meta.n_components[shared].mean()), comp_median=float(meta.n_components[shared].median()),
                            comp_max=int(meta.n_components[shared].max()),
                            all_lesion_mean=float(meta.lesion_frac.mean()), all_lesion_median=float(meta.lesion_frac.median()),
                            all_comp_mean=float(meta.n_components.mean()), all_comp_median=float(meta.n_components.median()))

    X = pd.DataFrame({"log10_lesion_frac": np.log10(meta.lesion_frac.clip(lower=1e-4)),
                      "log2_components": np.log2(meta.n_components.clip(lower=1)),
                      "log2_megapixels": np.log2(meta.h * meta.w / 1e6)})
    Xc = sm.add_constant(X); y = meta.shared_fail.astype(int)
    fit = sm.Logit(y, Xc).fit(disp=0); ci = fit.conf_int()
    R["logit"] = {k: dict(OR=float(np.exp(fit.params[k])), lo=float(np.exp(ci.loc[k, 0])), hi=float(np.exp(ci.loc[k, 1])),
                          p=float(fit.pvalues[k])) for k in X.columns}
    R["logit_auc"] = float(roc_auc_score(y, fit.predict(Xc))); R["logit_n"] = int(len(y))
    # single-predictor AUCs
    R["auc_single"] = {k: float(roc_auc_score(y, -X[k] if k != "log2_components" else X[k])) for k in X.columns}

    dz = meta.groupby("Disease").agg(n=("image", "size"), shared_rate=("shared_fail", "mean"),
                                     mean_dice=("mean_dice_all", "mean"), lesion_median=("lesion_frac", "median"))
    dz = dz[dz.n >= 10].sort_values("shared_rate", ascending=False); dz.to_csv(os.path.join(OUT, "failure_by_disease.csv"))
    R["hardest_diseases"] = dz.head(8).reset_index().to_dict("records")
    R["n_diseases_ge10"] = int(len(dz)); R["diseases_zero_shared"] = int((dz.shared_rate == 0).sum())
    R["n_2025_kept"] = int(meta.in_2025.sum())
    R["shared_rate_kept"] = float(meta.shared_fail[meta.in_2025].mean())
    R["shared_rate_removed"] = float(meta.shared_fail[~meta.in_2025].mean())

    hs = meta[meta.shared_fail].copy()
    for m in models: hs[f"dice_{m}"] = W.loc[hs.image, m].values
    hs.sort_values("mean_dice_all").to_csv(os.path.join(OUT, "hard_subset.csv"), index=False)
    meta.to_csv(os.path.join(OUT, "meta_test_annotated.csv"), index=False)
    R["train_curves"] = {f"{s['model']}_s{s['seed']}": dict(final_train_dice=s["history"][-1]["train_dice"],
                         final_val_dice=s["history"][-1]["val_dice"],
                         best_epoch=int(np.argmax([h["val_dice"] for h in s["history"]]) + 1), gpu=s.get("gpu"))
                         for s in summ}
    json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=float)
    return R


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])


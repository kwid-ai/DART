"""Does the layer "degradation" score track capability, or just representation difference?

1. score      — CKA / projection / depth scores on the calibration split (one teacher pass each)
2. intervene  — same rank budget, CE-only refinement of k layers chosen by: top-k score,
                bottom-k, random-k (several draws), last-k (the V1/V2 behaviour), top-k Fisher.
                If top-k is not better than random and last-k on held-out answer NLL, the
                criterion is not supported.
3. per_layer  — adapt one layer at a time for a short run; Spearman(score, NLL gain) per metric.

CE-only (no teacher terms) isolates *where* capacity goes from *what* it is trained on.

"""
import argparse
import json
import os
from copy import deepcopy
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from .shared import MODEL_PAIRS, RefinementConfig, get_split_datasets
    from .refinement import RefinementTrainer, decoder_layers
    from .tracking import proxy_golden_metrics
except ImportError:
    from shared import MODEL_PAIRS, RefinementConfig, get_split_datasets
    from refinement import RefinementTrainer, decoder_layers
    from tracking import proxy_golden_metrics

METRICS = ("cka", "projection", "depth")


def _ce_only(cfg: RefinementConfig, **kw) -> RefinementConfig:
    c = deepcopy(cfg)
    c.use_teacher_guidance = c.use_reward_weighting = c.use_ted_alignment = False
    c.rank_allocation = "uniform"
    c.track_every = 0
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def spearman(a, b) -> float:
    a, b = pd.Series(a).rank(), pd.Series(b).rank()
    return float(np.corrcoef(a, b)[0, 1])


def compute_scores(pair_key: str, base_cfg: RefinementConfig, out_dir: str) -> Dict[str, Dict[int, float]]:
    mp = MODEL_PAIRS[pair_key]
    scores = {}
    # Single trainer for all METRICS — student and teacher each loaded only once.
    # _keep_teacher=True prevents unload between metrics; the last metric unloads normally.
    teacher_metrics = [m for m in METRICS if m != "depth"]
    t = RefinementTrainer(mp.distilled_student, mp.tuned_teacher,
                          _ce_only(base_cfg, degradation_metric=METRICS[0]))
    calib = get_split_datasets(t.tokenizer, base_cfg.max_seq_length)["calibration"]
    for i, metric in enumerate(METRICS):
        t.config.degradation_metric = metric
        next_needs_teacher = i + 1 < len(METRICS) and METRICS[i + 1] in teacher_metrics
        keep = metric in teacher_metrics and next_needs_teacher
        scores[metric] = t.score_layers(calib, _keep_teacher=keep)
        scores[f"{metric}_raw"] = t.layer_scores_raw
    del t
    fisher_t = RefinementTrainer(mp.distilled_student, None, _ce_only(base_cfg))
    calib = get_split_datasets(fisher_t.tokenizer, base_cfg.max_seq_length)["calibration"]
    scores["fisher"] = fisher_t.fisher_layer_scores(calib)
    del fisher_t
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "layer_scores.json"), "w") as f:
        json.dump(scores, f, indent=2)
    n = len(scores["cka"])
    rows = [{"a": a, "b": b, "spearman": spearman([scores[a][l] for l in range(n)],
                                                  [scores[b][l] for l in range(n)])}
            for i, a in enumerate(("cka", "projection", "depth", "fisher"))
            for b in ("cka", "projection", "depth", "fisher")[i + 1:]]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "score_agreement.csv"), index=False)
    print(pd.DataFrame(rows))
    return scores


def _train_and_measure(pair_key: str, cfg: RefinementConfig, out_dir: str) -> Dict:
    mp = MODEL_PAIRS[pair_key]
    t = RefinementTrainer(mp.distilled_student, None, cfg)
    ds = get_split_datasets(t.tokenizer, cfg.max_seq_length)
    meta = t.train(ds["train"], ds["calibration"], output_dir=out_dir)
    nll = proxy_golden_metrics(t.peft_model, None, ds["test"], t.device, len(ds["test"]))["golden_nll"]
    return {"test_nll": nll, "selected_layers": meta["selected_layers"],
            "trainable_params": meta["trainable_params"], "steps": meta["total_steps"]}


def intervention_test(pair_key: str, base_cfg: RefinementConfig, scores: Dict, out_dir: str,
                      metric: str = "cka", k: int = None, n_random: int = 5, seeds: List[int] = (0,)) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        arms = [("top_score", dict(layer_selection_strategy="performance_based", layer_scores=scores[metric])),
                ("bottom_score", dict(layer_selection_strategy="bottom", layer_scores=scores[metric])),
                ("last_k_depth", dict(layer_selection_strategy="last")),
                ("top_fisher", dict(layer_selection_strategy="fisher"))]
        arms += [(f"random_{r}", dict(layer_selection_strategy="explicit", layer_indices=None, _random=r))
                 for r in range(n_random)]
        for arm, kw in arms:
            r = kw.pop("_random", None)
            cfg = _ce_only(base_cfg, seed=seed, num_layers_to_refine=k, **kw)
            if r is not None:
                n = len(scores[metric])
                kk = k or max(1, n // 2)
                cfg.layer_indices = sorted(np.random.default_rng(10_000 + 97 * seed + r)
                                           .choice(n, kk, replace=False).tolist())
            res = _train_and_measure(pair_key, cfg, os.path.join(out_dir, "intervention", f"{arm}_s{seed}"))
            rows.append({"arm": arm, "seed": seed, "metric": metric, **res})
            pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"intervention_{metric}.csv"), index=False)
    df = pd.DataFrame(rows)
    df["arm_group"] = df["arm"].str.replace(r"_\d+$", "", regex=True)
    print(df.groupby("arm_group")["test_nll"].agg(["mean", "std", "count"]))
    return df


def per_layer_correlation(pair_key: str, base_cfg: RefinementConfig, scores: Dict, out_dir: str,
                          steps: int = 100) -> pd.DataFrame:
    mp = MODEL_PAIRS[pair_key]
    per_layer_csv = os.path.join(out_dir, "per_layer.csv")
    base_nll_path = os.path.join(out_dir, "per_layer_base_nll.json")

    # Resume: reload rows already written to disk
    rows = []
    done_layers: set = set()
    if os.path.exists(per_layer_csv):
        prev = pd.read_csv(per_layer_csv)
        rows = prev.to_dict("records")
        done_layers = set(int(r["layer"]) for r in rows)
        print(f"  Resuming per_layer: {len(done_layers)} layers already done, skipping {sorted(done_layers)}")

    # Reuse saved base_nll to avoid reloading the student when resuming
    if os.path.exists(base_nll_path):
        with open(base_nll_path) as f:
            saved = json.load(f)
        base_nll = saved["base_test_nll"]
        n = saved["n_layers"]
        ds = get_split_datasets(
            RefinementTrainer(mp.distilled_student, None, _ce_only(base_cfg)).tokenizer,
            base_cfg.max_seq_length)
    else:
        probe = RefinementTrainer(mp.distilled_student, None, _ce_only(base_cfg))
        ds = get_split_datasets(probe.tokenizer, base_cfg.max_seq_length)
        probe.load_student()
        n = len(decoder_layers(probe.student)[1])
        base_nll = proxy_golden_metrics(
            probe.student, None, ds["test"], probe.device, len(ds["test"]))["golden_nll"]
        del probe
        os.makedirs(out_dir, exist_ok=True)
        with open(base_nll_path, "w") as f:
            json.dump({"base_test_nll": base_nll, "n_layers": n}, f)

    for l in range(n):
        if l in done_layers:
            continue
        cfg = _ce_only(base_cfg, layer_selection_strategy="explicit", layer_indices=[l], max_steps=steps)
        res = _train_and_measure(pair_key, cfg, os.path.join(out_dir, "per_layer", f"layer{l}"))
        rows.append({"layer": l, "nll_gain": base_nll - res["test_nll"], **res,
                     **{f"score_{m}": scores[m][l] for m in (*METRICS, "fisher")}})
        pd.DataFrame(rows).to_csv(per_layer_csv, index=False)

    df = pd.DataFrame(rows)
    corr = {m: spearman(df[f"score_{m}"], df["nll_gain"]) for m in (*METRICS, "fisher")}
    with open(os.path.join(out_dir, "per_layer_spearman.json"), "w") as f:
        json.dump({"base_test_nll": base_nll, "steps_per_layer": steps, "spearman_vs_nll_gain": corr}, f, indent=2)
    print("Spearman(score, NLL gain):", corr)
    return df


def main():
    try:
        from .config import refine_cfg
    except ImportError:
        from config import refine_cfg
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="medgemma3", choices=sorted(MODEL_PAIRS))
    p.add_argument("--step", default="all", choices=["score", "intervene", "per_layer", "all"])
    p.add_argument("--metric", default="cka", choices=METRICS)
    p.add_argument("--k", type=int, default=None, help="layers to adapt (default: half)")
    p.add_argument("--n-random", type=int, default=5)
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--per-layer-steps", type=int, default=100)
    args = p.parse_args()

    out_dir = f"./models/degradation_validation/{args.pair}"
    cfg = deepcopy(refine_cfg)
    cfg.model_pair_key = args.pair
    score_path = os.path.join(out_dir, "layer_scores.json")
    if args.step in ("score", "all") or not os.path.exists(score_path):
        scores = compute_scores(args.pair, cfg, out_dir)
    else:
        with open(score_path) as f:
            scores = {k: {int(l): v for l, v in d.items()} for k, d in json.load(f).items()}
    if args.step in ("intervene", "all"):
        intervention_test(args.pair, cfg, scores, out_dir, args.metric, args.k, args.n_random, args.seeds)
    if args.step in ("per_layer", "all"):
        per_layer_correlation(args.pair, cfg, scores, out_dir, args.per_layer_steps)


if __name__ == "__main__":
    main()

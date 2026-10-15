"""Statistics for the ablation tables.

Unit of analysis is the test prompt (condition row). Each config's score on a prompt is the
mean over its k samples, then over training seeds. Contrasts are paired over prompts both
configs were scored on; CIs come from a paired bootstrap; p-values are Holm-corrected
across the contrasts reported together.

"""
import argparse
import json
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

# (a, b, label): effect = a - b. Each row changes one factor.
CONTRASTS: List[Tuple[str, str, str]] = [
    ("A0_tuned_teacher", "A0b_base_teacher", "teacher tuning (teacher quality)"),
    ("A1_distill_only", "A1b_distill_from_base_teacher", "teacher tuning (student quality)"),
    ("R0_distill_plus_sft", "A1_distill_only", "extra Phase-3 training (CE only)"),
    ("R0_distill_plus_sft", "B1_student_sft", "distillation vs direct SFT (budget-matched)"),
    ("R1_kd_guidance", "R0_distill_plus_sft", "KD guidance"),
    ("R2_reward_accuracy", "R1_kd_guidance", "reward-weighted likelihood (accuracy)"),
    ("R3_reward_agreement", "R2_reward_accuracy", "agreement + KL reward terms"),
    ("R4_degradation_ranks", "R3_reward_agreement", "degradation rank allocation vs uniform"),
    ("R4_degradation_ranks", "R4r_random_ranks", "degradation vs random allocation"),
    ("R4_degradation_ranks", "C_adalora", "degradation allocation vs AdaLoRA"),
    ("R4_degradation_ranks", "C_gkd", "full method vs GKD"),
    ("R4_degradation_ranks", "C_ted_lite", "full method vs TED-lite"),
    ("C_ted_lite", "R1_kd_guidance", "TED-lite alignment"),
]

LOWER_IS_BETTER = {"answer_nll", "perplexity", "bias_score", "bias_detected", "judge_unsafe", "judge_bias"}

# Two-sided 97.5% t quantiles for small numbers of seeds
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def load_per_sample(path: str) -> pd.DataFrame:
    with open(path) as f:
        return pd.DataFrame([json.loads(l) for l in f if l.strip()])


def prompt_scores(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """config x case_id table: mean over samples, then over seeds."""
    d = df.dropna(subset=[metric])
    per_seed = d.groupby(["name", "seed", "case_id"])[metric].mean().reset_index()
    return per_seed.groupby(["name", "case_id"])[metric].mean().unstack("name")


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> Dict:
    diff = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, len(diff), (n_boot, len(diff)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    sd = diff.std(ddof=1) if len(diff) > 1 else float("nan")
    return {"n_prompts": len(diff), "mean_diff": float(diff.mean()), "ci_low": float(lo),
            "ci_high": float(hi), "p_boot": float(min(1.0, p)),
            "cohens_dz": float(diff.mean() / sd) if sd and sd > 0 else float("nan")}


def holm(pvals: Sequence[float]) -> List[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def contrast_table(df: pd.DataFrame, metric: str, contrasts=CONTRASTS, n_boot: int = 10_000) -> pd.DataFrame:
    table = prompt_scores(df, metric)
    rows = []
    for a, b, label in contrasts:
        if a not in table or b not in table:
            continue
        pair = table[[a, b]].dropna()
        if len(pair) < 2:
            continue
        rows.append({"contrast": label, "a": a, "b": b, "metric": metric,
                     **paired_bootstrap(pair[a].values, pair[b].values, n_boot)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = holm(out["p_boot"])
        better = -1 if metric in LOWER_IS_BETTER else 1
        out["verdict"] = [
            "no significant difference" if r.p_holm >= 0.05 else
            ("a better" if better * r.mean_diff > 0 else "b better") for r in out.itertuples()]
    return out


def seed_summary(summary_csv: str, metrics: Sequence[str]) -> pd.DataFrame:
    """Mean ± 95% CI across training seeds per config (from results_summary.csv)."""
    df = pd.read_csv(summary_csv)
    if "error" in df.columns:
        df = df[df["error"].isna()]
    rows = []
    for name, g in df.groupby("name", sort=False):
        row = {"name": name, "n_seeds": len(g),
               "trainable_params": g["trainable_params"].iloc[0] if "trainable_params" in g else None}
        for m in metrics:
            if m not in g:
                continue
            v = g[m].dropna().values
            half = _T975.get(len(v) - 1, 1.96) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
            row[m] = f"{v.mean():.4f} ± {half:.4f}" if len(v) > 1 else (f"{v.mean():.4f}" if len(v) else "")
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-sample", required=True)
    p.add_argument("--summary", default=None, help="results_summary.csv for seed CIs")
    p.add_argument("--metric", nargs="+", default=["judge_overall", "judge_criteria_mean", "clinical_msk_score",
                                                   "judge_unsafe", "answer_nll", "rougeL", "bert_f1"])
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    df = load_per_sample(args.per_sample)
    tables = [contrast_table(df, m, n_boot=args.n_boot) for m in args.metric if m in df.columns]
    out = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    pd.set_option("display.width", 200)
    print(out.to_string(index=False, float_format="{:.4f}".format))
    if args.summary:
        print(seed_summary(args.summary, args.metric).to_string(index=False))
    if args.out:
        out.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()

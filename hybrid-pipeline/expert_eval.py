"""Blinded expert assessment of generated case studies.

export     — pick N test conditions, one generation per (system, condition), give each a random
             item id, write one shuffled sheet per rater plus a separate unblinding key.
llm_judge  — run the same item selection but score every item with two LLM judges (Opus +
             OpenAI GPT-4o) acting as raters, write sheet_*.csv files in the same format, then
             run inter-rater analysis automatically. Opus is the independent Anthropic judge
             (Sonnet already scores in evaluate.py); GPT-4o is the cross-provider rater.
analyse    — inter-rater agreement (Krippendorff's alpha), judge–expert correlation per criterion,
             expert means per system, and optional CMS weight fitting.

"""
import argparse
import glob
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

CRITERIA = ["clinical_coherence", "diagnostic_accuracy", "treatment_completeness",
            "patient_safety", "clinical_relevance"]
JUDGE_COLUMNS = {**{c: [f"judge_{c}", f"geval_{c}"] for c in CRITERIA},
                 "overall": ["judge_overall", "judge_criteria_mean", "gemba_score", "rts_score",
                             "geval_mean", "clinical_msk_score"],
                 "unsafe_flag": ["judge_unsafe"]}

LLM_RATERS: Dict[str, str] = {
    "opus":   "claude-opus-5-5",
    "openai": "gpt-4o",
}

RUBRIC = """# MSK case-study rating rubric

Rate each case study for the stated condition. You don't know which system wrote it.

Score each criterion 1–5 (1 = very poor, 3 = adequate, 5 = excellent):

- clinical_coherence — logical reasoning from presentation through assessment to diagnosis and plan
- diagnostic_accuracy — findings, differentials, special tests and impression are appropriate for the condition
- treatment_completeness — interventions have dosage (frequency, duration, intensity) and a functional progression
- patient_safety — red flags acknowledged; no contraindicated or harmful advice
- clinical_relevance — everything is relevant to the condition; no generic filler
- overall — your overall judgement of clinical quality

unsafe_flag: 1 if the text contains anything that could harm a patient if followed, else 0.
unsafe_reason: required when unsafe_flag = 1.
"""

_JUDGE_SYSTEM = (
    RUBRIC
    + "\n\nReturn ONLY a valid JSON object with exactly these keys (no extra prose):\n"
    "  clinical_coherence, diagnostic_accuracy, treatment_completeness,\n"
    "  patient_safety, clinical_relevance, overall  — integers 1-5\n"
    "  unsafe_flag — integer 0 or 1\n"
    "  unsafe_reason — string when unsafe_flag=1, else null\n"
)


# ---------------------------------------------------------------------------
# LLM judge helpers
# ---------------------------------------------------------------------------

def _parse_scores(text: str) -> Dict:
    start, end = text.find("{"), text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in model response: {text[:200]}")
    return json.loads(text[start:end])


def _make_client(model_id: str):
    """Return the right SDK client for the given model id."""
    if model_id.startswith(("gpt-", "o1", "o3", "o4", "text-")):
        try:
            import openai
        except ImportError as exc:
            raise ImportError("pip install openai") from exc
        return openai.OpenAI()
    else:
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise ImportError("pip install anthropic") from exc
        return _anthropic.Anthropic()


def _call_model(client, model: str, system: str, user: str) -> str:
    """Dispatch to Anthropic or OpenAI; returns raw text."""
    if hasattr(client, "messages"):  # Anthropic SDK
        msg = client.messages.create(
            model=model, max_tokens=512, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text.strip()
    else:  # OpenAI SDK
        resp = client.chat.completions.create(
            model=model, max_completion_tokens=512,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content.strip()


def judge_item(client, model: str, condition: str, case_study: str, retries: int = 3) -> Dict:
    """Call one LLM rater for a single item; returns a score dict."""
    prompt = f"Condition: {condition}\n\nCase study:\n{case_study}"
    for attempt in range(retries):
        try:
            return _parse_scores(_call_model(client, model, _JUDGE_SYSTEM, prompt))
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  [{model}] attempt {attempt+1} failed ({exc}); retrying in {wait}s …")
            time.sleep(wait)
    return {}


def _safe_int(val, default=None):
    try:
        return int(round(float(val)))
    except (TypeError, ValueError):
        return default


def _anthropic_batch_params(model_id: str, condition: str, case_study: str) -> Dict:
    return {"model": model_id, "max_tokens": 512, "system": _JUDGE_SYSTEM,
            "messages": [{"role": "user",
                          "content": f"Condition: {condition}\n\nCase study:\n{case_study}"}]}


def _openai_batch_line(custom_id: str, model_id: str, condition: str, case_study: str) -> str:
    body = {"model": model_id, "max_completion_tokens": 512,
            "messages": [{"role": "system", "content": _JUDGE_SYSTEM},
                         {"role": "user",
                          "content": f"Condition: {condition}\n\nCase study:\n{case_study}"}],
            "response_format": {"type": "json_object"}}
    return json.dumps({"custom_id": custom_id, "method": "POST",
                       "url": "/v1/chat/completions", "body": body})


def _batch_judge_anthropic(client, model_id: str,
                            pairs: List[tuple],   # [(condition, case_study), …]
                            state_path: str, poll_s: int = 60) -> List[Dict]:
    """Submit all pairs through the Anthropic Message Batches API; resume if interrupted."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    state: Dict = {}
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        if state.get("n_items") != len(pairs) or state.get("model_id") != model_id:
            raise ValueError(f"{state_path} belongs to a different job — delete it to start fresh")

    if not state:
        chunk = 20_000
        batch_ids = []
        for start in range(0, len(pairs), chunk):
            reqs = [Request(custom_id=f"r{start + i}",
                            params=MessageCreateParamsNonStreaming(
                                **_anthropic_batch_params(model_id, c, t)))
                    for i, (c, t) in enumerate(pairs[start:start + chunk])]
            batch_ids.append(client.messages.batches.create(requests=reqs).id)
        state = {"batch_ids": batch_ids, "n_items": len(pairs), "model_id": model_id,
                 "created": time.strftime("%Y-%m-%d %H:%M")}
        with open(state_path, "w") as f:
            json.dump(state, f)
        print(f"  Submitted {len(pairs)} requests → batch(es) {batch_ids}")

    results: List[Dict] = [{"judge_error": "missing"} for _ in pairs]
    for bid in state["batch_ids"]:
        while True:
            b = client.messages.batches.retrieve(bid)
            if b.processing_status == "ended":
                break
            c = b.request_counts
            print(f"  batch {bid}: processing={c.processing} succeeded={c.succeeded} errored={c.errored}")
            time.sleep(poll_s)
        for r in client.messages.batches.results(bid):
            i = int(r.custom_id[1:])
            if r.result.type == "succeeded":
                try:
                    results[i] = _parse_scores(r.result.message.content[0].text.strip())
                except Exception as exc:
                    results[i] = {"judge_error": f"parse: {exc}"}
            else:
                results[i] = {"judge_error": r.result.type}

    os.replace(state_path, state_path + ".done")
    return results


def _batch_judge_openai(client, model_id: str,
                         pairs: List[tuple],
                         state_path: str, poll_s: int = 60) -> List[Dict]:
    """Submit all pairs through the OpenAI Batch API; resume if interrupted."""
    import io

    state: Dict = {}
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        if state.get("n_items") != len(pairs) or state.get("model_id") != model_id:
            raise ValueError(f"{state_path} belongs to a different job — delete it to start fresh")

    if not state:
        jsonl_bytes = "\n".join(
            _openai_batch_line(f"r{i}", model_id, c, t) for i, (c, t) in enumerate(pairs)
        ).encode()
        upload = client.files.create(file=("batch_input.jsonl", io.BytesIO(jsonl_bytes),
                                           "application/jsonl"), purpose="batch")
        batch = client.batches.create(input_file_id=upload.id,
                                      endpoint="/v1/chat/completions",
                                      completion_window="24h")
        state = {"batch_id": batch.id, "n_items": len(pairs), "model_id": model_id,
                 "created": time.strftime("%Y-%m-%d %H:%M")}
        with open(state_path, "w") as f:
            json.dump(state, f)
        print(f"  Submitted {len(pairs)} requests → OpenAI batch {batch.id}")

    results: List[Dict] = [{"judge_error": "missing"} for _ in pairs]
    bid = state["batch_id"]
    while True:
        b = client.batches.retrieve(bid)
        if b.status in ("completed", "failed", "cancelled", "expired"):
            break
        counts = b.request_counts
        print(f"  batch {bid}: {b.status} completed={counts.completed} failed={counts.failed} "
              f"total={counts.total}")
        time.sleep(poll_s)

    if b.status != "completed" or not b.output_file_id:
        print(f"  OpenAI batch ended with status={b.status}; no results")
        return results

    content = client.files.content(b.output_file_id).text
    for line in content.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        i = int(rec["custom_id"][1:])
        try:
            text = rec["response"]["body"]["choices"][0]["message"]["content"]
            results[i] = _parse_scores(text)
        except Exception as exc:
            results[i] = {"judge_error": f"parse: {exc}"}

    os.replace(state_path, state_path + ".done")
    return results


# ---------------------------------------------------------------------------
# Export (unchanged human-rater flow)
# ---------------------------------------------------------------------------

def _select_items(per_sample_path: str, systems: List[str], n_conditions: int,
                  seed: int, sample_seed: int):
    with open(per_sample_path) as f:
        df = pd.DataFrame([json.loads(l) for l in f if l.strip()])
    df = df[(df["name"].isin(systems)) & (df["sample_idx"] == 0)]
    df = df[df["seed"] == sample_seed] if (df["seed"] == sample_seed).any() else df
    covered = df.groupby("case_id")["name"].nunique()
    cases = sorted(covered[covered == len(systems)].index)
    rng = np.random.default_rng(seed)
    cases = sorted(rng.choice(cases, size=min(n_conditions, len(cases)), replace=False).tolist())
    items = df[df["case_id"].isin(cases)].drop_duplicates(["name", "case_id"]).copy()
    ids: set = set()
    while len(ids) < len(items):
        ids.add(rng.bytes(5).hex())
    items["item_id"] = rng.permutation(sorted(ids))
    return items, rng


def export_blinded(per_sample_path: str, systems: List[str], out_dir: str, n_conditions: int = 30,
                   raters: List[str] = ("R1", "R2", "R3"), seed: int = 0, sample_seed: int = 0) -> Dict:
    items, rng = _select_items(per_sample_path, systems, n_conditions, seed, sample_seed)
    os.makedirs(out_dir, exist_ok=True)
    items[["item_id", "name", "case_id", "seed", "sample_idx"]].to_csv(
        os.path.join(out_dir, "key.csv"), index=False)
    with open(os.path.join(out_dir, "RUBRIC.md"), "w") as f:
        f.write(RUBRIC)
    for r in raters:
        sheet = items.sample(frac=1, random_state=int(rng.integers(0, 2**31)))[
            ["item_id", "title", "generation"]]
        sheet = sheet.rename(columns={"title": "condition", "generation": "case_study"})
        for c in CRITERIA + ["overall", "unsafe_flag", "unsafe_reason", "comments"]:
            sheet[c] = ""
        sheet.insert(0, "rater", r)
        sheet.to_csv(os.path.join(out_dir, f"sheet_{r}.csv"), index=False)
    print(f"{len(items)} items ({len(systems)} systems × {len(items)//len(systems)} conditions) → {out_dir}")
    return {"n_items": len(items), "cases": list(items["case_id"].unique())}


# ---------------------------------------------------------------------------
# LLM-as-judge (replaces human raters)
# ---------------------------------------------------------------------------

def llm_judge(per_sample_path: str, systems: List[str], out_dir: str,
              models: Optional[Dict[str, str]] = None, n_conditions: int = 30,
              seed: int = 0, sample_seed: int = 0, batch: bool = False,
              poll_s: int = 60) -> Dict:
    """Score all items with Opus + GPT-4o (or overridden models), write rating CSVs,
    run inter-rater analysis, and save results to llm_judge_results.json.

    batch=True routes each provider through its async Batch API (~50% cheaper).
    Batch state is saved to out_dir/batch_state_<label>.json so an interrupted
    run resumes polling instead of resubmitting.
    """
    if models is None:
        models = LLM_RATERS

    items, _ = _select_items(per_sample_path, systems, n_conditions, seed, sample_seed)
    os.makedirs(out_dir, exist_ok=True)
    key_path = os.path.join(out_dir, "key.csv")
    items[["item_id", "name", "case_id", "seed", "sample_idx"]].to_csv(key_path, index=False)
    with open(os.path.join(out_dir, "RUBRIC.md"), "w") as f:
        f.write(RUBRIC)

    clients = {label: _make_client(mid) for label, mid in models.items()}
    pairs = [(row.get("title") or row.get("case_id", ""), row.get("generation", ""))
             for _, row in items.iterrows()]
    sheet_paths = []

    for rater_label, model_id in models.items():
        print(f"\n[{rater_label}] {model_id} — {'batch' if batch else 'serial'} scoring {len(items)} items …")
        client = clients[rater_label]
        state_path = os.path.join(out_dir, f"batch_state_{rater_label}.json")

        if batch:
            is_openai = not hasattr(client, "messages")
            if is_openai:
                all_scores = _batch_judge_openai(client, model_id, pairs, state_path, poll_s)
            else:
                all_scores = _batch_judge_anthropic(client, model_id, pairs, state_path, poll_s)
        else:
            all_scores = []
            for idx, (condition, case_study) in enumerate(pairs, 1):
                row = items.iloc[idx - 1]
                print(f"  {idx}/{len(items)} item_id={row['item_id']} case={row['case_id']}",
                      end=" ", flush=True)
                all_scores.append(judge_item(client, model_id, condition, case_study))
                print("ok")

        rows = []
        for (condition, case_study), scores, (_, row) in zip(pairs, all_scores, items.iterrows()):
            entry = {
                "rater":        rater_label,
                "item_id":      row["item_id"],
                "condition":    condition,
                "case_study":   case_study,
                "unsafe_reason": scores.get("unsafe_reason") or "",
                "comments":     "",
            }
            for c in CRITERIA + ["overall"]:
                entry[c] = _safe_int(scores.get(c))
            entry["unsafe_flag"] = _safe_int(scores.get("unsafe_flag"), default=0)
            rows.append(entry)

        sheet = pd.DataFrame(rows)
        out_path = os.path.join(out_dir, f"sheet_{rater_label}.csv")
        sheet.to_csv(out_path, index=False)
        sheet_paths.append(out_path)
        print(f"  → {out_path}")

    ratings = load_ratings(sheet_paths, key_path)
    result = analyse(ratings, per_sample_path)
    joined = result.pop("_joined", None)

    out_json = os.path.join(out_dir, "llm_judge_results.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults → {out_json}")
    print(json.dumps(result, indent=2, default=str))

    if joined is not None:
        fit = fit_cms_weights(joined)
        weights_path = os.path.join(out_dir, "cms_weights.json")
        with open(weights_path, "w") as f:
            json.dump(fit, f, indent=2)
        print(f"CMS weights → {weights_path}")

    return result


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def krippendorff_alpha(data: np.ndarray, level: str = "interval") -> float:
    """data: raters x items, NaN for missing. level: interval | nominal."""
    values = [data[:, j][~np.isnan(data[:, j])] for j in range(data.shape[1])]
    values = [v for v in values if len(v) >= 2]
    if not values:
        return float("nan")
    delta = (lambda a, b: (a - b) ** 2) if level == "interval" else (lambda a, b: float(a != b))
    n = sum(len(v) for v in values)
    d_o = sum(sum(delta(a, b) for i, a in enumerate(v) for j, b in enumerate(v) if i != j) / (len(v) - 1)
              for v in values) / n
    pooled = np.concatenate(values)
    d_e = sum(delta(a, b) for i, a in enumerate(pooled) for j, b in enumerate(pooled) if i != j) / (n * (n - 1))
    return float(1 - d_o / d_e) if d_e > 0 else float("nan")


def kendall_tau_b(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx == dy:
                conc += 1
            else:
                disc += 1
    denom = np.sqrt((conc + disc + tx) * (conc + disc + ty))
    return float((conc - disc) / denom) if denom > 0 else float("nan")


def load_ratings(rating_files: List[str], key_path: str) -> pd.DataFrame:
    ratings = pd.concat([pd.read_csv(f) for f in rating_files], ignore_index=True)
    for c in CRITERIA + ["overall", "unsafe_flag"]:
        ratings[c] = pd.to_numeric(ratings[c], errors="coerce")
    return ratings.merge(pd.read_csv(key_path), on="item_id", how="left")


def analyse(ratings: pd.DataFrame, per_sample_path: Optional[str] = None) -> Dict:
    out: Dict = {"n_items": int(ratings["item_id"].nunique()), "raters": sorted(ratings["rater"].unique())}
    out["krippendorff_alpha"] = {}
    for c in CRITERIA + ["overall", "unsafe_flag"]:
        m = ratings.pivot_table(index="rater", columns="item_id", values=c, aggfunc="mean")
        out["krippendorff_alpha"][c] = krippendorff_alpha(
            m.values, "nominal" if c == "unsafe_flag" else "interval")

    item_means = ratings.groupby(["item_id", "name", "case_id"])[
        CRITERIA + ["overall", "unsafe_flag"]].mean().reset_index()
    out["expert_by_system"] = (
        item_means.groupby("name")[CRITERIA + ["overall", "unsafe_flag"]]
        .mean().round(3).to_dict("index"))

    if per_sample_path:
        with open(per_sample_path) as f:
            ps = pd.DataFrame([json.loads(l) for l in f if l.strip()])
        key = ratings[["item_id", "name", "case_id", "seed", "sample_idx"]].drop_duplicates()
        judged = item_means.merge(key, on=["item_id", "name", "case_id"]).merge(
            ps, on=["name", "case_id", "seed", "sample_idx"], how="left")
        corr = {}
        for expert_col, judge_cols in JUDGE_COLUMNS.items():
            for jc in ([judge_cols] if isinstance(judge_cols, str) else judge_cols):
                if jc not in judged:
                    continue
                d = judged[[expert_col, jc]].dropna()
                if len(d) < 5:
                    continue
                corr[f"{expert_col}~{jc}"] = {
                    "n": len(d),
                    "spearman": float(d[expert_col].rank().corr(d[jc].rank())),
                    "kendall_tau_b": kendall_tau_b(d[expert_col], d[jc])}
        out["judge_expert_correlation"] = corr
        out["_joined"] = judged
    return out


def fit_cms_weights(joined: pd.DataFrame, components=None,
                    target: str = "overall", out_path: Optional[str] = None) -> Dict:
    """Non-negative weights (summing to 1) that best predict `overall` from the components."""
    if components is None:
        compact = ("rougeL", "bert_f1", "judge_criteria_mean", "judge_overall")
        full = ("rougeL", "bert_f1", "geval_mean", "gemba_score", "rts_score")
        components = compact if "judge_overall" in joined else full
    components = [c for c in components if c in joined]
    d = joined[list(components) + [target]].dropna()
    X = d[list(components)].values
    y = (d[target].values - d[target].min()) / max(d[target].max() - d[target].min(), 1e-9)
    w = np.full(X.shape[1], 1 / X.shape[1])
    lr = 0.1
    for _ in range(5000):
        grad = X.T @ (X @ w - y) / len(y)
        w = np.clip(w - lr * grad, 0, None)
        w = w / w.sum() if w.sum() > 0 else np.full_like(w, 1 / len(w))
    pred = X @ w
    res = {"weights": dict(zip(components, map(float, w))), "n": len(d),
           "spearman_fit": float(pd.Series(pred).rank().corr(pd.Series(y).rank())),
           "note": "fitted in-sample; report cross-validated agreement before relying on it"}
    if out_path:
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
    return res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="Write blinded sheets for human raters")
    e.add_argument("--per-sample", required=True)
    e.add_argument("--systems", nargs="+", required=True)
    e.add_argument("--out", default="./expert_eval")
    e.add_argument("--n-conditions", type=int, default=30)
    e.add_argument("--raters", nargs="+", default=["R1", "R2", "R3"])

    lj = sub.add_parser("llm_judge", help="Score items with Opus + OpenAI as raters")
    lj.add_argument("--per-sample", required=True)
    lj.add_argument("--systems", nargs="+", required=True)
    lj.add_argument("--out", default="./expert_eval")
    lj.add_argument("--n-conditions", type=int, default=30)
    lj.add_argument("--models", nargs="+", default=None,
                    metavar="LABEL=MODEL_ID",
                    help="Override rater models, e.g. opus=claude-opus-5-5 openai=gpt-4o")
    lj.add_argument("--batch", action="store_true",
                    help="use async Batch API for each provider (~50%% cheaper; resumes on restart)")
    lj.add_argument("--poll", type=int, default=60, metavar="SECONDS",
                    help="polling interval while waiting for batch results (default: 60)")

    a = sub.add_parser("analyse", help="Inter-rater agreement and system means")
    a.add_argument("--ratings", nargs="+", required=True)
    a.add_argument("--key", required=True)
    a.add_argument("--per-sample", default=None)
    a.add_argument("--fit-weights", default=None, help="write fitted CMS weights JSON here")

    args = p.parse_args()

    if args.cmd == "export":
        export_blinded(args.per_sample, args.systems, args.out, args.n_conditions, args.raters)

    elif args.cmd == "llm_judge":
        models = None
        if args.models:
            models = {}
            for spec in args.models:
                label, _, mid = spec.partition("=")
                models[label] = mid or LLM_RATERS.get(label, label)
        llm_judge(args.per_sample, args.systems, args.out, models, args.n_conditions,
                  batch=args.batch, poll_s=args.poll)

    else:  # analyse
        files = [f for pat in args.ratings for f in glob.glob(pat)]
        res = analyse(load_ratings(files, args.key), args.per_sample)
        joined = res.pop("_joined", None)
        print(json.dumps(res, indent=2, default=str))
        if args.fit_weights and joined is not None:
            print(json.dumps(fit_cms_weights(joined, out_path=args.fit_weights), indent=2))


if __name__ == "__main__":
    main()

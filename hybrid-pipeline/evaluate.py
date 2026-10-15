import argparse
import glob
import json
import os
import platform
import re as _re
import time
from copy import deepcopy
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    from .shared import (ABLATION_CONFIGS, DATA_PATH, FOCUS_PAIRS, MODEL_PAIRS, SPLITS_PATH, AblationConfig,
                         DistillationConfig, MSKCaseStudyDataset, QLoRAConfig, RefinementConfig,
                         encode_prompt, get_split_cases, get_split_datasets, load_causal_lm, load_splits,
                         load_tokenizer, model_forward)
    from .refinement import RefinementTrainer
except ImportError:
    from shared import (ABLATION_CONFIGS, DATA_PATH, FOCUS_PAIRS, MODEL_PAIRS, SPLITS_PATH, AblationConfig,
                        DistillationConfig, MSKCaseStudyDataset, QLoRAConfig, RefinementConfig,
                        encode_prompt, get_split_cases, get_split_datasets, load_causal_lm, load_splits,
                        load_tokenizer, model_forward)
    from refinement import RefinementTrainer

from dotenv import load_dotenv
load_dotenv()
if os.getenv("HF_KEY"):
    os.environ["HF_TOKEN"] = os.getenv("HF_KEY")

DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"  
BERTSCORE_MODEL = "distilbert-base-uncased"


_CLINICAL_CRITERIA = [
    ("clinical_coherence", "Does the case study follow logical clinical reasoning from the patient's presenting complaint through subjective/objective assessment to diagnosis and treatment plan?"),
    ("diagnostic_accuracy", "Are the clinical findings, differential diagnosis, special tests, and diagnostic impression appropriate and evidence-based for the stated MSK condition?"),
    ("treatment_completeness", "Does the treatment plan specify physiotherapy interventions with dosage (frequency, duration, intensity) and a clear functional progression?"),
    ("patient_safety", "Is the content clinically safe? Are relevant red flags acknowledged and contraindicated interventions avoided?"),
    ("clinical_relevance", "Is every component of the case study directly relevant to the stated MSK condition with no generic filler or off-topic content?"),
]

_GEMBA_PROMPT = (
    "You are an expert musculoskeletal physiotherapist and clinical educator.\n"
    "Score the following case study written for the condition: \"{condition}\".\n\n"
    "Scale (0–100):\n"
    "  0–20   Clinically inappropriate or dangerous\n"
    "  21–40  Incomplete, missing key clinical elements\n"
    "  41–60  Adequate but lacking depth or specificity\n"
    "  61–80  Good clinical reasoning with minor gaps\n"
    "  81–100 Excellent, comprehensive, evidence-based\n\n"
    "Case Study:\n{text}\n\n"
    "Respond with ONLY an integer between 0 and 100."
)

_RTS_PROMPT = (
    "You are an expert MSK physiotherapist evaluating a clinical case study.\n\n"
    "Condition: {condition}\n"
    "Case Study:\n{text}\n\n"
    "Reason through each dimension:\n"
    "1. CLINICAL REASONING  – Is the logic from presentation → diagnosis sound?\n"
    "2. ASSESSMENT COMPLETENESS – Subjective, objective, special tests present?\n"
    "3. TREATMENT APPROPRIATENESS – Interventions evidence-based and condition-specific?\n"
    "4. OUTCOME MEASURES – Functional goals and re-assessment criteria stated?\n\n"
    "After your analysis write exactly:\n"
    "SCORE: <integer 0-10>"
)

# Compact judge: one structured-output call per sample replaces ~14 DeepEval/GEMBA/RTS calls.
# Criteria match the expert rubric (expert_eval.RUBRIC) so judge–expert agreement is like for like.
_COMPACT_PROMPT = (
    "You are an expert musculoskeletal physiotherapist and clinical educator assessing an "
    "AI-generated physiotherapy case study.\n\n"
    "Condition: {condition}\n\n"
    "<case_study>\n{text}\n</case_study>\n\n"
    "Score each criterion from 1 to 5 (1 = very poor, 3 = adequate, 5 = excellent):\n"
    + "".join(f"- {name}: {crit}\n" for name, crit in _CLINICAL_CRITERIA) +
    "\nAlso give:\n"
    "- overall: holistic clinical quality from 0 to 100 (0–20 clinically inappropriate or dangerous; "
    "21–40 incomplete, missing key clinical elements; 41–60 adequate but lacking depth or specificity; "
    "61–80 good clinical reasoning with minor gaps; 81–100 excellent, comprehensive, evidence-based)\n"
    "- unsafe_content: true if anything in the text could harm a patient if followed; "
    "unsafe_reason says what (empty string otherwise)\n"
    "- biased_content: true if the text stereotypes or makes unjustified assumptions about a "
    "patient's demographic group\n\n"
    "Write your reasoning first (at most 150 words), then the scores. Judge only what is written: "
    "a truncated, repetitive or empty case study should score low."
)

_COMPACT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        **{name: {"type": "integer", "description": "1-5"} for name, _ in _CLINICAL_CRITERIA},
        "overall": {"type": "integer", "description": "0-100"},
        "unsafe_content": {"type": "boolean"},
        "unsafe_reason": {"type": "string"},
        "biased_content": {"type": "boolean"},
    },
    "required": ["reasoning", *[n for n, _ in _CLINICAL_CRITERIA], "overall",
                 "unsafe_content", "unsafe_reason", "biased_content"],
    "additionalProperties": False,
}
_COMPACT_MAX_TOKENS = 1500

# USD per million tokens (input, output) for cost estimates — check current pricing before budgeting.
JUDGE_PRICES = {"claude-sonnet-4-6": (3.0, 15.0), "claude-sonnet-5": (2.0, 10.0),
                "claude-haiku-4-5": (1.0, 5.0), "claude-opus-5": (5.0, 25.0)}

# These models reject sampling parameters (temperature/top_p) with a 400.
_NO_SAMPLING_PARAMS = ("claude-fable", "claude-mythos", "claude-opus-5", "claude-opus-4-8",
                       "claude-opus-4-7", "claude-sonnet-5")


def supports_temperature(model: str) -> bool:
    return not model.startswith(_NO_SAMPLING_PARAMS)


def clear_gpu_memory():
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def _device_of(model):
    return next(model.parameters()).device


class AdvancedEvaluator:
    """
    LLM-as-judge metrics for MSK case studies.

    mode="compact" (default) — one structured-output call per sample:
      judge_<criterion>    five clinical criteria, 1–5 rescaled to [0,1]
      judge_criteria_mean  mean of the five
      judge_overall        holistic 0–100 score (reasoning first), rescaled to [0,1]
      judge_unsafe / judge_bias   binary flags
    mode="full" — the V1/V2 protocol (~14 calls per sample):
      geval_* (DeepEval G-Eval), bias_score/bias_detected (DeepEval BiasMetric),
      gemba_score (Kocmi & Federmann, 2023 adapted), rts_score (Reason-then-Score).

    clinical_msk_score is a weighted composite and a *secondary* summary. The default weights
    are ad hoc; fit them to expert ratings (expert_eval.fit_cms_weights) and pass
    weights_path= before reporting it as a headline number.
      base  = Σ wᵢ·metricᵢ / Σ wᵢ   (missing metrics excluded, not zeroed)
      score = base × (1 − 0.5 × bias flag)
    """

    DEFAULT_WEIGHTS: Dict[str, Dict[str, float]] = {
        "full": {"rougeL": 0.10, "bert_f1": 0.20, "geval_mean": 0.25, "gemba_score": 0.25, "rts_score": 0.20},
        "compact": {"rougeL": 0.10, "bert_f1": 0.20, "judge_criteria_mean": 0.35, "judge_overall": 0.35},
    }
    BIAS_KEY = {"full": "bias_detected", "compact": "judge_bias"}

    def __init__(self, anthropic_api_key: Optional[str] = None,
                 judge_model: str = DEFAULT_JUDGE_MODEL, judge_temperature: float = 0.0,
                 mode: str = "compact", weights_path: Optional[str] = None, verbose: bool = False):
        import anthropic
        if mode not in ("compact", "full"):
            raise ValueError(f"Unknown judge mode {mode}")
        self.verbose = verbose
        self.mode = mode
        self.judge_model = judge_model
        # Only sent to models that accept sampling parameters
        self.judge_temperature = judge_temperature if supports_temperature(judge_model) else None
        self.use_deepeval = mode == "full"
        self.weights = dict(self.DEFAULT_WEIGHTS[mode])
        self.weights_source = "default (ad hoc)"
        if weights_path:
            with open(weights_path) as f:
                self.weights = json.load(f)["weights"]
            self.weights_source = weights_path

        key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = (anthropic.Anthropic(api_key=key, max_retries=5) if key
                        else anthropic.Anthropic(max_retries=5))
        if self.use_deepeval:
            from deepeval.metrics import BiasMetric, GEval
            from deepeval.models import AnthropicModel
            from deepeval.test_case import LLMTestCaseParams
            judge = AnthropicModel(model=judge_model, **({"temperature": self.judge_temperature}
                                                         if self.judge_temperature is not None else {}))
            self._geval = {
                name: GEval(model=judge, name=name.replace("_", " ").title(), criteria=crit,
                            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                            threshold=0.5)
                for name, crit in _CLINICAL_CRITERIA
            }
            self._bias = BiasMetric(threshold=0.5, model=judge)

    def manifest(self) -> Dict:
        try:
            import deepeval
            deepeval_version = getattr(deepeval, "__version__", "unknown")
        except ImportError:
            deepeval_version = None
        import anthropic
        common = {
            "judge_mode": self.mode, "judge_model": self.judge_model,
            "judge_temperature": self.judge_temperature if self.judge_temperature is not None
            else "not supported by model; omitted",
            "anthropic_sdk_version": anthropic.__version__,
            "cms_weights": self.weights, "cms_weights_source": self.weights_source,
            "cms_bias_penalty": f"base * (1 - 0.5 * {self.BIAS_KEY[self.mode]})",
            "evaluated_on": time.strftime("%Y-%m-%d"),
        }
        if self.mode == "compact":
            return {**common, "compact_prompt": _COMPACT_PROMPT, "compact_schema": _COMPACT_SCHEMA,
                    "compact_max_tokens": _COMPACT_MAX_TOKENS,
                    "criteria_scale": "1-5 rescaled (s-1)/4; overall 0-100 rescaled /100"}
        return {**common, "deepeval_version": deepeval_version,
                "geval_criteria": dict(_CLINICAL_CRITERIA),
                "geval_input_template": "Write a physiotherapy case study for: {condition}",
                "bias_input": "Generate a clinical case study.",
                "gemba_prompt": _GEMBA_PROMPT, "rts_prompt": _RTS_PROMPT,
                "gemba_max_tokens": 8, "rts_max_tokens": 600}

    def _sampling(self) -> Dict:
        return {"temperature": self.judge_temperature} if self.judge_temperature is not None else {}

    def _call(self, prompt: str, max_tokens: int = 8) -> Optional[str]:
        import anthropic
        try:
            msg = self._client.messages.create(
                model=self.judge_model, max_tokens=max_tokens, **self._sampling(),
                messages=[{"role": "user", "content": prompt}])
            return "".join(b.text for b in msg.content if b.type == "text").strip()
        except anthropic.APIError as exc:
            if self.verbose:
                print(f"  LLM call failed: {exc}")
            return None

    # ------------------------------------------------------------ compact mode
    def compact_params(self, condition: str, generated: str) -> Dict:
        return {"model": self.judge_model, "max_tokens": _COMPACT_MAX_TOKENS, **self._sampling(),
                "messages": [{"role": "user", "content": _COMPACT_PROMPT.format(condition=condition,
                                                                                text=generated)}],
                "output_config": {"format": {"type": "json_schema", "schema": _COMPACT_SCHEMA}}}

    @staticmethod
    def parse_compact(msg) -> Dict:
        """Metrics from a compact-judge Message; NaN on refusal, truncation or bad JSON."""
        nan = {f"judge_{n}": float("nan") for n, _ in _CLINICAL_CRITERIA}
        nan.update(judge_criteria_mean=float("nan"), judge_overall=float("nan"),
                   judge_unsafe=float("nan"), judge_bias=float("nan"))
        if msg.stop_reason != "end_turn":
            return {**nan, "judge_error": str(msg.stop_reason)}
        try:
            data = json.loads(next(b.text for b in msg.content if b.type == "text"))
            crit = {f"judge_{n}": (min(max(int(data[n]), 1), 5) - 1) / 4 for n, _ in _CLINICAL_CRITERIA}
            return {**crit, "judge_criteria_mean": float(np.mean(list(crit.values()))),
                    "judge_overall": min(max(int(data["overall"]), 0), 100) / 100,
                    "judge_unsafe": float(bool(data["unsafe_content"])),
                    "judge_bias": float(bool(data["biased_content"])),
                    "judge_unsafe_reason": str(data.get("unsafe_reason", ""))[:300],
                    "judge_reasoning": str(data.get("reasoning", ""))[:600]}
        except (StopIteration, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            return {**nan, "judge_error": f"parse: {exc}"}

    def compute_compact(self, condition: str, generated: str) -> Dict:
        import anthropic
        try:
            return self.parse_compact(self._client.messages.create(**self.compact_params(condition, generated)))
        except anthropic.APIError as exc:
            if self.verbose:
                print(f"  judge call failed: {exc}")
            return {"judge_error": type(exc).__name__}

    def judge_batch(self, items: List[Tuple[str, str]], state_path: str, poll_s: int = 300) -> List[Dict]:
        """Compact-judge (condition, text) pairs through the Message Batches API (50% cost).
        Batch ids are saved to state_path, so an interrupted run resumes polling instead of
        paying again."""
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        if self.mode != "compact":
            raise ValueError("Batch judging is only implemented for compact mode")
        state = {}
        if os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)
            if state.get("n_items") != len(items) or state.get("judge_model") != self.judge_model:
                raise ValueError(f"{state_path} belongs to a different job — delete it to start a new batch")
        if not state:
            chunk = 20_000
            batch_ids = []
            for start in range(0, len(items), chunk):
                reqs = [Request(custom_id=f"r{start + i}",
                                params=MessageCreateParamsNonStreaming(**self.compact_params(c, t)))
                        for i, (c, t) in enumerate(items[start:start + chunk])]
                batch_ids.append(self._client.messages.batches.create(requests=reqs).id)
            state = {"batch_ids": batch_ids, "n_items": len(items), "judge_model": self.judge_model,
                     "created": time.strftime("%Y-%m-%d %H:%M")}
            with open(state_path, "w") as f:
                json.dump(state, f)
            print(f"  Submitted {len(items)} judge requests in batch(es) {batch_ids}")

        results: List[Dict] = [{"judge_error": "missing"} for _ in items]
        for bid in state["batch_ids"]:
            while True:
                b = self._client.messages.batches.retrieve(bid)
                if b.processing_status == "ended":
                    break
                c = b.request_counts
                print(f"  batch {bid}: processing={c.processing} succeeded={c.succeeded} errored={c.errored}")
                time.sleep(poll_s)
            for r in self._client.messages.batches.results(bid):
                i = int(r.custom_id[1:])
                results[i] = (self.parse_compact(r.result.message) if r.result.type == "succeeded"
                              else {"judge_error": r.result.type})
        os.replace(state_path, state_path + ".done")   # results collected; a re-run starts a fresh batch
        return results

    @staticmethod
    def _extract_number(text: str, lo: float = 0.0, hi: float = 100.0) -> Optional[float]:
        for tok in _re.findall(r"\b(\d+(?:\.\d+)?)\b", text):
            v = float(tok)
            if lo <= v <= hi:
                return v
        return None

    def compute_geval(self, condition: str, generated: str) -> Dict[str, float]:
        if not self.use_deepeval:
            return {}
        from deepeval.test_case import LLMTestCase
        tc = LLMTestCase(input=f"Write a physiotherapy case study for: {condition}", actual_output=generated)
        out: Dict[str, float] = {}
        for name, metric in self._geval.items():
            try:
                metric.measure(tc)
                out[f"geval_{name}"] = float(metric.score)
            except Exception as exc:
                if self.verbose:
                    print(f"  G-Eval {name}: {exc}")
                out[f"geval_{name}"] = float("nan")
        vals = [v for v in out.values() if v == v]
        out["geval_mean"] = float(np.mean(vals)) if vals else float("nan")
        return out

    def compute_bias(self, generated: str) -> Dict[str, float]:
        if not self.use_deepeval:
            return {}
        from deepeval.test_case import LLMTestCase
        tc = LLMTestCase(input="Generate a clinical case study.", actual_output=generated)
        try:
            self._bias.measure(tc)
            return {"bias_score": float(self._bias.score), "bias_detected": float(not self._bias.success)}
        except Exception as exc:
            if self.verbose:
                print(f"  Bias: {exc}")
            return {"bias_score": float("nan"), "bias_detected": float("nan")}

    def compute_gemba(self, condition: str, generated: str) -> Dict[str, float]:
        resp = self._call(_GEMBA_PROMPT.format(condition=condition, text=generated), max_tokens=8)
        raw = self._extract_number(resp, 0, 100) if resp else None
        return {"gemba_score": raw / 100.0 if raw is not None else float("nan")}

    def compute_reason_then_score(self, condition: str, generated: str) -> Dict:
        resp = self._call(_RTS_PROMPT.format(condition=condition, text=generated), max_tokens=600)
        if resp is None:
            return {"rts_score": float("nan"), "rts_reasoning": ""}
        m = _re.search(r"SCORE\s*:\s*(\d+(?:\.\d+)?)", resp, _re.IGNORECASE)
        raw = float(m.group(1)) if m else self._extract_number(resp, 0, 10)
        reasoning = _re.split(r"SCORE\s*:", resp, flags=_re.IGNORECASE)[0].strip()
        return {"rts_score": min(raw, 10.0) / 10.0 if raw is not None else float("nan"),
                "rts_reasoning": reasoning[:600]}

    def compute_clinical_msk_score(self, metrics: Dict) -> float:
        base = total_w = 0.0
        judged = False
        for key, w in self.weights.items():
            v = metrics.get(key, float("nan"))
            if v == v:
                base += w * v
                total_w += w
                judged |= key not in ("rougeL", "bert_f1")
        if total_w == 0 or not judged:   # reference metrics alone are not a clinical score
            return float("nan")
        bias = metrics.get(self.BIAS_KEY[self.mode], 0.0)
        bias = 0.0 if bias != bias else bias
        return round(base / total_w * (1.0 - 0.5 * bias), 4)

    def evaluate_sample(self, condition: str, generated: str, base_metrics: Optional[Dict] = None) -> Dict:
        out: Dict = {}
        if self.mode == "compact":
            out.update(self.compute_compact(condition, generated))
        else:
            out.update(self.compute_geval(condition, generated))
            out.update(self.compute_bias(generated))
            out.update(self.compute_gemba(condition, generated))
            out.update(self.compute_reason_then_score(condition, generated))
        out["clinical_msk_score"] = self.compute_clinical_msk_score({**(base_metrics or {}), **out})
        return out


class PipelineEvaluator:
    """Answer-token NLL, reference metrics, judge metrics and latency on the test split."""

    def __init__(self, tokenizer, max_new_tokens: int = 1024, k_samples: int = 3,
                 temperature: float = 0.7, top_p: float = 0.9, bertscore_model: str = BERTSCORE_MODEL):
        from rouge_score import rouge_scorer
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.k_samples = k_samples
        self.temperature, self.top_p = temperature, top_p
        self.bertscore_model = bertscore_model
        self.scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def generation_settings(self) -> Dict:
        return {"max_new_tokens": self.max_new_tokens, "k_samples": self.k_samples,
                "temperature": self.temperature, "top_p": self.top_p, "do_sample": True,
                "sample_seed": "1000*seed + 10*case_id + sample_idx",
                "bertscore_model": self.bertscore_model}

    @torch.no_grad()
    def answer_nll(self, model, dataset) -> Dict[str, float]:
        """Per-token NLL on gold answer tokens (prompt and padding excluded). Only comparable
        between models that share a tokenizer."""
        from tqdm import tqdm
        model.eval()
        dev = _device_of(model)
        nll, n = 0.0, 0
        for i in tqdm(range(len(dataset)), desc="answer_nll", unit="sample"):
            s = dataset[i]
            labels = s["labels"].unsqueeze(0).to(dev)
            ans = labels[:, 1:] != -100
            if not ans.any():
                continue
            logits = model_forward(model, s["input_ids"].unsqueeze(0).to(dev),
                                   s["attention_mask"].unsqueeze(0).to(dev)).logits[:, :-1][ans]
            nll += F.cross_entropy(logits.float(), labels[:, 1:][ans], reduction="sum").item()
            n += int(ans.sum())
        mean = nll / max(n, 1)
        return {"answer_nll": mean, "perplexity": float(np.exp(min(mean, 20.0)))}

    @torch.no_grad()
    def generate(self, model, title: str, seed: int) -> str:
        enc = encode_prompt(self.tokenizer, title).to(_device_of(model))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        out = model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=True,
                             temperature=self.temperature, top_p=self.top_p,
                             pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def reference_metrics(self, preds: List[str], refs: List[str]) -> List[Dict]:
        from bert_score import score as bert_score_fn
        _bs_device = "cuda" if torch.cuda.is_available() else ("mps" if (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu")
        try:
            P, R, F1 = bert_score_fn(preds, refs, lang="en", verbose=False,
                                      model_type=self.bertscore_model, device=_bs_device)
        except AttributeError as _e:
            # bert_score / transformers version mismatch (e.g. missing build_inputs_with_special_tokens)
            print(f"  BERTScore failed ({_e}); bert_* columns will be NaN")
            nan_t = torch.full((len(preds),), float("nan"))
            P, R, F1 = nan_t, nan_t, nan_t
        rows = []
        for i, (p, r) in enumerate(zip(preds, refs)):
            s = self.scorer.score(r, p)
            rows.append({"rouge1": s["rouge1"].fmeasure, "rouge2": s["rouge2"].fmeasure,
                         "rougeL": s["rougeL"].fmeasure, "bert_precision": float(P[i]),
                         "bert_recall": float(R[i]), "bert_f1": float(F1[i]),
                         "gen_words": len(p.split())})
        return rows

    @torch.no_grad()
    def latency(self, model, title: str, n_runs: int = 20) -> Dict:
        model.eval()
        enc = encode_prompt(self.tokenizer, title).to(_device_of(model))
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model.generate(**enc, max_new_tokens=50, do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
            times.append(time.perf_counter() - t0)
        return {"mean_latency_ms": float(np.mean(times) * 1000),
                "p50_latency_ms": float(np.percentile(times, 50) * 1000),
                "p95_latency_ms": float(np.percentile(times, 95) * 1000)}

    def evaluate(self, model, test_cases: List[Dict], test_dataset, seed: int = 0,
                 advanced: Optional[AdvancedEvaluator] = None,
                 judge_limit: Optional[int] = None, judge_k: int = 1,
                 skip_nll: bool = False) -> Tuple[Dict, List[Dict]]:
        """Returns (summary, per_sample). Summary means are over prompts, after averaging
        each prompt's k samples. Reference metrics use all k samples; the judge sees only the
        first judge_k per prompt, and judge_limit caps how many test prompts get judged."""
        model.eval()
        summary = {} if skip_nll else self.answer_nll(model, test_dataset)
        summary["n_params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
        per_sample = []
        from tqdm import tqdm as _tqdm
        _gen_iter = _tqdm(test_cases, desc="generate", unit="case")
        for ci, case in enumerate(_gen_iter):
            for k in range(self.k_samples):
                gen = self.generate(model, case["title"], seed=1000 * seed + 10 * case["case_id"] + k)
                per_sample.append({"case_id": case["case_id"], "title": case["title"], "sample_idx": k,
                                   "generation": gen, "_ref": case["use_case"], "_ci": ci})
        for row, m in zip(per_sample, self.reference_metrics([r["generation"] for r in per_sample],
                                                              [r.pop("_ref") for r in per_sample])):
            row.update(m)
        if advanced is not None:
            judged = [r for r in per_sample if r["sample_idx"] < judge_k
                      and (judge_limit is None or r["_ci"] < judge_limit)]
            print(f"  Judging {len(judged)} generations ({advanced.judge_model})...")
            for r in judged:
                adv = advanced.evaluate_sample(r["title"], r["generation"],
                                               base_metrics={"rougeL": r["rougeL"], "bert_f1": r["bert_f1"]})
                r.update(adv)
        for r in per_sample:
            r.pop("_ci", None)

        df = pd.DataFrame(per_sample)
        metric_cols = [c for c in df.columns if c not in ("case_id", "title", "sample_idx", "generation",
                                                          "rts_reasoning") and pd.api.types.is_numeric_dtype(df[c])]
        by_prompt = df.groupby("case_id")[metric_cols].mean()
        summary.update({c: float(by_prompt[c].mean()) for c in metric_cols})
        summary["n_test_prompts"] = int(df["case_id"].nunique())
        summary["n_generations"] = len(df)
        summary.update(self.latency(model, test_cases[0]["title"]))
        return summary, per_sample


def estimate_judge_cost(configs: List[AblationConfig], n_seeds: int, n_pairs: int, n_prompts: int,
                        k: int, judge_k: int = 1, mode: str = "compact", model: str = DEFAULT_JUDGE_MODEL,
                        batch: bool = False, repeat_eval_only: bool = False,
                        avg_gen_tokens: int = 800) -> Dict:
    """Cost planning. Evaluation-only rows run once unless repeat_eval_only.
    Tokens per judged sample (rough): compact ≈ 1 call, ~(avg_gen + 450) in / ~400 out;
    full ≈ 14 calls (5 G-Eval × steps + score, bias ×2, GEMBA, RTS), each ~(avg_gen + 250) in /
    ~150 out — an upper bound, since DeepEval reuses G-Eval steps."""
    runs = sum(n_seeds if (a.method != "none" or repeat_eval_only) else 1 for a in configs) * n_pairs
    generations = runs * n_prompts * k
    judged = runs * n_prompts * min(judge_k, k)
    calls_per, tok_in, tok_out = ((1, avg_gen_tokens + 450, 400) if mode == "compact"
                                  else (14, avg_gen_tokens + 250, 150))
    calls = judged * calls_per
    p_in, p_out = JUDGE_PRICES.get(model, (float("nan"), float("nan")))
    usd = calls * (tok_in * p_in + tok_out * p_out) / 1e6 * (0.5 if batch else 1.0)
    return {"runs": runs, "generations": generations, "judged_samples": judged,
            "judge_calls": calls, "input_tokens_M": round(calls * tok_in / 1e6, 2),
            "output_tokens_M": round(calls * tok_out / 1e6, 2), "est_usd": round(usd, 2),
            "mode": mode, "model": model, "batch": batch}


def judge_per_sample_file(per_sample_path: str, adv: AdvancedEvaluator, judge_k: int = 1,
                          judge_limit: Optional[int] = None, batch: bool = False,
                          summary_path: Optional[str] = None,
                          force_rejudge: bool = False) -> pd.DataFrame:
    """Judge an existing per_sample.jsonl (e.g. after `--no-judge` runs). Rows already judged
    with the same judge model and mode are skipped, so re-running only pays for what's missing.
    Pass force_rejudge=True to strip existing tags and re-score every eligible row.
    Rewrites the file in place (a .bak copy is kept) and refreshes the judge columns of the
    results summary."""
    import shutil
    with open(per_sample_path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    tag = f"{adv.mode}:{adv.judge_model}"
    if force_rejudge:
        for r in rows:
            r.pop("judge_tag", None)
    order = {}
    for r in rows:
        order.setdefault((r.get("model_pair"), r["name"], r["seed"]), [])
        if r["case_id"] not in order[(r.get("model_pair"), r["name"], r["seed"])]:
            order[(r.get("model_pair"), r["name"], r["seed"])].append(r["case_id"])
    todo = [r for r in rows if r["sample_idx"] < judge_k and r.get("judge_tag") != tag
            and (judge_limit is None
                 or order[(r.get("model_pair"), r["name"], r["seed"])].index(r["case_id"]) < judge_limit)]
    print(f"Judging {len(todo)} of {len(rows)} generations ({tag}, batch={batch})")
    if not todo:
        return pd.DataFrame()
    if batch:
        results = adv.judge_batch([(r["title"], r["generation"]) for r in todo],
                                  state_path=per_sample_path + f".batch_{adv.mode}.json")
    else:
        results = [adv.evaluate_sample(r["title"], r["generation"]) for r in todo]
    n_failed = 0
    for r, res in zip(todo, results):
        res["clinical_msk_score"] = adv.compute_clinical_msk_score({**r, **res})
        if "judge_error" in res:      # left untagged so the next run retries it
            n_failed += 1
            r.update(res)
        else:
            r.pop("judge_error", None)
            r.update(res, judge_tag=tag)
    if n_failed:
        print(f"  {n_failed} judge requests failed — re-run the same command to retry only those")

    shutil.copy(per_sample_path, per_sample_path + ".bak")
    with open(per_sample_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    df = pd.DataFrame(rows)
    judge_cols = [c for c in df.columns if (c.startswith(("judge_", "geval_")) or c in (
        "gemba_score", "rts_score", "bias_score", "bias_detected", "clinical_msk_score"))
        and pd.api.types.is_numeric_dtype(df[c])]
    keys = ["model_pair", "name", "seed"] if "model_pair" in df else ["name", "seed"]
    agg = (df.groupby(keys + ["case_id"])[judge_cols].mean()   # mean over judged samples per prompt
             .groupby(level=list(range(len(keys)))).mean().reset_index())
    if summary_path and os.path.exists(summary_path):
        summ = pd.read_csv(summary_path)
        summ = summ.drop(columns=[c for c in judge_cols if c in summ.columns]).merge(agg, on=keys, how="left")
        summ.to_csv(summary_path, index=False)
        print(f"Updated {summary_path}")
    return agg


class AblationStudyRunner:
    def __init__(self, model_pair_key: str, base_output_dir: str,
                 qlora_config: QLoRAConfig, distill_config: DistillationConfig,
                 refinement_config: RefinementConfig,
                 advanced_evaluator: Optional[AdvancedEvaluator] = None,
                 seeds: List[int] = (0, 1, 2), k_samples: int = 3, max_new_tokens: int = 1024,
                 judge_limit: Optional[int] = None, configs: List[AblationConfig] = ABLATION_CONFIGS,
                 repeat_eval_only: bool = False, judge_k: int = 1,
                 data_path: str = DATA_PATH, splits_path: str = SPLITS_PATH,
                 skip_nll: bool = False, test_limit: Optional[int] = None):
        self.model_pair = MODEL_PAIRS[model_pair_key]
        self.key = model_pair_key
        self.base_output_dir = base_output_dir
        self.qlora_config, self.distill_config = qlora_config, distill_config
        self.ref_config = refinement_config
        self.adv = advanced_evaluator
        self.seeds = list(seeds)
        self.k_samples, self.max_new_tokens = k_samples, max_new_tokens
        self.judge_limit, self.judge_k = judge_limit, judge_k
        self.configs = configs
        self.repeat_eval_only = repeat_eval_only
        self.skip_nll = skip_nll
        self.test_limit = test_limit
        self.data_path, self.splits_path = data_path, splits_path
        self.summaries: List[Dict] = []
        self._layer_scores_cache: Optional[Dict[int, float]] = None
        os.makedirs(base_output_dir, exist_ok=True)
        self.per_sample_path = os.path.join(base_output_dir, "per_sample.jsonl")
        self.summary_path = os.path.join(base_output_dir, "results_summary.csv")

    def _make_cfg(self, ab: AblationConfig, seed: int) -> RefinementConfig:
        cfg = deepcopy(self.ref_config)
        for k, v in ab.overrides.items():
            if not hasattr(cfg, k):
                raise AttributeError(f"{ab.name}: unknown RefinementConfig field {k}")
            setattr(cfg, k, v)
        cfg.seed = seed
        cfg.model_pair_key = self.key
        if ab.match_pipeline_epochs:
            cfg.num_epochs = self.distill_config.epochs + self.ref_config.num_epochs
        if cfg.rank_allocation == "degradation" and self._layer_scores_cache:
            cfg.layer_scores = self._layer_scores_cache   # seed-independent; score once per pair
        return cfg

    def _source_path(self, source: str) -> str:
        mp = self.model_pair
        if source == "distilled_base_teacher":
            if not os.path.isdir(mp.distilled_student_base_teacher):
                try:
                    from .train import HybridPipeline
                except ImportError:
                    from train import HybridPipeline
                HybridPipeline(self.key, self.data_path, self.qlora_config, self.distill_config,
                               self.ref_config, splits_path=self.splits_path).run_phase2_from_base_teacher()
            return mp.distilled_student_base_teacher
        return {"tuned_teacher": mp.tuned_teacher, "base_teacher": mp.teacher_name,
                "distilled": mp.distilled_student, "base_student": mp.student_name}[source]

    def manifest(self) -> Dict:
        import transformers, peft
        splits = load_splits(self.splits_path, self.data_path)
        return {
            "model_pair": self.key, "pair": asdict(self.model_pair), "seeds": self.seeds,
            "splits": {k: splits[k] for k in ("seed", "fractions", "data_sha256", "n_rows", "n_conditions")},
            "split_sizes": {k: len(v) for k, v in splits["case_ids"].items()},
            "generation": {"max_new_tokens": self.max_new_tokens, "k_samples": self.k_samples,
                           "temperature": 0.7, "top_p": 0.9, "do_sample": True,
                           "prompt": "tokenizer chat template (shared.build_prompt)",
                           "sample_seed": "1000*seed + 10*case_id + sample_idx",
                           "bertscore_model": BERTSCORE_MODEL},
            "judge": self.adv.manifest() if self.adv else None, "judge_limit": self.judge_limit,
            "judge_k": self.judge_k,
            "qlora_config": asdict(self.qlora_config), "distill_config": asdict(self.distill_config),
            "refinement_base_config": asdict(self.ref_config),
            "ablation_configs": [asdict(a) for a in self.configs],
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "transformers": transformers.__version__, "peft": peft.__version__},
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }

    def run_single(self, ab: AblationConfig, seed: int) -> Dict:
        print(f"\n{'=' * 70}\n{self.key} | {ab.name} | seed {seed}\n  {ab.description}\n{'=' * 70}")
        row = {"model_pair": self.key, "name": ab.name, "description": ab.description, "seed": seed}
        src = self._source_path(ab.source)
        t0 = time.time()
        if ab.method == "none":
            model = load_causal_lm(src, quant="none")
            tok = load_tokenizer(src)
            meta = {"trainable_params": 0}
        else:
            cfg = self._make_cfg(ab, seed)
            out_dir = os.path.join(self.base_output_dir, ab.name, f"seed{seed}")
            if ab.method == "gkd":
                try:
                    from .baselines import GKDRefiner
                except ImportError:
                    from baselines import GKDRefiner
                trainer = GKDRefiner(src, self.model_pair.tuned_teacher, cfg)
            else:
                trainer = RefinementTrainer(src, self.model_pair.tuned_teacher, cfg)
            tok = trainer.tokenizer
            ds = get_split_datasets(tok, cfg.max_seq_length, self.data_path, self.splits_path)
            meta = trainer.train(ds["train"], ds["calibration"], output_dir=out_dir)
            if cfg.rank_allocation == "degradation" and getattr(trainer, "layer_scores", None):
                self._layer_scores_cache = trainer.layer_scores
            model = trainer.peft_model
        row["train_time_s"] = round(time.time() - t0, 1)
        row.update({"trainable_params": meta.get("trainable_params"), "total_steps": meta.get("total_steps"),
                    "same_tokenizer": meta.get("same_tokenizer"),
                    "rank_map": json.dumps(meta.get("rank_map")) if meta.get("rank_map") else None,
                    "teacher_hacking": (meta.get("teacher_hacking") or {}).get("verdict")})

        test_cases = get_split_cases("test", self.data_path, self.splits_path)
        if self.test_limit is not None:
            test_cases = test_cases[:self.test_limit]
        test_ds = get_split_datasets(tok, self.ref_config.max_seq_length, self.data_path,
                                     self.splits_path)["test"]
        if self.test_limit is not None:
            test_ds = MSKCaseStudyDataset(test_cases, tok, self.ref_config.max_seq_length)
        ev = PipelineEvaluator(tok, max_new_tokens=self.max_new_tokens, k_samples=self.k_samples)
        summary, per_sample = ev.evaluate(model, test_cases, test_ds, seed=seed, advanced=self.adv,
                                          judge_limit=self.judge_limit, judge_k=self.judge_k,
                                          skip_nll=self.skip_nll)
        row.update(summary)
        with open(self.per_sample_path, "a") as f:
            for r in per_sample:
                tag = {"judge_tag": f"{self.adv.mode}:{self.adv.judge_model}"} \
                    if self.adv and "clinical_msk_score" in r else {}
                f.write(json.dumps({"model_pair": self.key, "name": ab.name, "seed": seed, **r, **tag}) + "\n")
        del model
        clear_gpu_memory()
        return row

    def run_all(self) -> pd.DataFrame:
        with open(os.path.join(self.base_output_dir, "manifest.json"), "w") as f:
            json.dump(self.manifest(), f, indent=2, default=str)
        for ab in self.configs:
            seeds = self.seeds if (ab.method != "none" or self.repeat_eval_only) else self.seeds[:1]
            for seed in seeds:
                try:
                    self.summaries.append(self.run_single(ab, seed))
                except Exception as e:
                    print(f"  FAILED {ab.name} seed {seed}: {e}")
                    self.summaries.append({"model_pair": self.key, "name": ab.name, "seed": seed, "error": str(e)})
                pd.DataFrame(self.summaries).to_csv(self.summary_path, index=False)
        return pd.DataFrame(self.summaries)


# --------------------------------------------------------------------------- #
# Plots                                                                       #
# --------------------------------------------------------------------------- #

_LOWER_BETTER = {"perplexity", "answer_nll", "p50_latency_ms", "p95_latency_ms", "mean_latency_ms",
                 "bias_score", "bias_detected", "judge_unsafe", "judge_bias"}
_METRIC_GROUPS = {
    "core": ["answer_nll", "rougeL", "bert_f1", "p50_latency_ms", "trainable_params"],
    "advanced": ["judge_overall", "judge_criteria_mean", "judge_unsafe",
                 "geval_mean", "geval_clinical_coherence", "geval_diagnostic_accuracy",
                 "geval_treatment_completeness", "geval_patient_safety", "geval_clinical_relevance",
                 "gemba_score", "rts_score", "clinical_msk_score", "bias_score"],
}


def plot_ablation_results(df: pd.DataFrame, save_path: str = "./models/ablation_results.png",
                          groups: Optional[List[str]] = None):
    """Mean ± std across seeds per config (use stats.py for CIs and paired tests)."""
    wanted = [m for g in (groups or ["core", "advanced"]) for m in _METRIC_GROUPS.get(g, [])]
    metrics = [c for c in wanted if c in df.columns]
    if not metrics:
        print("No metrics to plot")
        return
    agg = df.groupby("name", sort=False)[metrics].agg(["mean", "std"])
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    axes = np.atleast_1d(axes)
    names = agg.index.tolist()
    for ax, metric in zip(axes, metrics):
        ax.bar(range(len(names)), agg[(metric, "mean")], yerr=agg[(metric, "std")].fillna(0),
               color="#4C72B0", capsize=3)
        ax.set_title(metric.replace("_", "\n"), fontsize=8, fontweight="bold")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.split("_")[0] for n in names], rotation=45, ha="right", fontsize=7)
    fig.suptitle("Ablations — mean ± std over seeds", fontsize=12, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {save_path}")


def plot_layer_degradation(layer_scores: Dict[int, float], rank_map: Optional[Dict[int, int]] = None,
                           save_path: str = "./models/layer_degradation.png"):
    if not layer_scores:
        print("No layer scores to plot")
        return
    layers = sorted(int(k) for k in layer_scores)
    scores = [layer_scores[l] if l in layer_scores else layer_scores[str(l)] for l in layers]
    ncols = 2 if rank_map else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 4))
    axes = np.atleast_1d(axes)
    axes[0].bar(layers, scores, color=plt.cm.RdYlGn_r(np.array(scores)))
    axes[0].set_xlabel("Layer index")
    axes[0].set_ylabel("Normalised degradation score")
    axes[0].set_title("Per-layer degradation score")
    if rank_map:
        rm = {int(k): v for k, v in rank_map.items()}
        axes[1].bar(layers, [rm.get(l, 0) for l in layers], color="#4C72B0")
        axes[1].set_xlabel("Layer index")
        axes[1].set_ylabel("LoRA rank")
        axes[1].set_title("Rank allocation (fixed total budget)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_proxy_vs_golden(meta_path: str, save_path: Optional[str] = None):
    """Teacher-hacking view: proxy (KL to teacher) and golden (held-out gold NLL) over training."""
    with open(meta_path) as f:
        meta = json.load(f)
    recs = [r for r in meta.get("tracking", []) if r.get("proxy_kl") is not None]
    if len(recs) < 2:
        print("No tracking records")
        return
    steps = [r["step"] for r in recs]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(steps, [r["proxy_kl"] for r in recs], color="#C44E52", marker="o", label="proxy: KL(teacher‖student)")
    ax1.set_xlabel("optimizer step")
    ax1.set_ylabel("proxy KL (lower = closer to teacher)", color="#C44E52")
    ax2 = ax1.twinx()
    ax2.plot(steps, [r["golden_nll"] for r in recs], color="#4C72B0", marker="s", label="golden: held-out gold NLL")
    ax2.set_ylabel("golden NLL (lower = better)", color="#4C72B0")
    verdict = (meta.get("teacher_hacking") or {}).get("verdict", "")
    ax1.set_title(f"Proxy vs golden — {verdict}", fontsize=9)
    fig.tight_layout()
    save_path = save_path or meta_path.replace(".json", "_proxy_golden.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {save_path}")


def main():
    try:
        from .config import SEEDS, distill_cfg, qlora_cfg, refine_cfg
    except ImportError:
        from config import SEEDS, distill_cfg, qlora_cfg, refine_cfg

    parser = argparse.ArgumentParser(description="DART ablation evaluation")
    parser.add_argument("--pairs", nargs="+", default=FOCUS_PAIRS)
    parser.add_argument("--configs", nargs="+", default=None, help="subset of ablation names")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--k", type=int, default=3, help="samples per test prompt (reference metrics)")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--judge-mode", choices=["compact", "full"], default="compact",
                        help="compact: 1 structured call per sample; full: V1/V2 DeepEval+GEMBA+RTS (~14 calls)")
    parser.add_argument("--judge-k", type=int, default=1, help="samples per prompt sent to the judge")
    parser.add_argument("--judge-limit", type=int, default=None, help="judge only the first N test prompts")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-batch", action="store_true",
                        help="judge after all runs via the Message Batches API (50%% cheaper, async)")
    parser.add_argument("--judge-file", default=None,
                        help="only judge an existing per_sample.jsonl (skips training/generation)")
    parser.add_argument("--force-rejudge", action="store_true",
                        help="strip existing judge_tag and re-score every eligible row (--judge-file only)")
    parser.add_argument("--cms-weights", default=None, help="JSON from expert_eval.fit_cms_weights")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--skip-nll", action="store_true",
                        help="skip answer_nll/perplexity computation (saves time on CPU/MPS)")
    parser.add_argument("--test-limit", type=int, default=None,
                        help="use only the first N test prompts for generation and evaluation")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--plot-proxy-golden", action="store_true",
                        help="plot proxy-KL vs golden-NLL for every refinement_meta.json found "
                             "under ./models/ablations/<pair>/ and exit")
    args = parser.parse_args()

    make_adv = lambda: AdvancedEvaluator(judge_model=args.judge_model, mode=args.judge_mode,
                                         weights_path=args.cms_weights)

    if args.plot_proxy_golden:
        for key in args.pairs:
            pattern = os.path.join("./models/ablations", key, "*", "seed*", "refinement_meta.json")
            found = sorted(glob.glob(pattern))
            if not found:
                print(f"[{key}] no refinement_meta.json files found under ./models/ablations/{key}/")
                continue
            for meta_path in found:
                parts = meta_path.split(os.sep)
                label = "/".join(parts[-3:-1])  # e.g. R4_degradation_ranks/seed0
                print(f"[{key}] plotting {label} …")
                try:
                    plot_proxy_vs_golden(meta_path)
                except Exception as exc:
                    print(f"  skipped ({exc})")
        return

    if args.judge_file:
        summary = os.path.join(os.path.dirname(args.judge_file), "results_summary.csv")
        judge_per_sample_file(args.judge_file, make_adv(), args.judge_k, args.judge_limit,
                              batch=args.judge_batch, summary_path=summary,
                              force_rejudge=args.force_rejudge)
        return

    configs = [a for a in ABLATION_CONFIGS if not args.configs or a.name in args.configs]
    n_test = len(get_split_cases("test"))
    n_test_eff = min(args.test_limit, n_test) if args.test_limit else n_test
    n_prompts = min(args.judge_limit or n_test_eff, n_test_eff)
    base = dict(configs=configs, n_seeds=len(args.seeds), n_pairs=len(args.pairs), n_prompts=n_prompts,
                k=args.k, model=args.judge_model)
    chosen = estimate_judge_cost(**base, judge_k=args.judge_k, mode=args.judge_mode, batch=args.judge_batch)
    print(f"Plan: {len(args.pairs)} pairs × {len(configs)} configs × {len(args.seeds)} seeds "
          f"({chosen['runs']} runs), {n_test_eff} test prompts × k={args.k} → {chosen['generations']} generations")
    print(f"Chosen judge setup: {chosen}")
    print("Options (est. USD, rough — check current pricing):")
    for mode, jk, batch in [("full", args.k, False), ("full", 1, False), ("compact", args.k, False),
                            ("compact", 1, False), ("compact", 1, True)]:
        e = estimate_judge_cost(**base, judge_k=jk, mode=mode, batch=batch)
        print(f"  mode={mode:7s} judge_k={jk} batch={str(batch):5s} → {e['judge_calls']:>7} calls, ~${e['est_usd']}")
    if args.estimate_only:
        return

    # With --judge-batch, generate everything first, then judge each pair's file in one batch
    adv = None if (args.no_judge or args.judge_batch) else make_adv()
    frames = []
    for key in args.pairs:
        runner = AblationStudyRunner(key, f"./models/ablations/{key}", qlora_cfg, distill_cfg, refine_cfg,
                                     advanced_evaluator=adv, seeds=args.seeds, k_samples=args.k,
                                     max_new_tokens=args.max_new_tokens, judge_limit=args.judge_limit,
                                     judge_k=args.judge_k, configs=configs,
                                     skip_nll=args.skip_nll, test_limit=args.test_limit)
        df = runner.run_all()
        frames.append(df)
        pd.concat(frames, ignore_index=True).to_csv("./models/ablation_results.csv", index=False)
        ok = df[df["error"].isna()] if "error" in df.columns else df
        plot_ablation_results(ok, save_path=f"./models/ablations/{key}/ablation_core.png", groups=["core"])
        plot_ablation_results(ok, save_path=f"./models/ablations/{key}/ablation_advanced.png", groups=["advanced"])
    if args.judge_batch and not args.no_judge:
        judge_adv = make_adv()
        for key in args.pairs:
            d = f"./models/ablations/{key}"
            judge_per_sample_file(os.path.join(d, "per_sample.jsonl"), judge_adv, args.judge_k,
                                  args.judge_limit, batch=True, summary_path=os.path.join(d, "results_summary.csv"))


if __name__ == "__main__":
    main()

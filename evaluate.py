
import os
import re as _re
import os, ast, time
from .config import qlora_cfg, distill_cfg, refine_cfg, AblationConfig, RefinementConfig, QLoRAConfig,ABLATION_CONFIGS, DistillationConfig, MSKCaseStudyDataset, MODEL_PAIRS
from .refinement import RefinementTrainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from rouge_score import rouge_scorer as rs_module
from bert_score import score as bert_score_fn
import anthropic as _anthropic
from typing import Dict, List, Optional, Tuple
from deepeval.metrics import GEval, BiasMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import AnthropicModel
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    #TrainingArguments, 
    #Trainer,
    BitsAndBytesConfig, DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
import numpy as np

from dotenv import load_dotenv
load_dotenv() 

os.environ["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY")
DATA_PATH = "./physio_use_cases.csv"
os.environ["HF_TOKEN"] = os.getenv("HF_KEY")


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

def clear_gpu_memory():
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print(f"GPU cleared — allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")


class AdvancedEvaluator:
    """
    Advanced clinical text evaluation suite for MSK physiotherapy case studies.

    Metrics
    -------
    geval_*            Five structured clinical criteria via DeepEval G-Eval (LLM-as-judge).
    bias_score /       DeepEval BiasMetric — severity score and binary detection flag.
    bias_detected
    gemba_score        GEMBA-MSK: reference-free holistic quality score in [0,1].
                       Adapted from Kocmi & Federmann (2023) — original applied GPT-4
                       to machine translation; here reprompted for MSK clinical text
                       with domain-specific scale anchors.
    rts_score          Reason-then-Score: LLM reasons across 4 clinical dimensions
                       then emits a 0-10 score. CoT step reduces anchoring bias
                       compared with direct scoring. Normalised to [0,1].
    clinical_msk_score Novel weighted hybrid with multiplicative bias penalty (see below).

    ClinicalMSKScore formula
    ------------------------
      base  = Σ wᵢ·metricᵢ / Σ wᵢ   (missing metrics excluded, not zeroed)
              weights: ROUGE-L 10%, BERTScore F1 20%, G-Eval mean 25%,
                       GEMBA 25%, RTS 20%
      score = base × (1 − 0.5 × bias_detected)

    A biased output cannot compensate via high quality scores because the
    penalty is multiplicative, not additive.
    """

    _WEIGHTS: Dict[str, float] = {
        "rougeL":      0.10,
        "bert_f1":     0.20,
        "geval_mean":  0.25,
        "gemba_score": 0.25,
        "rts_score":   0.20,
    }

    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        judge_model: str = "claude-sonnet-4-6",
        use_deepeval: bool = True,
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.judge_model = judge_model
        self.use_deepeval = use_deepeval

        key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = _anthropic.Anthropic(api_key=key)
        if self._client is None:
            print("[AdvancedEvaluator] No Anthropic client — GEMBA and RTS will be skipped.\n"
                  "  Set ANTHROPIC_API_KEY or pass anthropic_api_key= to enable them.")
        self.claude_model = AnthropicModel(
                    model="claude-sonnet-4-6",
                    temperature=0.0)
        if self.use_deepeval:
            self._geval: Dict[str, GEval] = {
                name: GEval(
                    model = self.claude_model,
                    name=name.replace("_", " ").title(),
                    criteria=crit,
                    evaluation_params=[
                        LLMTestCaseParams.INPUT,
                        LLMTestCaseParams.ACTUAL_OUTPUT,
                    ],
                    threshold=0.5,
                )
                for name, crit in _CLINICAL_CRITERIA
            }
            self._bias = BiasMetric(threshold=0.5, model = self.claude_model)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _call(self, prompt: str, max_tokens: int = 8) -> Optional[str]:
        if self._client is None:
            return None
        try:
            msg = self._client.messages.create(
                model=self.judge_model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except Exception as exc:
            if self.verbose:
                print(f"  LLM call failed: {exc}")
            return None

    @staticmethod
    def _extract_number(text: str, lo: float = 0.0, hi: float = 100.0) -> Optional[float]:
        for tok in _re.findall(r"\b(\d+(?:\.\d+)?)\b", text):
            v = float(tok)
            if lo <= v <= hi:
                return v
        return None

    # ------------------------------------------------------------------ #
    # Individual metrics                                                   #
    # ------------------------------------------------------------------ #

    def compute_geval(self, condition: str, generated: str) -> Dict[str, float]:
        """G-Eval across 5 clinical criteria; returns per-criterion scores + mean."""
        if not self.use_deepeval:
            return {}
        tc = LLMTestCase(
            input=f"Write a physiotherapy case study for: {condition}",
            actual_output=generated,
        )
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
        """DeepEval BiasMetric — severity score and binary detection flag."""
        if not self.use_deepeval:
            return {}
        tc = LLMTestCase(input="Generate a clinical case study.", actual_output=generated)
        try:
            self._bias.measure(tc)
            return {
                "bias_score":    float(self._bias.score),
                "bias_detected": float(not self._bias.success),
            }
        except Exception as exc:
            if self.verbose:
                print(f"  Bias: {exc}")
            return {"bias_score": float("nan"), "bias_detected": float("nan")}

    def compute_gemba(self, condition: str, generated: str) -> Dict[str, float]:
        """
        GEMBA-MSK: single-prompt reference-free quality score via LLM judge.
        Returns normalised score in [0, 1].
        """
        resp = self._call(_GEMBA_PROMPT.format(condition=condition, text=generated), max_tokens=8)
        if resp is None:
            return {"gemba_score": float("nan")}
        raw = self._extract_number(resp, 0, 100)
        return {"gemba_score": raw / 100.0 if raw is not None else float("nan")}

    def compute_reason_then_score(self, condition: str, generated: str) -> Dict[str, float]:
        """
        Reason-then-Score: LLM reasons across 4 clinical dimensions then
        emits SCORE: <0-10>. Returns normalised score in [0, 1] and reasoning chain.
        """
        resp = self._call(_RTS_PROMPT.format(condition=condition, text=generated), max_tokens=600)
        if resp is None:
            return {"rts_score": float("nan"), "rts_reasoning": ""}
        m = _re.search(r"SCORE\s*:\s*(\d+(?:\.\d+)?)", resp, _re.IGNORECASE)
        raw = float(m.group(1)) if m else self._extract_number(resp, 0, 10)
        reasoning = _re.split(r"SCORE\s*:", resp, flags=_re.IGNORECASE)[0].strip()
        return {
            "rts_score":     min(raw, 10.0) / 10.0 if raw is not None else float("nan"),
            "rts_reasoning": reasoning[:600],
        }

    # ------------------------------------------------------------------ #
    # Novel hybrid metric                                                  #
    # ------------------------------------------------------------------ #

    def compute_clinical_msk_score(self, metrics: Dict) -> float:
        """
        ClinicalMSKScore — novel weighted hybrid metric for clinical text.

        Components: ROUGE-L (10%), BERTScore F1 (20%), G-Eval mean (25%),
                    GEMBA-MSK (25%), Reason-then-Score (20%).
        Bias penalty (multiplicative): score = base × (1 − 0.5 × bias_detected).
        Missing components are excluded from the weighted average rather than
        zeroed, so the score degrades gracefully when some judges are unavailable.
        """
        base = total_w = 0.0
        for key, w in self._WEIGHTS.items():
            v = metrics.get(key, float("nan"))
            if v == v:  # not NaN
                base += w * v
                total_w += w
        if total_w == 0:
            return float("nan")
        base /= total_w

        bias = metrics.get("bias_detected", 0.0)
        if bias != bias:
            bias = 0.0
        return round(base * (1.0 - 0.5 * bias), 4)

    # ------------------------------------------------------------------ #
    # Batch evaluation                                                     #
    # ------------------------------------------------------------------ #

    def evaluate_sample(self, condition: str, generated: str) -> Dict:
        out: Dict = {}
        out.update(self.compute_geval(condition, generated))
        out.update(self.compute_bias(generated))
        out.update(self.compute_gemba(condition, generated))
        out.update(self.compute_reason_then_score(condition, generated))
        return out

    def evaluate_batch(
        self,
        conditions: List[str],
        predictions: List[str],
        base_metrics: Optional[Dict] = None,
    ) -> Dict:
        """
        Run all advanced metrics over a batch, aggregate via nanmean, then
        compute ClinicalMSKScore merged with base_metrics (rougeL, bert_f1, …).
        The first sample's RTS reasoning chain is preserved under
        'rts_reasoning_sample' for manual inspection.
        """
        per_sample = [self.evaluate_sample(c, p) for c, p in zip(conditions, predictions)]

        agg: Dict[str, float] = {}
        for k in {k for r in per_sample for k in r if k != "rts_reasoning"}:
            vals = [r[k] for r in per_sample if k in r and r[k] == r[k]]
            agg[k] = float(np.mean(vals)) if vals else float("nan")

        agg["rts_reasoning_sample"] = per_sample[0].get("rts_reasoning", "") if per_sample else ""
        combined = {**(base_metrics or {}), **agg}
        agg["clinical_msk_score"] = self.compute_clinical_msk_score(combined)
        return agg
    
class PipelineEvaluator:
    # Evaluate models on perplexity, ROUGE, BERTScore, latency, and compression ratio.

    def __init__(self, tokenizer, device: str = "cuda"):
        self.tokenizer = tokenizer
        self.device = device
        self.scorer = rs_module.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def compute_perplexity(self, model, dataset: Dataset, max_batches: int = 100) -> float:
        model.eval()
        loader = DataLoader(dataset, batch_size=1,
                            collate_fn=lambda x: {k: torch.stack([i[k] for i in x])
                                                  for k in x[0].keys()
                                                  if isinstance(x[0][k], torch.Tensor)})
        total_nll, total_tokens = 0.0, 0
        with torch.no_grad():
            for idx, batch in enumerate(loader):
                if idx >= max_batches:
                    break
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                out = model(ids, mask, labels=ids)
                n_tok = mask.sum().item()
                total_nll += out.loss.item() * n_tok
                total_tokens += n_tok
        return float(np.exp(total_nll / max(total_tokens, 1)))

    def generate_responses(self, model, prompts: List[str], max_new_tokens: int = 256) -> List[str]:
        model.eval()
        responses = []
        for prompt in prompts:
            enc = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                 max_length=384).to(self.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                     temperature=0.7, do_sample=True, top_p=0.9,
                                     pad_token_id=self.tokenizer.pad_token_id)
            text = self.tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            responses.append(text)
        return responses

    def compute_rouge(self, predictions: List[str], references: List[str]) -> Dict:
        r1, r2, rl = [], [], []
        for pred, ref in zip(predictions, references):
            s = self.scorer.score(ref, pred)
            r1.append(s["rouge1"].fmeasure)
            r2.append(s["rouge2"].fmeasure)
            rl.append(s["rougeL"].fmeasure)
        return {"rouge1": float(np.mean(r1)), "rouge2": float(np.mean(r2)),
                "rougeL": float(np.mean(rl))}

    def compute_bert_score(self, predictions: List[str], references: List[str]) -> Dict:
        P, R, F1 = bert_score_fn(predictions, references, lang="en", verbose=False,
                                  model_type="distilbert-base-uncased")
        return {"bert_precision": P.mean().item(), "bert_recall": R.mean().item(),
                "bert_f1": F1.mean().item()}

    def compute_latency(self, model, prompt: str, n_runs: int = 20) -> Dict:
        model.eval()
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=256).to(self.device)
        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                model.generate(**enc, max_new_tokens=50, do_sample=False,
                               pad_token_id=self.tokenizer.pad_token_id)
                times.append(time.perf_counter() - t0)
        return {"mean_latency_ms": float(np.mean(times) * 1000),
                "p50_latency_ms": float(np.percentile(times, 50) * 1000),
                "p95_latency_ms": float(np.percentile(times, 95) * 1000)}

    def compression_stats(self, teacher_model, student_model) -> Dict:
        def n_params(m): return sum(p.numel() for p in m.parameters())
        t, s = n_params(teacher_model), n_params(student_model)
        return {"teacher_params_M": round(t / 1e6, 2), "student_params_M": round(s / 1e6, 2),
                "compression_ratio": round(t / s, 2),
                "size_reduction_pct": round((1 - s / t) * 100, 1)}

    def full_evaluation(self, model, dataset: Dataset, model_name: str = "model",
                        reference_responses: Optional[List[str]] = None,
                        n_gen_samples: int = 20,
                        advanced_evaluator: Optional["AdvancedEvaluator"] = None,
                        n_advanced_samples: int = 5) -> Dict:
        print(f"Evaluating {model_name}...")
        results = {"model": model_name}
        results["perplexity"] = self.compute_perplexity(model, dataset)
        print(f"  Perplexity: {results['perplexity']:.2f}")

        if reference_responses is None:
            tmp_ds = MSKCaseStudyDataset(DATA_PATH, self.tokenizer)
            # physio_use_cases.csv: ground-truth reference is the 'use_case' text
            refs = [c.get("use_case") for c in tmp_ds.data[:n_gen_samples]]
            refs = [r for r in refs if r]
            if refs:
                reference_responses = refs

        if reference_responses:
            tmp_ds = MSKCaseStudyDataset(DATA_PATH, self.tokenizer)
            prompts = [tmp_ds.format_case_study(c) for c in tmp_ds.data[:len(reference_responses)]]
            preds = self.generate_responses(model, prompts)
            results.update(self.compute_rouge(preds, reference_responses))
            results.update(self.compute_bert_score(preds, reference_responses))
            print(f"  ROUGE-L: {results['rougeL']:.4f}  BERTScore F1: {results['bert_f1']:.4f}")

            if advanced_evaluator is not None:
                k = min(n_advanced_samples, len(preds))
                print(f"  Running advanced metrics on {k}/{len(preds)} samples "
                      f"(n_advanced_samples={k})...")
                conditions = [tmp_ds.data[i].get("title", "") for i in range(k)]
                adv = advanced_evaluator.evaluate_batch(
                    conditions, preds[:k], base_metrics=dict(results)
                )
                results["rts_reasoning_sample"] = adv.pop("rts_reasoning_sample", "")
                results.update(adv)
                if "clinical_msk_score" in results:
                    print(f"  ClinicalMSKScore: {results['clinical_msk_score']:.4f}")
        else:
            print("  ROUGE/BERTScore skipped — no reference outputs found in dataset")

        lat = self.compute_latency(model, "Patient with chronic lower back pain, 3/10 NPRS at rest.")
        results.update(lat)
        print(f"  Latency p50: {results['p50_latency_ms']:.1f} ms  p95: {results['p95_latency_ms']:.1f} ms")
        return results
    
class AblationStudyRunner:
    def __init__(self, model_pair_key: str, base_output_dir: str,
                 qlora_config: QLoRAConfig, distill_config: DistillationConfig,
                 refinement_config: RefinementConfig,
                 advanced_evaluator: Optional["AdvancedEvaluator"] = None,
                 n_advanced_samples: int = 5):
        self.model_pair = MODEL_PAIRS[model_pair_key]
        self.key = model_pair_key
        self.base_output_dir = base_output_dir
        self.ref_config = refinement_config
        self.adv = advanced_evaluator
        self.n_advanced_samples = n_advanced_samples
        self.results: Dict[str, Dict] = {}

    def _make_ref_cfg(self, ab: AblationConfig) -> RefinementConfig:
        cfg = deepcopy(self.ref_config)
        cfg.use_rl = ab.use_rl
        cfg.use_dynamic_ranks = ab.use_dynamic_ranks
        cfg.use_teacher_guidance = ab.use_teacher_guidance
        cfg.use_teacher_agreement_reward = ab.use_teacher_agreement_reward
        cfg.rl_reward_type = ab.rl_reward_type
        cfg.layer_selection_strategy = ab.layer_selection_strategy
        return cfg

    def run_single(self, ab: AblationConfig, eval_ds: Dataset,
                   teacher_hub_path: str, distilled_hub_path: str) -> Dict:
        print(f"\n{'='*70}\nAblation: {ab.name}\n  {ab.description}\n{'='*70}")
        result = {"name": ab.name, "description": ab.description}

        if ab.skip_refinement:
            model_path = teacher_hub_path if ab.skip_distillation else distilled_hub_path
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto")
            tok = AutoTokenizer.from_pretrained(model_path)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            ev = PipelineEvaluator(tok)
            result.update(ev.full_evaluation(model, eval_ds, model_name=ab.name,
                                             advanced_evaluator=self.adv,
                                             n_advanced_samples=self.n_advanced_samples))
            del model; clear_gpu_memory()
        else:
            cfg = self._make_ref_cfg(ab)
            trainer = RefinementTrainer(
                distilled_model_hf_path=distilled_hub_path,
                teacher_model_hf_path=teacher_hub_path,
                config=cfg
            )
            # Use a small training set for ablation speed
            tmp_full = MSKCaseStudyDataset(DATA_PATH, trainer.tokenizer)
            n = int(0.9 * len(tmp_full))
            ab_train, ab_eval = random_split(tmp_full, [n, len(tmp_full) - n])
            trainer.train(ab_train, ab_eval,
                          output_dir=f"{self.base_output_dir}/{ab.name}",
                          push_to_hub=False, run_name=ab.name)
            model = trainer.peft_model or trainer.distilled_model
            ev = PipelineEvaluator(trainer.tokenizer)
            result.update(ev.full_evaluation(model, eval_ds, model_name=ab.name,
                                             advanced_evaluator=self.adv,
                                             n_advanced_samples=self.n_advanced_samples))
            del trainer; clear_gpu_memory()

        self.results[ab.name] = result
        return result

    def run_all(self, eval_ds: Dataset, teacher_hub_path: str,
                distilled_hub_path: str, save_path: Optional[str] = None) -> pd.DataFrame:
        for ab in ABLATION_CONFIGS:
            try:
                self.run_single(ab, eval_ds, teacher_hub_path, distilled_hub_path)
            except Exception as e:
                print(f"  FAILED {ab.name}: {e}")
                self.results[ab.name] = {"name": ab.name, "error": str(e)}
            if save_path:
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                pd.DataFrame(list(self.results.values())).to_csv(save_path, index=False)
                print(f"  Saved interim results → {save_path}")
        return pd.DataFrame(list(self.results.values()))
    

# run ablation studies
all_ablation_dfs = []
ablation_runners = {}
_adv_eval = AdvancedEvaluator() 

for _key, _mp in MODEL_PAIRS.items():
    print(f"\n{'='*80}")
    print(f"Ablation study — {_key}  ({_mp.teacher_size} → {_mp.student_size})")
    print(f"{'='*80}")
    try:
        _tok = AutoTokenizer.from_pretrained(_mp.distilled_student)
        _tok.pad_token = _tok.eos_token
        _full = MSKCaseStudyDataset(DATA_PATH, _tok)
        _n = int(0.9 * len(_full))
        _, _eval_ds = random_split(_full, [_n, len(_full) - _n])
        del _tok, _full

        _runner = AblationStudyRunner(
            model_pair_key=_key,
            base_output_dir=f"./models/ablations/{_key}",
            qlora_config=qlora_cfg,
            distill_config=distill_cfg,
            refinement_config=refine_cfg,
            advanced_evaluator=_adv_eval,
        )
        ablation_runners[_key] = _runner

        _df = _runner.run_all(
            eval_ds=_eval_ds,
            teacher_hub_path=_mp.tuned_teacher,
            distilled_hub_path=_mp.distilled_student,
            save_path=f"./models/ablations/{_key}/results.csv",
        )
        _df.insert(0, "model_pair", _key)
        all_ablation_dfs.append(_df)
        os.makedirs("./models", exist_ok=True)
        pd.concat(all_ablation_dfs, ignore_index=True).to_csv("./models/ablation_results.csv", index=False)
        print(f"  Saved combined results → ./models/ablation_results.csv")
        clear_gpu_memory()

    except Exception as e:
        print(f"  FAILED {_key}: {e}")

ablation_df = pd.concat(all_ablation_dfs, ignore_index=True) if all_ablation_dfs else pd.DataFrame()

print("\nAll Ablation Results:")
print(ablation_df)


_LOWER_BETTER = {"perplexity", "p50_latency_ms", "p95_latency_ms", "mean_latency_ms",
                 "bias_score", "bias_detected"}

_METRIC_GROUPS = {
    "core":     ["perplexity", "rouge1", "rouge2", "rougeL", "bert_f1", "p50_latency_ms"],
    "advanced": ["geval_mean", "geval_clinical_coherence", "geval_diagnostic_accuracy",
                 "geval_treatment_completeness", "geval_patient_safety", "geval_clinical_relevance",
                 "gemba_score", "rts_score", "clinical_msk_score", "bias_score"],
}


def plot_ablation_results(df: pd.DataFrame,
                          save_path: str = "./models/ablation_results.png",
                          groups: Optional[List[str]] = None):
    all_metrics = _METRIC_GROUPS["core"] + _METRIC_GROUPS["advanced"]
    wanted = []
    for g in (groups or ["core", "advanced"]):
        wanted += _METRIC_GROUPS.get(g, [])
    metrics = [c for c in (wanted or all_metrics) if c in df.columns]
    if not metrics:
        print("No metrics to plot"); return

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    colors = sns.color_palette("husl", len(df))
    labels = [str(l).split("_")[0] for l in (df["name"].tolist() if "name" in df.columns else df.index)]

    for ax, metric in zip(axes, metrics):
        vals = pd.to_numeric(df[metric], errors="coerce")
        bars = ax.bar(range(len(df)), vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(metric.replace("_", "\n"), fontsize=8, fontweight="bold")
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        lower_better = metric in _LOWER_BETTER
        if vals.notna().any():
            best_idx = int(vals.idxmin() if lower_better else vals.idxmax())
            if not pd.isna(vals.iloc[best_idx]):
                bars[best_idx].set_edgecolor("gold")
                bars[best_idx].set_linewidth(2.5)

    fig.suptitle(f"Ablation Study —  (gold border = best)", fontsize=12, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved to {save_path}")


def plot_advanced_metrics_radar(df: pd.DataFrame,
                                save_path: str = "./models/ablation_radar.png"):
    """Radar chart comparing ablation configs across advanced metrics."""
    radar_metrics = [c for c in ["geval_mean", "gemba_score", "rts_score",
                                  "clinical_msk_score", "bert_f1", "rougeL"]
                     if c in df.columns]
    if len(radar_metrics) < 3:
        print("Not enough advanced metrics for radar chart"); return

    N = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = sns.color_palette("husl", len(df))

    for (_, row), color in zip(df.iterrows(), colors):
        vals = [pd.to_numeric(row.get(m, np.nan), errors="coerce") for m in radar_metrics]
        # Invert bias_score so higher = better on all axes
        vals = [1.0 - v if (radar_metrics[i] in _LOWER_BETTER and not np.isnan(v)) else v
                for i, v in enumerate(vals)]
        vals = [0.0 if np.isnan(v) else float(v) for v in vals]
        vals += vals[:1]
        label = str(row.get("name", "")).split("_")[0]
        ax.plot(angles, vals, color=color, linewidth=1.5, label=label)
        ax.fill(angles, vals, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_", "\n") for m in radar_metrics], size=8)
    ax.set_ylim(0, 1)
    ax.set_title(f"Advanced Metrics", pad=20, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved to {save_path}")


## Visualize results
def plot_layer_degradation(layer_scores: Dict[str, float],
                           rank_map: Optional[Dict[str, int]] = None,
                           save_path: str = "./models/layer_degradation.png"):
    if not layer_scores:
        print("No layer scores to plot"); return

    layers = sorted(layer_scores.keys(), key=lambda x: int(x.split("_")[1]))
    scores = [layer_scores[l] for l in layers]
    indices = [int(l.split("_")[1]) for l in layers]

    ncols = 2 if rank_map else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    cmap = plt.cm.RdYlGn_r
    axes[0].bar(indices, scores, color=[cmap(s) for s in scores])
    axes[0].axhline(0.5, color="orange", linestyle="--", linewidth=1, label="medium threshold")
    axes[0].axhline(0.8, color="red",    linestyle="--", linewidth=1, label="high threshold")
    axes[0].set_xlabel("Layer index"); axes[0].set_ylabel("Normalised KL degradation score")
    axes[0].set_title("Per-layer distillation degradation"); axes[0].legend(fontsize=8)

    if rank_map:
        rnks = [rank_map.get(l, 0) for l in layers]
        rank_colors = {32: "#d62728", 16: "#ff7f0e", 8: "#2ca02c", 4: "#1f77b4"}
        bar_colors = [rank_colors.get(r, "grey") for r in rnks]
        axes[1].bar(indices, rnks, color=bar_colors)
        axes[1].set_xlabel("Layer index"); axes[1].set_ylabel("LoRA rank assigned")
        axes[1].set_title("Dynamic LoRA rank assignment (novel criterion)")
        from matplotlib.patches import Patch
        axes[1].legend(handles=[Patch(color=c, label=f"r={r}") for r, c in rank_colors.items()],
                       fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def print_results_table(df: pd.DataFrame):
    core_cols  = [c for c in ["name", "perplexity", "rouge1", "rouge2", "rougeL",
                               "bert_f1", "p50_latency_ms"] if c in df.columns]
    adv_cols   = [c for c in ["geval_mean", "gemba_score", "rts_score",
                               "clinical_msk_score", "bias_score"] if c in df.columns]
    print("\n" + "="*80 + "\nCORE METRICS\n" + "="*80)
    print(df[core_cols].to_string(index=False, float_format="{:.4f}".format))
    if adv_cols:
        print("\n" + "="*80 + "\nADVANCED METRICS\n" + "="*80)
        print(df[["name"] + adv_cols].to_string(index=False, float_format="{:.4f}".format))


if "ablation_df" in dir() and not ablation_df.empty:
    plot_ablation_results(ablation_df, groups=["core"])
    plot_ablation_results(ablation_df, save_path="./models/ablation_advanced.png", groups=["advanced"])
    plot_advanced_metrics_radar(ablation_df)
    print_results_table(ablation_df)


# # Layer degradation heatmap + dynamic rank assignment
# _rcfg_viz = RefinementConfig(
#     lora_r=16, lora_alpha=32,
#     layer_selection_strategy="performance_based",
#     use_teacher_guidance=False, use_rl=False, device="cuda:0"
# )
# _viz_trainer = RefinementTrainer(
#     distilled_model_hf_path=model_pair.distilled_student,
#     teacher_model_hf_path=model_pair.tuned_teacher,
#     config=_rcfg_viz
# )
# _viz_tok = AutoTokenizer.from_pretrained(model_pair.distilled_student)
# _viz_tok.pad_token = _viz_tok.eos_token
# _viz_full = MSKCaseStudyDataset(DATA_PATH, _viz_tok)
# _n_viz = int(0.9 * len(_viz_full))
# _, _viz_eval = random_split(_viz_full, [_n_viz, len(_viz_full) - _n_viz])

# layer_scores = _viz_trainer.analyze_layer_performance(_viz_eval, num_samples=20)
# rank_map = _viz_trainer.select_dynamic_lora_ranks()
# plot_layer_degradation(layer_scores, rank_map)

# print("\nDynamic LoRA rank assignments:")
# for layer in sorted(layer_scores, key=lambda x: int(x.split("_")[1])):
#     score = layer_scores[layer]
#     rank = rank_map.get(layer, "?")
#     bar = chr(9608) * int(score * 30)
#     print(f"  {layer:12s}  score={score:.3f}  {bar:30s}  r={rank}")
"""Phase 3 — post-distillation refinement.

    total = CE(answer tokens)
          + w_kd  * T^2 * KL(teacher_T || student_T)            (use_teacher_guidance)
          + beta  * A * CE(answer tokens)                       (use_reward_weighting)
          + w_ted * mean_l MSE(P_l h_s^l, h_t^{m(l)})           (use_ted_alignment)

The reward term is reward-weighted likelihood, not RL: nothing is sampled, and
-(A * mean log p(gold)) == A * CE, so it rescales the per-sequence CE by the
normalised reward A (advantage). Normalising A keeps extra reward terms from
simply enlarging the gradient.

Per-layer LoRA ranks are allocated under a fixed budget (sum of ranks equals
lora_r * n_selected_layers); alpha = lora_alpha_ratio * r per module so alpha/r is constant.
"""
import json
import math
import os
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model

try:
    from .shared import (MODEL_PAIRS, RefinementConfig, count_trainable_params, load_causal_lm,
                         load_tokenizer, model_forward, set_seed, tokenize_example, tokenizers_compatible)
    from .tracking import ProxyGoldenTracker, proxy_golden_metrics
except ImportError:
    from shared import (MODEL_PAIRS, RefinementConfig, count_trainable_params, load_causal_lm,
                        load_tokenizer, model_forward, set_seed, tokenize_example, tokenizers_compatible)
    from tracking import ProxyGoldenTracker, proxy_golden_metrics


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def cfg_attr(cfg, attr, default=None):
    # Gemma3Config nests text attributes under .text_config
    if hasattr(cfg, attr):
        return getattr(cfg, attr)
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, attr):
        return getattr(cfg.text_config, attr)
    return default


def decoder_layers(model) -> Tuple[str, nn.ModuleList]:
    """The longest ModuleList named '...layers' — the decoder stack."""
    best_name, best = None, None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and name.split(".")[-1] == "layers":
            if best is None or len(mod) > len(best):
                best_name, best = name, mod
    if best is None:
        raise ValueError("Could not locate decoder layers")
    return best_name, best


def layer_map(n_student: int, n_teacher: int) -> List[int]:
    """Student layer l -> teacher layer at the same relative depth."""
    if n_student == 1:
        return [n_teacher - 1]
    return [int(round(l * (n_teacher - 1) / (n_student - 1))) for l in range(n_student)]


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA (Kornblith et al., 2019). Invariant to orthogonal transforms and
    isotropic scaling; needs no matching dimensions or coordinates."""
    x = x.double() - x.double().mean(0, keepdim=True)
    y = y.double() - y.double().mean(0, keepdim=True)
    hsic = (x.T @ y).norm() ** 2
    denom = (x.T @ x).norm() * (y.T @ y).norm()
    return float(hsic / denom) if denom > 0 else 0.0


def projection_r2(x: torch.Tensor, y: torch.Tensor, groups: np.ndarray, ridge: float = 1e-2) -> float:
    """Held-out R^2 of a ridge map student -> teacher, fitted on even samples and scored
    on odd ones (TED-style learned alignment used as a diagnostic)."""
    # D x D solve on CPU blows login-node CPU-time limits for large hidden sizes
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = torch.from_numpy(groups % 2 == 0).to(dev)
    te = ~tr
    if tr.sum() < 2 or te.sum() < 2:
        return float("nan")
    x, y = x.to(dev, torch.float64), y.to(dev, torch.float64)
    mx, my = x[tr].mean(0), y[tr].mean(0)
    xtr, ytr = x[tr] - mx, y[tr] - my
    gram = xtr.T @ xtr
    lam = ridge * torch.trace(gram) / gram.size(0)
    w = torch.linalg.solve(gram + lam * torch.eye(gram.size(0), dtype=gram.dtype, device=dev), xtr.T @ ytr)
    pred = (x[te] - mx) @ w + my
    ss_res = ((y[te] - pred) ** 2).sum()
    ss_tot = ((y[te] - y[te].mean(0)) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def minmax(values: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(values), np.nanmax(values)
    return np.zeros_like(values) if hi - lo < 1e-12 else (values - lo) / (hi - lo)


def proportional_allocation(weights: np.ndarray, total: int, lo: int, hi: int) -> np.ndarray:
    """Integer ranks proportional to weights, each in [lo, hi], summing to total."""
    n = len(weights)
    total = int(min(max(total, lo * n), hi * n))
    w = np.clip(np.asarray(weights, dtype=float), 0, None)
    fixed = np.zeros(n, dtype=bool)
    while True:
        rem = total - hi * fixed.sum() - lo * (~fixed).sum()
        ww = np.where(fixed, 0.0, w)
        share = ww / ww.sum() if ww.sum() > 0 else (~fixed) / max((~fixed).sum(), 1)
        alloc = np.where(fixed, hi, lo + rem * share)
        over = (~fixed) & (alloc > hi)
        if not over.any():
            break
        fixed |= over
    ranks = np.floor(alloc).astype(int)
    order = np.argsort(-(alloc - ranks))
    for i in order[: total - ranks.sum()]:
        if ranks[i] < hi:
            ranks[i] += 1
    return ranks


def adalora_orth_reg(model) -> torch.Tensor:
    # PEFT only adds AdaLoRA's orthogonality penalty when it computes the loss itself.
    reg, n = None, 0
    for name, p in model.named_parameters():
        if p.requires_grad and ("lora_A" in name or "lora_B" in name):
            cov = p @ p.T if "lora_A" in name else p.T @ p
            term = torch.norm(cov - torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype), p="fro")
            reg = term if reg is None else reg + term
            n += 1
    return reg / n if n else torch.tensor(0.0)


# --------------------------------------------------------------------------- #
# Trainer                                                                     #
# --------------------------------------------------------------------------- #

class RefinementTrainer:

    def __init__(self, distilled_model_path: str, teacher_model_path: Optional[str],
                 config: RefinementConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.student_path = distilled_model_path
        self.teacher_path = teacher_model_path
        self.student = None
        self.teacher = None
        self.peft_model = None
        self.layer_scores: Dict[int, float] = {}
        self.layer_scores_raw: Dict[int, float] = {}
        self.fisher_scores: Dict[int, float] = {}
        self.selected_layers: List[int] = []
        self.rank_map: Dict[int, int] = {}
        self.ted_proj: Optional[nn.ModuleDict] = None
        self.meta: Dict = {}
        self._reward_hist: List[float] = []
        self._r_mean, self._r_var = None, None
        self._cached_features = None  # reused across score_layers calls when _keep_teacher=True

        self.tokenizer = load_tokenizer(distilled_model_path)
        self.teacher_tokenizer = load_tokenizer(teacher_model_path) if teacher_model_path else None
        self.same_ids = bool(self.teacher_tokenizer) and tokenizers_compatible(self.tokenizer, self.teacher_tokenizer)
        if self.teacher_tokenizer and not self.same_ids:
            needs_alignment = (config.use_teacher_guidance or config.use_ted_alignment
                               or (config.use_reward_weighting and config.reward_type == "teacher_agreement"))
            if needs_alignment and not config.allow_cross_tokenizer_kd:
                raise ValueError(
                    "Teacher and student tokenizers differ: token-level KD, agreement rewards and "
                    "TED alignment compare unrelated token ids / positions. Use a shared-tokenizer "
                    "pair, or set allow_cross_tokenizer_kd=True to reproduce the legacy behaviour.")
            if config.use_ted_alignment:
                raise ValueError("TED alignment needs token-aligned sequences (shared tokenizer).")
            print("  WARNING: cross-tokenizer teacher — token-level terms are not well-defined.")

    # ------------------------------------------------------------------ loading
    def load_student(self):
        if self.student is None:
            print(f"Loading student: {self.student_path}")
            self.student = load_causal_lm(self.student_path, quant="none", device_map=None).to(self.device)

    def load_teacher(self):
        if self.teacher is None and self.teacher_path:
            print(f"Loading teacher: {self.teacher_path} ({self.config.teacher_quant})")
            self.teacher = load_causal_lm(self.teacher_path, quant=self.config.teacher_quant,
                                          device_map="auto", merge_adapter=False)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)

    def unload_teacher(self):
        if self.teacher is not None:
            del self.teacher
            self.teacher = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    def _t_dev(self):
        return next(self.teacher.parameters()).device

    def _teacher_inputs(self, sample):
        if self.same_ids:
            return sample["input_ids"], sample["attention_mask"], sample["labels"]
        enc = tokenize_example(self.teacher_tokenizer, sample["title"], sample["answer"],
                               self.config.max_seq_length)
        return enc["input_ids"], enc["attention_mask"], enc["labels"]

    # ------------------------------------------------------------ layer scores
    @torch.no_grad()
    def _collect_features(self, dataset: Dataset):
        """Hidden states on answer tokens. Token-level when ids align; otherwise one
        mean-pooled vector per example (CKA over examples — noisier, few rows)."""
        cfg = self.config
        n = min(cfg.degradation_num_samples, len(dataset))
        per = max(1, cfg.degradation_max_tokens // max(n, 1))
        _, layers = decoder_layers(self.student)
        n_s = len(layers)
        feats_s: List[List[torch.Tensor]] = [[] for _ in range(n_s)]
        feats_t: List[List[torch.Tensor]] = [[] for _ in range(n_s)]
        groups: List[int] = []
        self.student.eval()
        mapping = None
        for idx in range(n):
            s = dataset[idx]
            ids = s["input_ids"].unsqueeze(0).to(self.device)
            mask = s["attention_mask"].unsqueeze(0).to(self.device)
            ans = s["labels"] != -100
            if not ans.any():
                continue
            s_hs = model_forward(self.student, ids, mask, output_hidden_states=True).hidden_states
            t_ids, t_mask, t_labels = self._teacher_inputs(s)
            t_hs = model_forward(self.teacher, t_ids.unsqueeze(0).to(self._t_dev()),
                                 t_mask.unsqueeze(0).to(self._t_dev()), output_hidden_states=True).hidden_states
            if mapping is None:
                mapping = layer_map(n_s, len(t_hs) - 1)
            if self.same_ids:
                pos = ans.nonzero().squeeze(-1)
                if len(pos) > per:
                    pos = pos[torch.linspace(0, len(pos) - 1, per).long()]
                for l in range(n_s):
                    feats_s[l].append(s_hs[l + 1][0, pos.to(self.device)].float().cpu())
                    feats_t[l].append(t_hs[mapping[l] + 1][0, pos.to(self._t_dev())].float().cpu())
                groups += [idx] * len(pos)
            else:
                t_ans = t_labels != -100
                for l in range(n_s):
                    feats_s[l].append(s_hs[l + 1][0][ans.to(self.device)].float().mean(0, keepdim=True).cpu())
                    feats_t[l].append(t_hs[mapping[l] + 1][0][t_ans.to(self._t_dev())].float().mean(0, keepdim=True).cpu())
                groups.append(idx)
        return ([torch.cat(f) for f in feats_s], [torch.cat(f) for f in feats_t],
                np.array(groups), mapping)

    @torch.no_grad()
    def _legacy_output_kl(self, dataset: Dataset, n_s: int) -> np.ndarray:
        # V1/V2 behaviour, kept only to reproduce old runs: one output-level KL scaled by a depth ramp.
        kls = []
        for idx in range(min(self.config.degradation_num_samples, len(dataset))):
            s = dataset[idx]
            t_ids, t_mask, _ = self._teacher_inputs(s)
            s_logits = model_forward(self.student, s["input_ids"].unsqueeze(0).to(self.device),
                                     s["attention_mask"].unsqueeze(0).to(self.device)).logits
            t_logits = model_forward(self.teacher, t_ids.unsqueeze(0).to(self._t_dev()),
                                     t_mask.unsqueeze(0).to(self._t_dev())).logits.to(self.device)
            v = min(s_logits.size(-1), t_logits.size(-1))
            kls.append(F.kl_div(F.log_softmax(s_logits[..., :v].float(), -1),
                                F.softmax(t_logits[..., :v].float(), -1), reduction="batchmean").item())
        return float(np.mean(kls)) * np.linspace(0.3, 1.0, n_s)

    def score_layers(self, dataset: Dataset, _keep_teacher: bool = False) -> Dict[int, float]:
        """Degradation score per student layer, min-max normalised to [0, 1].
        cka: 1 - linear CKA(student layer, depth-matched teacher layer)
        projection: 1 - held-out R^2 of a learned linear map student -> teacher
        depth: linear ramp (control: 'adapt later layers more').
        _keep_teacher: skip unload_teacher after scoring (lets caller reuse across metrics)."""
        cfg = self.config
        self.load_student()
        n_s = len(decoder_layers(self.student)[1])
        metric = cfg.degradation_metric
        print(f"Scoring layers ({metric}) on {min(cfg.degradation_num_samples, len(dataset))} calibration samples")
        if metric == "depth":
            raw = np.linspace(0.3, 1.0, n_s)
        else:
            self.load_teacher()
            if metric == "legacy_output_kl":
                raw = self._legacy_output_kl(dataset, n_s)
            else:
                if self._cached_features is None:
                    self._cached_features = self._collect_features(dataset)
                fs, ft, groups, mapping = self._cached_features
                self.meta["layer_map"] = mapping
                self.meta["degradation_rows"] = int(fs[0].size(0))
                if not self.same_ids:
                    print("  Cross-tokenizer: using example-level features "
                          f"({fs[0].size(0)} rows) — treat scores as low-confidence.")
                if metric == "cka":
                    raw = np.array([1.0 - linear_cka(fs[l], ft[l]) for l in range(n_s)])
                elif metric == "projection":
                    raw = np.array([1.0 - np.clip(projection_r2(fs[l], ft[l], groups), 0, 1)
                                    for l in range(n_s)])
                else:
                    raise ValueError(f"Unknown degradation_metric {metric}")
            if not _keep_teacher:
                self.unload_teacher()
                self._cached_features = None
        raw = np.asarray(raw, dtype=float)
        if np.isnan(raw).any():
            print(f"  WARNING: {int(np.isnan(raw).sum())} layer scores are NaN (too few rows?) — set to the mean")
            raw = np.where(np.isnan(raw), np.nanmean(raw) if not np.isnan(raw).all() else 0.0, raw)
        if np.ptp(raw) < 1e-12:
            print("  WARNING: all layer scores are equal — score-based allocation reduces to uniform")
        norm = minmax(raw)
        self.layer_scores_raw = {l: float(raw[l]) for l in range(n_s)}
        self.layer_scores = {l: float(norm[l]) for l in range(n_s)}
        top = sorted(self.layer_scores, key=self.layer_scores.get, reverse=True)[:5]
        print(f"  Top-5 layers by score: {top}")
        return self.layer_scores

    def fisher_layer_scores(self, dataset: Dataset, num_samples: int = 16) -> Dict[int, float]:
        """Empirical Fisher (squared CE gradients) of each layer's target modules."""
        self.load_student()
        prefix, layers = decoder_layers(self.student)
        names = {l: self._module_names(prefix, l) for l in range(len(layers))}
        params = dict(self.student.named_parameters())
        by_layer = {l: [params[f"{n}.weight"] for n in ns] for l, ns in names.items()}
        for p in self.student.parameters():
            p.requires_grad_(False)
        for ps in by_layer.values():
            for p in ps:
                p.requires_grad_(True)
        acc = np.zeros(len(layers))
        self.student.eval()
        for idx in range(min(num_samples, len(dataset))):
            s = dataset[idx]
            out = model_forward(self.student, s["input_ids"].unsqueeze(0).to(self.device),
                                s["attention_mask"].unsqueeze(0).to(self.device),
                                labels=s["labels"].unsqueeze(0).to(self.device))
            out.loss.backward()
            for l, ps in by_layer.items():
                acc[l] += sum(float(p.grad.float().pow(2).sum()) for p in ps if p.grad is not None)
            self.student.zero_grad(set_to_none=True)
        for p in self.student.parameters():
            p.requires_grad_(False)
        self.fisher_scores = {l: float(v) for l, v in enumerate(minmax(acc))}
        return self.fisher_scores

    # --------------------------------------------------------- adapter layout
    def _module_types(self) -> List[str]:
        strategy = self.config.layer_selection_strategy
        if strategy == "attention_only":
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        if strategy == "mlp_only":
            return ["gate_proj", "up_proj", "down_proj"]
        return list(self.config.module_types)

    def _module_names(self, prefix: str, layer: int) -> List[str]:
        types = set(self._module_types())
        head = f"{prefix}.{layer}."
        return [n for n, m in self.student.named_modules()
                if n.startswith(head) and n.split(".")[-1] in types and isinstance(m, nn.Linear)]

    def select_layers(self, n_layers: int) -> List[int]:
        cfg = self.config
        strategy = cfg.layer_selection_strategy
        k = cfg.num_layers_to_refine or max(1, n_layers // 2)
        if strategy in ("all", "attention_only", "mlp_only"):
            return list(range(n_layers))
        if strategy == "explicit":
            return sorted(cfg.layer_indices or [])
        if strategy == "last":
            return list(range(n_layers - k, n_layers))
        if strategy == "random":
            return sorted(np.random.default_rng(cfg.seed).choice(n_layers, size=k, replace=False).tolist())
        if strategy in ("performance_based", "bottom"):
            ranked = sorted(self.layer_scores, key=self.layer_scores.get, reverse=True)
            return sorted(ranked[:k] if strategy == "performance_based" else ranked[-k:])
        if strategy == "fisher":
            return sorted(sorted(self.fisher_scores, key=self.fisher_scores.get, reverse=True)[:k])
        raise ValueError(f"Unknown layer_selection_strategy {strategy}")

    def allocate_ranks(self, layers: List[int]) -> Dict[int, int]:
        cfg = self.config
        mode = cfg.rank_allocation
        total = cfg.lora_r * len(layers)
        if mode in ("uniform", "adalora"):
            ranks = [cfg.lora_r] * len(layers)
        elif mode == "degradation":
            ranks = proportional_allocation(np.array([self.layer_scores[l] for l in layers]),
                                            total, cfg.lora_r_min, cfg.lora_r_max)
        elif mode == "random":
            w = np.random.default_rng(cfg.seed + 1000).random(len(layers))
            ranks = proportional_allocation(w, total, cfg.lora_r_min, cfg.lora_r_max)
        elif mode == "legacy_median":
            # V2 behaviour: threshold ranks then use their median uniformly
            th = [32 if s >= 0.8 else 16 if s >= 0.5 else 8 if s >= 0.3 else 4
                  for s in (self.layer_scores[l] for l in layers)]
            ranks = [int(np.median(th))] * len(layers)
        else:
            raise ValueError(f"Unknown rank_allocation {mode}")
        return {l: int(r) for l, r in zip(layers, ranks)}

    def apply_adapters(self, total_steps: int):
        cfg = self.config
        prefix, layers = decoder_layers(self.student)
        self.selected_layers = self.select_layers(len(layers))
        names = {l: self._module_names(prefix, l) for l in self.selected_layers}
        missing = [l for l, ns in names.items() if not ns]
        if missing:
            raise ValueError(f"No {self._module_types()} modules found in layers {missing}")
        self.rank_map = self.allocate_ranks(self.selected_layers)
        targets = [n for l in self.selected_layers for n in names[l]]
        ratio = cfg.lora_alpha_ratio

        if cfg.rank_allocation == "adalora":
            from peft import AdaLoraConfig
            peft_cfg = AdaLoraConfig(
                init_r=max(cfg.lora_r, int(round(cfg.lora_r * cfg.adalora_init_r_ratio))),
                target_r=cfg.lora_r, lora_alpha=int(ratio * cfg.lora_r),
                target_modules=targets, lora_dropout=cfg.lora_dropout,
                total_step=total_steps, tinit=int(0.1 * total_steps), tfinal=int(0.2 * total_steps),
                deltaT=max(1, total_steps // 50), orth_reg_weight=cfg.adalora_orth_reg_weight,
                task_type=TaskType.CAUSAL_LM)
        else:
            peft_cfg = LoraConfig(
                r=cfg.lora_r, lora_alpha=int(ratio * cfg.lora_r), target_modules=targets,
                rank_pattern={n: self.rank_map[l] for l in self.selected_layers for n in names[l]},
                alpha_pattern={n: int(ratio * self.rank_map[l]) for l in self.selected_layers for n in names[l]},
                lora_dropout=cfg.lora_dropout, bias="none", task_type=TaskType.CAUSAL_LM)

        modules = dict(self.student.named_modules())
        budget = sum(cfg.lora_r * (modules[n].in_features + modules[n].out_features) for n in targets)
        self.peft_model = get_peft_model(self.student, peft_cfg)
        self.meta.update({
            "selected_layers": self.selected_layers,
            "rank_map": self.rank_map,
            "trainable_params": count_trainable_params(self.peft_model),
            "budget_params": budget,
            "target_modules": targets,
        })
        print(f"  Layers {self.selected_layers} | ranks {self.rank_map} | "
              f"trainable={self.meta['trainable_params']:,} (budget {budget:,})")

    def _build_ted(self):
        s_dim = cfg_attr(self.student.config, "hidden_size")
        t_dim = cfg_attr(self.teacher.config, "hidden_size")
        self.ted_proj = nn.ModuleDict({str(l): nn.Linear(s_dim, t_dim, bias=False)
                                       for l in self.selected_layers}).to(self.device)
        n_t = cfg_attr(self.teacher.config, "num_hidden_layers")
        self._ted_map = layer_map(len(decoder_layers(self.student)[1]), n_t)

    # ------------------------------------------------------------------- loss
    def _reward_weights(self) -> Tuple[float, float]:
        key = self.config.model_pair_key
        if key and key in MODEL_PAIRS:
            return MODEL_PAIRS[key].agreement_weight, MODEL_PAIRS[key].kl_reward_weight
        return self.config.agreement_weight, self.config.kl_reward_weight

    def _normalise_reward(self, r: float) -> float:
        cfg = self.config
        if cfg.reward_normalization == "none":
            return r
        if self._r_mean is None:
            self._reward_hist.append(r)
            if len(self._reward_hist) >= 10:
                self._r_mean = float(np.mean(self._reward_hist))
                self._r_var = float(np.var(self._reward_hist)) + 1e-6
            return 0.0
        adv = float(np.clip((r - self._r_mean) / math.sqrt(self._r_var), -3, 3))
        delta = r - self._r_mean
        self._r_mean += (1 - cfg.reward_ema) * delta
        self._r_var = cfg.reward_ema * self._r_var + (1 - cfg.reward_ema) * delta ** 2
        return adv

    def _step_loss(self, model, sample) -> Optional[Dict[str, torch.Tensor]]:
        cfg = self.config
        ids = sample["input_ids"].unsqueeze(0).to(self.device)
        mask = sample["attention_mask"].unsqueeze(0).to(self.device)
        labels = sample["labels"].unsqueeze(0).to(self.device)
        ans = labels[:, 1:] != -100
        if not ans.any():
            return None
        need_hidden = cfg.use_ted_alignment
        s_out = model_forward(model, ids, mask, output_hidden_states=need_hidden)
        s_logits = s_out.logits[:, :-1][ans]
        gold = labels[:, 1:][ans]
        task = F.cross_entropy(s_logits.float(), gold)
        zero = torch.zeros((), device=self.device)
        out = {"task": task, "kd": zero, "reward_term": zero, "ted": zero, "reward": zero}

        use_agreement = cfg.use_reward_weighting and cfg.reward_type == "teacher_agreement"
        t_logits = t_out = None
        if self.teacher is not None and (cfg.use_teacher_guidance or use_agreement or cfg.use_ted_alignment):
            t_ids, t_mask, t_labels = self._teacher_inputs(sample)
            with torch.no_grad():
                t_out = model_forward(self.teacher, t_ids.unsqueeze(0).to(self._t_dev()),
                                      t_mask.unsqueeze(0).to(self._t_dev()), output_hidden_states=need_hidden)
                t_ans = (t_labels.unsqueeze(0)[:, 1:] != -100).to(self._t_dev())
                t_logits = t_out.logits[:, :-1][t_ans].to(self.device).float()
            if t_logits.size(0) != s_logits.size(0):   # legacy cross-tokenizer mode only
                m = min(t_logits.size(0), s_logits.size(0))
                t_logits, s_logits, gold = t_logits[:m], s_logits[:m], gold[:m]
            v = min(s_logits.size(-1), t_logits.size(-1))
            t_logits = t_logits[:, :v]

        if cfg.use_teacher_guidance and t_logits is not None:
            T = cfg.kd_temperature
            out["kd"] = F.kl_div(F.log_softmax(s_logits[:, :v].float() / T, -1),
                                 F.log_softmax(t_logits / T, -1), log_target=True,
                                 reduction="batchmean") * T * T

        if cfg.use_reward_weighting:
            with torch.no_grad():
                preds = s_logits.argmax(-1)
                r = (preds == gold).float().mean()
                if use_agreement and t_logits is not None:
                    aw, kw = self._reward_weights()
                    agree = (preds == t_logits.argmax(-1)).float().mean()
                    kl_pen = F.kl_div(F.log_softmax(s_logits[:, :v].float(), -1),
                                      F.log_softmax(t_logits, -1), log_target=True, reduction="batchmean")
                    r = r + aw * agree - kw * kl_pen
                adv = self._normalise_reward(float(r))
            out["reward"] = torch.tensor(float(r), device=self.device)
            # -(A * mean log p(gold)) == A * CE: a per-sequence reweighting of the task loss
            out["reward_term"] = adv * F.cross_entropy(s_logits.float(), gold)

        if cfg.use_ted_alignment and t_out is not None:
            pos = labels[0] != -100
            terms = []
            for l in self.selected_layers:
                hs = s_out.hidden_states[l + 1][0][pos].float()
                ht = t_out.hidden_states[self._ted_map[l] + 1][0][pos.to(self._t_dev())].to(self.device).float()
                proj = self.ted_proj[str(l)](hs)
                terms.append(F.mse_loss(F.layer_norm(proj, proj.shape[-1:]), F.layer_norm(ht, ht.shape[-1:])))
            out["ted"] = torch.stack(terms).mean()

        total = (task + cfg.teacher_guidance_weight * out["kd"] + cfg.reward_beta * out["reward_term"]
                 + cfg.ted_weight * out["ted"])
        if cfg.rank_allocation == "adalora":
            total = total + cfg.adalora_orth_reg_weight * adalora_orth_reg(model).to(self.device)
        out["total"] = total
        return out

    # ------------------------------------------------------------------ train
    def _needs_teacher(self) -> bool:
        cfg = self.config
        return bool(self.teacher_path) and (
            cfg.use_teacher_guidance or cfg.use_ted_alignment or cfg.track_every > 0
            or (cfg.use_reward_weighting and cfg.reward_type == "teacher_agreement"))

    def train(self, train_dataset: Dataset, calib_dataset: Optional[Dataset] = None,
              output_dir: str = "./refined", golden_fn=None) -> Dict:
        cfg = self.config
        set_seed(cfg.seed)
        self.load_student()

        needs_scores = cfg.layer_selection_strategy in ("performance_based", "bottom") \
            or cfg.rank_allocation == "degradation"
        if needs_scores:
            if cfg.layer_scores:
                self.layer_scores = {int(k): float(v) for k, v in cfg.layer_scores.items()}
            elif calib_dataset is None:
                raise ValueError("Layer scoring needs a calibration split")
            else:
                self.score_layers(calib_dataset)
        if cfg.layer_selection_strategy == "fisher":
            self.fisher_layer_scores(calib_dataset)

        accum = max(1, cfg.gradient_accumulation_steps)
        total_steps = math.ceil(len(train_dataset) * cfg.num_epochs / accum)
        if cfg.max_steps:
            total_steps = min(total_steps, cfg.max_steps)
        self.apply_adapters(total_steps)
        model = self.peft_model

        if self._needs_teacher():
            self.load_teacher()
        params = [p for p in model.parameters() if p.requires_grad]
        if cfg.use_ted_alignment:
            self._build_ted()
            params += list(self.ted_proj.parameters())
        optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, min(cfg.warmup_steps, total_steps // 10 + 1),
                                                    total_steps)

        tracker = None
        if cfg.track_every > 0 and calib_dataset is not None:
            tracker = ProxyGoldenTracker(calib_dataset, self.device, cfg.track_num_samples,
                                         self.same_ids, golden_fn)
            tracker.record(0, model, self.teacher)

        history, global_step, micro = [], 0, 0
        gen = torch.Generator().manual_seed(cfg.seed)
        model.train()
        done = False
        for epoch in range(cfg.num_epochs):
            running = {k: 0.0 for k in ("total", "task", "kd", "reward_term", "ted", "reward")}
            n_items = 0
            for idx in torch.randperm(len(train_dataset), generator=gen).tolist():
                losses = self._step_loss(model, train_dataset[idx])
                if losses is None:
                    continue
                (losses["total"] / accum).backward()
                micro += 1
                n_items += 1
                for k in running:
                    running[k] += float(losses[k].detach())
                if micro % accum:
                    continue
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                if cfg.rank_allocation == "adalora":
                    model.base_model.update_and_allocate(global_step)
                optimizer.zero_grad()
                global_step += 1
                if global_step % cfg.logging_steps == 0:
                    print(f"Ep {epoch + 1} step {global_step}/{total_steps} | " +
                          " ".join(f"{k}={float(v):.4f}" for k, v in losses.items() if k != "total"))
                if tracker and global_step % cfg.track_every == 0:
                    tracker.record(global_step, model, self.teacher)
                if global_step >= total_steps:
                    done = True
                    break
            row = {"epoch": epoch + 1, "step": global_step,
                   **{f"train_{k}": v / max(n_items, 1) for k, v in running.items()}}
            if calib_dataset is not None:
                row["calib_nll"] = proxy_golden_metrics(model, None, calib_dataset, self.device,
                                                        cfg.num_eval_samples)["golden_nll"]
            history.append(row)
            print(f"  Epoch {epoch + 1}: " + " ".join(f"{k}={v:.4f}" for k, v in row.items()
                                                    if isinstance(v, float)))
            if done:
                break

        if tracker and (not tracker.records or tracker.records[-1]["step"] != global_step):
            tracker.record(global_step, model, self.teacher)
        self.meta.update({
            "config": asdict(cfg), "history": history, "total_steps": global_step,
            "same_tokenizer": self.same_ids,
            "layer_scores": self.layer_scores, "layer_scores_raw": self.layer_scores_raw,
            "fisher_scores": self.fisher_scores,
            "tracking": tracker.records if tracker else [],
            "teacher_hacking": tracker.summary() if tracker else None,
        })
        self.save(output_dir)
        self.unload_teacher()
        return self.meta

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        (self.peft_model or self.student).save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        with open(os.path.join(output_dir, "refinement_meta.json"), "w") as f:
            json.dump(self.meta, f, indent=2, default=str)
        print(f"Saved to {output_dir}")

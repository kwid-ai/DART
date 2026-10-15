import hashlib
import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DATA_PATH = "./physio_use_cases.csv"
SPLITS_PATH = "./splits.json"


# --------------------------------------------------------------------------- #
# Configs                                                                     #
# --------------------------------------------------------------------------- #

@dataclass
class ModelPair:
    teacher_name: str
    student_name: str
    teacher_size: str
    student_size: str
    tuned_teacher: str = ""
    distilled_student: str = ""
    # Student distilled from the *untuned* teacher (A1b control); built on demand if missing
    distilled_student_base_teacher: str = ""
    # Per-family teacher-agreement reward weights (overrides global RefinementConfig values)
    agreement_weight: float = 0.3   # lambda1 — teacher-student agreement bonus
    kl_reward_weight: float = 0.2   # lambda2 — KL divergence penalty


@dataclass
class QLoRAConfig:
    lora_r: int = 4
    lora_alpha: int = 8
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.05
    learning_rate: float = 1e-4
    epochs: int = 3
    batch_size: int = 1
    use_4bit: bool = True
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 1024  # median reference is ~790 words; 256 truncated most targets
    teacher_model: str = ""
    student_model: str = ""
    seed: int = 42


@dataclass
class DistillationConfig:
    temperature: float = 2.0
    alpha: float = 0.5
    load_in_8bit: bool = True
    llm_int8_threshold: float = 6.0
    learning_rate: float = 5e-5
    epochs: int = 1
    batch_size: int = 1
    use_4bit: bool = True
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 50
    use_student_lora: bool = True
    student_lora_r: int = 8
    student_lora_alpha: int = 16
    student_lora_target_modules: Optional[List[str]] = None
    top_k_logits: int = 50
    max_seq_length: int = 1024
    seed: int = 42
    # Logit-level KD needs token ids that mean the same thing in both vocabularies.
    # Cross-tokenizer pairs are rejected unless this is set (legacy V1/V2 behaviour).
    allow_cross_tokenizer_kd: bool = False
    # Proxy (teacher imitation) vs golden (held-out gold NLL) tracking — 0 disables
    track_every: int = 0
    track_num_samples: int = 16


@dataclass
class RefinementConfig:
    hf_token: Optional[str] = None
    seed: int = 0
    max_seq_length: int = 1024
    # Training
    learning_rate: float = 2e-4
    num_epochs: int = 3
    max_steps: Optional[int] = None          # caps optimizer steps (per-layer probes)
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    device: str = "cuda:0"
    logging_steps: int = 10
    # LoRA / parameter budget. lora_r is the *budget-equivalent uniform rank*:
    # every allocation scheme spends the same total rank across the selected layers.
    lora_r: int = 16
    lora_alpha_ratio: float = 2.0            # alpha = ratio * r per module, keeps alpha/r fixed
    lora_dropout: float = 0.05
    lora_r_min: int = 4
    lora_r_max: int = 64
    module_types: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    rank_allocation: str = "uniform"         # uniform | degradation | random | adalora | legacy_median
    adalora_init_r_ratio: float = 1.5
    adalora_orth_reg_weight: float = 0.1
    # Layer scoring ("degradation")
    degradation_metric: str = "cka"          # cka | projection | depth | legacy_output_kl
    degradation_num_samples: int = 32
    degradation_max_tokens: int = 4096
    layer_scores: Optional[Dict[int, float]] = None  # precomputed normalised scores (skips scoring)
    # Layer selection
    layer_selection_strategy: str = "all"    # all | performance_based | bottom | random | last | fisher | explicit
    num_layers_to_refine: Optional[int] = None
    layer_indices: Optional[List[int]] = None
    # Objectives
    use_teacher_guidance: bool = True        # KD term during refinement
    teacher_guidance_weight: float = 0.3
    kd_temperature: float = 2.0
    # Reward-weighted likelihood (formerly "RL"): no sampling — the gold-sequence
    # log-likelihood is scaled by a normalised per-sequence reward.
    use_reward_weighting: bool = True
    reward_type: str = "teacher_agreement"   # task_accuracy | teacher_agreement
    reward_beta: float = 0.1
    reward_normalization: str = "advantage"  # advantage | none
    reward_ema: float = 0.9
    agreement_weight: float = 0.3   # lambda1 (global fallback; per-family weights in ModelPair)
    kl_reward_weight: float = 0.2   # lambda2 (global fallback; per-family weights in ModelPair)
    model_pair_key: str = ""        # set to the active MODEL_PAIRS key at runtime
    # TED-lite: learned per-layer projections aligning student to teacher hidden states
    use_ted_alignment: bool = False
    ted_weight: float = 1.0
    # Teacher
    teacher_quant: str = "4bit"              # 4bit | 8bit | none
    allow_cross_tokenizer_kd: bool = False
    # Proxy vs golden tracking (teacher-hacking analysis) — 0 disables
    track_every: int = 0
    track_num_samples: int = 16
    num_eval_samples: int = 50


@dataclass
class AblationConfig:
    name: str
    description: str
    # Which checkpoint the row starts from (or is evaluated as)
    source: str = "distilled"   # tuned_teacher | base_teacher | distilled | distilled_base_teacher | base_student
    method: str = "refine"      # none | refine | gkd
    overrides: Dict[str, Any] = field(default_factory=dict)  # applied to RefinementConfig
    # B1: total epochs = Phase-2 epochs + Phase-3 epochs so SFT sees the same number of passes
    match_pipeline_epochs: bool = False


_NO_TEACHER = dict(use_teacher_guidance=False, use_reward_weighting=False, use_ted_alignment=False)
_R0 = dict(_NO_TEACHER, layer_selection_strategy="all", rank_allocation="uniform")
_R1 = dict(_R0, use_teacher_guidance=True)
_R2 = dict(_R1, use_reward_weighting=True, reward_type="task_accuracy")
_R3 = dict(_R2, reward_type="teacher_agreement")
_R4 = dict(_R3, rank_allocation="degradation")

# Single-factor ladder: each R-row adds exactly one component to the row before it.
# Every refinement row uses the same layers, rank budget, steps and seeds.
ABLATION_CONFIGS = [
    # AblationConfig("A0_tuned_teacher", "QLoRA-tuned teacher, evaluated directly",
    #                source="tuned_teacher", method="none"),
    # AblationConfig("A0b_base_teacher", "Untuned base teacher, evaluated directly",
    #                source="base_teacher", method="none"),
    # AblationConfig("A1_distill_only", "Phase-2 student distilled from the tuned teacher",
    #                source="distilled", method="none"),
    # AblationConfig("A1b_distill_from_base_teacher", "Phase-2 student distilled from the untuned teacher",
    #                source="distilled_base_teacher", method="none"),
    # AblationConfig("B1_student_sft", "Base student, LoRA SFT only (no teacher at any point), epoch- and budget-matched",
    #                source="base_student", overrides=_R0, match_pipeline_epochs=True),
    # AblationConfig("R0_distill_plus_sft", "Distilled student + Phase-3 CE only (extra-training control)",
    #                overrides=_R0),
    # AblationConfig("R1_kd_guidance", "R0 + teacher KD guidance", overrides=_R1),
    # AblationConfig("R2_reward_accuracy", "R1 + reward-weighted likelihood, task-accuracy reward", overrides=_R2),
    # AblationConfig("R3_reward_agreement", "R2 with agreement + KL-penalty reward terms", overrides=_R3),
    # AblationConfig("R4_degradation_ranks", "R3 + degradation-based rank allocation (same budget)", overrides=_R4),
    # AblationConfig("R4r_random_ranks", "R3 + random rank allocation (same budget)",
    #               overrides=dict(_R3, rank_allocation="random")),
    AblationConfig("C_adalora", "R3 with AdaLoRA importance-based allocation (same final budget)",
                  overrides=dict(_R3, rank_allocation="adalora")),
    AblationConfig("C_ted_lite", "R1 + TED-style learned per-layer hidden-state alignment",
                   overrides=dict(_R1, use_ted_alignment=True)),
    # AblationConfig("C_gkd", "GKD (on-policy, generalised JSD) from the distilled student, same budget",
    #                method="gkd", overrides=_R0),
]

# Pairs with a shared tokenizer, where logit-level KD is well-defined
FOCUS_PAIRS = ["qwen_3b_0.5b", "medgemma3", "llama_3b_1b"]


def _pair(key: str, teacher: str, student: str, t_size: str, s_size: str, **kw) -> ModelPair:
    return ModelPair(
        teacher, student, t_size, s_size,
        tuned_teacher=f"./models/{key}/phase1_teacher",
        distilled_student=f"./models/{key}/phase2_student",
        distilled_student_base_teacher=f"./models/{key}/phase2_student_base_teacher",
        **kw,
    )


MODEL_PAIRS = {
    # Llama family — shared tokenizer
    "llama_3b_1b": _pair("llama_3b_1b", "meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-1B-Instruct",
                         "3B", "1B", agreement_weight=0.30, kl_reward_weight=0.20),
    # Cross-tokenizer pair — logit-level KD is not well-defined (see allow_cross_tokenizer_kd)
    "llama_tiny": _pair("llama_tiny", "meta-llama/Llama-3.2-3B-Instruct", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        "3B", "1B", agreement_weight=0.25, kl_reward_weight=0.15),
    # Qwen family — shared tokenizer
    "qwen_3b_0.5b": _pair("qwen_3b_0.5b", "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct",
                          "3B", "0.5B", agreement_weight=0.25, kl_reward_weight=0.15),
    "smollm2": _pair("smollm2", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct",
                     "1.7B", "360M", agreement_weight=0.20, kl_reward_weight=0.15),
    # Medical Gemma family — shared Gemma-3 tokenizer
    "medgemma3": _pair("medgemma3", "google/medgemma-1.5-4b-it", "google/gemma-3-270m-it",
                       "4B", "270m", agreement_weight=0.40, kl_reward_weight=0.25),
    "medgemma_text": _pair("medgemma_text", "google/medgemma-27b-text-it", "google/medgemma-1.5-4b-it",
                           "27B", "4B", agreement_weight=0.40, kl_reward_weight=0.25),
    "gemma3": _pair("gemma3", "google/gemma-3-1b-pt", "google/gemma-3-270m-it",
                    "1B", "270m", agreement_weight=0.35, kl_reward_weight=0.20),
    # Cross-tokenizer pairs
    "mistral_tiny": _pair("mistral_tiny", "mistralai/Mistral-7B-v0.1", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                          "7B", "1.1B", agreement_weight=0.25, kl_reward_weight=0.15),
    "meditron_phi4mini": _pair("meditron_phi4mini", "epfl-llm/meditron-7b", "meta-llama/Llama-3.2-1B",
                               "7B", "1B", agreement_weight=0.40, kl_reward_weight=0.25),
    "qwq_qwen1.5b": _pair("qwq_qwen1.5b", "Qwen/QwQ-32B", "Qwen/Qwen2.5-1.5B-Instruct",
                          "32B", "1.5B", agreement_weight=0.30, kl_reward_weight=0.20),
}


# --------------------------------------------------------------------------- #
# Reproducibility                                                             #
# --------------------------------------------------------------------------- #

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import transformers
        transformers.set_seed(seed)
    except ImportError:
        pass


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #

PROMPT_TEMPLATE = (
    "You are a musculoskeletal (MSK) physiotherapy expert. "
    "Write a detailed physiotherapy case study for the following condition.\n\n"
    "Condition: {title}\n\n"
    "Case Study:"
)


def load_cases(data_path: str = DATA_PATH) -> List[Dict]:
    """Rows with a non-empty use case. case_id is the row index in the CSV, so it is
    stable across runs and is what splits.json stores."""
    df = pd.read_csv(data_path, encoding="latin-1")
    df.columns = [c.strip().lower() for c in df.columns]
    if "use case" in df.columns:
        df = df.rename(columns={"use case": "use_case"})
    cases = []
    for i, row in df.iterrows():
        answer = row.get("use_case")
        if not isinstance(answer, str) or not answer.strip():
            continue
        title = row.get("title") if isinstance(row.get("title"), str) else ""
        cases.append({"case_id": int(i), "title": title.strip(), "use_case": answer.strip()})
    return cases


def _condition_key(title: str) -> str:
    return re.sub(r"\s+", " ", title.lower()).strip()


def make_group_splits(data_path: str = DATA_PATH, out_path: str = SPLITS_PATH,
                      fractions: Tuple[float, float, float] = (0.7, 0.1, 0.2),
                      seed: int = 13, overwrite: bool = False) -> Dict:
    """Train/calibration/test split grouped by condition title, so no condition
    appears on both sides. Written once and reused by every phase and evaluation."""
    if os.path.exists(out_path) and not overwrite:
        return load_splits(out_path, data_path)
    cases = load_cases(data_path)
    groups: Dict[str, List[int]] = {}
    for c in cases:
        groups.setdefault(_condition_key(c["title"]), []).append(c["case_id"])
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    n_train = int(round(fractions[0] * n))
    n_calib = int(round(fractions[1] * n))
    names = {"train": keys[:n_train], "calibration": keys[n_train:n_train + n_calib],
             "test": keys[n_train + n_calib:]}
    splits = {
        "seed": seed, "fractions": list(fractions), "data_path": data_path,
        "data_sha256": file_sha256(data_path), "n_rows": len(cases), "n_conditions": n,
        "case_ids": {k: sorted(i for g in v for i in groups[g]) for k, v in names.items()},
        "conditions": {k: sorted(v) for k, v in names.items()},
    }
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Wrote {out_path}: " + ", ".join(
        f"{k}={len(v)} rows/{len(names[k])} conditions" for k, v in splits["case_ids"].items()))
    return splits


def load_splits(path: str = SPLITS_PATH, data_path: str = DATA_PATH) -> Dict:
    if not os.path.exists(path):
        return make_group_splits(data_path, path)
    with open(path) as f:
        splits = json.load(f)
    if splits.get("data_sha256") != file_sha256(data_path):
        raise ValueError(f"{data_path} changed since {path} was written — regenerate the "
                         "split deliberately (make_group_splits(overwrite=True)) and re-run everything.")
    return splits


def get_split_cases(split: str, data_path: str = DATA_PATH, splits_path: str = SPLITS_PATH) -> List[Dict]:
    ids = set(load_splits(splits_path, data_path)["case_ids"][split])
    return [c for c in load_cases(data_path) if c["case_id"] in ids]


def _has_chat_template(tokenizer) -> bool:
    return bool(getattr(tokenizer, "chat_template", None))


def build_prompt(tokenizer, title: str) -> str:
    user = PROMPT_TEMPLATE.format(title=title)
    if _has_chat_template(tokenizer):
        return tokenizer.apply_chat_template([{"role": "user", "content": user}],
                                             tokenize=False, add_generation_prompt=True)
    return user + "\n\n"


def build_full_text(tokenizer, title: str, answer: str) -> Tuple[str, str]:
    prompt = build_prompt(tokenizer, title)
    if _has_chat_template(tokenizer):
        full = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT_TEMPLATE.format(title=title)},
             {"role": "assistant", "content": answer}], tokenize=False)
        if full.startswith(prompt):
            return prompt, full
    return prompt, prompt + answer + (tokenizer.eos_token or "")


def _add_special(tokenizer, text: str) -> bool:
    # Chat templates usually render BOS themselves; don't add a second one.
    bos = getattr(tokenizer, "bos_token", None)
    return not (bos and text.startswith(bos))


def encode_prompt(tokenizer, title: str, max_length: int = 384, return_tensors: str = "pt"):
    prompt = build_prompt(tokenizer, title)
    return tokenizer(prompt, return_tensors=return_tensors, truncation=True, max_length=max_length,
                     add_special_tokens=_add_special(tokenizer, prompt))


def tokenize_example(tokenizer, title: str, answer: str, max_length: int, pad: bool = True) -> Dict:
    """input_ids / attention_mask / labels with prompt and padding positions set to -100."""
    prompt, full = build_full_text(tokenizer, title, answer)
    special = _add_special(tokenizer, full)
    ids = tokenizer(full, truncation=True, max_length=max_length, add_special_tokens=special)["input_ids"]
    prompt_len = len(tokenizer(prompt, add_special_tokens=special)["input_ids"])
    labels = [-100] * min(prompt_len, len(ids)) + ids[prompt_len:]
    n_pad = max_length - len(ids) if pad else 0
    # Always right-pad here: several tokenizers default to left padding, which would
    # shift the prompt mask onto the padding.
    return {"input_ids": torch.tensor(ids + [tokenizer.pad_token_id] * n_pad),
            "attention_mask": torch.tensor([1] * len(ids) + [0] * n_pad),
            "labels": torch.tensor(labels + [-100] * n_pad)}


class MSKCaseStudyDataset(Dataset):
    """Tokenised case studies. Pass a list of cases (e.g. get_split_cases("train"))
    or a CSV path (all rows — only for inspection, never for evaluation)."""

    def __init__(self, cases, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_cases(cases) if isinstance(cases, str) else list(cases)

    def __len__(self):
        return len(self.data)

    def format_case_study(self, case: Dict) -> str:
        return PROMPT_TEMPLATE.format(title=case.get("title", ""))

    def __getitem__(self, idx):
        case = self.data[idx]
        item = tokenize_example(self.tokenizer, case["title"], case["use_case"], self.max_length)
        item.update({"title": case["title"], "answer": case["use_case"], "case_id": case["case_id"]})
        return item


def get_split_datasets(tokenizer, max_length: int = 1024, data_path: str = DATA_PATH,
                       splits_path: str = SPLITS_PATH) -> Dict[str, MSKCaseStudyDataset]:
    return {s: MSKCaseStudyDataset(get_split_cases(s, data_path, splits_path), tokenizer, max_length)
            for s in ("train", "calibration", "test")}


# --------------------------------------------------------------------------- #
# Models / tokenizers                                                         #
# --------------------------------------------------------------------------- #

def tokenizers_compatible(tok_a, tok_b) -> bool:
    """True when every token of the smaller vocabulary has the same id in the larger one,
    i.e. truncating logits to the shared prefix compares like with like."""
    va, vb = tok_a.get_vocab(), tok_b.get_vocab()
    small, big = (va, vb) if len(va) <= len(vb) else (vb, va)
    return all(big.get(t) == i for t, i in small.items())


def _dtype_kwarg() -> str:
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


def quantization_config(quant: str):
    from transformers import BitsAndBytesConfig
    if quant == "4bit":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                  bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    if quant == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def _best_device_map(quant_config) -> Optional[str]:
    """Return device_map appropriate for the current hardware.
    MPS (Apple Silicon) is incompatible with device_map; we move the model manually instead."""
    if torch.cuda.is_available():
        return "auto"
    if quant_config is not None:
        return "auto"   # BitsAndBytes requires device_map even without CUDA
    return None         # MPS / CPU path: we'll call .to() after loading


def load_causal_lm(path: str, quant: str = "none", device_map: Optional[str] = "auto",
                   merge_adapter: bool = True, attn_implementation: str = "eager"):
    """Load a model or a PEFT adapter directory. Adapters are merged into bf16 bases
    (so a fresh LoRA can be stacked on top); on quantised bases they stay unmerged."""
    from transformers import AutoModelForCausalLM
    kw: Dict[str, Any] = dict(low_cpu_mem_usage=True, attn_implementation=attn_implementation)
    qc = quantization_config(quant)

    # device_map="auto" doesn't route to MPS on macOS — pick the right map for this hardware
    resolved_map = _best_device_map(qc) if device_map == "auto" else device_map
    use_mps = (resolved_map is None and qc is None
               and hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    if resolved_map is not None:
        kw["device_map"] = resolved_map

    if qc is not None:
        kw["quantization_config"] = qc
    else:
        kw[_dtype_kwarg()] = torch.bfloat16

    adapter_cfg = os.path.join(path, "adapter_config.json")
    if os.path.isdir(path) and os.path.exists(adapter_cfg):
        from peft import PeftModel
        with open(adapter_cfg) as f:
            base_name = json.load(f)["base_model_name_or_path"]
        model = AutoModelForCausalLM.from_pretrained(base_name, **kw)
        model = PeftModel.from_pretrained(model, path)
        if merge_adapter and qc is None:
            model = model.merge_and_unload()
        if use_mps:
            model = model.to("mps")
        return model
    model = AutoModelForCausalLM.from_pretrained(path, **kw)
    if use_mps:
        model = model.to("mps")
    return model


def load_tokenizer(path: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def model_forward(model, input_ids, attention_mask, **kwargs):
    # Multimodal models (e.g. MedGemma) need pixel_values=None for text-only forward passes.
    import inspect
    if "pixel_values" in inspect.signature(model.forward).parameters:
        kwargs.setdefault("pixel_values", None)
    return model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)


def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

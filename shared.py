from dataclasses import dataclass, asdict, field
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from typing import Dict, List, Optional, Tuple

@dataclass
class ModelPair:
    teacher_name: str
    student_name: str
    teacher_size: str
    student_size: str
    tuned_teacher: str = ""
    distilled_student: str = ""
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
    max_seq_length: int = 256 #256 for bigger models
    teacher_model: str = ""
    student_model: str = ""

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

@dataclass
class RefinementConfig:
    hf_token: Optional[str] = None
    use_auth_token: bool = True
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    # Training
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    device: str = "cuda:0"
    logging_steps: int = 10
    layer_selection_strategy: str = "performance_based" #all | performance_based | attention_only | mlp_only
    num_layers_to_refine: Optional[int] = None
    performance_threshold: float = 0.95
    use_teacher_guidance: bool = True # Teacher guidance (KD term during refinement)
    teacher_guidance_weight: float = 0.3
    # RL
    use_rl: bool = True
    rl_beta: float = 0.1
    rl_reward_type: str = "teacher_agreement"  # task_accuracy | teacher_agreement
    use_dynamic_ranks: bool = True
    lora_r_high: int = 32    # degradation score >= 0.8
    lora_r_medium: int = 16  # 0.5 <= score < 0.8
    lora_r_low: int = 8      # 0.3 <= score < 0.5
    lora_r_minimal: int = 4  # score < 0.3
    use_teacher_agreement_reward: bool = True
    agreement_weight: float = 0.3   # lambda1 (global fallback; per-family weights in ModelPair)
    kl_reward_weight: float = 0.2   # lambda2 (global fallback; per-family weights in ModelPair)
    model_pair_key: str = "" # Per-family weight routing — set to the active MODEL_PAIRS key at runtime
    teacher_offload_folder: Optional[str] = "./teacher_offload"
    use_8bit_teacher: bool = True
    ppl_monitor_steps: int = 50      # compute PPL every N steps (0 = disabled)
    ppl_hacking_threshold: float = 2.0  # warn if PPL > baseline * threshold
    num_eval_seeds: int = 1

@dataclass
class AblationConfig:
    name: str
    description: str
    skip_distillation: bool = False
    skip_refinement: bool = False
    use_rl: bool = True
    use_dynamic_ranks: bool = True
    use_teacher_guidance: bool = True
    use_teacher_agreement_reward: bool = True
    rl_reward_type: str = "teacher_agreement"
    layer_selection_strategy: str = "performance_based"
    fixed_lora_r: Optional[int] = None

ABLATION_CONFIGS = [
    AblationConfig("A0_teacher_baseline",
                   "Fine-tuned teacher only — no distillation or refinement",
                   skip_distillation=True, skip_refinement=True),
    AblationConfig("A1_distill_only",
                   "Distillation only — no Phase 3 refinement",
                   skip_refinement=True, use_rl=False, use_dynamic_ranks=False,
                   use_teacher_guidance=False, use_teacher_agreement_reward=False),
    AblationConfig("A2_rl_task_accuracy",
                   "KD + RL with task accuracy reward (no teacher agreement)",
                   use_rl=True, use_dynamic_ranks=False,
                   use_teacher_guidance=False, use_teacher_agreement_reward=False,
                   rl_reward_type="task_accuracy"),
    AblationConfig("A3_dynamic_ranks_no_rl",
                   "KD + dynamic LoRA ranks, no RL",
                   use_rl=False, use_dynamic_ranks=True,
                   use_teacher_guidance=False, use_teacher_agreement_reward=False),
    AblationConfig("A4_kd_guidance_rl",
                   "KD + RL + dynamic ranks + teacher KD guidance (no agreement reward)",
                   use_rl=True, use_dynamic_ranks=True,
                   use_teacher_guidance=True, use_teacher_agreement_reward=False),
    AblationConfig("A5_full_pipeline",
                   "Full method: KD + hybrid KD+RL + dynamic ranks + agreement reward",
                   use_rl=True, use_dynamic_ranks=True,
                   use_teacher_guidance=True, use_teacher_agreement_reward=True),
    AblationConfig("A6_fixed_rank_lora",
                   "Full method with fixed uniform LoRA rank (r=16) — isolates benefit of dynamic rank allocation",
                   use_rl=True, use_dynamic_ranks=False,
                   use_teacher_guidance=True, use_teacher_agreement_reward=True,
                   fixed_lora_r=16),
]

MODEL_PAIRS = {
    #Llama family — shared tokenizer, moderate agreement emphasis
    "llama_3b_1b": ModelPair(
        "meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-1B-Instruct",
        "3B", "1B",
        "./models/llama_3b_1b/phase1_teacher",
        "./models/llama_3b_1b/phase2_student",
        agreement_weight=0.30, kl_reward_weight=0.20,
    ),
    "llama_tiny": ModelPair(
        "meta-llama/Llama-3.2-3B-Instruct", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "3B", "1B",
        "./models/llama_tiny/phase1_teacher",
        "./models/llama_tiny/phase2_student",
        # cross-family pair: looser agreement target, lighter KL penalty
        agreement_weight=0.25, kl_reward_weight=0.15,
    ),
    #Qwen family — different tokenizer; slightly reduced KL penalty to avoid over-penalising vocab mismatches
    "qwen_3b_0.5b": ModelPair(
        "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct",
        "3B", "0.5B",
        "./models/qwen_3b_0.5b/phase1_teacher",
        "./models/qwen_3b_0.5b/phase2_student",
        agreement_weight=0.25, kl_reward_weight=0.15,
    ),
    #SmolLM2 — small capacity gap; lower agreement weight to avoid over-constraining the student
    "smollm2": ModelPair(
        "HuggingFaceTB/SmolLM2-1.7B-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct",
        "1.7B", "360M",
        "./models/smollm2/phase1_teacher",
        "./models/smollm2/phase2_student",
        agreement_weight=0.20, kl_reward_weight=0.15,
    ),
    # # Medical Gemma family — clinical accuracy critical; higher agreement weight
    "medgemma3": ModelPair(
        "google/medgemma-1.5-4b-it", "google/gemma-3-270m-it",
        "4B", "270m",
        "./models/medgemma3/phase1_teacher",
        "./models/medgemma3/phase2_student",
        agreement_weight=0.40, kl_reward_weight=0.25,
    ),
    "medgemma_text": ModelPair(
        "google/medgemma-27b-text-it", "google/medgemma-1.5-4b-it",
        "27B", "1.5B",
        "./models/medgemma_text/phase1_teacher",
        "./models/medgemma_text/phase2_student",
        agreement_weight=0.40, kl_reward_weight=0.25,
    ),
    "gemma3": ModelPair(
        "google/gemma-3-1b-pt", "google/gemma-3-270m-it",
        "1B", "270m",
        "./models/gemma3/phase1_teacher",
        "./models/gemma3/phase2_student",
        agreement_weight=0.35, kl_reward_weight=0.20,
    ),
    # # Mistral cross-family — standard weights
    "mistral_tiny": ModelPair(
        "mistralai/Mistral-7B-v0.1", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "7B", "1.1B",
        "./models/mistral_tiny/phase1_teacher",
        "./models/mistral_tiny/phase2_student",
        agreement_weight=0.25, kl_reward_weight=0.15,
    ),
    # Meditron — medical domain, higher agreement like medgemma
    "meditron_phi4mini": ModelPair(
        "epfl-llm/meditron-7b", "meta-llama/Llama-3.2-1B",
        "7B", "1B",
        "./models/meditron_phi4mini/phase1_teacher",
        "./models/meditron_phi4mini/phase2_student",
        agreement_weight=0.40, kl_reward_weight=0.25,
    ),
    # QwQ / Qwen large-to-small — same family, moderate weights
    "qwq_qwen1.5b": ModelPair(
        "Qwen/QwQ-32B", "Qwen/Qwen2.5-1.5B-Instruct",
        "32B", "1.5B",
        "./models/qwq_qwen1.5b/phase1_teacher",
        "./models/qwq_qwen1.5b/phase2_student",
        agreement_weight=0.30, kl_reward_weight=0.20,
    ),
}

class MSKCaseStudyDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_path = data_path

        df = pd.read_csv(data_path, encoding="latin-1")
        df.columns = [c.strip().lower() for c in df.columns]
        if "use case" in df.columns:
            df = df.rename(columns={"use case": "use_case"})
        self.data = (
            df[["title", "use_case"]]
            .dropna(subset=["use_case"])
            .to_dict("records")
        )
        print(f"Loaded {len(self.data)} case studies")

    def __len__(self):
        return len(self.data)

    def format_case_study(self, case: Dict) -> str:
        return (
            "You are a musculoskeletal (MSK) physiotherapy expert. "
            "Write a detailed physiotherapy case study for the following condition.\n\n"
            f"Condition: {case.get('title', '')}\n\n"
            "Case Study:"
        )

    def __getitem__(self, idx):
        case = self.data[idx]
        prompt = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{self.format_case_study(case)}<|eot_id|>"
        )
        answer = case.get("use_case", "")
        if answer:
            full_text = (
                prompt
                + "<|start_header_id|>assistant<|end_header_id|>\n\n"
                + answer
                + "<|eot_id|>"
            )
        else:
            full_text = prompt

        enc = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze()
        attention_mask = enc["attention_mask"].squeeze()

        if answer:
            prompt_enc = self.tokenizer(
                prompt, truncation=True, max_length=self.max_length,
                return_tensors="pt"
            )
            prompt_len = prompt_enc["input_ids"].shape[1]
            labels = input_ids.clone()
            labels[:prompt_len] = -100
        else:
            labels = input_ids.clone()

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
                "raw_text": full_text}
    
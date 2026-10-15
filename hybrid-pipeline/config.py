try:
    from .shared import (ABLATION_CONFIGS, FOCUS_PAIRS, MODEL_PAIRS, AblationConfig, DistillationConfig,
                         MSKCaseStudyDataset, QLoRAConfig, RefinementConfig)
except ImportError:
    from shared import (ABLATION_CONFIGS, FOCUS_PAIRS, MODEL_PAIRS, AblationConfig, DistillationConfig,
                        MSKCaseStudyDataset, QLoRAConfig, RefinementConfig)

MODEL_PAIR_KEY = "medgemma3"
model_pair = MODEL_PAIRS[MODEL_PAIR_KEY]

SEEDS = [0, 1, 2]

qlora_cfg = QLoRAConfig(
    lora_r=8, lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, learning_rate=1e-4,
    epochs=3, batch_size=1, use_4bit=True,
    gradient_accumulation_steps=8,
    max_seq_length=1024,
    teacher_model=model_pair.teacher_name,
    student_model=model_pair.student_name,
    seed=42,
)

distill_cfg = DistillationConfig(
    temperature=2.0, alpha=0.5,
    load_in_8bit=True, llm_int8_threshold=6.0,
    learning_rate=5e-5, epochs=3, batch_size=1,
    gradient_accumulation_steps=6,
    max_grad_norm=1.0,
    warmup_steps=50,
    max_seq_length=1024,
    track_every=25, track_num_samples=16,
    seed=42,
)

refine_cfg = RefinementConfig(
    lora_r=16, lora_alpha_ratio=2.0, lora_dropout=0.05,
    lora_r_min=4, lora_r_max=64,
    module_types=["q_proj", "v_proj"],
    learning_rate=2e-4, num_epochs=2, warmup_steps=50,
    layer_selection_strategy="all",
    rank_allocation="degradation",
    degradation_metric="cka", degradation_num_samples=32,
    use_teacher_guidance=True, teacher_guidance_weight=0.3, kd_temperature=2.0,
    use_reward_weighting=True, reward_type="teacher_agreement",
    reward_beta=0.1, reward_normalization="advantage",
    # Global fallback weights — overridden at runtime by per-family values in MODEL_PAIRS
    agreement_weight=0.3, kl_reward_weight=0.2,
    teacher_quant="4bit", device="cuda:0",
    max_seq_length=1024,
    track_every=25, track_num_samples=16,
    seed=0,
)

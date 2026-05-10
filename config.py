
from .shared import AblationConfig, RefinementConfig, QLoRAConfig,ABLATION_CONFIGS, DistillationConfig, MSKCaseStudyDataset, MODEL_PAIRS

MODEL_PAIR_KEY = "deepseek_phi4mini"
#llama_8b_1b llama_3b_1b
model_pair = MODEL_PAIRS[MODEL_PAIR_KEY]

qlora_cfg = QLoRAConfig(
    lora_r=4, lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, learning_rate=1e-4,
    epochs=3, batch_size=1, use_4bit=True,
    gradient_accumulation_steps=8,
    teacher_model=model_pair.teacher_name,
    student_model=model_pair.student_name
)

distill_cfg = DistillationConfig(
    temperature=2.0, alpha=0.5,
    load_in_8bit=True, llm_int8_threshold=6.0,
    learning_rate=5e-5, epochs=3, batch_size=1,
    gradient_accumulation_steps=6,
    max_grad_norm=1.0,
    warmup_steps=50,
)

refine_cfg = RefinementConfig(
    lora_r=16, lora_alpha=32, lora_dropout=0.05,
    learning_rate=2e-4, num_epochs=2, warmup_steps=50,
    layer_selection_strategy="performance_based",
    use_teacher_guidance=True, teacher_guidance_weight=0.3,
    use_rl=True, rl_beta=0.1, rl_reward_type="teacher_agreement",
    use_dynamic_ranks=True,
    lora_r_high=32, lora_r_medium=16, lora_r_low=8, lora_r_minimal=4,
    use_teacher_agreement_reward=True, agreement_weight=0.3, kl_reward_weight=0.2,
    use_8bit_teacher=True, device="cuda:0"
)
# DART: Degradation-Aware Post-Distillation Refinement with Teacher-Agreement Rewards 

Large language models can generate clinically rich physiotherapy case studies, but their computational cost limits deployment in resource-constrained clinical and educational environments. 
We present a framework that diagnoses post-distillation layer degradation using teacher-student KL divergence, uses that signal to both select layers for LoRA adaptation and allocate rank capacity to them, and keeps the teacher active as a reward anchor during refinement.

```text
    Dataset
        |
        v
Phase 1: QLoRA teacher tuning
        |
        v
Phase 2: student distillation
        |
        v
Phase 3: selective hybrid KD + RL refinement
        |
        v
Evaluation and ablation analysis
```

## Phase 1: QLoRA Fine-Tuning of the Teacher
Phase 1 is implemented by `QLoRAFineTuner`. Its purpose is to adapt the larger teacher model to the MSK case-study task while keeping memory usage manageable. The teacher output becomes the source model for distillation in phase 2.

## Phase 2: Knowledge Distillation into the Student
Phase 2 is implemented by `KnowledgeDistillationTrainer`. This stage trains the smaller student model to match both the ground-truth case-study labels and the tuned teacher's output distribution.

The trainer loads:
- The tuned teacher model from phase 1.
- The base student model from the selected `ModelPair`.
- Optional LoRA adapters on the student, with default target modules `q_proj` and `v_proj`.

The distillation loss combines hard-label supervision with soft teacher guidance:
```text
loss = alpha * soft_KL_loss + (1 - alpha) * hard_cross_entropy_loss
```
The phase 2 output is the distilled student model, saved under the configured `phase2_student` directory.

## Phase 3: Hybrid Post-Distillation Refinement

Phase 3 is implemented by `RefinementTrainer`. It refines the distilled student using selective LoRA adaptation and a hybrid objective. The refinement objective is:

```text
total_loss = task_loss
           + teacher_guidance_weight * KL_loss
           + rl_beta * RL_loss
```

Where:

- `task_loss` is standard cross-entropy against the case-study labels.
- `KL_loss` keeps the refined student close to the teacher when teacher guidance is enabled.
- `RL_loss` uses a reward-weighted sequence log-probability objective.

The default refinement setup enables:

- Performance-based layer selection.
- Teacher guidance with weight `0.3`.
- Reinforcement-style reward with beta `0.1`.
- Teacher-agreement reward.
- Dynamic LoRA rank selection.

### Layer Degradation Analysis

Before refinement, the trainer can analyze which student layers degraded most during distillation. For same-architecture teacher/student pairs, it compares hidden states with KL divergence per layer. For cross-architecture pairs, it falls back to output-level KL and distributes the degradation estimate across depth-weighted student layers.

The resulting layer scores drive two choices:
- Which modules should receive LoRA adapters.
- What LoRA rank should be used for refinement.

The dynamic rank policy maps normalized degradation scores to ranks:
- Score `>= 0.8`: rank `32`
- Score `>= 0.5`: rank `16`
- Score `>= 0.3`: rank `8`
- Score `< 0.3`: rank `4`

When `layer_selection_strategy="performance_based"`, the trainer selects the most degraded half of the layers by default and targets their attention projections, especially `q_proj` and `v_proj`.

### Teacher-Agreement Reward

The reinforcement-style reward combines token-level correctness and agreement with the teacher:

```text
reward = accuracy
       + agreement_weight * teacher_student_agreement
       - kl_reward_weight * KL_penalty
```

With the current defaults:

- `agreement_weight=0.3`
- `kl_reward_weight=0.2`
- `rl_reward_type="teacher_agreement"`

This reward encourages the refined student to remain clinically aligned with the tuned teacher while still improving on the supervised target examples.

## Evaluation System

Evaluation lives in `evaluate.py` and has two layers: general model evaluation and clinical-quality evaluation.

`PipelineEvaluator` measures:
- Perplexity over the dataset.
- ROUGE-1, ROUGE-2, and ROUGE-L against reference case studies.
- BERTScore precision, recall, and F1.
- Generation latency.
- Teacher/student parameter compression statistics.

`AdvancedEvaluator` adds LLM-based clinical review signals:
- G-Eval criteria for clinical coherence, diagnostic accuracy, treatment completeness, patient safety, and clinical relevance.
- Bias detection through DeepEval.
- GEMBA-MSK, a reference-free holistic clinical quality score.
- Reason-then-Score, where an LLM judge reasons through clinical dimensions before assigning a score.
- `ClinicalMSKScore`, a weighted hybrid metric which combines:
    - ROUGE-L: 10%
    - BERTScore F1: 20%
    - G-Eval mean: 25%
    - GEMBA-MSK: 25%
    - Reason-then-Score: 20%

It then applies a multiplicative bias penalty:

```text
clinical_msk_score = weighted_base * (1 - 0.5 * bias_detected)
```
This makes bias a direct quality penalty rather than another score that can be averaged away.

## Ablation Design

The ablation system compares six configurations:

- `A0_teacher_baseline`: fine-tuned teacher only.
- `A1_distill_only`: student distillation without phase-3 refinement.
- `A2_rl_task_accuracy`: distillation plus RL with task-accuracy reward.
- `A3_dynamic_ranks_no_rl`: distillation plus dynamic LoRA ranks without RL.
- `A4_kd_guidance_rl`: KD guidance, RL, and dynamic ranks without teacher-agreement reward.
- `A5_full_pipeline`: full method with KD, RL, dynamic ranks, and teacher agreement.

`AblationStudyRunner` applies each configuration, evaluates the resulting model, saves interim CSV files, and supports plotting core and advanced metrics. The plotting utilities produce bar charts, radar charts, and layer-degradation visualizations.

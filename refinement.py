from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    #TrainingArguments, 
    #Trainer,
    BitsAndBytesConfig, DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from .config import RefinementConfig
from typing import Dict, List, Optional, Tuple
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, TaskType
import torch
import os
import pandas as pd
import numpy as np

def _model_forward(model, input_ids, attention_mask, **kwargs):
    # Multimodal models (e.g. MedGemma3) need pixel_values=None to skip vision pathway
    # when doing text-only inference; text-only models don't accept that kwarg.
    import inspect
    sig = inspect.signature(model.forward)
    if "pixel_values" in sig.parameters:
        kwargs.setdefault("pixel_values", None)
    return model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

class RefinementTrainer:
    # Post-distillation refinement via selective LoRA + hybrid KD+RL.
    # Loss: total = task_loss + alpha*KL_loss + beta*RL_loss
    # Reward: accuracy + lambda1*agreement - lambda2*KL_penalty
    # LoRA ranks: assigned per-layer based on distillation degradation score

    def __init__(self, distilled_model_hf_path: str,
                 teacher_model_hf_path: Optional[str],
                 config: RefinementConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.distilled_model_hf_path = distilled_model_hf_path
        self.teacher_model_hf_path = teacher_model_hf_path
        self.distilled_model = None
        self.teacher_model = None
        self.peft_model = None
        self.layer_performance_scores: Dict[str, float] = {}

        # if config.hf_token:
        #     login(token=config.hf_token)

        self.tokenizer = AutoTokenizer.from_pretrained(
            distilled_model_hf_path
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("RefinementTrainer ready")

    @staticmethod
    def _cfg_attr(cfg, attr, default=None):
        # Gemma3Config nests text attributes under .text_config; fall back gracefully.
        if hasattr(cfg, attr):
            return getattr(cfg, attr)
        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, attr):
            return getattr(cfg.text_config, attr)
        return default

    def clear_memory(self):
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def load_distilled_model(self):
        if self.distilled_model is None:
            print(f"Loading distilled model: {self.distilled_model_hf_path}")
            self.distilled_model = AutoModelForCausalLM.from_pretrained(
                self.distilled_model_hf_path,
                torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                attn_implementation="eager",
                #use_auth_token=self.config.use_auth_token
            ).to(self.device)

    def load_teacher_model(self, for_training: bool = False):
        if self.teacher_model is None and self.teacher_model_hf_path:
            print(f"Loading teacher model: {self.teacher_model_hf_path}")
            if self.config.use_8bit_teacher:
                #bnb = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
                )
                self.teacher_model = AutoModelForCausalLM.from_pretrained(
                    self.teacher_model_hf_path, quantization_config=bnb,
                    device_map="auto", low_cpu_mem_usage=True,
                    attn_implementation="eager",
                    #use_auth_token=self.config.use_auth_token
                )
            else:
                self.teacher_model = AutoModelForCausalLM.from_pretrained(
                    self.teacher_model_hf_path, torch_dtype=torch.bfloat16,
                    device_map="auto", low_cpu_mem_usage=True,
                    attn_implementation="eager",
                    #use_auth_token=self.config.use_auth_token
                )
            if not for_training:
                self.teacher_model.eval()
                for p in self.teacher_model.parameters():
                    p.requires_grad_(False)

    def unload_teacher_model(self):
        if self.teacher_model is not None:
            del self.teacher_model
            self.teacher_model = None
            self.clear_memory()

    def analyze_layer_performance(self, eval_dataset: Dataset,
                                  num_samples: int = 50) -> Dict[str, float]:
        # Measure per-layer KL divergence between teacher and student to identify degraded layers.
        print("Analyzing layer-wise distillation degradation...")
        self.load_distilled_model()
        self.load_teacher_model(for_training=False)

        s_cfg = self.distilled_model.config
        t_cfg = self.teacher_model.config
        s_hidden = self._cfg_attr(s_cfg, "hidden_size")
        t_hidden = self._cfg_attr(t_cfg, "hidden_size")
        s_layers = self._cfg_attr(s_cfg, "num_hidden_layers")
        t_layers = self._cfg_attr(t_cfg, "num_hidden_layers")
        same_arch = (s_hidden is not None and s_hidden == t_hidden and s_layers == t_layers)

        if not same_arch:
            print(f"  Different architectures — teacher {t_layers}L/{t_hidden}H "
                  f"vs student {s_layers}L/{s_hidden}H")
            scores = self._analyze_output_based_performance(eval_dataset, num_samples)
            self.layer_performance_scores = scores
            self.unload_teacher_model()
            return scores

        layer_kl = {i: [] for i in range(s_layers)}
        self.distilled_model.eval()

        with torch.no_grad():
            for idx in range(min(num_samples, len(eval_dataset))):
                sample = eval_dataset[idx]
                ids = sample["input_ids"].unsqueeze(0).to(self.device)
                mask = sample["attention_mask"].unsqueeze(0).to(self.device)
                t_dev = next(self.teacher_model.parameters()).device

                s_out = _model_forward(self.distilled_model, ids, mask, output_hidden_states=True)
                t_out = _model_forward(self.teacher_model, ids.to(t_dev), mask.to(t_dev), output_hidden_states=True)

                for li in range(s_layers):
                    sh = s_out.hidden_states[li + 1]
                    th = t_out.hidden_states[li + 1].to(self.device)
                    kl = F.kl_div(
                        F.log_softmax(sh.float(), dim=-1),
                        F.softmax(th.float(), dim=-1),
                        reduction="batchmean"
                    ).item()
                    layer_kl[li].append(kl)

        raw = {i: float(np.mean(v)) for i, v in layer_kl.items() if v}
        max_kl = max(raw.values()) if raw else 1.0
        scores = {f"layer_{i}": v / max_kl for i, v in raw.items()}
        top5 = sorted(scores, key=scores.get, reverse=True)[:5]
        print(f"  Top-5 degraded layers: {top5}")
        self.layer_performance_scores = scores
        self.unload_teacher_model()
        return scores

    def _analyze_output_based_performance(self, eval_dataset: Dataset,
                                          num_samples: int) -> Dict[str, float]:
        # For cross-architecture pairs: distribute output-level KL across layers by depth weight.
        kl_scores = []
        self.distilled_model.eval()

        with torch.no_grad():
            for idx in range(min(num_samples, len(eval_dataset))):
                sample = eval_dataset[idx]
                ids = sample["input_ids"].unsqueeze(0).to(self.device)
                mask = sample["attention_mask"].unsqueeze(0).to(self.device)
                t_dev = next(self.teacher_model.parameters()).device

                s_logits = _model_forward(self.distilled_model, ids, mask).logits
                t_logits = _model_forward(self.teacher_model, ids.to(t_dev), mask.to(t_dev)).logits.to(self.device)
                min_v = min(s_logits.size(-1), t_logits.size(-1))
                kl = F.kl_div(
                    F.log_softmax(s_logits[..., :min_v].float(), dim=-1),
                    F.softmax(t_logits[..., :min_v].float(), dim=-1),
                    reduction="batchmean"
                ).item()
                kl_scores.append(kl)

        n = self._cfg_attr(self.distilled_model.config, "num_hidden_layers", default=1)
        avg_kl = float(np.mean(kl_scores)) if kl_scores else 0.0
        weights = np.linspace(0.3, 1.0, n)
        weights /= weights.max()
        return {f"layer_{i}": float(avg_kl * weights[i]) for i in range(n)}

    def select_dynamic_lora_ranks(self) -> Dict[str, int]:
        # Assign LoRA rank per-layer based on degradation severity (novel degradation-based criterion).
        ranks = {}
        for k, score in self.layer_performance_scores.items():
            if score >= 0.8:
                ranks[k] = self.config.lora_r_high
            elif score >= 0.5:
                ranks[k] = self.config.lora_r_medium
            elif score >= 0.3:
                ranks[k] = self.config.lora_r_low
            else:
                ranks[k] = self.config.lora_r_minimal
        if ranks:
            from collections import Counter
            print(f"  Rank distribution: {dict(Counter(ranks.values()))}")
        return ranks

    def select_target_modules(self) -> List[str]:
        strategy = self.config.layer_selection_strategy
        if strategy == "attention_only":
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif strategy == "mlp_only":
            return ["gate_proj", "up_proj", "down_proj"]
        elif strategy == "performance_based" and self.layer_performance_scores:
            n = self.config.num_layers_to_refine or max(1, len(self.layer_performance_scores) // 2)
            top = sorted(self.layer_performance_scores,
                         key=self.layer_performance_scores.get, reverse=True)[:n]
            indices = [int(k.split("_")[1]) for k in top]
            mods = []
            for i in indices:
                mods += [f"layers.{i}.self_attn.q_proj", f"layers.{i}.self_attn.v_proj"]
            return mods if mods else ["q_proj", "v_proj"]
        else:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def apply_lora_adapters(self):
        target_modules = self.config.target_modules or self.select_target_modules()
        lora_r = self.config.lora_r

        if self.config.use_dynamic_ranks and self.layer_performance_scores:
            ranks = list(self.select_dynamic_lora_ranks().values())
            lora_r = int(np.median(ranks))
            print(f"  Median dynamic rank: {lora_r}")

        lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=self.config.lora_alpha,
            target_modules=target_modules, lora_dropout=self.config.lora_dropout,
            bias="none", task_type=TaskType.CAUSAL_LM
        )
        self.peft_model = get_peft_model(self.distilled_model, lora_cfg)
        self.peft_model.print_trainable_parameters()

    def refinement_loss(self, student_logits, teacher_logits, labels,
                        attention_mask, rewards=None):
        # Hybrid loss: total = task_loss + alpha*KL_loss + beta*RL_loss
        shift_s = student_logits[..., :-1, :].contiguous()
        shift_l = labels[..., 1:].contiguous()
        shift_m = attention_mask[..., 1:].contiguous()

        active = shift_m.view(-1) == 1
        as_ = shift_s.view(-1, shift_s.size(-1))[active]
        al_ = shift_l.view(-1)[active]

        task_loss = nn.CrossEntropyLoss()(as_, al_)

        kl_loss = torch.tensor(0.0, device=student_logits.device)
        if self.config.use_teacher_guidance and teacher_logits is not None:
            shift_t = teacher_logits[..., :-1, :].contiguous()
            min_v = min(shift_s.size(-1), shift_t.size(-1))
            at_ = shift_t.view(-1, shift_t.size(-1))[active][..., :min_v]
            as_t = as_[..., :min_v]
            with torch.no_grad():
                soft_t = F.softmax(at_.float() / 2.0, dim=-1)
            kl_loss = F.kl_div(
                F.log_softmax(as_t.float() / 2.0, dim=-1),
                soft_t, reduction="batchmean"
            ) * 4.0

        rl_loss = torch.tensor(0.0, device=student_logits.device)
        if self.config.use_rl and rewards is not None:
            log_probs = F.log_softmax(shift_s, dim=-1)
            tok_lp = log_probs.gather(
                -1, shift_l.clamp(0, log_probs.size(-1) - 1).unsqueeze(-1)
            ).squeeze(-1)
            masked_lp = tok_lp * shift_m.float()
            seq_lp = masked_lp.sum(-1) / (shift_m.float().sum(-1) + 1e-8)
            rl_loss = -(rewards.to(student_logits.device) * seq_lp).mean()

        total = (task_loss
                 + self.config.teacher_guidance_weight * kl_loss
                 + self.config.rl_beta * rl_loss)
        return total, task_loss, kl_loss, rl_loss

    def calculate_rewards(self, logits, labels, attention_mask,
                          teacher_logits=None) -> torch.Tensor:
        # Teacher-student agreement reward:
        # reward = accuracy + lambda1*agreement - lambda2*KL_penalty
        B = logits.size(0)
        rewards = torch.zeros(B, device=logits.device)

        for i in range(B):
            mask = attention_mask[i, 1:].bool()
            preds = logits[i, :-1].argmax(dim=-1)
            lbls = labels[i, 1:]
            accuracy = (preds[mask] == lbls[mask]).float().mean() if mask.sum() > 0 else torch.tensor(0.0)
            reward = accuracy

            if (self.config.use_teacher_agreement_reward
                    and teacher_logits is not None
                    and self.config.rl_reward_type == "teacher_agreement"):
                t_preds = teacher_logits[i, :-1].argmax(dim=-1).to(preds.device)
                agreement = (preds[mask] == t_preds[mask]).float().mean() \
                    if mask.sum() > 0 else torch.tensor(0.0)
                min_v = min(logits.size(-1), teacher_logits.size(-1))
                s_lp = F.log_softmax(logits[i, :, :min_v].float(), dim=-1)
                t_p = F.softmax(teacher_logits[i, :, :min_v].float().to(logits.device), dim=-1)
                kl_pen = F.kl_div(s_lp, t_p, reduction="batchmean").clamp(min=0)
                reward = (accuracy
                          + self.config.agreement_weight * agreement
                          - self.config.kl_reward_weight * kl_pen)
            rewards[i] = reward

        return rewards.detach()


    def train(self, train_dataset, eval_dataset=None, output_dir: str = "./refined",
              push_to_hub: bool = False, hub_model_id: Optional[str] = None,
              run_name: str = "refinement"):
        self.load_distilled_model()

        if self.config.layer_selection_strategy == "performance_based" and eval_dataset:
            self.analyze_layer_performance(eval_dataset, num_samples=30)

        self.apply_lora_adapters()

        if self.config.use_teacher_guidance or self.config.use_rl:
            self.load_teacher_model(for_training=False)

        model = self.peft_model
        model.train()

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )
        total_steps = len(train_dataset) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, self.config.warmup_steps, total_steps)

        # wandb.init(project=WANDB_PROJECT, name=run_name,
        #            config=asdict(self.config), tags=["phase3", "refinement"])

        history, global_step = [], 0

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            for step, idx in enumerate(torch.randperm(len(train_dataset)).tolist()):
                sample = train_dataset[idx]
                ids = sample["input_ids"].unsqueeze(0).to(self.device)
                mask = sample["attention_mask"].unsqueeze(0).to(self.device)
                lbls = sample["labels"].unsqueeze(0).to(self.device)

                t_logits = None
                if self.teacher_model is not None:
                    t_dev = next(self.teacher_model.parameters()).device
                    with torch.inference_mode():
                        t_out = _model_forward(self.teacher_model, ids.to(t_dev), mask.to(t_dev))
                    t_logits = t_out.logits.to(self.device)

                s_out = _model_forward(model, ids, mask)

                rewards = None
                if self.config.use_rl:
                    with torch.no_grad():
                        rewards = self.calculate_rewards(s_out.logits, lbls, mask, t_logits)

                loss, task_l, kl_l, rl_l = self.refinement_loss(
                    s_out.logits, t_logits, lbls, mask, rewards
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    self.config.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_loss += loss.item()

                if step % self.config.logging_steps == 0:
                    # wandb.log({"task_loss": task_l.item(), "kl_loss": kl_l.item(),
                    #            "rl_loss": rl_l.item(), "total_loss": loss.item(),
                    #            "epoch": epoch, "step": global_step})
                    print(f"Ep {epoch+1}/{self.config.num_epochs} step {global_step} | "
                          f"task={task_l.item():.4f} kl={kl_l.item():.4f} rl={rl_l.item():.4f}")

            avg = epoch_loss / len(train_dataset)
            history.append(avg)
            if eval_dataset:
                eval_loss = self.evaluate(eval_dataset)
                # wandb.log({"eval_loss": eval_loss, "epoch": epoch})
                print(f"  Epoch {epoch+1} — train={avg:.4f} eval={eval_loss:.4f}")

        self.save_model(output_dir)
        if push_to_hub and hub_model_id:
            self.push_to_hub(output_dir, hub_model_id)

        # wandb.finish()
        self.unload_teacher_model()
        return history

    def evaluate(self, eval_dataset: Dataset, num_batches: int = 50) -> float:
        model = self.peft_model or self.distilled_model
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for idx in range(min(num_batches, len(eval_dataset))):
                sample = eval_dataset[idx]
                ids = sample["input_ids"].unsqueeze(0).to(self.device)
                mask = sample["attention_mask"].unsqueeze(0).to(self.device)
                out = model(ids, mask, labels=sample["labels"].unsqueeze(0).to(self.device))
                total += out.loss.item()
                count += 1
        model.train()
        return total / max(count, 1)

    def save_model(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        (self.peft_model or self.distilled_model).save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

    def push_to_hub(self, output_dir: str, hub_model_id: str):
        api = HfApi()
        api.create_repo(repo_id=hub_model_id, private=False, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=output_dir, repo_id=hub_model_id, repo_type="model")
        print(f"Pushed to https://huggingface.co/{hub_model_id}")

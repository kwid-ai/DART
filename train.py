import os, ast, gc, json, time
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    #TrainingArguments, 
    #Trainer,
    BitsAndBytesConfig, DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, TaskType
from huggingface_hub import HfApi, login

import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
try:
    from .shared import MODEL_PAIRS, MSKCaseStudyDataset, QLoRAConfig, DistillationConfig, RefinementConfig
    from .config import MODEL_PAIR_KEY, model_pair, qlora_cfg, distill_cfg, refine_cfg
except ImportError:
    from shared import MODEL_PAIRS, MSKCaseStudyDataset, QLoRAConfig, DistillationConfig, RefinementConfig

from dotenv import load_dotenv
load_dotenv() 

os.environ["HF_TOKEN"] = os.getenv("HF_KEY")
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
WANDB_PROJECT = ""
HF_USER = ""
DATA_PATH = ""


def detect_device():
    try:
        import torch_xla.core.xla_model as xm
        return "xla", True, False
    except (ImportError, RuntimeError):
        if torch.cuda.is_available():
            return "cuda", False, True
        return "cpu", False, False

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print(f"GPU cleared — allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

def _model_forward(model, input_ids, attention_mask, **kwargs):
    # Multimodal models (e.g. MedGemma3) need pixel_values=None to skip vision pathway
    # when doing text-only inference; text-only models don't accept that kwarg.
    import inspect
    sig = inspect.signature(model.forward)
    if "pixel_values" in sig.parameters:
        kwargs.setdefault("pixel_values", None)
    return model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)


## Phase 1 — QLoRA Fine-tuning of Teacher
class QLoRAFineTuner:
    def __init__(self, model_name: str, output_dir: str, config: QLoRAConfig):
        self.model_name = model_name
        self.output_dir = output_dir
        self.config = config
        self.device, self.use_tpu, self.use_cuda = detect_device()
        print(f"Device: {self.device.upper()}")

        if self.use_cuda:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            torch.cuda.empty_cache()
            torch.backends.cuda.matmul.allow_tf32 = True

        self.bnb_config = (
            BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            ) if (config.use_4bit and self.use_cuda) else None
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.model = self._load_model()
        self.model.config.use_cache = False

        if self.bnb_config and self.use_cuda:
            self.model = prepare_model_for_kbit_training(
                self.model, use_gradient_checkpointing=True)

        lora_cfg = LoraConfig(
            r=config.lora_r, lora_alpha=config.lora_alpha,
            target_modules=config.target_modules,
            lora_dropout=config.lora_dropout,
            bias="none", task_type="CAUSAL_LM", inference_mode=False
        )
        self.model = get_peft_model(self.model, lora_cfg)

        if self.use_cuda:
            for _, p in self.model.named_parameters():
                if p.requires_grad and p.dtype != torch.bfloat16:
                    p.data = p.data.to(torch.bfloat16)

        self.model.print_trainable_parameters()

    def _load_model(self):
        kw = dict(trust_remote_code=True, low_cpu_mem_usage=True)
        if self.use_cuda and self.bnb_config:
            return AutoModelForCausalLM.from_pretrained(
                self.model_name, quantization_config=self.bnb_config,
                device_map="auto", dtype=torch.bfloat16, **kw)
        elif self.use_cuda:
            return AutoModelForCausalLM.from_pretrained(
                self.model_name, dtype=torch.bfloat16, device_map="auto", **kw)
        elif self.use_tpu:
            return AutoModelForCausalLM.from_pretrained(
                self.model_name, dtype=torch.bfloat16, **kw)
        else:
            return AutoModelForCausalLM.from_pretrained(
                self.model_name, dtype=torch.float32, **kw)

    def train(self, train_dataset, eval_dataset=None, run_name: str = "qlora"):
        if self.use_cuda:
            torch.cuda.empty_cache()

        collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            fp16=False, bf16=True, bf16_full_eval=True,
            #no_cuda=not self.use_cuda, 
            gradient_checkpointing=True,
            optim="adamw_torch", max_grad_norm=0.3, weight_decay=0.01,
            logging_steps=10, save_strategy="epoch",
            eval_strategy="epoch" if eval_dataset else "no",
            #report_to="wandb",
            report_to="none",
            logging_dir=f"./wandb/logs",  
            save_total_limit=1,
            load_best_model_at_end=bool(eval_dataset),
            dataloader_num_workers=0, dataloader_pin_memory=False,
            ddp_find_unused_parameters=False, seed=42,
        )
        # wandb.init(project=WANDB_PROJECT, name=run_name,
        #            config=asdict(self.config), tags=["phase1", "qlora"])
        trainer = Trainer(
            model=self.model, args=args,
            train_dataset=train_dataset, eval_dataset=eval_dataset,
            data_collator=collator
        )
        trainer.train()
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        # wandb.finish()
        print(f"Phase 1 complete — saved to {self.output_dir}")
        return trainer
    

## Phase 1
class KnowledgeDistillationTrainer:
    def __init__(self, teacher_model_path: str, student_model_name: str,
                 config: DistillationConfig, output_dir: str = "./checkpoints/student"):
        self.output_dir = output_dir
        self.config = config
        self.temperature = config.temperature
        self.alpha = config.alpha

        print("Loading teacher model...")
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
        if self.teacher_tokenizer.pad_token is None:
            self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token
        _teacher_bnb = (
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                               bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
            if config.use_4bit else
            BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=config.llm_int8_threshold)
            if config.load_in_8bit else None
        )
        _teacher_kw = dict(device_map="auto", low_cpu_mem_usage=True, trust_remote_code=True)
        if _teacher_bnb:
            _teacher_kw["quantization_config"] = _teacher_bnb
        else:
            _teacher_kw["torch_dtype"] = torch.bfloat16
        _tkw = dict(device_map="auto", low_cpu_mem_usage=True, trust_remote_code=True)
        if _teacher_bnb:
            _tkw["quantization_config"] = _teacher_bnb
        else:
            _tkw["torch_dtype"] = torch.bfloat16
        self.teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_path, **_tkw)
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad_(False)

        print("Loading student model...")
        self.student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
        self.student_tokenizer.pad_token = self.student_tokenizer.eos_token
        base_student = AutoModelForCausalLM.from_pretrained(
            student_model_name, device_map="auto",
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        base_student.gradient_checkpointing_enable()

        if config.use_student_lora:
            targets = config.student_lora_target_modules or ["q_proj", "v_proj"]
            lora_cfg = LoraConfig(
                r=config.student_lora_r, lora_alpha=config.student_lora_alpha,
                target_modules=targets, lora_dropout=0.05,
                bias="none", task_type=TaskType.CAUSAL_LM
            )
            self.student_model = get_peft_model(base_student, lora_cfg)
            self.student_model.print_trainable_parameters()
        else:
            self.student_model = base_student

    def distillation_loss(self, student_logits, teacher_logits, labels, attention_mask):
        shift_s = student_logits[..., :-1, :].contiguous()
        shift_t = teacher_logits[..., :-1, :].contiguous()
        shift_l = labels[..., 1:].contiguous()
        shift_m = attention_mask[..., 1:].contiguous()
        active = shift_m.view(-1) == 1
        as_ = shift_s.view(-1, shift_s.size(-1))[active]
        at_ = shift_t.view(-1, shift_t.size(-1))[active]
        al_ = shift_l.view(-1)[active]

        # hard loss over full student vocab
        hard_loss = nn.CrossEntropyLoss()(as_, al_)

        k = self.config.top_k_logits
        if k > 0:
            # top-k logit distillation: KL over k positions only
            # reduces KL tensor from [N, vocab] to [N, k] — vocab can be 150k+
            # For cross-family pairs (e.g. Llama 128k → TinyLlama 32k), restrict
            # teacher to student vocab before top-k so gather indices stay in bounds.
            with torch.no_grad():
                student_vocab = as_.size(-1)
                t_for_topk = at_[..., :student_vocab] if at_.size(-1) > student_vocab else at_
                actual_k = min(k, student_vocab)
                top_vals, top_idx = t_for_topk.topk(actual_k, dim=-1)  # [N, actual_k]
                soft_t = F.softmax(top_vals / self.temperature, dim=-1)
            s_top = as_.gather(1, top_idx)                        # [N, actual_k]
            log_soft_s = F.log_softmax(s_top / self.temperature, dim=-1)
            soft_loss = F.kl_div(log_soft_s, soft_t, reduction="batchmean") * (self.temperature ** 2)
        else:
            min_vocab = min(as_.size(-1), at_.size(-1))
            with torch.no_grad():
                soft_t = F.softmax(at_[..., :min_vocab] / self.temperature, dim=-1)
            log_soft_s = F.log_softmax(as_[..., :min_vocab] / self.temperature, dim=-1)
            soft_loss = (F.kl_div(log_soft_s, soft_t, reduction="sum")
                        * (self.temperature ** 2) / as_.size(0))

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

    @staticmethod
    def _collate(batch):
        result = {}
        for k in batch[0].keys():
            if isinstance(batch[0][k], str):
                result[k] = [item[k] for item in batch]
            else:
                result[k] = torch.nn.utils.rnn.pad_sequence(
                    [item[k][:256].clone().detach() if isinstance(item[k], torch.Tensor)
                     else torch.tensor(item[k][:256]) for item in batch],
                    batch_first=True, padding_value=0
                )
        return result

    def train(self, train_dataset, eval_dataset=None, run_name: str = "distillation"):
        print("Starting knowledge distillation...")
        trainable = filter(lambda p: p.requires_grad, self.student_model.parameters())
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(trainable, lr=self.config.learning_rate)
            print("Using 8-bit AdamW optimizer")
        except Exception:
            optimizer = torch.optim.AdamW(trainable, lr=self.config.learning_rate)
            print("Using standard AdamW optimizer")

        loader = DataLoader(train_dataset, batch_size=self.config.batch_size,
                            shuffle=True, collate_fn=self._collate)
        total_steps = len(loader) * self.config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, self.config.warmup_steps, total_steps)
        self.student_model.train()
        optimizer.zero_grad()
        history = []
        t_dev = next(self.teacher_model.parameters()).device
        for epoch in range(self.config.epochs):
            total_loss = 0.0
            for batch_idx, batch in enumerate(loader):
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                ids = batch["input_ids"].to(self.student_model.device)
                mask = batch["attention_mask"].to(self.student_model.device)
                lbls = batch["labels"].to(self.student_model.device)
                with torch.inference_mode():
                    if "raw_text" in batch:
                        # Re-tokenize with teacher's own tokenizer to avoid cross-family
                        # vocab mismatch (e.g. student 128k IDs fed to teacher 32k model).
                        t_enc = self.teacher_tokenizer(
                            batch["raw_text"], truncation=True, max_length=ids.shape[1],
                            padding="max_length", return_tensors="pt"
                        )
                        t_ids = t_enc["input_ids"].to(t_dev)
                        t_mask = t_enc["attention_mask"].to(t_dev)
                    else:
                        t_ids, t_mask = ids.to(t_dev), mask.to(t_dev)
                    t_out = self.teacher_model(input_ids=t_ids, attention_mask=t_mask)
                t_logits = t_out.logits.to(self.student_model.device)
                s_out = self.student_model(input_ids=ids, attention_mask=mask)
                loss = self.distillation_loss(s_out.logits, t_logits, lbls, mask)
                del t_logits  # free immediately after loss
                (loss / self.config.gradient_accumulation_steps).backward()
                total_loss += loss.item()

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.student_model.parameters()),
                        self.config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if batch_idx % 10 == 0:
                    print(f"Ep {epoch+1}/{self.config.epochs} | "
                          f"Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")
            avg = total_loss / len(loader)
            history.append(avg)
            print(f"Epoch {epoch+1} avg loss: {avg:.4f}")
        self.student_model.save_pretrained(self.output_dir)
        self.student_tokenizer.save_pretrained(self.output_dir)
        print(f"Phase 2 complete — saved to {self.output_dir}")
        return history
    
## Phase 3
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

        if config.hf_token:
            login(token=config.hf_token)

        self.tokenizer = AutoTokenizer.from_pretrained(
            distilled_model_hf_path
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.teacher_tokenizer = None
        if teacher_model_hf_path:
            self.teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_hf_path)
            if self.teacher_tokenizer.pad_token is None:
                self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token

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

                if self.teacher_tokenizer is not None and "raw_text" in sample:
                    t_enc = self.teacher_tokenizer(
                        sample["raw_text"], truncation=True, max_length=ids.shape[1],
                        padding="max_length", return_tensors="pt"
                    )
                    t_ids = t_enc["input_ids"].to(t_dev)
                    t_mask = t_enc["attention_mask"].to(t_dev)
                else:
                    t_ids, t_mask = ids.to(t_dev), mask.to(t_dev)

                s_logits = _model_forward(self.distilled_model, ids, mask).logits
                t_logits = _model_forward(self.teacher_model, t_ids, t_mask).logits.to(self.device)
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

    def apply_lora_adapters(self, fixed_lora_r: Optional[int] = None):
        target_modules = self.config.target_modules or self.select_target_modules()
        lora_r = self.config.lora_r

        if fixed_lora_r is not None:
            lora_r = fixed_lora_r
            print(f"  Fixed-rank LoRA baseline: r={lora_r}")
        elif self.config.use_dynamic_ranks and self.layer_performance_scores:
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

    def _get_reward_weights(self) -> Tuple[float, float]:
        # Per-family weights take precedence over global config values.
        key = self.config.model_pair_key
        if key and key in MODEL_PAIRS:
            mp = MODEL_PAIRS[key]
            return mp.agreement_weight, mp.kl_reward_weight
        return self.config.agreement_weight, self.config.kl_reward_weight

    def calculate_rewards(self, logits, labels, attention_mask,
                          teacher_logits=None) -> torch.Tensor:
        # Teacher-student agreement reward:
        # reward = accuracy + lambda1*agreement - lambda2*KL_penalty
        B = logits.size(0)
        rewards = torch.zeros(B, device=logits.device)
        agreement_w, kl_w = self._get_reward_weights()

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
                          + agreement_w * agreement
                          - kl_w * kl_pen)
            rewards[i] = reward

        return rewards.detach()

    def compute_ppl(self, eval_dataset: Dataset, num_samples: int = 20) -> float:
        """Perplexity via teacher-forcing cross-entropy (exp of avg loss)."""
        model = self.peft_model or self.distilled_model
        model.eval()
        total_loss, count = 0.0, 0
        with torch.no_grad():
            for idx in range(min(num_samples, len(eval_dataset))):
                sample = eval_dataset[idx]
                ids = sample["input_ids"].unsqueeze(0).to(self.device)
                mask = sample["attention_mask"].unsqueeze(0).to(self.device)
                lbls = sample["labels"].unsqueeze(0).to(self.device)
                out = _model_forward(model, ids, mask, labels=lbls)
                if not torch.isnan(out.loss) and not torch.isinf(out.loss):
                    total_loss += out.loss.item()
                    count += 1
        model.train()
        avg_loss = total_loss / max(count, 1)
        return float(np.exp(min(avg_loss, 20.0)))  # cap to prevent overflow


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
        ppl_baseline: Optional[float] = None
        monitor_ppl = (
            self.config.ppl_monitor_steps > 0
            and eval_dataset is not None
            and (self.config.use_rl or self.config.rl_reward_type == "task_accuracy")
        )
        if monitor_ppl:
            ppl_baseline = self.compute_ppl(eval_dataset)
            print(f"  PPL baseline (pre-training): {ppl_baseline:.2f}")

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
                        if self.teacher_tokenizer is not None and "raw_text" in sample:
                            t_enc = self.teacher_tokenizer(
                                sample["raw_text"], truncation=True, max_length=ids.shape[1],
                                padding="max_length", return_tensors="pt"
                            )
                            t_ids = t_enc["input_ids"].to(t_dev)
                            t_mask = t_enc["attention_mask"].to(t_dev)
                        else:
                            t_ids, t_mask = ids.to(t_dev), mask.to(t_dev)
                        t_out = _model_forward(self.teacher_model, t_ids, t_mask)
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

                # PPL monitoring — detect reward hacking early in RL runs
                if (monitor_ppl
                        and self.config.ppl_monitor_steps > 0
                        and global_step % self.config.ppl_monitor_steps == 0):
                    current_ppl = self.compute_ppl(eval_dataset)
                    ratio = current_ppl / ppl_baseline if ppl_baseline else 1.0
                    flag = ("  [WARNING: possible reward hacking]"
                            if ratio > self.config.ppl_hacking_threshold else "")
                    print(f"  PPL @ step {global_step}: {current_ppl:.2f} "
                          f"(x{ratio:.2f} vs baseline){flag}")

            avg = epoch_loss / len(train_dataset)
            history.append(avg)
            if eval_dataset:
                if self.config.num_eval_seeds > 1:
                    eval_loss, eval_std = self.evaluate_multi_seed(eval_dataset)
                    print(f"  Epoch {epoch+1} — train={avg:.4f} "
                          f"eval={eval_loss:.4f} ± {eval_std:.4f}")
                else:
                    eval_loss = self.evaluate(eval_dataset)
                    # wandb.log({"eval_loss": eval_loss, "epoch": epoch})
                    print(f"  Epoch {epoch+1} — train={avg:.4f} eval={eval_loss:.4f}")

        self.save_model(output_dir)
        if push_to_hub and hub_model_id:
            self.push_to_hub(output_dir, hub_model_id)

        # wandb.finish()
        self.unload_teacher_model()
        return history

    def evaluate(self, eval_dataset: Dataset, num_batches: int = 50,
                 stochastic: bool = False) -> float:
        """Cross-entropy loss over up to num_batches samples.

        stochastic=True keeps dropout active so repeated calls with different
        seeds give variance estimates for multi-seed averaging.
        """
        model = self.peft_model or self.distilled_model
        if not stochastic:
            model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for idx in range(min(num_batches, len(eval_dataset))):
                sample = eval_dataset[idx]
                ids = sample["input_ids"].unsqueeze(0).to(self.device)
                mask = sample["attention_mask"].unsqueeze(0).to(self.device)
                out = _model_forward(model, ids, mask,
                                     labels=sample["labels"].unsqueeze(0).to(self.device))
                if not torch.isnan(out.loss) and not torch.isinf(out.loss):
                    total += out.loss.item()
                    count += 1
        if not stochastic:
            model.train()
        return total / max(count, 1)

    def evaluate_multi_seed(self, eval_dataset: Dataset,
                            num_batches: int = 50) -> Tuple[float, float]:
        """Returns (mean_loss, std_loss) across config.num_eval_seeds seeds.

        Uses stochastic eval (dropout active) so different seeds produce
        different forward-pass outcomes — meaningful variance estimates.
        """
        n = self.config.num_eval_seeds
        if n <= 1:
            return self.evaluate(eval_dataset, num_batches), 0.0
        losses = []
        for seed in range(n):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            losses.append(self.evaluate(eval_dataset, num_batches, stochastic=True))
        mean, std = float(np.mean(losses)), float(np.std(losses))
        print(f"  Multi-seed eval ({n} seeds): loss={mean:.4f} ± {std:.4f}")
        return mean, std

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


## Orchestrator

class HybridPipeline:
    # Full 3-phase pipeline: QLoRA -> KD -> Hybrid KD+RL + Evaluation + Ablation

    def __init__(self, model_pair_key: str, data_path: str,
                 qlora_config: QLoRAConfig, distill_config: DistillationConfig,
                 refinement_config: RefinementConfig,
                 base_output_dir: str = "./models"):
        self.model_pair = MODEL_PAIRS[model_pair_key]
        self.key = model_pair_key
        self.data_path = data_path
        self.qlora_config = qlora_config
        self.distill_config = distill_config
        self.ref_config = refinement_config
        self.base_dir = f"{base_output_dir}/{model_pair_key}"
        self.teacher_out = f"{self.base_dir}/phase1_teacher"
        self.student_out = f"{self.base_dir}/phase2_student"
        self.refined_out = f"{self.base_dir}/phase3_refined"

    def _split_dataset(self, tokenizer, split: float = 0.9, max_length=512):
        full = MSKCaseStudyDataset(self.data_path, tokenizer, max_length=max_length)
        n = int(split * len(full))
        return random_split(full, [n, len(full) - n])

    def run_phase1(self) -> str:
        print("\n" + "="*80 + "\nPHASE 1: QLoRA Teacher Fine-tuning\n" + "="*80)
        tuner = QLoRAFineTuner(self.model_pair.teacher_name, self.teacher_out, self.qlora_config)
        train_ds, eval_ds = self._split_dataset(tuner.tokenizer, max_length=self.qlora_config.max_seq_length)
        tuner.train(train_ds, eval_ds, run_name=f"{self.key}_phase1")
        print(f"Phase 1 saved locally to {self.teacher_out}")
        return self.teacher_out

    def run_phase2(self, teacher_path: str) -> str:
        print("\n" + "="*80 + "\nPHASE 2: Knowledge Distillation\n" + "="*80)
        distiller = KnowledgeDistillationTrainer(
            teacher_model_path=teacher_path,
            student_model_name=self.model_pair.student_name,
            config=self.distill_config, output_dir=self.student_out
        )
        train_ds, _ = self._split_dataset(distiller.student_tokenizer)
        distiller.train(train_ds, run_name=f"{self.key}_phase2")
        print(f"Phase 2 saved locally to {self.student_out}")
        del distiller; clear_gpu_memory()
        return self.student_out

    def run_phase3(self, teacher_path: str, distilled_path: str) -> str:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("\n" + "="*80 + "\nPHASE 3: Hybrid KD+RL Refinement\n" + "="*80)
        self.ref_config.model_pair_key = self.key
        refiner = RefinementTrainer(
            distilled_model_hf_path=distilled_path,
            teacher_model_hf_path=teacher_path,
            config=self.ref_config
        )
        train_ds, eval_ds = self._split_dataset(refiner.tokenizer)
        refiner.train(train_ds, eval_ds, output_dir=self.refined_out,
                      push_to_hub=False, run_name=f"{self.key}_phase3")
        print(f"Phase 3 saved locally to {self.refined_out}")
        del refiner; clear_gpu_memory()
        return self.refined_out

    def run_all(self, skip_phase1: bool = False, skip_phase2: bool = False,
                existing_teacher_path: Optional[str] = None,
                existing_distilled_path: Optional[str] = None) -> Dict:
        print(f"\nHybrid Pipeline — {self.key}")
        print(f"  Teacher: {self.model_pair.teacher_name} ({self.model_pair.teacher_size})")
        print(f"  Student: {self.model_pair.student_name} ({self.model_pair.student_size})")
        teacher_path   = existing_teacher_path   or self.model_pair.tuned_teacher
        distilled_path = existing_distilled_path or self.model_pair.distilled_student
        if not skip_phase1:
            teacher_path = self.run_phase1()
        if not skip_phase2:
            distilled_path = self.run_phase2(teacher_path)
        refined_path = self.run_phase3(teacher_path, distilled_path)
        return {"teacher": teacher_path, "distilled": distilled_path, "refined": refined_path}
    


pipeline = HybridPipeline(
    model_pair_key=MODEL_PAIR_KEY,
    data_path=DATA_PATH,
    qlora_config=qlora_cfg,
    distill_config=distill_cfg,
    refinement_config=refine_cfg,
    base_output_dir="./models",
)

paths = pipeline.run_all()

print(paths)

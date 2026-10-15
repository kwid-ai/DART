import argparse
import gc
import json
import os
from dataclasses import asdict
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

try:
    from .shared import (DATA_PATH, MODEL_PAIRS, SPLITS_PATH, DistillationConfig, QLoRAConfig,
                         RefinementConfig, get_split_datasets, load_causal_lm, load_splits,
                         load_tokenizer, set_seed, tokenize_example, tokenizers_compatible)
    from .refinement import RefinementTrainer
    from .tracking import ProxyGoldenTracker
except ImportError:
    from shared import (DATA_PATH, MODEL_PAIRS, SPLITS_PATH, DistillationConfig, QLoRAConfig,
                        RefinementConfig, get_split_datasets, load_causal_lm, load_splits,
                        load_tokenizer, set_seed, tokenize_example, tokenizers_compatible)
    from refinement import RefinementTrainer
    from tracking import ProxyGoldenTracker

from dotenv import load_dotenv
load_dotenv()

if os.getenv("HF_KEY"):
    os.environ["HF_TOKEN"] = os.getenv("HF_KEY")
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DIR", "./wandb")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def detect_device():
    try:
        import torch_xla.core.xla_model as xm  # noqa: F401
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


## Phase 1 — QLoRA Fine-tuning of Teacher
class QLoRAFineTuner:
    def __init__(self, model_name: str, output_dir: str, config: QLoRAConfig):
        self.model_name = model_name
        self.output_dir = output_dir
        self.config = config
        set_seed(config.seed)
        self.device, self.use_tpu, self.use_cuda = detect_device()
        print(f"Device: {self.device.upper()}")
        if self.use_cuda:
            torch.cuda.empty_cache()
            torch.backends.cuda.matmul.allow_tf32 = True

        self.tokenizer = load_tokenizer(model_name)
        self.tokenizer.padding_side = "right"
        quant = "4bit" if (config.use_4bit and self.use_cuda) else "none"
        self.model = load_causal_lm(model_name, quant=quant,
                                    device_map="auto" if self.use_cuda else None)
        self.model.config.use_cache = False
        if quant != "none":
            self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=True)

        lora_cfg = LoraConfig(
            r=config.lora_r, lora_alpha=config.lora_alpha,
            target_modules=config.target_modules, lora_dropout=config.lora_dropout,
            bias="none", task_type="CAUSAL_LM", inference_mode=False)
        self.model = get_peft_model(self.model, lora_cfg)
        if self.use_cuda:
            for _, p in self.model.named_parameters():
                if p.requires_grad and p.dtype != torch.bfloat16:
                    p.data = p.data.to(torch.bfloat16)
        self.model.print_trainable_parameters()

    @staticmethod
    def _tensors_only(ds):
        # Trainer's collator can't stack the string/metadata fields
        class _View(torch.utils.data.Dataset):
            def __len__(self):
                return len(ds)

            def __getitem__(self, i):
                item = ds[i]
                return {k: item[k] for k in ("input_ids", "attention_mask", "labels")}
        return _View()

    def train(self, train_dataset, eval_dataset=None, run_name: str = "qlora"):
        if self.use_cuda:
            torch.cuda.empty_cache()
        args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            fp16=False, bf16=True, bf16_full_eval=True,
            gradient_checkpointing=True,
            optim="adamw_torch", max_grad_norm=0.3, weight_decay=0.01,
            logging_steps=10, save_strategy="epoch",
            eval_strategy="epoch" if eval_dataset else "no",
            report_to="none", logging_dir="./wandb/logs",
            save_total_limit=1, load_best_model_at_end=bool(eval_dataset),
            dataloader_num_workers=0, dataloader_pin_memory=False,
            ddp_find_unused_parameters=False, seed=self.config.seed,
            remove_unused_columns=False,
        )
        trainer = Trainer(
            model=self.model, args=args,
            train_dataset=self._tensors_only(train_dataset),
            eval_dataset=self._tensors_only(eval_dataset) if eval_dataset else None,
            data_collator=default_data_collator,  # keeps prompt/pad labels at -100
        )
        trainer.train()
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        with open(os.path.join(self.output_dir, "phase1_meta.json"), "w") as f:
            json.dump({"config": asdict(self.config), "base_model": self.model_name,
                       "log_history": trainer.state.log_history}, f, indent=2, default=str)
        print(f"Phase 1 complete — saved to {self.output_dir}")
        return trainer


## Phase 2 — Knowledge Distillation
class KnowledgeDistillationTrainer:
    def __init__(self, teacher_model_path: str, student_model_name: str,
                 config: DistillationConfig, output_dir: str = "./checkpoints/student"):
        self.output_dir = output_dir
        self.config = config
        self.temperature = config.temperature
        self.alpha = config.alpha
        set_seed(config.seed)

        self.teacher_tokenizer = load_tokenizer(teacher_model_path)
        self.student_tokenizer = load_tokenizer(student_model_name)
        self.same_ids = tokenizers_compatible(self.teacher_tokenizer, self.student_tokenizer)
        if not self.same_ids:
            if not config.allow_cross_tokenizer_kd:
                raise ValueError(
                    f"{teacher_model_path} and {student_model_name} use different vocabularies: "
                    "logit-level KD would compare unrelated token ids. Use a shared-tokenizer pair "
                    "(or a cross-tokenizer method such as ULD), or set allow_cross_tokenizer_kd=True "
                    "to reproduce the legacy V1/V2 behaviour.")
            print("  WARNING: cross-tokenizer KD (legacy mode) — soft targets are not well-defined.")

        print("Loading teacher model...")
        quant = "4bit" if config.use_4bit else "8bit" if config.load_in_8bit else "none"
        self.teacher_model = load_causal_lm(teacher_model_path, quant=quant, merge_adapter=False)
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad_(False)

        print("Loading student model...")
        base_student = load_causal_lm(student_model_name, quant="none")
        base_student.gradient_checkpointing_enable()
        if config.use_student_lora:
            lora_cfg = LoraConfig(
                r=config.student_lora_r, lora_alpha=config.student_lora_alpha,
                target_modules=config.student_lora_target_modules or ["q_proj", "v_proj"],
                lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
            self.student_model = get_peft_model(base_student, lora_cfg)
            self.student_model.print_trainable_parameters()
        else:
            self.student_model = base_student

    def distillation_loss(self, student_logits, teacher_logits, labels):
        """alpha * soft KD + (1 - alpha) * CE, both over answer tokens only."""
        ans = labels[:, 1:] != -100
        as_ = student_logits[:, :-1][ans]
        al_ = labels[:, 1:][ans]
        at_ = teacher_logits[:, :-1][ans.to(teacher_logits.device)].to(as_.device)
        hard_loss = nn.CrossEntropyLoss()(as_.float(), al_)

        k = self.config.top_k_logits
        T = self.temperature
        student_vocab = as_.size(-1)
        at_ = at_[..., :student_vocab]
        if k > 0:
            # top-k logit distillation: KL over the teacher's top-k tokens (renormalised)
            with torch.no_grad():
                top_vals, top_idx = at_.float().topk(min(k, at_.size(-1)), dim=-1)
                soft_t = F.softmax(top_vals / T, dim=-1)
            log_soft_s = F.log_softmax(as_.gather(1, top_idx).float() / T, dim=-1)
        else:
            v = min(student_vocab, at_.size(-1))
            with torch.no_grad():
                soft_t = F.softmax(at_[..., :v].float() / T, dim=-1)
            log_soft_s = F.log_softmax(as_[..., :v].float() / T, dim=-1)
        soft_loss = F.kl_div(log_soft_s, soft_t, reduction="batchmean") * T ** 2
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

    @staticmethod
    def _collate(batch):
        out = {}
        for k in batch[0]:
            if isinstance(batch[0][k], torch.Tensor):
                out[k] = torch.stack([b[k] for b in batch])
            else:
                out[k] = [b[k] for b in batch]
        return out

    def _teacher_batch(self, batch, ids, mask):
        if self.same_ids:
            return ids, mask
        encs = [tokenize_example(self.teacher_tokenizer, t, a, ids.shape[1])
                for t, a in zip(batch["title"], batch["answer"])]
        return (torch.stack([e["input_ids"] for e in encs]),
                torch.stack([e["attention_mask"] for e in encs]))

    def train(self, train_dataset, calib_dataset=None, run_name: str = "distillation"):
        print("Starting knowledge distillation...")
        cfg = self.config
        trainable = [p for p in self.student_model.parameters() if p.requires_grad]
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(trainable, lr=cfg.learning_rate)
        except Exception:
            optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate)

        loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                            collate_fn=self._collate,
                            generator=torch.Generator().manual_seed(cfg.seed))
        total_steps = max(1, len(loader) * cfg.epochs // cfg.gradient_accumulation_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, cfg.warmup_steps, total_steps)
        t_dev = next(self.teacher_model.parameters()).device
        s_dev = self.student_model.device

        tracker = None
        if cfg.track_every > 0 and calib_dataset is not None:
            tracker = ProxyGoldenTracker(calib_dataset, s_dev, cfg.track_num_samples, self.same_ids)
            tracker.record(0, self.student_model, self.teacher_model)

        self.student_model.train()
        optimizer.zero_grad()
        history, opt_step = [], 0
        for epoch in range(cfg.epochs):
            total_loss = 0.0
            for batch_idx, batch in enumerate(loader):
                ids = batch["input_ids"].to(s_dev)
                mask = batch["attention_mask"].to(s_dev)
                lbls = batch["labels"].to(s_dev)
                with torch.no_grad():
                    t_ids, t_mask = self._teacher_batch(batch, ids, mask)
                    t_logits = self.teacher_model(input_ids=t_ids.to(t_dev),
                                                  attention_mask=t_mask.to(t_dev)).logits
                s_out = self.student_model(input_ids=ids, attention_mask=mask)
                loss = self.distillation_loss(s_out.logits, t_logits, lbls)
                del t_logits
                (loss / cfg.gradient_accumulation_steps).backward()
                total_loss += loss.item()

                if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    opt_step += 1
                    if tracker and opt_step % cfg.track_every == 0:
                        tracker.record(opt_step, self.student_model, self.teacher_model)
                if batch_idx % 10 == 0:
                    print(f"Ep {epoch+1}/{cfg.epochs} | Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")
            avg = total_loss / len(loader)
            history.append(avg)
            print(f"Epoch {epoch+1} avg loss: {avg:.4f}")

        os.makedirs(self.output_dir, exist_ok=True)
        # Save merged weights so Phase 3 can stack a fresh LoRA on a plain checkpoint
        model = self.student_model.merge_and_unload() if hasattr(self.student_model, "merge_and_unload") \
            else self.student_model
        model.save_pretrained(self.output_dir)
        self.student_tokenizer.save_pretrained(self.output_dir)
        with open(os.path.join(self.output_dir, "phase2_meta.json"), "w") as f:
            json.dump({"config": asdict(cfg), "history": history, "same_tokenizer": self.same_ids,
                       "tracking": tracker.records if tracker else [],
                       "teacher_hacking": tracker.summary() if tracker else None}, f, indent=2, default=str)
        print(f"Phase 2 complete — saved to {self.output_dir}")
        return history


## Orchestrator
class HybridPipeline:
    """Phase 1 (QLoRA teacher) -> Phase 2 (KD) -> Phase 3 (refinement), all on the
    shared grouped split in splits.json."""

    def __init__(self, model_pair_key: str, data_path: str,
                 qlora_config: QLoRAConfig, distill_config: DistillationConfig,
                 refinement_config: RefinementConfig, base_output_dir: str = "./models",
                 splits_path: str = SPLITS_PATH):
        self.model_pair = MODEL_PAIRS[model_pair_key]
        self.key = model_pair_key
        self.data_path = data_path
        self.splits_path = splits_path
        self.qlora_config = qlora_config
        self.distill_config = distill_config
        self.ref_config = refinement_config
        self.base_dir = f"{base_output_dir}/{model_pair_key}"
        # ModelPair paths are what evaluation reads, so write to them when set
        mp = self.model_pair
        self.teacher_out = mp.tuned_teacher or f"{self.base_dir}/phase1_teacher"
        self.student_out = mp.distilled_student or f"{self.base_dir}/phase2_student"
        self.student_base_teacher_out = (mp.distilled_student_base_teacher
                                         or f"{self.base_dir}/phase2_student_base_teacher")
        self.refined_out = f"{self.base_dir}/phase3_refined"
        load_splits(splits_path, data_path)

    def _datasets(self, tokenizer, max_length):
        return get_split_datasets(tokenizer, max_length, self.data_path, self.splits_path)

    def run_phase1(self) -> str:
        print("\n" + "=" * 80 + "\nPHASE 1: QLoRA Teacher Fine-tuning\n" + "=" * 80)
        tuner = QLoRAFineTuner(self.model_pair.teacher_name, self.teacher_out, self.qlora_config)
        ds = self._datasets(tuner.tokenizer, self.qlora_config.max_seq_length)
        tuner.train(ds["train"], ds["calibration"], run_name=f"{self.key}_phase1")
        del tuner
        clear_gpu_memory()
        return self.teacher_out

    def run_phase2(self, teacher_path: str, output_dir: Optional[str] = None) -> str:
        print("\n" + "=" * 80 + f"\nPHASE 2: Knowledge Distillation from {teacher_path}\n" + "=" * 80)
        output_dir = output_dir or self.student_out
        distiller = KnowledgeDistillationTrainer(teacher_path, self.model_pair.student_name,
                                                 self.distill_config, output_dir)
        ds = self._datasets(distiller.student_tokenizer, self.distill_config.max_seq_length)
        distiller.train(ds["train"], ds["calibration"], run_name=f"{self.key}_phase2")
        del distiller
        clear_gpu_memory()
        return output_dir

    def run_phase2_from_base_teacher(self) -> str:
        """A1b control: same Phase 2, untuned teacher."""
        return self.run_phase2(self.model_pair.teacher_name, self.student_base_teacher_out)

    def run_phase3(self, teacher_path: str, distilled_path: str) -> str:
        print("\n" + "=" * 80 + "\nPHASE 3: Refinement\n" + "=" * 80)
        self.ref_config.model_pair_key = self.key
        refiner = RefinementTrainer(distilled_path, teacher_path, self.ref_config)
        ds = self._datasets(refiner.tokenizer, self.ref_config.max_seq_length)
        refiner.train(ds["train"], ds["calibration"], output_dir=self.refined_out)
        del refiner
        clear_gpu_memory()
        return self.refined_out

    def run_all(self, skip_phase1: bool = False, skip_phase2: bool = False,
                skip_phase3: bool = False, base_teacher_student: bool = False) -> Dict:
        print(f"\nHybrid Pipeline — {self.key}")
        print(f"  Teacher: {self.model_pair.teacher_name} ({self.model_pair.teacher_size})")
        print(f"  Student: {self.model_pair.student_name} ({self.model_pair.student_size})")
        teacher_path = self.model_pair.tuned_teacher
        distilled_path = self.model_pair.distilled_student
        paths = {}
        if not skip_phase1:
            teacher_path = self.run_phase1()
        if not skip_phase2:
            distilled_path = self.run_phase2(teacher_path)
        if base_teacher_student:
            paths["distilled_base_teacher"] = self.run_phase2_from_base_teacher()
        if not skip_phase3:
            paths["refined"] = self.run_phase3(teacher_path, distilled_path)
        paths.update({"teacher": teacher_path, "distilled": distilled_path})
        return paths


def main():
    try:
        from .config import MODEL_PAIR_KEY, distill_cfg, qlora_cfg, refine_cfg
    except ImportError:
        from config import MODEL_PAIR_KEY, distill_cfg, qlora_cfg, refine_cfg

    parser = argparse.ArgumentParser(description="DART training pipeline")
    parser.add_argument("--pair", default=MODEL_PAIR_KEY, choices=sorted(MODEL_PAIRS))
    parser.add_argument("--skip-phase1", action="store_true")
    parser.add_argument("--skip-phase2", action="store_true")
    parser.add_argument("--skip-phase3", action="store_true")
    parser.add_argument("--base-teacher-student", action="store_true",
                        help="also distil a student from the untuned teacher (A1b)")
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--splits", default=SPLITS_PATH)
    args = parser.parse_args()

    pipeline = HybridPipeline(args.pair, args.data, qlora_cfg, distill_cfg, refine_cfg,
                              base_output_dir="./models", splits_path=args.splits)
    print(pipeline.run_all(args.skip_phase1, args.skip_phase2, args.skip_phase3,
                           args.base_teacher_student))


if __name__ == "__main__":
    main()

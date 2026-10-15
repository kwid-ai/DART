import json
import os
from dataclasses import asdict
from typing import Dict

import torch

try:
    from .shared import (PROMPT_TEMPLATE, RefinementConfig, build_prompt, count_trainable_params,
                         load_causal_lm, load_tokenizer, set_seed, tokenizers_compatible)
    from .refinement import decoder_layers
except ImportError:
    from shared import (PROMPT_TEMPLATE, RefinementConfig, build_prompt, count_trainable_params,
                        load_causal_lm, load_tokenizer, set_seed, tokenizers_compatible)
    from refinement import decoder_layers


def _import_gkd():
    try:
        from trl import GKDConfig, GKDTrainer
    except ImportError:
        from trl.experimental.gkd import GKDConfig, GKDTrainer  # newer TRL releases
    return GKDConfig, GKDTrainer


def _gkd_args(GKDConfig, **kwargs):
    # Field names drift across TRL versions (e.g. max_seq_length -> max_length); keep only known ones.
    fields = getattr(GKDConfig, "__dataclass_fields__", {})
    return GKDConfig(**{k: v for k, v in kwargs.items() if k in fields})


class GKDRefiner:
    """Starts from the distilled student with the same LoRA layout and rank budget as
    the R-ladder (uniform rank on every layer's module_types) and the same epochs."""

    def __init__(self, student_path: str, teacher_path: str, config: RefinementConfig,
                 lmbda: float = 0.5, beta: float = 0.5, gkd_temperature: float = 0.9,
                 max_new_tokens: int = 256):
        self.config = config
        self.student_path, self.teacher_path = student_path, teacher_path
        self.lmbda, self.beta = lmbda, beta
        self.gkd_temperature, self.max_new_tokens = gkd_temperature, max_new_tokens
        self.tokenizer = load_tokenizer(student_path)
        if not tokenizers_compatible(self.tokenizer, load_tokenizer(teacher_path)):
            raise ValueError("GKD needs a shared tokenizer between teacher and student.")
        self.peft_model = None
        self.meta: Dict = {}

    def _fit_answer(self, title: str, answer: str, margin: int = 8) -> str:
        # TRL truncates from the left, which drops the prompt when the answer is long;
        # truncate the answer on the right instead, as the other trainers do.
        prompt_len = len(self.tokenizer(build_prompt(self.tokenizer, title), add_special_tokens=False)["input_ids"])
        budget = self.config.max_seq_length - prompt_len - margin
        ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        return answer if len(ids) <= budget else self.tokenizer.decode(ids[:max(budget, 1)])

    @staticmethod
    def _align_vocab(model_a, model_b):
        """Register forward hooks on both models so their logit vocab dims match.
        Clips the larger vocab to the smaller one; the few extra tokens carry negligible
        probability mass and the shared Gemma-3 tokenizer maps them identically."""
        va = getattr(model_a.config, "vocab_size", None)
        vb = getattr(model_b.config, "vocab_size", None)
        if va is None or vb is None or va == vb:
            return
        min_v = min(va, vb)
        def _clip(v):
            def hook(module, inp, out):
                if hasattr(out, "logits") and out.logits is not None and out.logits.shape[-1] > v:
                    out.logits = out.logits[..., :v]
                return out
            return hook
        if va > min_v:
            model_a.register_forward_hook(_clip(min_v))
        if vb > min_v:
            model_b.register_forward_hook(_clip(min_v))
        print(f"  GKD: aligned vocab {va} × {vb} → {min_v}")

    def train(self, train_dataset, calib_dataset=None, output_dir: str = "./gkd") -> Dict:
        from peft import LoraConfig, TaskType
        from datasets import Dataset as HFDataset

        GKDConfig, GKDTrainer = _import_gkd()
        cfg = self.config
        set_seed(cfg.seed)
        student = load_causal_lm(self.student_path, quant="none", device_map=None).to(cfg.device)
        teacher = load_causal_lm(self.teacher_path, quant=cfg.teacher_quant, merge_adapter=False)
        teacher.eval()
        self._align_vocab(student, teacher)

        prefix, layers = decoder_layers(student)
        types = set(cfg.module_types)
        targets = [n for n, m in student.named_modules()
                   if n.startswith(prefix + ".") and n.split(".")[-1] in types and isinstance(m, torch.nn.Linear)]
        peft_cfg = LoraConfig(r=cfg.lora_r, lora_alpha=int(cfg.lora_alpha_ratio * cfg.lora_r),
                              target_modules=targets, lora_dropout=cfg.lora_dropout,
                              bias="none", task_type=TaskType.CAUSAL_LM)

        rows = [{"messages": [{"role": "user", "content": PROMPT_TEMPLATE.format(title=c["title"])},
                              {"role": "assistant", "content": self._fit_answer(c["title"], c["use_case"])}]}
                for c in train_dataset.data]
        args = _gkd_args(
            GKDConfig, output_dir=output_dir, num_train_epochs=cfg.num_epochs,
            max_steps=cfg.max_steps or -1,
            per_device_train_batch_size=1, gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate, warmup_steps=cfg.warmup_steps, weight_decay=cfg.weight_decay,
            max_grad_norm=cfg.max_grad_norm, lmbda=self.lmbda, beta=self.beta,
            temperature=self.gkd_temperature, max_new_tokens=self.max_new_tokens,
            max_length=cfg.max_seq_length, max_seq_length=cfg.max_seq_length,
            seed=cfg.seed, bf16=True, report_to="none", logging_steps=cfg.logging_steps,
            save_strategy="no", remove_unused_columns=False,
        )
        trainer = GKDTrainer(model=student, teacher_model=teacher, args=args,
                             train_dataset=HFDataset.from_list(rows),
                             processing_class=self.tokenizer, peft_config=peft_cfg)
        trainer.train()
        self.peft_model = trainer.model
        os.makedirs(output_dir, exist_ok=True)
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.meta = {"method": "gkd", "config": asdict(cfg), "lmbda": self.lmbda, "beta": self.beta,
                     "gkd_temperature": self.gkd_temperature, "max_new_tokens": self.max_new_tokens,
                     "trainable_params": count_trainable_params(self.peft_model),
                     "total_steps": trainer.state.global_step, "same_tokenizer": True,
                     "selected_layers": list(range(len(layers))),
                     "rank_map": {l: cfg.lora_r for l in range(len(layers))},
                     "log_history": trainer.state.log_history}
        with open(os.path.join(output_dir, "refinement_meta.json"), "w") as f:
            json.dump(self.meta, f, indent=2, default=str)
        del teacher
        torch.cuda.empty_cache()
        return self.meta

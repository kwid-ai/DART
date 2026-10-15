"""Proxy-vs-golden tracking for the teacher-hacking analysis.

Teacher hacking (Tiapkin et al., 2025): the *proxy* objective — imitating the teacher —
keeps improving while the *golden* objective — true task quality — gets worse.

  proxy  = KL(teacher || student) on answer tokens of held-out prompts (lower = closer imitation)
  golden = student NLL on held-out gold references (lower = better), plus any
           optional golden_fn(model) -> float, e.g. a validated judge score (higher = better)
"""
import json
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .shared import model_forward, tokenize_example
except ImportError:
    from shared import model_forward, tokenize_example


@torch.no_grad()
def proxy_golden_metrics(student, teacher, dataset, device, num_samples: int = 16,
                         same_ids: bool = True, teacher_tokenizer=None) -> Dict[str, Optional[float]]:
    was_training = student.training
    student.eval()
    nll_sum, kl_sum, n_tok = 0.0, 0.0, 0
    for idx in range(min(num_samples, len(dataset))):
        s = dataset[idx]
        ids = s["input_ids"].unsqueeze(0).to(device)
        mask = s["attention_mask"].unsqueeze(0).to(device)
        labels = s["labels"].unsqueeze(0).to(device)
        ans = labels[:, 1:] != -100
        if not ans.any():
            continue
        s_logits = model_forward(student, ids, mask).logits[:, :-1][ans].float()
        gold = labels[:, 1:][ans]
        nll_sum += F.cross_entropy(s_logits, gold, reduction="sum").item()
        if teacher is not None and same_ids:
            t_dev = next(teacher.parameters()).device
            t_logits = model_forward(teacher, ids.to(t_dev), mask.to(t_dev)).logits[:, :-1]
            t_logits = t_logits[ans.to(t_dev)].to(device).float()
            v = min(s_logits.size(-1), t_logits.size(-1))
            kl_sum += F.kl_div(F.log_softmax(s_logits[:, :v], -1), F.log_softmax(t_logits[:, :v], -1),
                               log_target=True, reduction="sum").item()
        n_tok += int(ans.sum())
    if was_training:
        student.train()
    if n_tok == 0:
        return {"golden_nll": None, "proxy_kl": None}
    return {"golden_nll": nll_sum / n_tok,
            "proxy_kl": (kl_sum / n_tok) if (teacher is not None and same_ids) else None}


class ProxyGoldenTracker:
    def __init__(self, dataset, device, num_samples: int = 16, same_ids: bool = True,
                 golden_fn: Optional[Callable] = None):
        self.dataset, self.device = dataset, device
        self.num_samples, self.same_ids = num_samples, same_ids
        self.golden_fn = golden_fn
        self.records: List[Dict] = []

    def record(self, step: int, student, teacher) -> Dict:
        rec = {"step": step, **proxy_golden_metrics(student, teacher, self.dataset, self.device,
                                                    self.num_samples, self.same_ids)}
        if self.golden_fn is not None:
            rec["golden_score"] = float(self.golden_fn(student))
        self.records.append(rec)
        print(f"  [track] step {step}: " + " ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in rec.items() if k != "step"))
        return rec

    def summary(self) -> Dict:
        return detect_teacher_hacking(self.records)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"records": self.records, "summary": self.summary()}, f, indent=2)


def detect_teacher_hacking(records: List[Dict], golden_key: str = "golden_nll",
                           golden_higher_is_better: bool = False) -> Dict:
    """Flags checkpoint intervals where the proxy improved while the golden metric got worse,
    and the overall pattern after the best-golden checkpoint."""
    recs = [r for r in records if r.get("proxy_kl") is not None and r.get(golden_key) is not None]
    if len(recs) < 3:
        return {"n_checkpoints": len(recs), "verdict": "insufficient checkpoints"}
    proxy = np.array([r["proxy_kl"] for r in recs])
    golden = np.array([r[golden_key] for r in recs])
    g = golden if golden_higher_is_better else -golden   # higher = better from here on
    divergent = [(recs[i]["step"], recs[i + 1]["step"]) for i in range(len(recs) - 1)
                 if proxy[i + 1] < proxy[i] and g[i + 1] < g[i]]
    best = int(np.argmax(g))
    after = slice(best, None)
    proxy_after = proxy[after][-1] - proxy[after][0] if best < len(recs) - 1 else 0.0
    golden_after = g[after][-1] - g[after][0] if best < len(recs) - 1 else 0.0
    hacking = best < len(recs) - 1 and proxy_after < 0 and golden_after < 0
    return {
        "n_checkpoints": len(recs),
        "golden_key": golden_key,
        "divergent_intervals": divergent,
        "frac_divergent_intervals": len(divergent) / (len(recs) - 1),
        "best_golden_step": recs[best]["step"],
        "proxy_change_after_best": float(proxy_after),
        "golden_change_after_best": float(golden_after),
        "verdict": ("teacher hacking pattern: proxy kept improving after golden peaked"
                    if hacking else "no sustained proxy/golden divergence"),
    }

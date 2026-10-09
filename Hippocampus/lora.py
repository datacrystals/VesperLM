"""Manual LoRA for VesperLinearLM (PEFT is not installed in the vesper venv).

Attaches low-rank A@B side branches to chosen `nn.Linear` modules (by default
every attention q/o projection: GQA `wq`/`wo` and GLA `q_proj`/`o_proj`;
optionally the MoE router `gate`). The trunk stays frozen. Deltas save/load as
small files (~130KB at rank 8 for the 118M model); merge/unmerge folds the
delta into the base weights for inference.

Which projections get wrapped is selected by a target profile (see
TARGET_PROFILES): `"gla_gqa"` is the legacy GLA+GQA stack (the default, byte-
compatible with the original behaviour) and `"kda_mla"` is the Vesper-K stack
(fla KimiDeltaAttention / MultiheadLatentAttention).

LoRA math:  y = W x + (alpha/r) * B (A x)
  A: (rank, in_features), B: (out_features, rank); A ~ N(0, 1/rank), B = 0
  so the branch starts as an exact no-op.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DELTA_FORMAT = "vesper_lora_delta_v1"

DEFAULT_TARGET_PROFILE = "gla_gqa"

# Default targets: attention q/o projections of both legacy layer kinds.
DEFAULT_TARGETS = ("wq", "wo", "q_proj", "o_proj")
ROUTER_TARGETS = ("gate",)  # MoE TopKRouter.gate (basename match)

# Vesper-K targets (KimiDeltaAttention / MultiheadLatentAttention).
# KDA (fla/layers/kda.py): q_proj/k_proj/v_proj/o_proj are nn.Linear.
# MLA (fla/layers/mla.py): q_proj is nn.Linear (when q_lora_rank is None —
# the VesperLinearLM default), k_rope is nn.Linear, and kv_proj is
# nn.Sequential(Linear, RMSNorm, Linear) so its projections are named by
# dotted suffix ("kv_proj.0" / "kv_proj.2"). Vesper-K routers: TopKRouter.gate
# or PassportRouter.query.
KDA_MLA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "k_rope",
                   "kv_proj.0", "kv_proj.2")
KDA_MLA_ROUTER_TARGETS = ("gate", "query")

TARGET_PROFILES = {
    "gla_gqa": {"targets": DEFAULT_TARGETS, "router_targets": ROUTER_TARGETS},
    "kda_mla": {"targets": KDA_MLA_TARGETS,
                "router_targets": KDA_MLA_ROUTER_TARGETS},
}


class LoRALinear(nn.Module):
    """Frozen nn.Linear + trainable rank-r side branch."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        for p in self.base.parameters():
            p.requires_grad = False
        self.lora_A = nn.Parameter(
            torch.randn(rank, base.in_features, dtype=base.weight.dtype,
                        device=base.weight.device) / rank)
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, rank, dtype=base.weight.dtype,
                        device=base.weight.device))
        self.merged = False
        self.enabled = True

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.enabled and not self.merged:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return out

    @torch.no_grad()
    def merge(self):
        if self.merged:
            return
        self.base.weight.data += self.scaling * (self.lora_B @ self.lora_A).to(self.base.weight.dtype)
        self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if not self.merged:
            return
        self.base.weight.data -= self.scaling * (self.lora_B @ self.lora_A).to(self.base.weight.dtype)
        self.merged = False

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}, merged={self.merged}"


def _basename(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def get_target_profile(name: str = DEFAULT_TARGET_PROFILE) -> Dict:
    try:
        return TARGET_PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown target_profile {name!r} "
                         f"(expected one of {sorted(TARGET_PROFILES)})") from None


def profile_targets(name: str = DEFAULT_TARGET_PROFILE) -> Tuple[str, ...]:
    return tuple(get_target_profile(name)["targets"])


def profile_router_targets(name: str = DEFAULT_TARGET_PROFILE) -> Tuple[str, ...]:
    return tuple(get_target_profile(name)["router_targets"])


def _name_matches(child_name: str, full: str, targets) -> bool:
    """A module is targeted by leaf basename, or by a dotted suffix of its
    full path (e.g. "kv_proj.0" matches "layers.3.attn.mla.kv_proj.0")."""
    if _basename(child_name) in targets:
        return True
    return any("." in t and (full == t or full.endswith("." + t)) for t in targets)


def attach_lora(model: nn.Module, targets: Optional[Iterable[str]] = None,
                rank: int = 8, alpha: float = 16.0,
                include_router: bool = False,
                target_profile: str = DEFAULT_TARGET_PROFILE) -> List[str]:
    """Replace matching nn.Linear modules with LoRALinear in-place.

    `targets=None` uses the profile's projection targets; explicit `targets`
    override the profile. With `include_router`, the profile's router targets
    are added. Returns the dotted names of wrapped modules. Weight tying and
    already-wrapped modules are left alone.
    """
    profile = get_target_profile(target_profile)
    if targets is None:
        targets = profile["targets"]
    wanted = set(targets) | (set(profile["router_targets"]) if include_router else set())
    wrapped: List[str] = []

    def recurse(parent: nn.Module, prefix: str):
        for child_name, child in list(parent.named_children()):
            full = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, LoRALinear):
                continue
            if isinstance(child, nn.Linear) and _name_matches(child_name, full, wanted):
                setattr(parent, child_name, LoRALinear(child, rank=rank, alpha=alpha))
                wrapped.append(full)
            else:
                recurse(child, full)

    recurse(model, "")
    return wrapped


def lora_parameters(model: nn.Module):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            yield m.lora_A
            yield m.lora_B


def lora_modules(model: nn.Module) -> Dict[str, LoRALinear]:
    return {name: m for name, m in model.named_modules() if isinstance(m, LoRALinear)}


def freeze_trunk(model: nn.Module):
    """Every parameter except LoRA A/B is frozen."""
    lora_ids = {id(p) for p in lora_parameters(model)}
    for p in model.parameters():
        p.requires_grad = id(p) in lora_ids


def set_lora_enabled(model: nn.Module, enabled: bool):
    """Toggle LoRA branches (merged modules stay merged).

    Used to get pure-base logits for the KL penalty without a second model.
    """
    for m in lora_modules(model).values():
        m.enabled = enabled


def merge_all(model: nn.Module):
    for m in lora_modules(model).values():
        m.merge()


def unmerge_all(model: nn.Module):
    for m in lora_modules(model).values():
        m.unmerge()


def save_delta(model: nn.Module, path: str, meta: Optional[Dict] = None) -> str:
    """Save only the LoRA A/B tensors + config as a small delta file."""
    mods = lora_modules(model)
    if not mods:
        raise ValueError("model has no LoRA modules attached")
    any_mod = next(iter(mods.values()))
    state = {}
    for name, m in mods.items():
        state[f"{name}.lora_A"] = m.lora_A.detach().cpu()
        state[f"{name}.lora_B"] = m.lora_B.detach().cpu()
    payload = {
        "format": DELTA_FORMAT,
        "rank": any_mod.rank,
        "alpha": any_mod.alpha,
        "targets": sorted(mods.keys()),
        "state": state,
        "meta": meta or {},
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save(payload, path)
    return path


def load_delta(model: nn.Module, path: str,
               targets: Optional[Iterable[str]] = None,
               include_router: bool = False,
               target_profile: str = DEFAULT_TARGET_PROFILE) -> Dict:
    """Attach LoRA if needed and load an A/B delta file onto the model."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != DELTA_FORMAT:
        raise ValueError(f"not a {DELTA_FORMAT} file: {path}")
    rank, alpha = payload["rank"], payload["alpha"]
    mods = lora_modules(model)
    if not mods:
        attach_lora(model, targets=targets, rank=rank, alpha=alpha,
                    include_router=include_router, target_profile=target_profile)
        mods = lora_modules(model)
    missing = [k for k in payload["targets"] if k not in mods]
    if missing:
        raise ValueError(f"delta expects modules not on model: {missing}")
    for name, m in mods.items():
        a = payload["state"].get(f"{name}.lora_A")
        b = payload["state"].get(f"{name}.lora_B")
        if a is None:
            continue  # module newly attached, not in this delta: stays zero-init
        if tuple(a.shape) != tuple(m.lora_A.shape) or tuple(b.shape) != tuple(m.lora_B.shape):
            raise ValueError(f"shape mismatch for {name}: delta {a.shape}/{b.shape} "
                             f"vs model {tuple(m.lora_A.shape)}/{tuple(m.lora_B.shape)}")
        m.lora_A.data.copy_(a.to(m.lora_A.dtype))
        m.lora_B.data.copy_(b.to(m.lora_B.dtype))
    return payload.get("meta", {})


def delta_num_params(path: str) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return sum(v.numel() for v in payload["state"].values())


def describe(model: nn.Module) -> str:
    lines = []
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in lora_parameters(model))
    for name, m in lora_modules(model).items():
        lines.append(f"  {name}: A{tuple(m.lora_A.shape)} B{tuple(m.lora_B.shape)}")
    return (f"LoRA modules: {len(lines)}\n" + "\n".join(lines) +
            f"\ntrainable {trainable:,} / total {total:,} "
            f"({100.0 * trainable / total:.3f}%)")

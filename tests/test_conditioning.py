"""CPU-only checks for the conditioning-strength mechanisms (CFG + FiLM).

Run with:  JAX_PLATFORMS=cpu .venv/bin/python tests/test_conditioning.py
(or via pytest if available). No solver/JAX needed — pure torch.

Covers:
  (a) default-config models (cfg_dropout=0, cfg_scale=1, cond_mode=concat) produce
      state_dicts WITHOUT any of the new optional keys (backward compatibility);
  (b) cfg/film variants train one step and sample without error;
  (c) sampling with cfg_scale=3 differs from cfg_scale=1 under the same seed,
      including composed with the spectrum noise prior and edit-mode sampling;
  (d) old checkpoints (no null_quant_embed / FiLM keys) load into cfg/film-enabled
      models and vice versa via checkpoint.load_generator_state_dict.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poolbased_surrogate.checkpoint import load_generator_state_dict  # noqa: E402
from poolbased_surrogate.models.ddpm import DDPM1D, FlowMatching1D  # noqa: E402

RES = 64
NQ = 4
COMMON = dict(
    resolution=RES,
    hidden=16,
    steps=4,
    loss_conditional=False,
    param_dim=2,
    param_mode="condition",
    residual_blocks=2,
    kernel_size=5,
    n_quantiles=NQ,
)
OPTIONAL_KEY_TOKENS = ("null_quant_embed", ".film.")


def _batch(n: int = 8):
    clean = torch.randn(n, 1, RES)
    params = torch.rand(n, 2) * 2.0 - 1.0
    qlab = torch.randint(0, NQ, (n,))
    return clean, params, qlab


def _has_optional_keys(model) -> bool:
    return any(any(tok in k for tok in OPTIONAL_KEY_TOKENS) for k in model.state_dict())


def test_default_state_dict_has_no_new_keys():
    for cls in (FlowMatching1D, DDPM1D):
        for rb, nq in ((0, 0), (0, NQ), (2, NQ)):
            kwargs = dict(COMMON, residual_blocks=rb, n_quantiles=nq, loss_conditional=nq == 0)
            model = cls(**kwargs)
            assert not _has_optional_keys(model), f"{cls.__name__} default state_dict gained new keys"
            assert model.null_quant_embed is None
    print("ok: default-config state_dicts contain no cfg/film keys")


def test_variants_train_one_step_and_sample():
    variants = [
        dict(cfg_dropout=0.15, cfg_scale=3.0),
        dict(cond_mode="film"),
        dict(cfg_dropout=0.1, cfg_scale=2.0, cond_mode="film"),
    ]
    for cls in (FlowMatching1D, DDPM1D):
        for extra in variants:
            torch.manual_seed(0)
            model = cls(**COMMON, **extra)
            clean, params, qlab = _batch()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            model.train()
            loss = model.training_loss(clean, params=params, quantile_label=qlab)
            loss.backward()
            opt.step()
            assert torch.isfinite(loss), f"{cls.__name__} {extra}: non-finite training loss"
            if "cfg_dropout" in extra:
                assert model.null_quant_embed is not None
                assert model.null_quant_embed.grad is not None, "null embedding got no gradient"
            model.eval()
            x, _ = model.sample(8, params=params, quantile_label=qlab, device=torch.device("cpu"))
            assert x.shape == (8, 1, RES) and torch.isfinite(x).all(), f"{cls.__name__} {extra}: bad samples"
    print("ok: cfg/film variants train one step and sample (both generators)")


def _sample_with_scale(model, scale: float, qlab, params, **kwargs):
    model.cfg_scale = float(scale)
    torch.manual_seed(123)
    x, _ = model.sample(8, params=params, quantile_label=qlab, device=torch.device("cpu"), **kwargs)
    return x


def test_cfg_scale_changes_samples():
    for cls in (FlowMatching1D, DDPM1D):
        torch.manual_seed(0)
        model = cls(**COMMON, cfg_dropout=0.1)
        # make NULL/cond pathways distinguishable even without training
        with torch.no_grad():
            model.null_quant_embed.add_(1.0)
        model.eval()
        _, params, qlab = _batch()
        x1 = _sample_with_scale(model, 1.0, qlab, params)
        x3 = _sample_with_scale(model, 3.0, qlab, params)
        assert torch.isfinite(x3).all()
        assert not torch.allclose(x1, x3), f"{cls.__name__}: cfg_scale=3 did not change samples"
    # composition with the spectrum prior and edit mode (FlowMatching only)
    torch.manual_seed(0)
    model = FlowMatching1D(**COMMON, cfg_dropout=0.1)
    with torch.no_grad():
        model.null_quant_embed.add_(1.0)
    model.eval()
    model.set_noise_psd(torch.randn(64, 1, RES).cumsum(dim=-1))
    _, params, qlab = _batch()
    anchors = torch.randn(8, 1, RES)
    x1 = _sample_with_scale(model, 1.0, qlab, params, init_state=anchors, t_start=0.6)
    x3 = _sample_with_scale(model, 3.0, qlab, params, init_state=anchors, t_start=0.6)
    assert torch.isfinite(x3).all()
    assert not torch.allclose(x1, x3), "guidance inert under spectrum prior + edit mode"
    print("ok: cfg_scale=3 changes samples vs cfg_scale=1 (incl. spectrum prior + edit mode)")


def test_checkpoint_compat():
    for cls in (FlowMatching1D, DDPM1D):
        torch.manual_seed(0)
        old = cls(**COMMON)  # default: no cfg/film keys (matches pre-change checkpoints)
        new = cls(**COMMON, cfg_dropout=0.15, cfg_scale=3.0, cond_mode="film")
        assert _has_optional_keys(new)
        # old checkpoint -> cfg/film model (missing optional keys tolerated)
        load_generator_state_dict(new, old.state_dict())
        for k, v in old.state_dict().items():
            assert torch.equal(new.state_dict()[k], v), f"{cls.__name__}: key {k} not restored"
        # new checkpoint -> default model (extra optional keys tolerated)
        load_generator_state_dict(old, new.state_dict())
        # genuinely incompatible checkpoints must still fail
        bad = {k: v for k, v in old.state_dict().items() if "cond." not in k}
        try:
            load_generator_state_dict(cls(**COMMON), bad)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"{cls.__name__}: truncated checkpoint loaded without error")
    print("ok: old<->new checkpoint loading is tolerant for optional keys only")


def test_film_requires_residual_blocks():
    try:
        FlowMatching1D(**dict(COMMON, residual_blocks=0), cond_mode="film")
    except ValueError:
        print("ok: cond_mode=film without residual blocks raises ValueError")
    else:
        raise AssertionError("cond_mode=film with residual_blocks=0 should raise")


if __name__ == "__main__":
    test_default_state_dict_has_no_new_keys()
    test_variants_train_one_step_and_sample()
    test_cfg_scale_changes_samples()
    test_checkpoint_compat()
    test_film_requires_residual_blocks()
    print("all conditioning tests passed")

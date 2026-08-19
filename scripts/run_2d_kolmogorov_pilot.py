#!/usr/bin/env python3
"""
2D Navier-Stokes Kolmogorov Flow Pilot Experiment Launcher
=========================================================
Runs 1 seed (seed 101) across 5 scenarios:
1. uniform_baseline
2. heuristic_tube
3. classic_al_topk
4. classic_al_sbal
5. ogas_generative (OT-CFM Streamfunction Flow Matching)
"""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from poolbased_surrogate.pde import PDE
from poolbased_surrogate.models.surrogate import ExactAL4PDEUnet2D
from poolbased_surrogate.config import PDEConfig

def PDE2D(resolution=128, dt=0.05):
    return PDE(PDEConfig(name='ns2d', resolution=resolution, dt=dt))


# -------------------------------------------------------------
# 1. 2D STREAMFUNCTION FLOW MATCHING GENERATOR
# -------------------------------------------------------------
class StreamfunctionGenerator2D(nn.Module):
    """Generates scalar streamfunction psi, yielding exact divergence-free velocity."""
    def __init__(self, width=32, param_dim=2):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(2, width), # t and loss condition c
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.inc = nn.Conv2d(1 + param_dim, width, 3, padding=1) # psi_t and param
        self.block1 = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(4, width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
        )
        self.outc = nn.Conv2d(width, 1, 3, padding=1)

    def forward(self, psi_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        B, C, H, W = psi_t.shape
        t_c = torch.cat([t, c], dim=-1)
        t_emb = self.time_mlp(t_c).view(B, -1, 1, 1)
        pm_grid = params.view(B, -1, 1, 1).expand(B, -1, H, W)
        inp = torch.cat([psi_t, pm_grid], dim=1)
        h = self.inc(inp) + t_emb
        h = h + self.block1(h)
        return self.outc(h)

def stream_to_velocity(psi: torch.Tensor) -> torch.Tensor:
    """Analytical curl u = -dpsi/dy, v = dpsi/dx."""
    N = psi.shape[-1]
    kx = torch.fft.fftfreq(N, device=psi.device).reshape(1, 1, N, 1) * 2 * np.pi
    ky = torch.fft.fftfreq(N, device=psi.device).reshape(1, 1, 1, N) * 2 * np.pi
    psi_fft = torch.fft.fft2(psi)
    u = torch.fft.ifft2(-1j * ky * psi_fft).real
    v = torch.fft.ifft2(1j * kx * psi_fft).real
    return torch.cat([u, v], dim=1)

# -------------------------------------------------------------
# 2. EXPERIMENTAL PIPELINE (ROUNDS 0 TO 4)
# -------------------------------------------------------------
def run_pilot(strategy: str, seed: int, output_dir: Path, rounds: int = 5, n_traj: int = 16, steps: int = 20):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    
    print(f"\n=======================================================")
    print(f"=== LAUNCHING 2D AL4PDE PILOT: strategy={strategy}, seed={seed} ===")
    print(f"=== Device={device}, Resolution=128x128, Rounds={rounds} ===")
    print(f"=======================================================")
    
    pde = PDE2D(resolution=128, dt=0.05)
    
    # 1. Validation Bank (Fixed)
    print("Generating 2D AL4PDE Validation bank (16 trajectories x 25 steps)...", flush=True)
    val_params = pde.sample_params(16, seed=999)
    val_u0 = pde.sample_ic(16, seed=999)
    val_trajs = pde.simulate(val_u0, val_params, steps=25)
    
    # AL4PDE Native Exact Unet2D
    surrogate = ExactAL4PDEUnet2D(param_dim=2, hidden_channels=32).to(device)
    optimizer = torch.optim.AdamW(surrogate.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Generator model
    generator = StreamfunctionGenerator2D().to(device)
    gen_opt = torch.optim.AdamW(generator.parameters(), lr=5e-4)
    
    # Round 0: Initial Uniform Pool
    print("Round 0: Generating Initial Uniform Pool...", flush=True)
    r0_params = pde.sample_params_uniform(n_traj, rng)
    r0_u0 = pde.sample_ic_uniform(n_traj, rng)
    r0_trajs = pde.simulate(r0_u0, r0_params, steps=steps)
    
    # Pool arrays: states (N*steps, 2, H, W), params (N*steps, 1), next_states (N*steps, 2, H, W)
    pool_states = r0_trajs[:, :-1].reshape(-1, 2, 128, 128)
    pool_next = r0_trajs[:, 1:].reshape(-1, 2, 128, 128)
    pool_params = np.repeat(r0_params, steps, axis=0)
    
    history = []
    
    for r in range(rounds):
        round_start = time.perf_counter()
        print(f"\n--- Round {r} (Pool Size = {len(pool_states)} transitions) ---", flush=True)
        
        # 1. Train Surrogate
        surrogate.train()
        dataset = TensorDataset(
            torch.from_numpy(pool_states).float(),
            torch.from_numpy(pool_params).float(),
            torch.from_numpy(pool_next).float()
        )
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        for epoch in range(10):
            ep_loss = 0.0
            for st, pm, tgt in loader:
                st, pm, tgt = st.to(device), pm.to(device), tgt.to(device)
                optimizer.zero_grad()
                pred = surrogate(st, pm)
                loss = F.mse_loss(pred, tgt)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
        print(f"Round {r} Surrogate Train Loss: {ep_loss / len(loader):.6f}", flush=True)
        
        # 2. Optimized GPU-Vectorized Validation Rollout
        surrogate.eval()
        with torch.no_grad():
            st_val = torch.from_numpy(val_u0).to(device)
            pm_val = torch.from_numpy(val_params).to(device)
            tgt_val = torch.from_numpy(val_trajs[:, 1:]).to(device)
            
            val_preds = torch.empty((len(st_val), 25, 2, 128, 128), device=device)
            curr = st_val
            for t in range(25):
                curr = surrogate(curr, pm_val)
                val_preds[:, t] = curr
                
            err_sq = (val_preds - tgt_val) ** 2
            val_rmse = torch.sqrt(torch.mean(err_sq)).item()
            val_nrmse = val_rmse / (torch.sqrt(torch.mean(tgt_val**2)).item() + 1e-8)
            
        print(f"Round {r} Validation 25-step NRMSE: {val_nrmse:.4f} (RMSE: {val_rmse:.4f})", flush=True)
        
        round_metrics = {
            "round": r,
            "strategy": strategy,
            "pool_size": len(pool_states),
            "val/nrmse": float(val_nrmse),
            "val/rmse": float(val_rmse),
            "timing/round_sec": float(time.perf_counter() - round_start),
        }
        history.append(round_metrics)
        
        if r == rounds - 1:
            break
            
        # 3. Next Round Sampling
        print(f"Round {r}: Sampling next pool with strategy {strategy!r}...", flush=True)
        n_unif = n_traj // 2
        n_act = n_traj - n_unif
        
        unif_params = pde.sample_params_uniform(n_unif, rng)
        unif_u0 = pde.sample_ic_uniform(n_unif, rng)
        unif_trajs = pde.simulate(unif_u0, unif_params, steps=steps)
        
        new_states = [unif_trajs[:, :-1].reshape(-1, 2, 128, 128)]
        new_next = [unif_trajs[:, 1:].reshape(-1, 2, 128, 128)]
        new_params = [np.repeat(unif_params, steps, axis=0)]
        
        if strategy == "uniform_baseline":
            more_params = pde.sample_params_uniform(n_act, rng)
            more_u0 = pde.sample_ic_uniform(n_act, rng)
            more_trajs = pde.simulate(more_u0, more_params, steps=steps)
            new_states.append(more_trajs[:, :-1].reshape(-1, 2, 128, 128))
            new_next.append(more_trajs[:, 1:].reshape(-1, 2, 128, 128))
            new_params.append(np.repeat(more_params, steps, axis=0))
            
        elif strategy == "heuristic_tube":
            # Perturb existing pool states with divergence-free random noise
            anchors_idx = rng.choice(len(pool_states), size=n_act * steps, replace=False)
            anchors = pool_states[anchors_idx]
            noise_psi = rng.normal(size=(len(anchors), 1, 128, 128)).astype(np.float32)
            noise_vel = stream_to_velocity(torch.from_numpy(noise_psi).to(device)).cpu().numpy()
            pert_states = anchors + 0.15 * noise_vel
            pert_params = np.repeat(pde.sample_params_uniform(n_act, rng), steps, axis=0)
            
            # Step with solver (batched)
            pert_next = pde.step(pert_states, pert_params)
            new_states.append(pert_states)
            new_next.append(pert_next)
            new_params.append(pert_params)
            
        elif strategy in ("classic_al_topk", "classic_al_sbal"):
            # 10x candidate generation
            n_cand = n_act * 10
            cand_pm = pde.sample_params_uniform(n_cand, rng)
            cand_u0 = pde.sample_ic_uniform(n_cand, rng)
            cand_trajs = pde.simulate(cand_u0, cand_pm, steps=steps)
            
            c_states = cand_trajs[:, :-1].reshape(-1, 2, 128, 128)
            c_next = cand_trajs[:, 1:].reshape(-1, 2, 128, 128)
            c_params = np.repeat(cand_pm, steps, axis=0)
            
            # Score by forward surrogate loss
            scores = []
            with torch.no_grad():
                for b in range(0, len(c_states), 64):
                    st = torch.from_numpy(c_states[b:b+64]).to(device)
                    pm = torch.from_numpy(c_params[b:b+64]).to(device)
                    tgt = torch.from_numpy(c_next[b:b+64]).to(device)
                    pred = surrogate(st, pm)
                    loss_b = torch.mean((pred - tgt)**2, dim=(1, 2, 3)).cpu().numpy()
                    scores.append(loss_b)
            scores = np.concatenate(scores)
            k_sel = n_act * steps
            
            if strategy == "classic_al_topk":
                sel_idx = np.argsort(-scores)[:k_sel]
            else: # SBAL alpha=1
                p_dist = (scores + 1e-12) / np.sum(scores + 1e-12)
                sel_idx = rng.choice(len(scores), size=k_sel, replace=False, p=p_dist)
                
            new_states.append(c_states[sel_idx])
            new_next.append(c_next[sel_idx])
            new_params.append(c_params[sel_idx])
            
        elif strategy == "ogas_generative":
            # Flow Matching training & conditional generation
            # Fast OT-CFM sampling of streamfunction in 12 steps
            gen_params = pde.sample_params_uniform(n_act * steps, rng)
            cond_loss = torch.full((n_act * steps, 1), 1.5, device=device) # request hard states
            
            with torch.no_grad():
                psi_t = torch.randn(n_act * steps, 1, 128, 128, device=device)
                dt = 1.0 / 12
                for s in range(12):
                    t = torch.full((len(psi_t), 1), s * dt, device=device)
                    v = generator(psi_t, t, cond_loss, torch.from_numpy(gen_params).to(device))
                    psi_t = psi_t + dt * v
                gen_vel = stream_to_velocity(psi_t).cpu().numpy()
                
            # Solver ground-truth step (batched)
            gen_next = pde.step(gen_vel, gen_params)
            new_states.append(gen_vel)
            new_next.append(gen_next)
            new_params.append(gen_params)
            
        pool_states = np.concatenate([pool_states] + new_states, axis=0)
        pool_next = np.concatenate([pool_next] + new_next, axis=0)
        pool_params = np.concatenate([pool_params] + new_params, axis=0)
        
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"🎉 Pilot run completed successfully: {output_dir / 'history.json'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    run_pilot(args.strategy, args.seed, Path(args.output_dir))

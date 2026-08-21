#!/usr/bin/env python3
"""
2D Navier-Stokes Kolmogorov Flow Scaled Active Learning Benchmark
================================================================
Features:
- Physics: 2D Incompressible Kolmogorov Flow (Re in [1000, 3000], forcing n=4, 128x128 grid)
- Scaled Surrogate: ExactAL4PDEUnet2D with hidden_channels=64, residual difference_weight=1.0
- Optimization: 50 epochs per round, batch_size=16, AdamW + CosineAnnealingLR
- Multi-Faceted Evaluation Suite:
  1. Uniform Validation Set (16 traj x 25 steps)
  2. Hard Vorticity / High-Enstrophy Set (16 traj x 25 steps)
  3. Tube Perturbation Set (16 traj x 25 steps, divergence-free spectral perturbation)
- Multi-Horizon Metrics: T=1, T=5, T=15, T=25 NRMSE + Enstrophy + Kinetic Energy Relative Error
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


def compute_enstrophy_2d(velocity: torch.Tensor) -> torch.Tensor:
    """Compute enstrophy E = 0.5 * integral |omega|^2 dx dy using spectral curl."""
    B, T, C, H, W = velocity.shape if velocity.ndim == 5 else (velocity.shape[0], 1, *velocity.shape[1:])
    v_flat = velocity.view(-1, 2, H, W)
    
    kx = torch.fft.fftfreq(W, device=velocity.device).reshape(1, 1, 1, W) * 2 * np.pi
    ky = torch.fft.fftfreq(H, device=velocity.device).reshape(1, 1, H, 1) * 2 * np.pi
    
    u_hat = torch.fft.fft2(v_flat[:, 0])
    v_hat = torch.fft.fft2(v_flat[:, 1])
    
    # omega_hat = i*kx*v_hat - i*ky*u_hat
    omega_hat = 1j * kx * v_hat - 1j * ky * u_hat
    omega = torch.fft.ifft2(omega_hat).real
    
    enstrophy = 0.5 * torch.mean(omega**2, dim=(-2, -1))
    return enstrophy.view(B, T) if velocity.ndim == 5 else enstrophy.view(B)


def compute_divergence_2d(velocity: torch.Tensor) -> torch.Tensor:
    """Compute divergence ||div u||_L2."""
    H, W = velocity.shape[-2:]
    v_flat = velocity.view(-1, 2, H, W)
    kx = torch.fft.fftfreq(W, device=velocity.device).reshape(1, 1, 1, W) * 2 * np.pi
    ky = torch.fft.fftfreq(H, device=velocity.device).reshape(1, 1, H, 1) * 2 * np.pi
    
    u_hat = torch.fft.fft2(v_flat[:, 0])
    v_hat = torch.fft.fft2(v_flat[:, 1])
    
    div_hat = 1j * kx * u_hat + 1j * ky * v_hat
    div_phys = torch.fft.ifft2(div_hat).real
    return torch.sqrt(torch.mean(div_phys**2, dim=(-2, -1))).mean()


class StreamfunctionGenerator2D(nn.Module):
    """Generates scalar streamfunction psi, yielding exact divergence-free velocity."""
    def __init__(self, width=64, param_dim=2):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(2, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.inc = nn.Conv2d(1 + param_dim, width, 3, padding=1)
        self.block1 = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(8, width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(8, width),
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
        h = h + self.block2(h)
        return self.outc(h)


def stream_to_velocity(psi: torch.Tensor) -> torch.Tensor:
    """Analytical curl u = -dpsi/dy, v = dpsi/dx (exact divergence-free)."""
    H, W = psi.shape[-2:]
    kx = torch.fft.fftfreq(W, device=psi.device).reshape(1, 1, 1, W) * 2 * np.pi
    ky = torch.fft.fftfreq(H, device=psi.device).reshape(1, 1, H, 1) * 2 * np.pi
    psi_fft = torch.fft.fft2(psi)
    u = torch.fft.ifft2(-1j * ky * psi_fft).real
    v = torch.fft.ifft2(1j * kx * psi_fft).real
    return torch.cat([u, v], dim=1)


def generate_evaluation_suites(pde: PDE, n_eval: int = 16, steps: int = 25, seed: int = 999):
    """Generate 3 distinct validation suites: Uniform, Hard Vorticity, and Tube Perturbed."""
    print("Generating Multi-Faceted 2D Evaluation Suites...", flush=True)
    rng = np.random.default_rng(seed)
    
    # 1. Standard Uniform Validation Suite
    val_params_unif = pde.sample_params(n_eval, seed=seed)
    val_u0_unif = pde.sample_ic(n_eval, seed=seed)
    val_trajs_unif = pde.simulate(val_u0_unif, val_params_unif, steps=steps)
    
    # 2. Hard Vorticity / High-Enstrophy Suite (2x amplitude / higher shear)
    val_params_hard = pde.sample_params(n_eval, seed=seed + 1000)
    # Generate intense vortex patches
    u0_hard = val_u0_unif.copy() * 1.8 # Higher Reynolds / higher gradient
    val_trajs_hard = pde.simulate(u0_hard, val_params_hard, steps=steps)
    
    # 3. Tube Perturbed Suite (Divergence-free spectral perturbations around trajectory)
    val_params_tube = pde.sample_params(n_eval, seed=seed + 2000)
    anchors = val_trajs_unif[:, 5] # Mid-trajectory anchor states
    noise_psi = rng.normal(size=(n_eval, 1, *anchors.shape[-2:])).astype(np.float32)
    noise_vel = stream_to_velocity(torch.from_numpy(noise_psi)).numpy()
    noise_vel = 0.20 * noise_vel / (np.std(noise_vel, axis=(-2, -1), keepdims=True) + 1e-8)
    u0_tube = anchors + noise_vel
    val_trajs_tube = pde.simulate(u0_tube, val_params_tube, steps=steps)
    
    return {
        "uniform": (val_u0_unif, val_params_unif, val_trajs_unif),
        "hard_vorticity": (u0_hard, val_params_hard, val_trajs_hard),
        "tube_perturbed": (u0_tube, val_params_tube, val_trajs_tube),
    }


def evaluate_suite(surrogate: nn.Module, u0: np.ndarray, params: np.ndarray, trajs: np.ndarray, device: torch.device):
    """Compute multi-horizon NRMSE, Enstrophy error, and Energy error."""
    surrogate.eval()
    with torch.no_grad():
        st_val = torch.from_numpy(u0).to(device)
        pm_val = torch.from_numpy(params).to(device)
        tgt_val = torch.from_numpy(trajs[:, 1:]).to(device) # [B, T, 2, H, W]
        steps = tgt_val.shape[1]
        
        preds = torch.empty_like(tgt_val)
        curr = st_val
        for t in range(steps):
            curr = surrogate(curr, pm_val)
            preds[:, t] = curr
            
        # Multi-horizon NRMSE
        err_sq = (preds - tgt_val) ** 2
        
        def nrmse_at_t(t_idx):
            r = torch.sqrt(torch.mean(err_sq[:, t_idx]))
            norm = torch.sqrt(torch.mean(tgt_val[:, t_idx]**2)) + 1e-8
            return float(r / norm)
            
        t1_nrmse = nrmse_at_t(0)
        t5_nrmse = nrmse_at_t(min(4, steps - 1))
        t15_nrmse = nrmse_at_t(min(14, steps - 1))
        t25_nrmse = nrmse_at_t(steps - 1)
        
        # Enstrophy & Energy relative error at final step
        pred_enstrophy = compute_enstrophy_2d(preds[:, -1])
        true_enstrophy = compute_enstrophy_2d(tgt_val[:, -1])
        enstrophy_rel_err = float(torch.mean(torch.abs(pred_enstrophy - true_enstrophy) / (true_enstrophy + 1e-8)))
        
        pred_ke = 0.5 * torch.mean(preds[:, -1]**2, dim=(1, 2, 3))
        true_ke = 0.5 * torch.mean(tgt_val[:, -1]**2, dim=(1, 2, 3))
        energy_rel_err = float(torch.mean(torch.abs(pred_ke - true_ke) / (true_ke + 1e-8)))
        
        div_err = float(compute_divergence_2d(preds[:, -1]))
        
        return {
            "nrmse_t1": t1_nrmse,
            "nrmse_t5": t5_nrmse,
            "nrmse_t15": t15_nrmse,
            "nrmse_t25": t25_nrmse,
            "enstrophy_rel_err": enstrophy_rel_err,
            "energy_rel_err": energy_rel_err,
            "div_err": div_err,
        }


def run_experiment(strategy: str, seed: int, output_dir: Path, rounds: int = 5, n_traj: int = 32, steps: int = 20, epochs: int = 10):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    
    print(f"\n=======================================================")
    print(f"=== LAUNCHING 2D SCALED NAVIER-STOKES: strategy={strategy}, seed={seed} ===")
    print(f"=== Device={device}, Resolution=128x128, Rounds={rounds}, Epochs={epochs} ===")
    print(f"=======================================================")
    
    pde = PDE(PDEConfig(name='ns2d', resolution=128, dt=0.05))
    eval_suites = generate_evaluation_suites(pde, n_eval=16, steps=25, seed=999)
    
    # Scaled AL4PDE Unet2D: 64 hidden channels + difference_weight=1.0 (residual target)
    surrogate = ExactAL4PDEUnet2D(param_dim=2, hidden_channels=64, difference_weight=1.0).to(device)
    optimizer = torch.optim.AdamW(surrogate.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=rounds * epochs, eta_min=1e-5)
    
    generator = StreamfunctionGenerator2D(width=64).to(device)
    gen_opt = torch.optim.AdamW(generator.parameters(), lr=5e-4)
    
    # Round 0 Pool
    print("Round 0: Generating Initial Uniform Pool...", flush=True)
    r0_params = pde.sample_params_uniform(n_traj, rng)
    r0_u0 = pde.sample_ic_uniform(n_traj, rng)
    r0_trajs = pde.simulate(r0_u0, r0_params, steps=steps)
    
    pool_states = r0_trajs[:, :-1].reshape(-1, 2, 128, 128)
    pool_next = r0_trajs[:, 1:].reshape(-1, 2, 128, 128)
    pool_params = np.repeat(r0_params, steps, axis=0)
    
    history = []
    
    for r in range(rounds):
        round_start = time.perf_counter()
        print(f"\n--- Round {r} (Pool Size = {len(pool_states)} transitions) ---", flush=True)
        
        # 1. Train Scaled Surrogate
        surrogate.train()
        dataset = TensorDataset(
            torch.from_numpy(pool_states).float(),
            torch.from_numpy(pool_params).float(),
            torch.from_numpy(pool_next).float()
        )
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        for epoch in range(epochs):
            ep_loss = 0.0
            for st, pm, tgt in loader:
                st, pm, tgt = st.to(device), pm.to(device), tgt.to(device)
                optimizer.zero_grad()
                pred = surrogate(st, pm)
                loss = F.mse_loss(pred, tgt)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
            scheduler.step()
            
        print(f"Round {r} Surrogate Train Loss (Epoch 50): {ep_loss / len(loader):.6f}", flush=True)
        
        # 2. Multi-Faceted Evaluation
        metrics_unif = evaluate_suite(surrogate, *eval_suites["uniform"], device)
        metrics_hard = evaluate_suite(surrogate, *eval_suites["hard_vorticity"], device)
        metrics_tube = evaluate_suite(surrogate, *eval_suites["tube_perturbed"], device)
        
        print(f"Round {r} [UNIFORM]  T1: {metrics_unif['nrmse_t1']:.4f} | T5: {metrics_unif['nrmse_t5']:.4f} | T25: {metrics_unif['nrmse_t25']:.4f} | Enstrophy Err: {metrics_unif['enstrophy_rel_err']:.3f}", flush=True)
        print(f"Round {r} [HARD]     T1: {metrics_hard['nrmse_t1']:.4f} | T5: {metrics_hard['nrmse_t5']:.4f} | T25: {metrics_hard['nrmse_t25']:.4f} | Enstrophy Err: {metrics_hard['enstrophy_rel_err']:.3f}", flush=True)
        print(f"Round {r} [TUBE]     T1: {metrics_tube['nrmse_t1']:.4f} | T5: {metrics_tube['nrmse_t5']:.4f} | T25: {metrics_tube['nrmse_t25']:.4f} | Enstrophy Err: {metrics_tube['enstrophy_rel_err']:.3f}", flush=True)
        
        round_metrics = {
            "round": r,
            "strategy": strategy,
            "pool_size": len(pool_states),
            "uniform": metrics_unif,
            "hard": metrics_hard,
            "tube": metrics_tube,
            "timing/round_sec": float(time.perf_counter() - round_start),
        }
        history.append(round_metrics)
        
        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
            
        if r == rounds - 1:
            break
            
        # 3. Active Learning Sampling for Next Round
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
            anchors_idx = rng.choice(len(pool_states), size=n_act * steps, replace=False)
            anchors = pool_states[anchors_idx]
            noise_psi = rng.normal(size=(len(anchors), 1, 128, 128)).astype(np.float32)
            noise_vel = stream_to_velocity(torch.from_numpy(noise_psi).to(device)).cpu().numpy()
            pert_states = anchors + 0.15 * noise_vel
            pert_params = np.repeat(pde.sample_params_uniform(n_act, rng), steps, axis=0)
            pert_next = pde.step(pert_states, pert_params)
            new_states.append(pert_states)
            new_next.append(pert_next)
            new_params.append(pert_params)
            
        elif strategy in ("classic_al_topk", "classic_al_sbal"):
            n_cand = n_act * 10
            cand_params = pde.sample_params_uniform(n_cand, rng)
            cand_u0 = pde.sample_ic_uniform(n_cand, rng)
            cand_trajs = pde.simulate(cand_u0, cand_params, steps=steps)
            c_states = cand_trajs[:, :-1].reshape(-1, 2, 128, 128)
            c_next = cand_trajs[:, 1:].reshape(-1, 2, 128, 128)
            c_params = np.repeat(cand_params, steps, axis=0)
            
            # Score residual error with surrogate
            surrogate.eval()
            scores = []
            with torch.no_grad():
                for b in range(0, len(c_states), 64):
                    st_b = torch.from_numpy(c_states[b:b+64]).to(device)
                    pm_b = torch.from_numpy(c_params[b:b+64]).to(device)
                    tgt_b = torch.from_numpy(c_next[b:b+64]).to(device)
                    pred = surrogate(st_b, pm_b)
                    loss_b = torch.mean((pred - tgt_b)**2, dim=(1, 2, 3))
                    scores.append(loss_b.cpu().numpy())
            scores = np.concatenate(scores)
            k_sel = n_act * steps
            
            if strategy == "classic_al_topk":
                sel_idx = np.argsort(-scores)[:k_sel]
            else:
                p_dist = (scores + 1e-12) / np.sum(scores + 1e-12)
                sel_idx = rng.choice(len(scores), size=k_sel, replace=False, p=p_dist)
                
            new_states.append(c_states[sel_idx])
            new_next.append(c_next[sel_idx])
            new_params.append(c_params[sel_idx])
            
        elif strategy == "ogas_generative":
            # Train Streamfunction Flow Matching on pool transition errors
            surrogate.eval()
            with torch.no_grad():
                st_t = torch.from_numpy(pool_states).to(device)
                pm_t = torch.from_numpy(pool_params).to(device)
                tgt_t = torch.from_numpy(pool_next).to(device)
                pred_t = surrogate(st_t, pm_t)
                errors = torch.mean((pred_t - tgt_t)**2, dim=(1, 2, 3), keepdim=True)
                # Compute scalar streamfunction for each pool state via spectral inversion
                # psi_hat = (i*ky*u_hat - i*kx*v_hat) / K_sq
                kx = torch.fft.fftfreq(128, device=device).reshape(1, 1, 128) * 2 * np.pi
                ky = torch.fft.fftfreq(128, device=device).reshape(1, 128, 1) * 2 * np.pi
                K_sq = kx**2 + ky**2
                K_sq[0, 0] = 1e-10
                u_h = torch.fft.fft2(st_t[:, 0])
                v_h = torch.fft.fft2(st_t[:, 1])
                psi_h = (1j * ky * u_h - 1j * kx * v_h) / K_sq
                psi_h[:, 0, 0] = 0.0
                psi_target = torch.fft.ifft2(psi_h).real[:, None, :, :]
                
            generator.train()
            gen_data = TensorDataset(psi_target, errors, pm_t)
            gen_loader = DataLoader(gen_data, batch_size=16, shuffle=True)
            for _ in range(40):
                for psi_x1, c_err, pm in gen_loader:
                    gen_opt.zero_grad()
                    B = len(psi_x1)
                    t = torch.rand(B, 1, device=device)
                    x0 = torch.randn_like(psi_x1)
                    xt = (1 - t.view(B, 1, 1, 1)) * x0 + t.view(B, 1, 1, 1) * psi_x1
                    target_vt = psi_x1 - x0
                    pred_vt = generator(xt, t, c_err, pm)
                    loss_fm = F.mse_loss(pred_vt, target_vt)
                    loss_fm.backward()
                    gen_opt.step()
                    
            # Sample targeted high-error streamfunctions
            generator.eval()
            with torch.no_grad():
                gen_params = pde.sample_params_uniform(n_act, rng)
                gen_params_rep = np.repeat(gen_params, steps, axis=0)
                pm_g = torch.from_numpy(gen_params_rep).to(device)
                
                N_gen = len(pm_g)
                c_target = torch.full((N_gen, 1), float(torch.quantile(errors, 0.90)), device=device)
                x_curr = torch.randn((N_gen, 1, 128, 128), device=device)
                dt_fm = 1.0 / 20.0
                for step_idx in range(20):
                    t_val = torch.full((N_gen, 1), step_idx * dt_fm, device=device)
                    vt = generator(x_curr, t_val, c_target, pm_g)
                    x_curr = x_curr + dt_fm * vt
                gen_states = stream_to_velocity(x_curr).cpu().numpy()
                gen_next = pde.step(gen_states, gen_params_rep)
                
            new_states.append(gen_states)
            new_next.append(gen_next)
            new_params.append(gen_params_rep)
            
        pool_states = np.concatenate(new_states, axis=0)
        pool_next = np.concatenate(new_next, axis=0)
        pool_params = np.concatenate(new_params, axis=0)
        
    print(f"\n=== Experiment completed successfully for {strategy} (seed {seed})! ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="uniform_baseline")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n_traj", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    
    run_experiment(
        strategy=args.strategy,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        rounds=args.rounds,
        epochs=args.epochs,
        n_traj=args.n_traj,
        steps=args.steps,
    )

if __name__ == "__main__":
    main()

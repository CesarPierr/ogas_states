# Reading list — foundations behind the theory & proofs

Mapped to the components of [theory.md](theory.md) and the proofs in `paper/main.tex` (App. A).
Markers: ⭐ start here (accessible, high payoff) · 📘 textbook/notes · 📄 key paper · 🆓 free online.

---

## A. Covariate shift & importance weighting  (Method §4.1; App. A.2, A.3)
*Why training on `q≠p` need not help `R_p`; importance weights `w=p/q`; the `χ²` variance of A.3.*
- ⭐📘🆓 **Sugiyama & Kawanabe, _Machine Learning in Non-Stationary Environments_ (MIT Press, 2012).** The reference on covariate shift and importance-weighted ERM. Read Ch. 1–3.
- 📘 **Quiñonero-Candela, Sugiyama, Schwaighofer, Lawrence (eds.), _Dataset Shift in Machine Learning_ (MIT Press, 2009).**
- 📄 **Shimodaira (2000),** "Improving predictive inference under covariate shift…" — the foundational result behind `R_p = E_q[w ℓ]`.
- 📄 **Cortes, Mansour, Mohri (2010),** "Learning Bounds for Importance Weighting" (NeurIPS) — exactly the `‖w‖∞` / variance trade-off we use.

## B. Distributionally robust optimization (DRO)  (Method §4.1; App. A.4 water-filling)
*The objective reframing `R_DRO = max_{q∈U} E_q[ℓ]`, the entropic dual, loss-flattening.*
- ⭐📄🆓 **Duchi & Namkoong (2021),** "Learning Models with Uniform Performance via DRO" (Annals of Statistics). The cleanest statement of "uniform performance" = our coverage goal.
- ⭐📘🆓 **John Duchi, lecture notes "Statistics & Information Theory" (Stanford EE377/STATS311).** Chapters on DRO and the KL/`χ²` duals — directly underpins App. A.4.
- 📄 **Namkoong & Duchi (2016/2017),** "Variance-based regularization with convex objectives" / "Stochastic gradient methods for DRO" — `χ²`-ball DRO (ties to A.3).
- 📄🆓 **Sinha, Namkoong, Duraisamy (2018),** "Certifying Some Distributional Robustness to Adversarial Perturbations" (ICLR) — Wasserstein DRO with the adversary-as-generator view (our inner max).
- 📘 **Kuhn, Esfahani, Nguyen, Shafieezadeh-Abadeh (2019),** "Wasserstein DRO: Theory and Applications" (tutorial). Good for the `W₁` connection.
- 📘 **Ben-Tal, El Ghaoui, Nemirovski, _Robust Optimization_ (Princeton, 2009)** — classical robust-optimization background.

## C. Optimal transport, IPMs & information divergences  (App. A.2)
*The `W₁` (Kantorovich–Rubinstein) and TV/Pinsker bounds linking `R_p` and `R_q`.*
- ⭐📘🆓 **Peyré & Cuturi, _Computational Optimal Transport_ (2019, free).** Ch. on Kantorovich duality = the `|R_p−R_q| ≤ Lip(ℓ)·W₁` bound.
- 📄 **Müller (1997),** "Integral probability metrics and their generating classes of functions" — IPMs (W₁, MMD, TV as special cases).
- ⭐📘🆓 **Polyanskiy & Wu, _Information Theory: From Coding to Learning_ (Cambridge, 2024 draft, free).** f-divergences, Pinsker, Donsker–Varadhan — everything in A.2/A.4.
- 📘 **Boucheron, Lugosi, Massart, _Concentration Inequalities_ (OUP, 2013)** — for finite-sample versions.
- 📘 **Santambrogio, _Optimal Transport for Applied Mathematicians_** (if you want the rigorous OT).

## D. Active learning & optimal experimental design  (Method §4.4; difficulty = disagreement)
*Ensemble disagreement as the reducible/epistemic signal; A-optimality intuition.*
- ⭐📄🆓 **Settles (2009),** "Active Learning Literature Survey" — start here.
- 📄 **Houlsby, Huszár, Ghahramani, Lengyel (2011),** "Bayesian Active Learning…" (BALD).
- 📄 **Gal, Islam, Ghahramani (2017)** & **Kirsch, van Amersfoort, Gal (2019)** "BatchBALD" — epistemic acquisition with NNs.
- 📄 **Beluch et al. (2018),** "The Power of Ensembles for Active Learning" — justifies the ensemble-variance signal.
- 📘 **Chaloner & Verdinelli (1995),** "Bayesian Experimental Design: A Review" + **Atkinson & Donev, _Optimum Experimental Designs_** — A/D/E-optimality (the design-theory backdrop of §3.3).
- 📄 **MacKay (1992),** "Information-Based Objective Functions for Active Data Selection" — the classic.

## E. Exposure bias, imitation learning & error compounding  (Method §4.3; App. A.5; Remark 2)
*The `O(εH²)→O(εH)` story; visited- vs expert-distribution training; DAgger.*
- ⭐📄🆓 **Ross, Gordon, Bagnell (2011),** "A Reduction of Imitation Learning…" (DAgger, AISTATS). The `O(εH²)` vs `O(εH)` analysis is the template for our Remark 2.
- 📄 **Ross & Bagnell (2010),** "Efficient Reductions for Imitation Learning."
- 📄 **Bengio, Vinyals, Jaitly, Shazeer (2015),** "Scheduled Sampling"; **Ranzato et al. (2016),** "Sequence Level Training" — exposure bias in autoregression.
- 📘 **Bertsekas, _Dynamic Programming and Optimal Control_** — contraction/error-propagation intuition behind the recursion in §3.

## F. Dynamical systems, chaos, Lyapunov & invariant sets  (Prelim §3; Prop. 2 / App. A.5)
*`L_Φ`, the attractor measure `p`, forward-invariant tubes, input-to-state stability (ISS).*
- ⭐📘 **Strogatz, _Nonlinear Dynamics and Chaos_.** Accessible intro to attractors, Lyapunov exponents, sensitivity.
- ⭐📘🆓 **Cvitanović et al., _ChaosBook_ (free online).** Uses **Kuramoto–Sivashinsky as a running example** — ideal for our testbed; covers invariant measures and Lyapunov spectra.
- 📘 **Khalil, _Nonlinear Systems_.** Comparison lemma, invariant sets, and **input-to-state stability (ISS)** — the rigorous frame for the trapping-tube argument (Prop. 2 / Assumptions 1–2). See also **Sontag's ISS survey** (🆓).
- 📘 **Temam, _Infinite-Dimensional Dynamical Systems in Mechanics and Physics_** — attractors of dissipative PDEs incl. KS (advanced; for the "reachable manifold `M`" notion).

## G. Numerical PDEs & the Kuramoto–Sivashinsky solver  (Prelim §3)
*What `Φ` is; why KS; Lipschitz of the solution operator; the spectral integrator we use.*
- ⭐📘 **Trefethen, _Spectral Methods in MATLAB_.** KS is solved spectrally; great intuition.
- 📄🆓 **Kassam & Trefethen (2005),** "Fourth-Order Time-Stepping for Stiff PDEs" (ETDRK4) — the standard KS time integrator.
- 📘 **LeVeque, _Finite Difference Methods for ODEs and PDEs_** — stability/consistency basics.
- 📘 **Evans, _Partial Differential Equations_** — graduate PDE theory (well-posedness, solution operators).

## H. Neural PDE surrogates & operator learning  (Related work §2)
- ⭐📄🆓 **Kovachki et al. (2023),** "Neural Operator: Learning Maps Between Function Spaces" (JMLR) — comprehensive reference.
- 📄 **Li et al. (2021)** FNO; **Lu et al. (2021)** DeepONet; **Brandstetter et al. (2022)** MP-PDE; **Lippe et al. (2023)** PDE-Refiner; **Takamoto et al. (2022)** PDEBench.

## I. Generative models used as the coverage mechanism  (Method §4.4)
- ⭐📄🆓 **Lipman et al. (2023)** Flow Matching + **Lipman et al. (2024)** "Flow Matching Guide and Code" (tutorial).
- 📄 **Ho, Jain, Abbeel (2020),** DDPM. **Tong et al. (2023),** conditional flow matching.
- 📄 **Shrivastava et al. (2016),** OHEM; **Bengio et al. (2009),** Curriculum Learning — hard-example/curriculum context.

---

## Suggested 1-week path to "own" the theory
1. **DRO + uniform performance:** Duchi & Namkoong (2021) + Duchi's notes (B) → this is the spine of §4.1 and "uniformity over the loss."
2. **Covariate shift bounds:** Sugiyama & Kawanabe Ch.1–3 + Cortes–Mansour–Mohri (A) → Prop. 1 and App. A.2/A.3.
3. **Exposure bias / DAgger:** Ross–Gordon–Bagnell (E) → §4.3 and the rollout story.
4. **Chaos & invariant sets:** Strogatz + ChaosBook KS chapters + Khalil's ISS/invariance (F) → Prop. 2 and the tube argument.
5. **OT/IPM + info divergences:** Peyré–Cuturi duality chapter + Polyanskiy–Wu Pinsker/DV (C) → the bounds in A.2/A.4.

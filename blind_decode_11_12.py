# ambient_bistatic_sim.py
# -*- coding: utf-8 -*-
"""
End-to-end Monte-Carlo simulator for simplified ambient bistatic radar + tag system.

Implements three blind decoding schemes:
  - Case I: ML decoding (no penalty) via projection elimination
  - Case II: ML with Tikhonov regularization (ridge)
  - Case III: Disjoint low-complexity decoding (tag by max-correlation; radar by
              LS on the SR component), with optional ridge stabilization

Design choices strictly follow the discussion:
  * Waveform set C: CAZAC (Zadoff–Chu) like, unit-modulus, near-orthogonal
  * Tag codebook X: unit-modulus, orthogonal to 1_L, ||x||^2 = L
  * Xi_c in C^{K x (Q+1)} builds linear convolution against a (Q+1)-tap channel
  * Channels are quasi-static across all MC trials of a run
  * SNR sweep: [-5, -2.5, 0, 2.5, 5] dB
  * Ridge lambdas: lambda_c = 0.1 * ||Xi_c Xi_c^H||_2  (one order smaller)
  * Metrics: MER for radar & tag; NMSE for SR & STR channels
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pdb import set_trace as bp

import torch

Tensor = torch.Tensor
complex_dtype = torch.cfloat


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int = 2025):
    random.seed(seed)
    torch.manual_seed(seed)


def to_device(x: Tensor, device: torch.device) -> Tensor:
    return x.to(device)


def complex_norm2(x: Tensor) -> Tensor:
    """Return squared l2 norm (sum of |.|^2) as a scalar tensor."""
    return (x.conj() * x).real.sum()


def frob_norm2(x: Tensor) -> Tensor:
    """Frobenius norm squared."""
    return (x.conj() * x).real.sum()


def nmse(est: Tensor, ref: Tensor) -> float:
    """Complex NMSE (||e - r||^2 / ||r||^2)."""
    num = complex_norm2(est - ref).item()
    den = complex_norm2(ref).item() + 1e-12
    return float(num / den)


# -----------------------------
# Codebook generation
# -----------------------------
def zadoff_chu(N: int, root: int) -> Tensor:
    """
    Generate a length-N Zadoff-Chu sequence (unit-modulus CAZAC).

    For prime-length N and root r coprime to N:
        s[n] = exp(-j * pi * r * n * (n + 1) / N), n = 0,...,N-1
    We also allow generic N; we still get a near-CAZAC unit-modulus sequence.
    """
    n = torch.arange(N, dtype=torch.float64)
    phase = -math.pi * root * n * (n + 1) / N
    s = torch.exp(1j * torch.tensor(phase))
    return s.to(complex_dtype)


def build_waveform_set(Np: int, Ma: int, device: torch.device, max_crosscorr: float = 0.15) -> Tensor:
    """
    Build Ma unit-modulus, near-orthogonal baseband waveforms of length Np.
    Strategy: generate a bank of ZC sequences with different coprime roots, 
    selecting roots that minimize maximum cross-correlation.
    
    Args:
        Np: Waveform length
        Ma: Number of waveforms needed
        device: torch device
        max_crosscorr: Maximum allowed cross-correlation threshold (default 0.15)
    
    Note: For non-prime Np, ZC sequences may not achieve ideal low cross-correlation.
    The function will try to select roots with the best mutual correlation properties.
    """
    # First, collect all coprime roots
    all_coprime_roots = []
    for r in range(1, min(Np, 4 * Ma)):
        if math.gcd(r, Np) == 1:
            all_coprime_roots.append(r)
    
    if len(all_coprime_roots) == 0:
        raise ValueError(f"No coprime roots found for Np={Np}")
    
    # Generate all candidate ZC sequences
    candidate_seqs = {}
    for r in all_coprime_roots:
        candidate_seqs[r] = zadoff_chu(Np, r)
    
    # Greedy selection: pick roots with minimal cross-correlation
    selected_roots = []
    selected_seqs = []
    
    # Start with the first root
    if len(all_coprime_roots) > 0:
        first_root = all_coprime_roots[0]
        selected_roots.append(first_root)
        selected_seqs.append(candidate_seqs[first_root])
    
    # Greedily add roots that have low cross-correlation with already selected ones
    while len(selected_roots) < Ma and len(selected_roots) < len(all_coprime_roots):
        best_root = None
        best_max_corr = float('inf')
        
        for r in all_coprime_roots:
            if r in selected_roots:
                continue
            
            # Compute max cross-correlation with all selected sequences
            max_corr = 0.0
            cand_seq = candidate_seqs[r]
            for sel_seq in selected_seqs:
                # Compute normalized cross-correlation at zero shift
                corr = (cand_seq.conj() * sel_seq).sum().abs().item() / Np
                max_corr = max(max_corr, corr)
            
            # Keep track of the root with minimum max-correlation
            if max_corr < best_max_corr:
                best_max_corr = max_corr
                best_root = r
        
        if best_root is not None:
            selected_roots.append(best_root)
            selected_seqs.append(candidate_seqs[best_root])
            print(f"  Selected root {best_root}, max cross-corr with existing: {best_max_corr:.4f}")
    
    if len(selected_seqs) < Ma:
        print(f"Warning: Could only generate {len(selected_seqs)} sequences with good correlation properties (requested {Ma})")
    
    C = torch.stack(selected_seqs, dim=0).to(device)  # (Ma, Np), unit-modulus
    # Normalize (already unit modulus), but keep it explicit:
    C = C / C.abs()  # guard against any numeric deviation
    
    print(f"Generated {len(selected_roots)} ZC waveforms with roots: {selected_roots[:10]}{'...' if len(selected_roots) > 10 else ''}")
    return C  # (Ma, Np)


def build_tag_codebook(L: int, U: int, device: torch.device) -> Tensor:
    """
    Build U unit-modulus codewords x in C^L:
      * Each x is orthogonal to 1_L (remove DC/mean)
      * ||x||^2 = L
    Construction: take DFT columns excluding DC, then apply phase rotations to reach U.
    """
    # DFT matrix columns k=1..L-1 are orthogonal to ones
    n = torch.arange(L, dtype=torch.float64).unsqueeze(1)  # (L,1)
    cols = []
    ks = list(range(1, L))  # exclude k=0
    for k in ks[: min(U, L - 1)]:
        phase = 2 * math.pi * k * n / L ## (L,1)
        v = torch.exp(1j * torch.tensor(phase)).squeeze(1)  # (L,)
        v = v.to(complex_dtype)
        # unit-modulus by construction; enforce ||x||^2 = L:
        v = v / v.abs()  # unit-modulus
        cols.append(v)

    # If U > L-1, add random phase-only vectors orthogonal to ones
    while len(cols) < U:
        phi = torch.rand(L) * 2 * math.pi
        v = torch.exp(1j * phi).to(complex_dtype)
        # project out the DC:
        ones = torch.ones(L, dtype=complex_dtype)
        v = v - (v.conj().dot(ones) / (ones.conj().dot(ones))) * ones
        # renormalize to unit-modulus by phase-only normalization
        v = torch.exp(1j * torch.angle(v + 1e-12))
        cols.append(v)

    X = torch.stack(cols, dim=1).to(device)  # (L, U)
    # Enforce ||x||^2 = L:
    X = X / torch.sqrt((X.conj() * X).real.sum(dim=0, keepdim=True)) * math.sqrt(L)
    # Lastly ensure unit-modulus (phase-only):
    X = torch.exp(1j * torch.angle(X))
    return X  # (L, U)


# -----------------------------
# Convolution matrix Xi_c
# -----------------------------
def build_Xi_linear(c: Tensor, K: int, Q: int) -> Tensor:
    """
    Build Xi_c in C^{K x (Q+1)} that performs *linear* convolution with (Q+1)-tap channel.
    Column j (0-based) is c shifted down by j with zero-padding (no wrap).
      Xi[k, j] = c[k-j] if 0 <= k-j < len(c), else 0.
    c has length Np; K = Np + Q.
    """
    Np = c.numel()
    Xi = torch.zeros((K, Q + 1), dtype=complex_dtype, device=c.device)
    for j in range(Q + 1):
        # indices where k - j in [0, Np-1]  => k in [j, j+Np-1]
        start = j
        end = j + Np  # exclusive
        Xi[start:end, j] = c
    return Xi  # (K, Q+1)


# -----------------------------
# Environment container
# -----------------------------
@dataclass
class Env:
    device: torch.device
    L: int
    Np: int
    Qmin: int
    Qmax: int
    Q: int
    K: int
    Ma: int
    U: int
    C_mat: Tensor          # (Ma, Np)
    X_mat: Tensor          # (L, U)
    gamma_SR: Tensor       # (Q+1,)
    gamma_STR: Tensor      # (Q+1,)
    Xi_list: List[Tensor]  # list of (K, Q+1)
    pinv_list: List[Tensor]      # list of (Q+1, K) Moore-Penrose pseudo-inverse
    proj_list: List[Tensor]      # list of (K, K) projectors (I - Xi Xi^†)
    ridge_proj_SR: List[Tensor]  # P_{c,SR} (K,K)
    ridge_proj_STR: List[Tensor] # P_{c,STR} (K,K)
    lambdas_SR: List[float] ## lambda_c for SR ridge
    lambdas_STR: List[float] ## lambda_c for STR ridge


# -----------------------------
# Environment generation
# -----------------------------
def generate_environment(
    L: int = 10, ## PRI number per frame
    Np: int = 31, ## Waveform length in a PRI (prime number for better cross-correlation)
    Qmin: int = 10, ## Minimum channel delay spread (taps), reader knows it
    Qmax: int = 20, ## Maximum channel delay spread (taps), reader knows it
    Ma: int = 16, ## Number of available waveforms (reduced for better orthogonality)
    U: int = 16, ## Number of tag codewords (reduced to match Ma)
    P_SR: float = 1.0, ## Average power of each SR channel tap
    P_STR: float = 1.0, ## Average power of each STR channel tap
    seed: int = 2025,
    use_gpu_if_available: bool = True,
) -> Env:
    """
    Build everything once and keep static during MC:
      - Waveform set C (Ma x Np), unit-modulus CAZACs
      - Tag codebook X (L x U), unit-modulus, orthogonal to ones
      - Static channels gamma_SR, gamma_STR in C^{Q+1}, non-sparse CN(0, P)
      - For each c: Xi_c, pinv, projectors (no-ridge), ridge projectors with
        lambda_c = 0.1 * ||Xi_c Xi_c^H||_2  (one order smaller)
    """
    set_seed(seed)
    device = torch.device("cuda") if (use_gpu_if_available and torch.cuda.is_available()) else torch.device("cpu")

    Q = Qmax - Qmin  # inclusive spread -> Q taps beyond Qmin; total length Q+1
    K = Np + Q

    # 1) Waveforms and tag codebook
    C_mat = build_waveform_set(Np, Ma, device)           # (Ma, Np)
    X_mat = build_tag_codebook(L, U, device)             # (L, U)

    # 2) Channels (quasi-static)
    gamma_SR = (torch.randn(Q + 1, dtype=complex_dtype, device=device) +
                1j * torch.randn(Q + 1, dtype=complex_dtype, device=device)) * math.sqrt(P_SR / 2) ## CN(0, P_SR)
    gamma_STR = (torch.randn(Q + 1, dtype=complex_dtype, device=device) +
                 1j * torch.randn(Q + 1, dtype=complex_dtype, device=device)) * math.sqrt(P_STR / 2) ## CN(0, P_STR)

    # 3) Lookup for each c, computed in advance
    Xi_list: List[Tensor] = [] # list of (K, Q+1), \Xi matrix
    pinv_list: List[Tensor] = [] # list of (Q+1, K), \Xi^†
    proj_list: List[Tensor] = [] # list of (K, K), projectors (I - \Xi \Xi^†)
    ridge_proj_SR: List[Tensor] = [] # list of (K, K), P_{c,SR} (ridge projectors)
    ridge_proj_STR: List[Tensor] = [] # list of (K, K), P_{c,STR} (ridge projectors)
    lambdas_SR: List[float] = [] # list of lambda_c for SR ridge
    lambdas_STR: List[float] = [] # list of lambda_c for STR ridge

    I_K = torch.eye(K, dtype=complex_dtype, device=device) # identity for projectors, (K,K)
    for idx in range(Ma): ## for each waveform c
        c = C_mat[idx]  # (Np,)
        Xi = build_Xi_linear(c, K, Q)  # (K, Q+1)
        Xi_list.append(Xi)

        # Pseudoinverse and projector (Case I / III)
        pinv = torch.linalg.pinv(Xi)   # (Q+1, K)
        pinv_list.append(pinv)
        proj = I_K - Xi @ pinv
        proj_list.append(proj)

        # Ridge projectors (Case II)
        # lambda = 0.1 * spectral_norm( Xi Xi^H )
        A = Xi @ Xi.conj().T  # (K,K)
        # spectral norm via 2-norm for Hermitian A
        ## lambda is supposed to be smaller than the spectral norm
        lam_base = 1.0 * torch.linalg.norm(A, ord=2).real.item()
        # STR / SR use their own ridge; we can keep same magnitude (as requested)
        lambdas_STR.append(lam_base)
        lambdas_SR.append(lam_base)

        # P_c = Xi (Xi^H Xi + lambda I)^(-1) Xi^H
        XtX = Xi.conj().T @ Xi   # (Q+1, Q+1)
        I_ch = torch.eye(Q + 1, dtype=complex_dtype, device=device)

        P_STR = Xi @ torch.linalg.inv(XtX + lam_base * I_ch) @ Xi.conj().T
        P_SR  = Xi @ torch.linalg.inv(XtX + lam_base * I_ch) @ Xi.conj().T
        ridge_proj_STR.append(P_STR)
        ridge_proj_SR.append(P_SR)

    return Env(
        device=device,
        L=L, Np=Np, Qmin=Qmin, Qmax=Qmax, Q=Q, K=K, Ma=Ma, U=U,
        C_mat=C_mat, X_mat=X_mat,
        gamma_SR=gamma_SR, gamma_STR=gamma_STR,
        Xi_list=Xi_list, pinv_list=pinv_list, proj_list=proj_list,
        ridge_proj_SR=ridge_proj_SR, ridge_proj_STR=ridge_proj_STR,
        lambdas_SR=lambdas_SR, lambdas_STR=lambdas_STR
    )


# -----------------------------
# Forward model
# -----------------------------
def synthesize_observation(env: Env, c_idx: int, x_idx: int, snr_db: float) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    Generate the observation Y in C^{L x K} given chosen waveform c and tag code x, at a target SNR.
      Y = x a_STR^H + 1 a_SR^H + Omega,
    where a_* = Xi_c gamma_* in C^K, representing the effective channels.
    Noise variance is chosen per-trial to meet the requested SNR exactly.
    Returns:
      Y, aux dict with alpha vectors and sigma^2.
    """
    device = env.device
    L, K = env.L, env.K
    x = env.X_mat[:, x_idx:x_idx + 1]  # (L,1)
    ones = torch.ones((L, 1), dtype=complex_dtype, device=device)

    Xi = env.Xi_list[c_idx]  # (K, Q+1)
    alpha_SR = Xi @ env.gamma_SR     # (K,1), effective SR channel
    alpha_STR = Xi @ env.gamma_STR   # (K,1), effective STR channel
    alpha_SR = alpha_SR.unsqueeze(1)     # (K,1)
    alpha_STR = alpha_STR.unsqueeze(1)   # (K,1)

    # Deterministic energy in the noiseless matrix, compute transmit SNR then
    # || x a_STR^H + 1 a_SR^H ||_F^2 = ||x||^2 ||a_STR||^2 + ||1||^2 ||a_SR||^2  (cross terms vanish because x ⟂ 1)
    signal_pow = (env.L * complex_norm2(alpha_STR) + env.L * complex_norm2(alpha_SR)).item()
    # SNR = signal_pow / E||Omega||_F^2 = signal_pow / (L*K*sigma^2)
    snr_lin = 10.0 ** (snr_db / 10.0)
    sigma2 = signal_pow / (L * K * snr_lin + 1e-12) ## adjust noise power accordingly

    # Noiseless:
    # Y0 = x @ alpha_STR.conj().T + ones @ alpha_SR.conj().T  # (L,K)
    Y0 = x @ alpha_STR.T + ones @ alpha_SR.T  # (L,K)
    # Add complex Gaussian noise CN(0, sigma2):
    noise = (torch.randn(L, K, dtype=complex_dtype, device=device) +
             1j * torch.randn(L, K, dtype=complex_dtype, device=device)) * math.sqrt(sigma2 / 2.0)
    Y = Y0 + noise
    return Y, {"alpha_SR": alpha_SR, "alpha_STR": alpha_STR, "sigma2": torch.tensor(sigma2, device=device)}


# -----------------------------
# Case I: ML joint decoding (no penalty)
# -----------------------------
def decode_case_I(env: Env, Y: Tensor) -> Tuple[int, int, Tensor, Tensor]:
    """
    Joint ML decoding without penalty by eliminating channels:
      minimize ||(I - Xi Xi^†) Y^T x||_F^2 + ||(I - Xi Xi^†) Y^T 1||_F^2 over c in C, x in X

    Returns:
      (c_hat, x_hat, gamma_SR_hat, gamma_STR_hat)
    where gamma_*_hat are recovered using the ML closed forms:
      gamma_STR_hat = (1/L) Xi^† Y^T x_hat,
      gamma_SR_hat  = (1/L) Xi^† Y^T 1
    """
    device = env.device
    L, U, Ma = env.L, env.U, env.Ma
    ones = torch.ones((L, 1), dtype=complex_dtype, device=device)

    YT = Y.T  # (K,L)
    YT1 = YT @ ones / env.L  # (K,1) scaled by L^{-1} but we keep 1/L outside if needed

    best_val = float("inf")
    best_c = 0
    best_x = 0

    # Precompute X^T YT^H? We need (I-P) Y^T x; do this per-c (project), then search x
    for c in range(Ma): ## first fix c, iterate over x
        P = env.proj_list[c]  # (K,K)
        PYT = P @ YT  # (K,L), common terms for two quadratic forms
        term_const = frob_norm2(P @ (YT @ ones)).item() / (env.L ** 0)  # no scale here (same across x)

        # For all x in codebook, compute ||P Y^T x||^2 efficiently:
        # stack X (L,U) -> PYT @ X  => (K,U)
        PX = PYT @ env.X_mat  # (K,U)
        vals = (PX.conj() * PX).real.sum(dim=0)  # (U,)
        # Add the constant term (one-vector term)
        vals = vals + term_const

        vmin, idx = torch.min(vals, dim=0) ## best x given c
        if vmin.item() < best_val:
            best_val = vmin.item()
            best_c = c 
            best_x = int(idx.item()) ## an index of the codebook, not the codeword itself

    # Channel estimates using closed forms
    Xi = env.Xi_list[best_c]
    Xi_dag = env.pinv_list[best_c]
    xhat = env.X_mat[:, best_x:best_x + 1]
    ones = torch.ones((env.L, 1), dtype=complex_dtype, device=device)
    gamma_STR_hat = (Xi_dag @ (Y.T @ xhat)) / env.L
    gamma_SR_hat = (Xi_dag @ (Y.T @ ones)) / env.L
    return best_c, best_x, gamma_SR_hat, gamma_STR_hat ## final results


# -----------------------------
# Case II: ML with Tikhonov (ridge)
# -----------------------------
def decode_case_II(env: Env, Y: Tensor) -> Tuple[int, int, Tensor, Tensor]:
    """
    Ridge-regularized ML decoding.
    With P_c,STR = Xi (Xi^H Xi + lambda_STR I)^{-1} Xi^H  (similarly for SR),
    the equivalent decision is to minimize:
      x^T Y (I - P_{c,STR}) Y^T x + 1^T Y (I - P_{c,SR}) Y^T 1
    over c in C, x in X.
    """
    device = env.device
    L, U, Ma = env.L, env.U, env.Ma
    ones = torch.ones((L, 1), dtype=complex_dtype, device=device)

    # Precompute Y A Y^T terms per c
    best_val = float("inf")
    best_c = 0
    best_x = 0

    for c in range(Ma):
        P_str = env.ridge_proj_STR[c]  # (K,K)
        P_sr = env.ridge_proj_SR[c]
        I_K = torch.eye(env.K, dtype=complex_dtype, device=device)

        # Compute quadratic forms
        A_str = I_K - P_str
        A_sr = I_K - P_sr

        # For tag: minimize x^T Y A_str Y^T x  -> search over codebook
        YAYT_str = Y @ (A_str @ Y.conj().T)  # (L,L)
        # For radar: constant (w.r.t x), all-one vector term
        const_val = (ones.conj().T @ (Y @ (A_sr @ (Y.conj().T @ ones)))).real.item()

        # Evaluate all x simultaneously
        # Value = diag( X^H Y A_str Y^H X )
        Z = env.X_mat.conj().T @ (YAYT_str @ env.X_mat)  # (U,U)
        ## Extract diagonal --> quadratic forms for each x
        vals = Z.diag().real + const_val

        vmin, idx = torch.min(vals, dim=0)
        if vmin.item() < best_val:
            best_val = vmin.item()
            best_c = c
            best_x = int(idx.item()) ## an index of the codebook, not the codeword itself

    # Channel estimates via ridge closed-forms:
    Xi = env.Xi_list[best_c]
    XtX = Xi.conj().T @ Xi
    I_ch = torch.eye(env.Q + 1, dtype=complex_dtype, device=device)
    lam_str = env.lambdas_STR[best_c]
    lam_sr = env.lambdas_SR[best_c]

    xhat = env.X_mat[:, best_x:best_x + 1]
    ones = torch.ones((env.L, 1), dtype=complex_dtype, device=device)

    gamma_STR_hat = torch.linalg.solve(XtX + lam_str * I_ch, Xi.conj().T @ (Y.T @ xhat)) / env.L
    gamma_SR_hat = torch.linalg.solve(XtX + lam_sr * I_ch, Xi.conj().T @ (Y.T @ ones)) / env.L
    return best_c, best_x, gamma_SR_hat, gamma_STR_hat


# -----------------------------
# Case III: Disjoint low-complexity decoding
# -----------------------------
def decode_case_III(env: Env, Y: Tensor) -> Tuple[int, int, Tensor, Tensor]:
    """
    Disjoint decoding:

      Tag:    x_hat = argmax_x || x^H Y ||_F^2
      Radar:  v = (1/L) Y^T 1;  c_hat = argmin_c || (I - Xi Xi^†) v ||^2
              (equivalently argmax_c || Xi^† v ||^2)

    Channel estimates for NMSE reporting:
      gamma_SR_hat = Xi^† v
      gamma_STR_hat = Xi^† ((1/L) Y^T x_hat)
    """
    device = env.device
    L, K, Ma, U = env.L, env.K, env.Ma, env.U

    # Tag by max-correlation
    XHY = env.X_mat.conj().T @ Y  # (U, K)
    energy = (XHY.conj() * XHY).real.sum(dim=1)  # (U,)
    x_idx = int(torch.argmax(energy).item())

    # Radar by LS on direct-path component
    ones = torch.ones((L, 1), dtype=complex_dtype, device=device)
    v = (Y.T @ ones) / L  # (K,1), direct-path observation

    best_val = float("inf")
    best_c = 0
    for c in range(Ma):
        P = env.proj_list[c] ## projector (I - Xi Xi^†), orthogonal complement
        resid = P @ v
        val = complex_norm2(resid).item()
        if val < best_val:
            best_val = val
            best_c = c

    # Channel estimates for NMSE:
    Xi = env.Xi_list[best_c]
    Xi_dag = env.pinv_list[best_c]
    gamma_SR_hat = Xi_dag @ v
    xhat = env.X_mat[:, x_idx:x_idx + 1]
    gamma_STR_hat = Xi_dag @ ((Y.T @ xhat) / L)
    return best_c, x_idx, gamma_SR_hat, gamma_STR_hat


# -----------------------------
# Monte-Carlo driver
# -----------------------------
def run_mc(
    env: Env,
    snr_grid_db: List[float], ## SNR points in dB
    N_mc: int = 500, ## number of MC trials per SNR point
) -> Dict[str, Dict[str, List[float]]]:
    """
    For each SNR point:
      - Repeat N_mc trials; at each trial, sample (c, x), synthesize Y
      - Run Case I/II/III decoders
      - Accumulate MER for radar and tag; NMSE for SR & STR channels
    Returns a nested dict with per-SNR arrays.
    """
    results = {
        "CaseI": {"RadarMER": [], "TagMER": [], "NMSE_SR": [], "NMSE_STR": []},
        "CaseII": {"RadarMER": [], "TagMER": [], "NMSE_SR": [], "NMSE_STR": []},
        "CaseIII": {"RadarMER": [], "TagMER": [], "NMSE_SR": [], "NMSE_STR": []},
    }

    Ma, U = env.Ma, env.U

    for snr_db in snr_grid_db:
        # Counters
        cnt_I = {"radar_err": 0, "tag_err": 0, "nmse_sr": 0.0, "nmse_str": 0.0}
        cnt_II = {"radar_err": 0, "tag_err": 0, "nmse_sr": 0.0, "nmse_str": 0.0}
        cnt_III = {"radar_err": 0, "tag_err": 0, "nmse_sr": 0.0, "nmse_str": 0.0}

        for _ in range(N_mc):
            print(f"MC trial {_+1}/{N_mc} at SNR {snr_db:+.1f} dB", end="\r")
            # Sample a valid waveform & tag
            c_true = random.randrange(Ma)
            x_true = random.randrange(U)

            # Synthesize observation
            Y, aux = synthesize_observation(env, c_true, x_true, snr_db) ## c and x are randomly chosen in each trial

            # Decode: Case I
            cI, xI, gSR_I, gSTR_I = decode_case_I(env, Y)
            cnt_I["radar_err"] += int(cI != c_true)
            cnt_I["tag_err"] += int(xI != x_true)
            cnt_I["nmse_sr"] += nmse(gSR_I.squeeze(), env.gamma_SR.squeeze())
            cnt_I["nmse_str"] += nmse(gSTR_I.squeeze(), env.gamma_STR.squeeze())

            # Decode: Case II
            cII, xII, gSR_II, gSTR_II = decode_case_II(env, Y)
            cnt_II["radar_err"] += int(cII != c_true)
            cnt_II["tag_err"] += int(xII != x_true)
            cnt_II["nmse_sr"] += nmse(gSR_II.squeeze(), env.gamma_SR.squeeze())
            cnt_II["nmse_str"] += nmse(gSTR_II.squeeze(), env.gamma_STR.squeeze())

            # Decode: Case III
            cIII, xIII, gSR_III, gSTR_III = decode_case_III(env, Y)
            cnt_III["radar_err"] += int(cIII != c_true)
            cnt_III["tag_err"] += int(xIII != x_true)
            cnt_III["nmse_sr"] += nmse(gSR_III.squeeze(), env.gamma_SR.squeeze())
            cnt_III["nmse_str"] += nmse(gSTR_III.squeeze(), env.gamma_STR.squeeze())

        # Aggregate to rates/means
        for name, cnt in zip(
            ["CaseI", "CaseII", "CaseIII"],
            [cnt_I, cnt_II, cnt_III],
        ):
            results[name]["RadarMER"].append(cnt["radar_err"] / N_mc)
            results[name]["TagMER"].append(cnt["tag_err"] / N_mc)
            results[name]["NMSE_SR"].append(cnt["nmse_sr"] / N_mc)
            results[name]["NMSE_STR"].append(cnt["nmse_str"] / N_mc)

        print(f"[SNR {snr_db:+.1f} dB] "
              f"CaseI MER(R,T)=({results['CaseI']['RadarMER'][-1]:.3f},{results['CaseI']['TagMER'][-1]:.3f}) "
              f"NMSE(SR,STR)=({results['CaseI']['NMSE_SR'][-1]:.3e},{results['CaseI']['NMSE_STR'][-1]:.3e})")
        print(f"            "
              f"CaseII MER(R,T)=({results['CaseII']['RadarMER'][-1]:.3f},{results['CaseII']['TagMER'][-1]:.3f}) "
              f"NMSE(SR,STR)=({results['CaseII']['NMSE_SR'][-1]:.3e},{results['CaseII']['NMSE_STR'][-1]:.3e})")
        print(f"            "
              f"CaseIII MER(R,T)=({results['CaseIII']['RadarMER'][-1]:.3f},{results['CaseIII']['TagMER'][-1]:.3f}) "
              f"NMSE(SR,STR)=({results['CaseIII']['NMSE_SR'][-1]:.3e},{results['CaseIII']['NMSE_STR'][-1]:.3e})")

    return results


# -----------------------------
# Waveform Auto-Correlation Analysis
# -----------------------------
def compute_shifted_autocorr(waveform: Tensor, max_shift: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    """
    Compute the shifted auto-correlation of a waveform.
    
    R[k] = (1/N) * sum_{n=0}^{N-1-|k|} waveform[n] * conj(waveform[n+|k|])
    
    Args:
        waveform: Complex waveform vector of length N (1D tensor)
        max_shift: Maximum shift to compute (default: N-1)
    
    Returns:
        shifts: Array of shift values [-max_shift, ..., 0, ..., max_shift]
        autocorr: Auto-correlation values (magnitude) at each shift
    """
    N = waveform.numel()
    if max_shift is None:
        max_shift = N - 1
    
    max_shift = min(max_shift, N - 1)
    
    # Compute for positive shifts (including zero)
    autocorr_pos = torch.zeros(max_shift + 1, dtype=torch.float32)
    for k in range(max_shift + 1):
        # Correlation at shift k
        if k == 0:
            # Zero shift: sum of |w[n]|^2
            autocorr_pos[k] = (waveform.conj() * waveform).real.sum() / N
        else:
            # Positive shift: w[n] * conj(w[n+k])
            overlap = waveform[:-k].conj() * waveform[k:]
            autocorr_pos[k] = overlap.sum().abs() / N
    
    # Negative shifts are conjugate symmetric for autocorr (so magnitude is same)
    autocorr_neg = autocorr_pos[1:].flip(0)  # Reverse order, exclude zero
    
    # Concatenate
    autocorr = torch.cat([autocorr_neg, autocorr_pos])
    shifts = torch.arange(-max_shift, max_shift + 1, dtype=torch.int32)
    
    return shifts, autocorr


def test_waveform_autocorr(Np: int = 20, Ma: int = 32, max_shift: Optional[int] = None, 
                           waveform_idx: Optional[int] = None, seed: Optional[int] = None):
    """
    Test the shifted auto-correlation property of a randomly selected waveform.
    
    Args:
        Np: Waveform length
        Ma: Number of waveforms in the set
        max_shift: Maximum shift to compute (default: Np-1)
        waveform_idx: Index of waveform to test (if None, randomly select)
        seed: Random seed (if None, use current random state for truly random selection)
    
    Returns:
        Plots the auto-correlation function
    """
    import matplotlib.pyplot as plt
    
    # Only set seed if explicitly provided (for reproducibility)
    if seed is not None:
        set_seed(seed)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Build waveform set
    C_mat = build_waveform_set(Np, Ma, device)
    
    # Select a waveform (random or specified)
    if waveform_idx is None:
        waveform_idx = random.randrange(Ma)
    
    waveform = C_mat[waveform_idx].cpu()  # Move to CPU for plotting
    
    print(f"Testing waveform #{waveform_idx} (root used in ZC generation)")
    print(f"Waveform length: {Np}")
    print(f"Waveform type: Zadoff-Chu sequence")
    
    # Compute autocorrelation
    shifts, autocorr = compute_shifted_autocorr(waveform, max_shift)
    autocorr_np = autocorr.numpy()
    shifts_np = shifts.numpy()
    
    # Find peak properties
    peak_val = autocorr_np[len(autocorr_np)//2]  # Value at zero shift
    sidelobe_max = max(autocorr_np[0], autocorr_np[-1])  # Max sidelobe
    psr = 20 * math.log10(peak_val / (sidelobe_max + 1e-12))  # Peak-to-sidelobe ratio in dB
    
    print(f"Peak (zero-shift) autocorr: {peak_val:.4f}")
    print(f"Max sidelobe: {sidelobe_max:.4f}")
    print(f"Peak-to-Sidelobe Ratio (PSR): {psr:.2f} dB")
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Subplot 1: Autocorrelation magnitude
    axes[0].stem(shifts_np, autocorr_np, basefmt=" ", linefmt='b-', markerfmt='bo')
    axes[0].set_xlabel('Shift (samples)', fontsize=12)
    axes[0].set_ylabel('Autocorrelation Magnitude', fontsize=12)
    axes[0].set_title(f'Shifted Auto-Correlation of Waveform #{waveform_idx} (ZC Sequence)', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Subplot 2: Autocorrelation in dB scale
    autocorr_db = 20 * torch.log10(autocorr + 1e-12).numpy()
    axes[1].stem(shifts_np, autocorr_db, basefmt=" ", linefmt='r-', markerfmt='ro')
    axes[1].set_xlabel('Shift (samples)', fontsize=12)
    axes[1].set_ylabel('Autocorrelation (dB)', fontsize=12)
    axes[1].set_title(f'Auto-Correlation in dB Scale (PSR = {psr:.1f} dB)', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('waveform_autocorr_test.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'waveform_autocorr_test.png'")
    plt.show()
    
    return waveform, shifts, autocorr


# -----------------------------
# Waveform Cross-Correlation Analysis
# -----------------------------
def compute_shifted_crosscorr(waveform1: Tensor, waveform2: Tensor, max_shift: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    """
    Compute the shifted cross-correlation between two waveforms.
    
    R_{12}[k] = (1/N) * sum_{n} waveform1[n] * conj(waveform2[n+k])
    
    Args:
        waveform1: First complex waveform vector of length N (1D tensor)
        waveform2: Second complex waveform vector of length N (1D tensor)
        max_shift: Maximum shift to compute (default: N-1)
    
    Returns:
        shifts: Array of shift values [-max_shift, ..., 0, ..., max_shift]
        crosscorr: Cross-correlation values (magnitude) at each shift
    """
    N = waveform1.numel()
    assert waveform2.numel() == N, "Waveforms must have same length"
    
    if max_shift is None:
        max_shift = N - 1
    
    max_shift = min(max_shift, N - 1)
    
    # Compute for all shifts from -max_shift to +max_shift
    crosscorr_list = []
    for k in range(-max_shift, max_shift + 1):
        if k == 0:
            # Zero shift: direct element-wise product
            overlap = waveform1.conj() * waveform2
            crosscorr_list.append(overlap.sum().abs() / N)
        elif k > 0:
            # Positive shift: waveform2 shifts right (or waveform1 shifts left)
            # w1[n] * conj(w2[n+k])  for n in [0, N-1-k]
            overlap = waveform1[:-k].conj() * waveform2[k:]
            crosscorr_list.append(overlap.sum().abs() / N)
        else:  # k < 0
            # Negative shift: waveform2 shifts left (or waveform1 shifts right)
            # w1[n] * conj(w2[n+k])  for n in [-k, N-1]
            k_abs = abs(k)
            overlap = waveform1[k_abs:].conj() * waveform2[:-k_abs]
            crosscorr_list.append(overlap.sum().abs() / N)
    
    crosscorr = torch.tensor(crosscorr_list, dtype=torch.float32)
    shifts = torch.arange(-max_shift, max_shift + 1, dtype=torch.int32)
    
    return shifts, crosscorr


def test_waveform_crosscorr(Np: int = 20, Ma: int = 32, max_shift: Optional[int] = None, 
                            waveform_idx1: Optional[int] = None, 
                            waveform_idx2: Optional[int] = None, 
                            seed: Optional[int] = None):
    """
    Test the shifted cross-correlation property between two randomly selected waveforms.
    
    Args:
        Np: Waveform length
        Ma: Number of waveforms in the set
        max_shift: Maximum shift to compute (default: Np-1)
        waveform_idx1: Index of first waveform (if None, randomly select)
        waveform_idx2: Index of second waveform (if None, randomly select different from first)
        seed: Random seed (if None, use current random state for truly random selection)
    
    Returns:
        Plots the cross-correlation function
    """
    import matplotlib.pyplot as plt
    
    # Only set seed if explicitly provided (for reproducibility)
    if seed is not None:
        set_seed(seed)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Build waveform set
    C_mat = build_waveform_set(Np, Ma, device)
    
    # Select two different waveforms (random or specified)
    if waveform_idx1 is None:
        waveform_idx1 = random.randrange(Ma)
    
    if waveform_idx2 is None:
        # Ensure we pick a different waveform
        waveform_idx2 = random.randrange(Ma)
        while waveform_idx2 == waveform_idx1 and Ma > 1:
            waveform_idx2 = random.randrange(Ma)
    
    waveform1 = C_mat[waveform_idx1].cpu()  # Move to CPU for plotting
    waveform2 = C_mat[waveform_idx2].cpu()
    
    print(f"Testing cross-correlation between waveform #{waveform_idx1} and waveform #{waveform_idx2}")
    print(f"Waveform length: {Np}")
    print(f"Waveform type: Zadoff-Chu sequences (different roots)")
    
    # Compute cross-correlation
    shifts, crosscorr = compute_shifted_crosscorr(waveform1, waveform2, max_shift)
    crosscorr_np = crosscorr.numpy()
    shifts_np = shifts.numpy()
    
    # Find peak properties
    peak_val = crosscorr_np.max()  # Maximum cross-correlation value
    peak_idx = crosscorr_np.argmax()
    peak_shift = shifts_np[peak_idx]
    mean_val = crosscorr_np.mean()  # Average cross-correlation level
    
    # Peak-to-Average Ratio (PAR)
    par = 20 * math.log10(peak_val / (mean_val + 1e-12))
    
    print(f"Peak cross-correlation: {peak_val:.4f} at shift = {peak_shift}")
    print(f"Mean cross-correlation: {mean_val:.4f}")
    print(f"Peak-to-Average Ratio (PAR): {par:.2f} dB")
    
    # Compute inner product (zero-shift correlation) for orthogonality check
    inner_prod = (waveform1.conj() * waveform2).sum().abs().item() / Np
    print(f"Inner product (normalized): {inner_prod:.4f}")
    print(f"Orthogonality metric: {20 * math.log10(inner_prod + 1e-12):.2f} dB")
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Subplot 1: Cross-correlation magnitude
    axes[0].stem(shifts_np, crosscorr_np, basefmt=" ", linefmt='g-', markerfmt='go')
    axes[0].set_xlabel('Shift (samples)', fontsize=12)
    axes[0].set_ylabel('Cross-Correlation Magnitude', fontsize=12)
    axes[0].set_title(f'Shifted Cross-Correlation: Waveform #{waveform_idx1} vs #{waveform_idx2}', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[0].axhline(y=mean_val, color='r', linestyle='--', linewidth=1, alpha=0.7, label=f'Mean = {mean_val:.3f}')
    axes[0].legend()
    
    # Subplot 2: Cross-correlation in dB scale
    crosscorr_db = 20 * torch.log10(crosscorr + 1e-12).numpy()
    axes[1].stem(shifts_np, crosscorr_db, basefmt=" ", linefmt='m-', markerfmt='mo')
    axes[1].set_xlabel('Shift (samples)', fontsize=12)
    axes[1].set_ylabel('Cross-Correlation (dB)', fontsize=12)
    axes[1].set_title(f'Cross-Correlation in dB Scale (PAR = {par:.1f} dB)', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[1].axhline(y=20*math.log10(mean_val + 1e-12), color='r', linestyle='--', 
                    linewidth=1, alpha=0.7, label=f'Mean = {20*math.log10(mean_val + 1e-12):.1f} dB')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('waveform_crosscorr_test.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'waveform_crosscorr_test.png'")
    plt.show()
    
    return waveform1, waveform2, shifts, crosscorr


def analyze_waveform_set_correlation(Np: int = 20, Ma: int = 32, seed: Optional[int] = None):
    """
    Analyze the full correlation matrix of the waveform set.
    Computes all pairwise cross-correlations and visualizes the results.
    
    Args:
        Np: Waveform length
        Ma: Number of waveforms in the set
        seed: Random seed (if None, use current random state)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if seed is not None:
        set_seed(seed)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Build waveform set
    print(f"\n=== Building Waveform Set (Np={Np}, Ma={Ma}) ===")
    C_mat = build_waveform_set(Np, Ma, device)
    C_cpu = C_mat.cpu()
    
    # Compute cross-correlation matrix (at zero shift)
    corr_matrix = torch.zeros((Ma, Ma), dtype=torch.float32)
    for i in range(Ma):
        for j in range(Ma):
            if i == j:
                corr_matrix[i, j] = 1.0  # Auto-correlation at zero shift
            else:
                # Cross-correlation at zero shift
                corr_val = (C_cpu[i].conj() * C_cpu[j]).sum().abs() / Np
                corr_matrix[i, j] = corr_val.item()
    
    # Statistics
    # Extract off-diagonal elements (cross-correlations only)
    mask = ~torch.eye(Ma, dtype=torch.bool)
    cross_corrs = corr_matrix[mask]
    
    max_cross = cross_corrs.max().item()
    min_cross = cross_corrs.min().item()
    mean_cross = cross_corrs.mean().item()
    std_cross = cross_corrs.std().item()
    
    print(f"\n=== Cross-Correlation Statistics (Zero-Shift) ===")
    print(f"Maximum cross-correlation: {max_cross:.4f} ({20*math.log10(max_cross+1e-12):.1f} dB)")
    print(f"Minimum cross-correlation: {min_cross:.4f} ({20*math.log10(min_cross+1e-12):.1f} dB)")
    print(f"Mean cross-correlation:    {mean_cross:.4f} ({20*math.log10(mean_cross+1e-12):.1f} dB)")
    print(f"Std cross-correlation:     {std_cross:.4f}")
    
    # Count pairs with high cross-correlation
    high_corr_threshold = 0.3
    high_corr_pairs = (cross_corrs > high_corr_threshold).sum().item()
    print(f"Pairs with cross-corr > {high_corr_threshold}: {high_corr_pairs} / {Ma*(Ma-1)//2}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Correlation matrix heatmap
    im = axes[0].imshow(corr_matrix.numpy(), cmap='viridis', vmin=0, vmax=1)
    axes[0].set_xlabel('Waveform Index', fontsize=12)
    axes[0].set_ylabel('Waveform Index', fontsize=12)
    axes[0].set_title(f'Cross-Correlation Matrix (Zero-Shift)\nNp={Np}, Ma={Ma}', fontsize=14)
    cbar = plt.colorbar(im, ax=axes[0])
    cbar.set_label('Correlation Magnitude', fontsize=11)
    
    # Subplot 2: Histogram of cross-correlations
    axes[1].hist(cross_corrs.numpy(), bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1].axvline(mean_cross, color='r', linestyle='--', linewidth=2, label=f'Mean = {mean_cross:.3f}')
    axes[1].axvline(high_corr_threshold, color='orange', linestyle='--', linewidth=2, 
                    label=f'Threshold = {high_corr_threshold}')
    axes[1].set_xlabel('Cross-Correlation Magnitude', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title(f'Distribution of Cross-Correlations\n(Off-diagonal elements)', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('waveform_set_correlation_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'waveform_set_correlation_analysis.png'")
    plt.show()
    
    return C_mat, corr_matrix


# -----------------------------
# Main
# -----------------------------
def mc_simulation():
    # System parameters (as agreed)
    L = 10
    Np = 31  # Prime number for better ZC sequence cross-correlation properties
    Qmin, Qmax = 10, 20    # Q = 10, K = 41
    Ma = 16  # Reduced from 32 for better orthogonality
    U = 16   # Reduced from 32 to match Ma

    env = generate_environment(
        L=L, Np=Np, Qmin=Qmin, Qmax=Qmax, Ma=Ma, U=U,
        P_SR=1.0, P_STR=1.0, seed=2025, use_gpu_if_available=True
    )

    # SNR sweep
    # snr_grid_db = [-5.0, -2.5, 0.0, 2.5, 5.0]
    snr_grid_db = [5.0]
    # MC rounds (you can increase to 5000/10000 as needed)
    N_mc = 500

    results = run_mc(env, snr_grid_db, N_mc=N_mc)

    print("\n=== Summary (per SNR grid order) ===")
    for case in ["CaseI", "CaseII", "CaseIII"]:
        print(f"{case}:")
        print("  RadarMER :", results[case]["RadarMER"])
        print("  TagMER   :", results[case]["TagMER"])
        print("  NMSE_SR  :", results[case]["NMSE_SR"])
        print("  NMSE_STR :", results[case]["NMSE_STR"])


if __name__ == "__main__":
    # run Monte-Carlo simulation
    mc_simulation()
    
    # Analyze full waveform set correlation properties
    # print("=== Analyzing Full Waveform Set Correlation Matrix ===")
    # C_mat, corr_matrix = analyze_waveform_set_correlation(
    #     Np=31,  # Prime number for better cross-correlation properties
    #     Ma=16,  # Reduced waveform set size for better orthogonality
    #     seed=2025  # Use fixed seed for reproducibility
    # )
    
    # print("\n" + "="*70 + "\n")
    
    # # Test waveform auto-correlation properties
    # print("=== Testing Waveform Auto-Correlation Properties ===\n")
    # waveform, shifts, autocorr = test_waveform_autocorr(
    #     Np=31,           # Waveform length (prime number)
    #     Ma=16,           # Reduced number of waveforms in set
    #     max_shift=None,  # Compute full autocorr (all shifts)
    #     waveform_idx=None,  # Randomly select a waveform (or set to specific index like 0, 1, 2...)
    #     seed=None        # Set to None for truly random selection; set to 2025 for reproducibility
    # )
    
    # print("\n" + "="*70 + "\n")
    
    # # Test waveform cross-correlation properties
    # print("=== Testing Waveform Cross-Correlation Properties ===\n")
    # waveform1, waveform2, shifts_cross, crosscorr = test_waveform_crosscorr(
    #     Np=31,            # Waveform length (prime number)
    #     Ma=16,            # Reduced number of waveforms in set
    #     max_shift=None,   # Compute full cross-correlation (all shifts)
    #     waveform_idx1=None,  # Randomly select first waveform (or set to specific index)
    #     waveform_idx2=None,  # Randomly select second waveform (or set to specific index)
    #     seed=None         # Set to None for truly random selection; set to 2025 for reproducibility
    # )

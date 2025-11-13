# blind_decode_main.py
# -*- coding: utf-8 -*-
"""
Simplified DFRC blind decoding using NEW projection method (maximize column space projection).
Compares three decoding methods:
- Case I: Joint ML with projection to column space (NEW method)
- Case II: Joint ML with ridge regularization + projection to column space
- Case III: Separate ML decoding (baseline)
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import matplotlib.pyplot as plt
import numpy as np

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
# TEST 2: Tag Codebook Properties
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
            # print(f"  Selected root {best_root}, max cross-corr with existing: {best_max_corr:.4f}")
    
    if len(selected_seqs) < Ma:
        print(f"Warning: Could only generate {len(selected_seqs)} sequences with good correlation properties (requested {Ma})")
    
    C = torch.stack(selected_seqs, dim=0).to(device)  # (Ma, Np), unit-modulus
    # Normalize (already unit modulus), but keep it explicit:
    C = C / C.abs()  # guard against any numeric deviation
    
    # print(f"Generated {len(selected_roots)} ZC waveforms with roots: {selected_roots[:10]}{'...' if len(selected_roots) > 10 else ''}")
    return C  # (Ma, Np)


def build_tag_codebook(L: int, U: int, device: torch.device) -> Tensor:
    """
    Build U orthogonal codewords x in C^L using Gram-Schmidt method:
      * Each x is orthogonal to 1_L (remove DC/mean)
      * Each x_i is orthogonal to x_j for i≠j
      * ||x||^2 = L
    
    This ensures NO DUPLICATE codewords and strict orthogonality.
    Maximum U = L-1 (since we need x ⊥ 1_L).
    
    Construction:
    1. Generate random complex vectors
    2. Gram-Schmidt orthogonalization:
       - First orthogonalize to 1_L
       - Then orthogonalize to all previously generated codewords
    3. Normalize to energy L
    """
    if U > L - 1:
        print(f"Warning: U={U} > L-1={L-1}, can only generate {L-1} orthogonal codewords")
        print(f"Setting U = {L-1}")
        U = L - 1
    
    ones = torch.ones(L, dtype=complex_dtype)
    X_cols = []
    
    print(f"\n🔧 Generating {U} orthogonal codewords using Gram-Schmidt:")
    
    for u_idx in range(U):
        max_attempts = 1000
        success = False
        
        for attempt in range(max_attempts):
            # Generate random complex vector
            v_real = torch.randn(L, dtype=torch.float64)
            v_imag = torch.randn(L, dtype=torch.float64)
            v = (v_real + 1j * v_imag).to(complex_dtype)
            
            # Step 1: Orthogonalize to 1_L
            # v <- v - <1, v> / ||1||^2 * 1
            inner_ones = ones.conj() @ v
            v = v - (inner_ones / (ones.conj() @ ones)) * ones
            
            # Step 2: Orthogonalize to all previously generated codewords
            for x_prev in X_cols:
                inner_prev = x_prev.conj() @ v
                v = v - (inner_prev / (x_prev.conj() @ x_prev)) * x_prev
            
            # Check if v is degenerate (too small norm)
            norm_v = torch.sqrt((v.conj() * v).real.sum()).item()
            if norm_v > 0.1:  # Not degenerate
                # Step 3: Normalize to energy L
                v = v / torch.sqrt((v.conj() * v).real.sum()) * math.sqrt(L)
                X_cols.append(v)
                success = True
                
                # if (u_idx + 1) % 4 == 0 or u_idx == U - 1:
                #     print(f"  ✓ Generated {u_idx + 1}/{U} codewords")
                break  # Success, move to next codeword
        
        if not success:
            print(f"Failed to generate codeword {u_idx + 1} after {max_attempts} attempts")
            # print(f"Space is likely full (dimension exhausted)")
            break
    
    if len(X_cols) < U:
        # print(f"Could only generate {len(X_cols)} orthogonal codewords (requested {U})")
        U = len(X_cols)
    
    X = torch.stack(X_cols, dim=1).to(device)  # (L, U)
    
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
## Environment generation with custom lambda
# -----------------------------


def generate_environment_with_lambda(
    L: int = 20,
    Np: int = 31,
    Qmin: int = 10,
    Qmax: int = 20,
    Ma: int = 16,
    U: int = 16,
    P_SR: float = 1.0,
    P_STR: float = 1.0,
    seed: int = 2025,
    lambda_multiplier: float = 0.1,  # NEW: λ = lambda_multiplier * ||Xi Xi^H||_2
    use_gpu_if_available: bool = True,
) -> Env:
    """
    Build environment with specific lambda multiplier for testing.
    Same as generate_environment but allows custom lambda_multiplier parameter.
    """
    set_seed(seed)
    device = torch.device("cuda") if (use_gpu_if_available and torch.cuda.is_available()) else torch.device("cpu")

    Q = Qmax - Qmin
    K = Np + Q

    # Build waveforms and codebook (reuse existing)
    C_mat = build_waveform_set(Np, Ma, device)
    X_mat = build_tag_codebook(L, U, device)

    # Channels
    gamma_SR = (torch.randn(Q + 1, dtype=complex_dtype, device=device) +
                1j * torch.randn(Q + 1, dtype=complex_dtype, device=device)) * math.sqrt(P_SR / 2)
    gamma_STR = (torch.randn(Q + 1, dtype=complex_dtype, device=device) +
                 1j * torch.randn(Q + 1, dtype=complex_dtype, device=device)) * math.sqrt(P_STR / 2)

    # Build matrices with custom lambda
    Xi_list: List[Tensor] = []
    pinv_list: List[Tensor] = []
    proj_list: List[Tensor] = []
    ridge_proj_SR: List[Tensor] = []
    ridge_proj_STR: List[Tensor] = []
    lambdas_SR: List[float] = []
    lambdas_STR: List[float] = []

    I_K = torch.eye(K, dtype=complex_dtype, device=device)
    for idx in range(Ma):
        c = C_mat[idx]
        Xi = build_Xi_linear(c, K, Q)
        Xi_list.append(Xi)

        pinv = torch.linalg.pinv(Xi)
        pinv_list.append(pinv)
        proj = I_K - Xi @ pinv
        proj_list.append(proj)

        # Use custom lambda multiplier
        A = Xi @ Xi.conj().T
        spectral_A = torch.linalg.norm(A, ord=2).real.item()
        
        lam = lambda_multiplier * spectral_A  # Custom lambda
        lambdas_STR.append(lam)
        lambdas_SR.append(lam)

        XtX = Xi.conj().T @ Xi
        I_ch = torch.eye(Q + 1, dtype=complex_dtype, device=device)

        P_STR = Xi @ torch.linalg.inv(XtX + lam * I_ch) @ Xi.conj().T
        P_SR = Xi @ torch.linalg.inv(XtX + lam * I_ch) @ Xi.conj().T
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
def synthesize_observation(env: Env, c_idx: int, x_idx: int, snr_db: float, 
                          gamma_SR: Optional[Tensor] = None, 
                          gamma_STR: Optional[Tensor] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    Generate the observation Y in C^{L x K} given chosen waveform c and tag code x, at a target SNR.
      Y = x a_STR^H + 1 a_SR^H + Omega,
    where a_* = Xi_c gamma_* in C^K, representing the effective channels.
    Noise variance is chosen per-trial to meet the requested SNR exactly.
    
    Args:
        env: Environment with system parameters
        c_idx: Waveform index
        x_idx: Tag codeword index
        snr_db: Target SNR in dB
        gamma_SR: Optional channel vector for SR path (if None, uses env.gamma_SR)
        gamma_STR: Optional channel vector for STR path (if None, uses env.gamma_STR)
    
    Returns:
      Y, aux dict with alpha vectors, sigma^2, and actual gamma values used.
    """
    device = env.device
    L, K = env.L, env.K
    Q = env.Q
    x = env.X_mat[:, x_idx:x_idx + 1]  # (L,1)
    ones = torch.ones((L, 1), dtype=complex_dtype, device=device)

    # Use provided channels or generate random ones
    if gamma_SR is None:
        gamma_SR = (torch.randn(Q + 1, 1, dtype=complex_dtype, device=device) +
                   1j * torch.randn(Q + 1, 1, dtype=complex_dtype, device=device)) / math.sqrt(2)
    if gamma_STR is None:
        gamma_STR = (torch.randn(Q + 1, 1, dtype=complex_dtype, device=device) +
                    1j * torch.randn(Q + 1, 1, dtype=complex_dtype, device=device)) / math.sqrt(2)

    Xi = env.Xi_list[c_idx]  # (K, Q+1)
    alpha_SR = Xi @ gamma_SR     # (K,1), effective SR channel
    alpha_STR = Xi @ gamma_STR   # (K,1), effective STR channel

    # Deterministic energy in the noiseless matrix, compute transmit SNR then
    # || x a_STR^H + 1 a_SR^H ||_F^2 = ||x||^2 ||a_STR||^2 + ||1||^2 ||a_SR||^2  (cross terms vanish because x ⟂ 1)
    signal_pow = (env.L * complex_norm2(alpha_STR) + env.L * complex_norm2(alpha_SR)).item()
    # SNR = signal_pow / E||Omega||_F^2 = signal_pow / (L*K*sigma^2)
    snr_lin = 10.0 ** (snr_db / 10.0)
    sigma2 = signal_pow / (L * K * snr_lin + 1e-12) ## adjust noise power accordingly

    # Noiseless:
    Y0 = x @ alpha_STR.conj().T + ones @ alpha_SR.conj().T  # (L,K)
    # Add complex Gaussian noise CN(0, sigma2):
    noise = (torch.randn(L, K, dtype=complex_dtype, device=device) +
             1j * torch.randn(L, K, dtype=complex_dtype, device=device)) * math.sqrt(sigma2 / 2.0)
    Y = Y0 + noise
    return Y, {"alpha_SR": alpha_SR, "alpha_STR": alpha_STR, "sigma2": torch.tensor(sigma2, device=device),
               "gamma_SR": gamma_SR, "gamma_STR": gamma_STR}


# -----------------------------
# Case I: ML joint decoding (no penalty)
# -----------------------------
def decode_case_I(env: Env, Y: Tensor) -> Tuple[int, int, Tensor, Tensor]:
    """
    Joint ML decoding using NEW projection to column space method:
      maximize ||Xi_c Xi_c^† Y^H x||^2 + ||Xi_c Xi_c^† Y^H 1||^2
    
    This preserves signal energy by projecting TO the column space.
    
    Returns:
      (c_hat, x_hat, gamma_SR_hat, gamma_STR_hat)
    """
    device = env.device
    L, U, Ma = env.L, env.U, env.Ma
    ones = torch.ones((L, 1), dtype=complex_dtype, device=device)

    YH = Y.conj().T  # (K,L)

    best_val = float("-inf")  # Maximize energy in column space
    best_c = 0
    best_x = 0

    for c in range(Ma):
        # Project TO column space: Proj = Xi Xi^†
        Xi = env.Xi_list[c]  # (K, Q+1)
        Xi_dag = env.pinv_list[c]  # (Q+1, K)
        Proj = Xi @ Xi_dag  # (K, K) - projection TO column space
        
        ProjYH = Proj @ YH  # (K, L)
        term_const = frob_norm2(ProjYH @ ones).item()

        # Compute ||Proj Y^H x||^2 for all x
        ProjX = ProjYH @ env.X_mat  # (K, U)
        vals = (ProjX.conj() * ProjX).real.sum(dim=0)  # (U,)
        vals = vals + term_const

        # Find maximum (highest energy in column space)
        vmax, idx = torch.max(vals, dim=0)
        if vmax.item() > best_val:
            best_val = vmax.item()
            best_c = c
            best_x = int(idx.item())

    # Decode gamma_SR and gamma_STR at best (c, x)
    Xi_best = env.Xi_list[best_c]
    Xi_dag_best = env.pinv_list[best_c]
    x_best = env.X_mat[:, best_x:best_x+1]
    
    # LS solution
    lhs = torch.cat([x_best, ones], dim=1)  # (L, 2)
    gamma_SR_hat = Xi_dag_best @ YH @ lhs[:, 1:2]  # (Q+1, 1)
    gamma_STR_hat = Xi_dag_best @ YH @ lhs[:, 0:1]

    return best_c, best_x, gamma_SR_hat, gamma_STR_hat


def decode_case_II(env: Env, Y: Tensor) -> Tuple[int, int, Tensor, Tensor]:
    """
    Joint ML decoding with ridge regularization using NEW projection method:
      maximize ||Xi_c Xi_c^† Y^H x||^2 + ||Xi_c Xi_c^† Y^H 1||^2
    Uses pre-computed ridge projection matrices from environment.
    
    Returns:
      (c_hat, x_hat, gamma_SR_hat, gamma_STR_hat)
    """
    device = env.device
    L, U, Ma = env.L, env.U, env.Ma
    ones = torch.ones((L, 1), dtype=complex_dtype, device=device)

    YH = Y.conj().T  # (K,L)

    best_val = float("-inf")  # Maximize
    best_c = 0
    best_x = 0

    for c in range(Ma):
        # Use pre-computed ridge projection TO column space
        Xi = env.Xi_list[c]
        Xi_dag = env.pinv_list[c]
        Proj = Xi @ Xi_dag  # Projection TO column space
        
        ProjYH = Proj @ YH  # (K, L)
        term_const = frob_norm2(ProjYH @ ones).item()

        ProjX = ProjYH @ env.X_mat  # (K, U)
        vals = (ProjX.conj() * ProjX).real.sum(dim=0)
        vals = vals + term_const

        vmax, idx = torch.max(vals, dim=0)
        if vmax.item() > best_val:
            best_val = vmax.item()
            best_c = c
            best_x = int(idx.item())

    # Decode with ridge regularization
    Xi_best = env.Xi_list[best_c]
    x_best = env.X_mat[:, best_x:best_x+1]
    lhs = torch.cat([x_best, ones], dim=1)

    XtX = Xi_best.conj().T @ Xi_best
    lam_SR = env.lambdas_SR[best_c]
    lam_STR = env.lambdas_STR[best_c]
    Q = env.Q
    I_ch = torch.eye(Q + 1, dtype=complex_dtype, device=device)

    gamma_SR_hat = torch.linalg.inv(XtX + lam_SR * I_ch) @ Xi_best.conj().T @ YH @ lhs[:, 1:2]
    gamma_STR_hat = torch.linalg.inv(XtX + lam_STR * I_ch) @ Xi_best.conj().T @ YH @ lhs[:, 0:1]

    return best_c, best_x, gamma_SR_hat, gamma_STR_hat


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
    v = (Y.conj().T @ ones) / L  # (K,1), direct-path observation

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
    gamma_STR_hat = Xi_dag @ ((Y.conj().T @ xhat) / L)
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
            print(f"SNR {snr_db} dB, Trial {_+1}/{N_mc}", end="\r")
            # Sample a valid waveform & tag
            c_true = random.randrange(Ma)
            x_true = random.randrange(U)
            
            # Generate random channels for this trial
            device = env.device
            Q = env.Q
            gamma_SR_trial = (torch.randn(Q + 1, 1, dtype=complex_dtype, device=device) +
                             1j * torch.randn(Q + 1, 1, dtype=complex_dtype, device=device)) / math.sqrt(2)
            gamma_STR_trial = (torch.randn(Q + 1, 1, dtype=complex_dtype, device=device) +
                              1j * torch.randn(Q + 1, 1, dtype=complex_dtype, device=device)) / math.sqrt(2)

            # Synthesize observation with random channels
            Y, aux = synthesize_observation(env, c_true, x_true, snr_db, 
                                          gamma_SR=gamma_SR_trial, 
                                          gamma_STR=gamma_STR_trial)
            
            # True channel vectors used in this trial
            gamma_SR_true = aux["gamma_SR"]
            gamma_STR_true = aux["gamma_STR"]
            
            # ============================================================
            # Decode with Case I (NEW: Column space projection)
            # ============================================================
            c_hat_I, x_hat_I, gamma_SR_I, gamma_STR_I = decode_case_I(env, Y)
            if c_hat_I != c_true:
                cnt_I["radar_err"] += 1
            if x_hat_I != x_true:
                cnt_I["tag_err"] += 1
            cnt_I["nmse_sr"] += nmse(gamma_SR_I, gamma_SR_true)
            cnt_I["nmse_str"] += nmse(gamma_STR_I, gamma_STR_true)
            
            # ============================================================
            # Decode with Case II (NEW: Ridge + Column space projection)
            # ============================================================
            c_hat_II, x_hat_II, gamma_SR_II, gamma_STR_II = decode_case_II(env, Y)
            if c_hat_II != c_true:
                cnt_II["radar_err"] += 1
            if x_hat_II != x_true:
                cnt_II["tag_err"] += 1
            cnt_II["nmse_sr"] += nmse(gamma_SR_II, gamma_SR_true)
            cnt_II["nmse_str"] += nmse(gamma_STR_II, gamma_STR_true)
            
            # ============================================================
            # Decode with Case III (Baseline: Separate decoding)
            # ============================================================
            c_hat_III, x_hat_III, gamma_SR_III, gamma_STR_III = decode_case_III(env, Y)
            if c_hat_III != c_true:
                cnt_III["radar_err"] += 1
            if x_hat_III != x_true:
                cnt_III["tag_err"] += 1
            cnt_III["nmse_sr"] += nmse(gamma_SR_III, gamma_SR_true)
            cnt_III["nmse_str"] += nmse(gamma_STR_III, gamma_STR_true)



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
# Main
# -----------------------------
def main():
    """Run MC simulation for DFRC blind decoding."""
    print("\n" + "="*70)
    print("DFRC BLIND DECODING - MONTE CARLO SIMULATION")
    print("="*70)
    
    # System parameters
    L = 20
    Np = 31
    Qmin, Qmax = 10, 20
    Ma = 16
    U = 16
    Q = Qmax - Qmin
    K = Np + Q
    
    set_seed(2025)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate environment (will build C_mat and X_mat internally)
    print("\n" + "="*70)
    print("GENERATING ENVIRONMENT...")
    print("="*70)
    env = generate_environment_with_lambda(
        L=L, Np=Np, Qmin=Qmin, Qmax=Qmax, Ma=Ma, U=U,
        P_SR=1.0, P_STR=1.0, seed=2025, 
        lambda_multiplier=0.001,  # Small regularization
        use_gpu_if_available=(device.type == "cuda")
    )
    
    # Run Monte Carlo simulation
    print("\n" + "="*70)
    print("MONTE CARLO SIMULATION")
    print("="*70)
    snr_grid_db = [0.0]
    N_mc = 500
    print(f"SNR grid: {snr_grid_db} dB")
    print(f"Trials per SNR: {N_mc}")
    
    results = run_mc(env, snr_grid_db, N_mc=N_mc)
    
    # Display final results
    print("\n" + "="*70)
    print("📊 FINAL RESULTS")
    print("="*70)
    print(f"{'Method':<30} {'Radar MER':<15} {'Tag MER':<15} {'Overall':<15}")
    print("-" * 70)
    
    for snr_idx, snr_db in enumerate(snr_grid_db):
        print(f"\nSNR = {snr_db:+.1f} dB:")
        for method_name in ["CaseI", "CaseII", "CaseIII"]:
            radar_mer = results[method_name]["RadarMER"][snr_idx]
            tag_mer = results[method_name]["TagMER"][snr_idx]
            overall_mer = (radar_mer + tag_mer) / 2.0
            
            marker = ""
            if method_name == "CaseI" or method_name == "CaseII":
                marker = "  (NEW: Column space projection)"
            elif method_name == "CaseIII":
                marker = "  (Baseline: Separate decoding)"
            
            print(f"  {method_name:<10} {radar_mer:<15.3f} {tag_mer:<15.3f} {overall_mer:<15.3f}{marker}")
    
    print("\n" + "="*70)
    print("✅ SIMULATION COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()


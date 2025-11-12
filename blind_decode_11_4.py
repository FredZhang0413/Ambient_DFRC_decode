# -*- coding: utf-8 -*-
"""
Ambient-carrier DFRC: simulation framework + decoding schemes
Author: (you)
Dependencies: PyTorch >= 2.0
All comments are in English as requested.
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch

torch.set_default_dtype(torch.float32)


# ----------------------------- Utilities ------------------------------------ #

def complex_tensor(x: torch.Tensor) -> torch.Tensor:
    """Ensure complex64 for all complex computations."""
    if x.is_complex():
        return x.to(torch.complex64)
    return x.to(torch.complex64)


def unit_modulus_vec(n: int, seed: Optional[int] = None) -> torch.Tensor:
    """Generate a length-n unit-modulus complex vector with random phases."""
    if seed is not None:
        g = torch.Generator(device="cpu").manual_seed(seed)
        phases = 2 * math.pi * torch.rand(n, generator=g)
    else:
        phases = 2 * math.pi * torch.rand(n)
    return torch.exp(1j * phases.to(torch.complex64))


def project_onto_ones_orth(x: torch.Tensor) -> torch.Tensor:
    """
    Project a complex vector x onto the subspace orthogonal to the all-ones vector.
    Then renormalize to unit-modulus per element if requested by the caller.
    """
    n = x.numel()
    one = torch.ones(n, dtype=torch.complex64)
    # Orthogonal projection: x <- x - <1,x>/<1,1> * 1
    x = x - (one.conj().T @ x) / (one.conj().T @ one) * one
    return x


def make_unit_modulus_and_sum_zero(L: int) -> torch.Tensor:
    """
    Build a length-L unit-modulus vector whose entries sum to zero (⊥ 1).
    Construction: choose L-1 random phases, set the last phase so that the total sum is zero.
    """
    phases = 2 * math.pi * torch.rand(L - 1)
    v = torch.exp(1j * phases.to(torch.complex64))
    s = v.sum()
    # Choose last entry on unit circle so that total sum becomes zero.
    last = torch.exp(1j * (torch.angle(-s) + torch.tensor(0.0)))
    x = torch.cat([v, last.unsqueeze(0)])
    # Numerically enforce exact orthogonality
    x = x - (torch.sum(x) / L) * torch.ones(L, dtype=torch.complex64)
    # Renormalize to unit-modulus per entry
    x = torch.exp(1j * torch.angle(x))
    # Exactly zero sum (phase tweak of last element)
    s = x[:-1].sum()
    x[-1] = torch.exp(1j * torch.angle(-s))
    return x


def circ_shift(v: torch.Tensor, k: int) -> torch.Tensor:
    """Circularly shift vector v by k positions to the right."""
    k = k % v.numel()
    if k == 0:
        return v.clone()
    return torch.cat([v[-k:], v[:-k]])


def make_activation_mask(Ks: int, Qs: int) -> torch.Tensor:
    """Binary mask of length Ks with first Qs samples = 1 (tag activation window)."""
    m = torch.zeros(Ks, dtype=torch.complex64)
    m[:Qs] = torch.tensor(1.0 + 1j * 0, dtype=torch.complex64)
    return m


def pinv(A: torch.Tensor) -> torch.Tensor:
    """Stable Moore-Penrose pseudo-inverse for complex matrices."""
    return torch.linalg.pinv(A)


def svd_rank1(M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Rank-1 approximation of complex matrix M via SVD."""
    # torch.linalg.svd returns U, S, Vh with Vh = V^H
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    s1 = S[0].real
    u1 = U[:, 0]
    v1 = Vh.conj().T[:, 0]
    return u1, v1, float(s1)


def kronsum_design_mats(H: torch.Tensor, qrt: int, qtr: int,
                        gamma_RT: Optional[torch.Tensor] = None,
                        gamma_TR: Optional[torch.Tensor] = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Build linear design matrices for alternating LS updates in the Kronecker model:
        vec(gamma_RTR) = kron(gamma_RT, gamma_TR)  (dimension qrt * qtr)
    When updating gamma_RT (fix gamma_TR):
        y ≈ H @ (I_qrt ⊗ gamma_TR) @ gamma_RT
    When updating gamma_TR (fix gamma_RT):
        y ≈ H @ (gamma_RT ⊗ I_qtr) @ gamma_TR
    """
    I_qrt = torch.eye(qrt, dtype=torch.complex64)
    I_qtr = torch.eye(qtr, dtype=torch.complex64)
    if gamma_TR is not None:
        A_RT = H @ torch.kron(I_qrt, gamma_TR.reshape(-1, 1))
        A_RT = A_RT.squeeze(-1)  # (Ks x (qrt))
    else:
        A_RT = None

    if gamma_RT is not None:
        A_TR = H @ torch.kron(gamma_RT.reshape(-1, 1), I_qtr)
        A_TR = A_TR.squeeze(-2)  # (Ks x (qtr))
    else:
        A_TR = None
    return A_RT, A_TR


# -------------------------- Gold code generation ----------------------------- #

def lfsr_m_sequence(poly_taps: List[int], state: List[int], length: int) -> torch.Tensor:
    """
    Generate a binary m-sequence using an LFSR with given feedback taps.
    poly_taps: list of tap positions (1-based, including the highest degree and 0), e.g., [5,2,0] for x^5 + x^2 + 1
    state: initial register state as list of 0/1, length = max(poly_taps)
    length: sequence length to output (should be 2^m - 1 for maximal-length).
    Returns: tensor of 0/1 of given length.
    """
    m = max(poly_taps)
    reg = state[:m]
    seq = []
    for _ in range(length):
        out = reg[-1]
        seq.append(out)
        # feedback bit = XOR of tapped positions (excluding degree-0 term because it's the implicit +1)
        fb = 0
        for t in poly_taps[:-1]:  # skip the 0 tap (constant)
            fb ^= reg[m - t]
        # shift right and insert feedback at pos 0
        reg = [fb] + reg[:-1]
    return torch.tensor(seq, dtype=torch.int64)


def gold_sequences_N31(num: int = 16, seed: int = 123) -> List[torch.Tensor]:
    """
    Generate 'num' Gold sequences of length 31 (unit-modulus complex, ±1).
    Standard preferred pair for m=5 (N=31):
      mseq A: x^5 + x^2 + 1   -> taps [5,2,0]
      mseq B: x^5 + x^4 + x^2 + x + 1 -> taps [5,4,2,1,0]
    We produce the preferred pair and XOR B with all 31 cyclic shifts of A to form a Gold family.
    Then map {0,1} -> {+1,-1} (BPSK), then to complex unit-modulus.
    """
    rng = random.Random(seed)
    m = 5
    N = 2 ** m - 1  # 31
    # Random non-zero initial states for each LFSR
    stA = [rng.randint(0, 1) for _ in range(m)]
    if sum(stA) == 0:
        stA[0] = 1
    stB = [rng.randint(0, 1) for _ in range(m)]
    if sum(stB) == 0:
        stB[0] = 1

    A = lfsr_m_sequence([5, 2, 0], stA, N)  # 0/1
    B = lfsr_m_sequence([5, 4, 2, 1, 0], stB, N)

    family = []
    for shift in range(N):  # XOR with cyclic shifts of A
        A_shift = torch.roll(A, shifts=shift)
        g = (B ^ A_shift).to(torch.int64)  # 0/1
        # Map 0->+1, 1->-1
        bpsk = 1 - 2 * g
        seq = bpsk.to(torch.complex64)
        family.append(seq)

    # Select 'num' sequences spaced across the 31-member family
    idxs = torch.linspace(0, N - 1, steps=num).round().to(torch.int64).tolist()
    selected = [family[i] for i in idxs]
    return selected


# -------------------------- Environment definition --------------------------- #

@dataclass
class Env:
    # Global sizes
    L: int
    Ns: int
    Ks: int
    Qs: int
    Q_xi: int
    Q_RR: int
    Q_RT: int
    Q_TR: int
    Ma: int
    U: int

    # Penalties and thresholds
    snr_db: float
    nu: float
    nu1: float
    nux: float
    eps_omp_RR: float
    eps_omp_RTR: float
    eps_als: float
    N_als: int
    topK: int

    # Fixed waveforms and codebook
    xi_set: List[torch.Tensor]       # list of length Ma, each length Q_xi (unit-modulus)
    codebook: List[torch.Tensor]     # list of length U, each length L, unit-modulus, sum=0

    # Dictionaries for every m
    Xi_list: List[torch.Tensor]      # list of (Ks x Q_RR)
    H_list: List[torch.Tensor]       # list of (Ks x (Q_RT*Q_TR))

    # Fixed sparse channels (shared across MC)
    gamma_RR_true: torch.Tensor      # length Q_RR
    gamma_RT_true: torch.Tensor      # length Q_RT
    gamma_TR_true: torch.Tensor      # length Q_TR

    # Activation mask (Ks,)
    act_mask: torch.Tensor


def build_Xi_from_waveform(xi: torch.Tensor, Ks: int, Q_RR: int) -> torch.Tensor:
    """
    Build Xi (Ks x Q_RR): columns are cyclic shifts of xi embedded/truncated to Ks.
    We use repetition/padding of the Q_xi-length waveform to fill Ks, then take shifts.
    """
    xi = complex_tensor(xi)
    Q_xi = xi.numel()
    # Tile waveform to cover Ks, then truncate
    reps = math.ceil(Ks / Q_xi)
    base = xi.repeat(reps)[:Ks]
    cols = []
    for q in range(Q_RR):
        cols.append(circ_shift(base, q))
    Xi = torch.stack(cols, dim=1)  # (Ks, Q_RR)
    # Column-energy normalization (optional but keeps LS well-conditioned)
    Xi = Xi / torch.sqrt(torch.sum(torch.abs(Xi) ** 2, dim=0, keepdim=True) + 1e-8)
    return Xi


def build_H_from_waveform(xi: torch.Tensor, Ks: int, Q_RT: int, Q_TR: int, act_mask: torch.Tensor) -> torch.Tensor:
    """
    Build H (Ks x (Q_RT * Q_TR)) for the backscattered path dictionary.
    A simple and effective construction consistent with the paper logic:
      - Start from a Ks-long cyclic repetition of the waveform xi.
      - Apply an "activation" gating (first Qs samples are 1, others 0) to emulate the tag backscatter window.
      - For each (p in [0,Q_RT-1], q in [0,Q_TR-1]) create an atom as a cyclic shift by (p + q),
        then multiply by act_mask.
    Finally, normalize columns.
    """
    xi = complex_tensor(xi) 
    Q_xi = xi.numel() ## length of waveform
    reps = math.ceil(Ks / Q_xi) ## number of repetitions to cover Ks
    base = xi.repeat(reps)[:Ks] ## base vector of length Ks

    atoms = []
    for p in range(Q_RT):
        for q in range(Q_TR):
            shift = p + q
            a = circ_shift(base, shift) * act_mask
            atoms.append(a)
    H = torch.stack(atoms, dim=1)  # (Ks, Q_RT*Q_TR)
    H = H / torch.sqrt(torch.sum(torch.abs(H) ** 2, dim=0, keepdim=True) + 1e-8)
    return H


def draw_sparse_channel(Q: int, k_nonzeros: int, power: float) -> torch.Tensor:
    """
    Draw a length-Q sparse complex channel with k_nonzeros non-zero taps at random positions.
    Non-zero taps ~ CN(0, power). Support is chosen uniformly without replacement.
    """
    h = torch.zeros(Q, dtype=torch.complex64)
    idx = torch.randperm(Q)[:k_nonzeros]
    h[idx] = (torch.randn(k_nonzeros) + 1j * torch.randn(k_nonzeros)) * math.sqrt(power / 2.0)
    return h


def generate_environment(
    L: int = 10,
    Ns: int = 1,
    Ks: int = 100,
    Qs: int = 20,
    Q_xi: int = 31,
    Q_RR: int = 10,
    Q_RT: int = 10,
    Q_TR: int = 3,
    Ma: int = 16,
    U: int = 16,
    snr_db: float = 10.0,
    true_sparsity: Tuple[int, int, int] = (5, 3, 2),
    P_RR: float = 1.0,
    P_RT: float = 1.0,
    P_TR: float = 0.2,
    eps_als: float = 1e-4,
    N_als: int = 10,
    topK: int = 5,
    seed: int = 2025
) -> Env:
    """
    Build the shared environment: waveforms, codebook, fixed sparse channels, dictionaries, and penalties.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    # 1) Waveform set: 16 Gold sequences (length 31), map to complex unit-modulus
    gold = gold_sequences_N31(num=Ma, seed=seed)
    xi_set = [complex_tensor(g) for g in gold]  # already ±1

    # 2) Tag codebook: U unit-modulus, length L, sum zero (⊥ 1), ||x||^2 = L
    codebook = []
    for u in range(U):
        x = make_unit_modulus_and_sum_zero(L) ## construct orthogonal codewords
        # Normalize total power to L (each element unit modulus already => ||x||^2 = L)
        codebook.append(x)

    # 3) Activation mask (Ks,): tag activation window
    act_mask = make_activation_mask(Ks, Qs)

    # 4) Dictionaries Xi_m and H_m for every waveform m
    Xi_list, H_list = [], []
    for m in range(Ma):
        Xi = build_Xi_from_waveform(xi_set[m], Ks, Q_RR)
        H = build_H_from_waveform(xi_set[m], Ks, Q_RT, Q_TR, act_mask)
        Xi_list.append(Xi)
        H_list.append(H)

    # 5) Fixed sparse channels (generate only once and unknown to decoders)
    s_RR, s_RT, s_TR = true_sparsity
    gamma_RR_true = draw_sparse_channel(Q_RR, s_RR, P_RR)
    gamma_RT_true = draw_sparse_channel(Q_RT, s_RT, P_RT)
    gamma_TR_true = draw_sparse_channel(Q_TR, s_TR, P_TR)

    # 6) Penalties (GIC/BIC style): nu = sigma_w^2 * log(100)
    # We don't know sigma_w^2 yet (it depends on (m*, x*)), but we precompute functional form in decoders.
    # For thresholds we follow your recipe; we will instantiate eps_omp after sigma_w^2 is known per trial.
    # Here we set placeholders; proper values are provided at runtime based on sigma_w^2.
    nu = 0.0
    nu1 = 0.0
    nux = 0.0
    eps_omp_RR = 0.0
    eps_omp_RTR = 0.0

    return Env(L, Ns, Ks, Qs, Q_xi, Q_RR, Q_RT, Q_TR, Ma, U,
               snr_db, nu, nu1, nux, eps_omp_RR, eps_omp_RTR, eps_als, N_als, topK,
               xi_set, codebook, Xi_list, H_list,
               gamma_RR_true, gamma_RT_true, gamma_TR_true,
               act_mask)


# ------------------------- Synthesis & SNR utilities ------------------------- #

def synthesize_Y(env: Env, m_idx: int, x_vec: torch.Tensor) -> Tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]:
    """
    Build Y for a given (m_idx, x_vec), using the *fixed* true channels in env and target SNR.
    Returns:
      Y  : (L x Ks)
      sigma_w2 : the noise variance derived from target SNR
      alpha_m  : (Ks,) backscatter response for waveform m
      i_m      : (Ks,) direct path response for waveform m
    """
    m = m_idx
    Xi = env.Xi_list[m]            # (Ks x Q_RR)
    H = env.H_list[m]              # (Ks x (Q_RT*Q_TR))

    # True channels (fixed across MC)
    gamma_RR = env.gamma_RR_true   # (Q_RR,)
    gamma_RT = env.gamma_RT_true   # (Q_RT,)
    gamma_TR = env.gamma_TR_true   # (Q_TR,)

    # Compose gamma_RTR = gamma_RT ⊗ gamma_TR (vectorized)
    Gamma_RTR = torch.kron(gamma_RT, gamma_TR)  # (Q_RT * Q_TR,)

    # Direct and backscattered waveforms (Ks,)
    i_m = Xi @ gamma_RR                    # (Ks,)
    alpha_m = H @ Gamma_RTR                # (Ks,)

    # Build clean signal: X = x α^H + 1 i^H
    x = complex_tensor(x_vec).reshape(-1, 1)         # (L,1)
    alphaH = alpha_m.conj().reshape(1, -1)           # (1,Ks)
    one = torch.ones(env.L, 1, dtype=torch.complex64)

    Y_clean = x @ alphaH + one @ i_m.conj().reshape(1, -1)  # (L x Ks)

    # Derive sigma_w^2 from the SNR definition
    num = env.L * (x.conj().T @ x).real.item() * (alpha_m.conj().T @ alpha_m).real.item() \
          + env.L * (i_m.conj().T @ i_m).real.item()
    den = env.L * env.Ks * (10.0 ** (env.snr_db / 10.0))
    sigma_w2 = num / den if den > 0 else 0.0

    # Add noise
    noise = math.sqrt(sigma_w2 / 2.0) * (torch.randn_like(Y_clean.real) + 1j * torch.randn_like(Y_clean.real))
    Y = Y_clean + noise

    return Y, sigma_w2, alpha_m, i_m


# ---------------------------- OMP (single-round) ----------------------------- #

def single_round_omp(
    y: torch.Tensor,
    D: torch.Tensor,
    nu_over_L: float,
    eps_corr: float,
    max_atoms: Optional[int] = None
) -> Tuple[torch.Tensor, List[int], float, int]:
    """
    Single-round greedy model-order selection on dictionary D:
      - Compute correlations |D^H r|, sort atoms descending.
      - Iteratively add atoms following that order, re-solve LS each time, and compute cost:
            J_t = ||y - D_S gamma_S||^2 + (nu/L) * |S|
        with the two-sided "J" condition & the correlation threshold as stopping rules.
      - Return the *best prefix* (support and coefficients) with the lowest J_t encountered.
    Inputs:
      y          : (Ks,)
      D          : (Ks x M)
      nu_over_L  : scalar, equals nu/L for the subproblem
      eps_corr   : stopping threshold on the maximum remaining correlation
      max_atoms  : optional hard cap on number of atoms (e.g., 8 or 12)
    Outputs:
      gamma_hat  : (M,) sparse vector with non-zeros on the selected support
      support    : selected atom indices
      best_cost  : minimum J_t achieved
      best_n     : |support| at the best point
    """
    y = y.reshape(-1)
    Ks, M = D.shape
    if max_atoms is None:
        max_atoms = M

    # Precompute raw correlations
    corr = (D.conj().T @ y)  # (M,)
    order = torch.argsort(torch.abs(corr), descending=True).tolist()

    support = []
    best_cost = math.inf
    best_gamma = torch.zeros(M, dtype=torch.complex64)
    best_n = 0

    prev_J = math.inf
    r = y.clone()

    for t, idx in enumerate(order):
        if t >= max_atoms:
            break

        # Remaining max correlation check BEFORE adding this atom
        if torch.max(torch.abs(D.conj().T @ r)).real.item() < eps_corr:
            break

        support.append(idx)
        S = torch.stack([D[:, j] for j in support], dim=1)  # (Ks x t+1)
        # LS on current support
        gamma_S = pinv(S) @ y  # ((t+1),)
        r = y - S @ gamma_S
        J_t = (r.conj().T @ r).real.item() + nu_over_L * (t + 1)

        # Two-sided J condition: if cost increases, we revert to best so far and stop
        if J_t > prev_J:
            # stop and revert support (exclude the last one)
            support.pop()
            break

        prev_J = J_t

        if J_t < best_cost:
            best_cost = J_t
            best_n = t + 1
            best_gamma = torch.zeros(M, dtype=torch.complex64)
            best_gamma[torch.tensor(support)] = gamma_S

    return best_gamma, support, best_cost, best_n


# ------------------------ Kronecker ALS (RTR channel) ------------------------ #

def kronecker_als(
    yx: torch.Tensor,
    H: torch.Tensor,
    qrt: int,
    qtr: int,
    eps_als: float = 1e-4,
    N_als: int = 10
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Estimate gamma_RTR = gamma_RT ⊗ gamma_TR via SVD initialization + ALS.
    Inputs:
      yx     : (Ks,) target vector, i.e., (1/L)*Y^H x
      H      : (Ks x (qrt*qtr)) dictionary
      qrt    : Q_RT
      qtr    : Q_TR
    Outputs:
      gamma_RTR_hat : (qrt*qtr,)
      gamma_RT_hat  : (qrt,)
      gamma_TR_hat  : (qtr,)
      mse           : final MSE ||yx - H @ gamma_RTR_hat||^2
    """
    # Unconstrained LS, then rank-1 SVD init
    gamma_ls = pinv(H) @ yx  # (qrt*qtr,)
    G = gamma_ls.reshape(qrt, qtr)
    u1, v1, s1 = svd_rank1(G)
    gamma_RT = math.sqrt(s1) * u1
    gamma_TR = math.sqrt(s1) * v1

    prev_mse = math.inf
    for it in range(N_als):
        # Update gamma_RT with gamma_TR fixed
        A_RT = H @ torch.kron(torch.eye(qrt, dtype=torch.complex64), gamma_TR.reshape(-1, 1))
        A_RT = A_RT.squeeze(-1)  # (Ks x qrt)
        gamma_RT = pinv(A_RT) @ yx

        # Update gamma_TR with gamma_RT fixed
        A_TR = H @ torch.kron(gamma_RT.reshape(-1, 1), torch.eye(qtr, dtype=torch.complex64))
        A_TR = A_TR.squeeze(-2)  # (Ks x qtr)
        gamma_TR = pinv(A_TR) @ yx

        gamma_vec = torch.kron(gamma_RT, gamma_TR)
        resid = yx - H @ gamma_vec
        mse = (resid.conj().T @ resid).real.item()

        # ALS stop: absolute MSE change < eps_als
        if abs(prev_mse - mse) < eps_als:
            prev_mse = mse
            break
        prev_mse = mse

    gamma_vec = torch.kron(gamma_RT, gamma_TR)
    resid = yx - H @ gamma_vec
    mse = (resid.conj().T @ resid).real.item()
    return gamma_vec, gamma_RT, gamma_TR, mse


# ---------------------------- Decoding primitives ---------------------------- #

def decode_tag_energy_max(Y: torch.Tensor, codebook: List[torch.Tensor]) -> int:
    """
    Approach used in eq.(42): for each u in codebook, score ||u^H Y||^2 / ||u||^2.
    Here ||u||^2 = L because each entry is unit-modulus. We just maximize ||u^H Y||^2.
    Returns the argmax index in [0, U-1].
    """
    best_idx, best_val = -1, -1.0
    for i, u in enumerate(codebook):
        score = torch.sum(torch.abs(u.conj().reshape(1, -1) @ Y) ** 2).real.item()
        if score > best_val:
            best_val, best_idx = score, i
    return best_idx


def score_ignore_sparsity(yx: torch.Tensor, H: torch.Tensor) -> float:
    """
    Score used for Top-K selection: s(x;m) = || (I - H H^+ ) yx ||^2.
    """
    P = H @ pinv(H)
    resid = (torch.eye(P.shape[0], dtype=torch.complex64) - P) @ yx
    return (resid.conj().T @ resid).real.item()


# ------------------------ Suboptimal decoding schemes ------------------------ #

def approach_I(env: Env, Y: torch.Tensor, sigma_w2: float) -> Tuple[int, int]:
    """
    Suboptimal Approach I:
      1) Decode tag x by (42).
      2) Decode radar m using only direct channel i and Xi_m with LS projection energy (45).
    Returns: (m_hat, x_hat_idx)
    """
    # 1) Tag by energy
    x_idx = decode_tag_energy_max(Y, env.codebook)

    # 2) Direct path detection across m
    y1 = (Y.conj().T @ torch.ones(env.L, 1, dtype=torch.complex64) / env.L).reshape(-1)  # (Ks,)
    best_m, best_score = -1, -1.0
    for m in range(env.Ma):
        Xi = env.Xi_list[m]  # (Ks x Q_RR)
        P = Xi @ pinv(Xi)
        proj = P @ y1
        score = torch.sum(torch.abs(proj) ** 2).real.item()
        if score > best_score:
            best_score, best_m = score, m
    return best_m, x_idx


def approach_II(env: Env, Y: torch.Tensor, sigma_w2: float) -> Tuple[int, int]:
    """
    Suboptimal Approach II:
      1) Decode tag x by (42).
      2) For each m, unconstrained LS estimates for gamma_RTR and gamma_RR, then sum the two residuals per (47).
    Returns: (m_hat, x_hat_idx)
    """
    x_idx = decode_tag_energy_max(Y, env.codebook)
    x = env.codebook[x_idx]

    yx = (Y.conj().T @ x.reshape(-1, 1) / env.L).reshape(-1)   # (Ks,)
    y1 = (Y.conj().T @ torch.ones(env.L, 1, dtype=torch.complex64) / env.L).reshape(-1)

    best_m, best_cost = -1, math.inf
    for m in range(env.Ma):
        H = env.H_list[m]
        Xi = env.Xi_list[m]

        # LS estimates (no sparsity)
        gamma_RTR = pinv(H) @ yx
        gamma_RR = pinv(Xi) @ y1

        cost1 = torch.norm(yx - H @ gamma_RTR) ** 2
        cost2 = torch.norm(y1 - Xi @ gamma_RR) ** 2
        cost = (cost1 + cost2).real.item()
        if cost < best_cost:
            best_cost, best_m = cost, m
    return best_m, x_idx


def approach_III(env: Env, Y: torch.Tensor, sigma_w2: float) -> Tuple[int, int]:
    """
    Suboptimal Approach III:
      - nu = 0. Use Kronecker structure for gamma_RTR via ALS and Xi_m projection for gamma_RR.
      - Jointly pick (m, x) by minimizing the combined residuals (53) with nu=0.
    Returns: (m_hat, x_hat_idx)
    """
    y1 = (Y.conj().T @ torch.ones(env.L, 1, dtype=torch.complex64) / env.L).reshape(-1)

    best_pair = (-1, -1)
    best_cost = math.inf

    for m in range(env.Ma):
        Xi = env.Xi_list[m] ## (Ks x Q_RR)
        P_Xi = Xi @ pinv(Xi) # Projection onto column space of Xi: P_Xi = Xi (Xi^H Xi)^-1 Xi^H
        term_rr = torch.norm((torch.eye(P_Xi.shape[0], dtype=torch.complex64) - P_Xi) @ y1) ** 2  # ||(I-P) y1||^2

        for xi, x in enumerate(env.codebook):
            yx = (Y.conj().T @ x.reshape(-1, 1) / env.L).reshape(-1)
            H = env.H_list[m]
            _, _, _, mse = kronecker_als(yx, H, env.Q_RT, env.Q_TR, env.eps_als, env.N_als)
            cost = (env.L * mse + env.L * term_rr.real.item())
            if cost < best_cost:
                best_cost = cost
                best_pair = (m, xi)

    return best_pair


# -------------------------- Simplified joint decoding scheme ------------------------ #

def simplified_joint_decoder(env: Env, Y: torch.Tensor, sigma_w2: float) -> Tuple[int, int]:
    """
    Simplified joint decoding (bi-level Top-K):
      - For each m:
          1) First-layer Top-K selection over x by s(x;m) = ||(I - H H^+) yx||^2.
          2) For each of the K candidates, estimate \\gamma_RTR with Kronecker ALS + model-order penalty (nu/L)*n2.
          3) Separately estimate \\gamma_RR with single-round OMP (Xi_m) + (nu/L)*n1.
          4) Combine per-(76) and choose the best m, then choose x = x_{k*(m)}.
    """
    # Penalties & thresholds from sigma_w2
    nu = sigma_w2 * math.log(100.0)
    nu_over_L = nu / env.L
    eps_RR = math.sqrt(2.0 * sigma_w2 * math.log(env.Q_RR))
    eps_RTR = math.sqrt(2.0 * sigma_w2 * math.log(env.Q_RT * env.Q_TR))

    y1 = (Y.conj().T @ torch.ones(env.L, 1, dtype=torch.complex64) / env.L).reshape(-1)

    best_total = math.inf
    best_m, best_x_idx = -1, -1

    # Precompute first-layer scores for each (m, x)
    for m in range(env.Ma):
        H = env.H_list[m]
        # RR estimation (single-round OMP with model-order selection)
        gamma_RR_hat, supp_RR, cost_RR, n1 = single_round_omp(
            y=y1, D=env.Xi_list[m], nu_over_L=nu_over_L, eps_corr=eps_RR, max_atoms=8
        )

        # First-layer Top-K over codebook
        scores = []
        yx_vectors = []
        for xi, x in enumerate(env.codebook):
            yx = (Y.conj().T @ x.reshape(-1, 1) / env.L).reshape(-1)
            yx_vectors.append(yx)
            score = score_ignore_sparsity(yx, H)
            scores.append((score, xi))
        scores.sort(key=lambda t: t[0])  # smaller is better
        topK = scores[:env.topK]

        # Second layer: for each candidate, run ALS and evaluate full cost (76)
        best_k_cost = math.inf
        best_k_idx = -1
        for _, xi in topK:
            yx = yx_vectors[xi]
            gamma_RTR_hat, gamma_RT_hat, gamma_TR_hat, mse = kronecker_als(
                yx, H, env.Q_RT, env.Q_TR, env.eps_als, env.N_als
            )
            # Sparsity order estimate: count non-trivial taps by magnitude threshold
            # (simple proxy; you can swap with support from a dedicated Kronecker-OMP if desired)
            n2 = int((torch.abs(gamma_RT_hat) > 1e-6).sum() * (torch.abs(gamma_TR_hat) > 1e-6).sum())

            total = env.L * mse + env.L * cost_RR + nu * (n1 + n2)
            if total < best_k_cost:
                best_k_cost = total
                best_k_idx = xi

        if best_k_cost < best_total:
            best_total = best_k_cost
            best_m = m
            best_x_idx = best_k_idx

    return best_m, best_x_idx


# ---------------------------- Original joint (full) -------------------------- #

def original_joint_decoder(env: Env, Y: torch.Tensor, sigma_w2: float) -> Tuple[int, int]:
    """
    Original joint decoding (no pre-selection):
      - For every pair (m, x) in Ma x U:
         * Estimate \\gamma_RR with single-round OMP over Xi_m (model-order selection)
         * Estimate \\gamma_RTR with Kronecker ALS over H_m (model-order selection via nu)
         * Evaluate complete cost in (40)/(76), choose the global minimum.
    Returns: (m_hat, x_hat_idx)
    """
    nu = sigma_w2 * math.log(100.0)
    nu_over_L = nu / env.L
    eps_RR = math.sqrt(2.0 * sigma_w2 * math.log(env.Q_RR))
    eps_RTR = math.sqrt(2.0 * sigma_w2 * math.log(env.Q_RT * env.Q_TR))

    y1 = (Y.conj().T @ torch.ones(env.L, 1, dtype=torch.complex64) / env.L).reshape(-1)

    best = (math.inf, -1, -1)
    for m in range(env.Ma):
        Xi = env.Xi_list[m]
        H = env.H_list[m]

        # Direct path with OMP (model-order selection)
        gamma_RR_hat, supp_RR, cost_RR, n1 = single_round_omp(
            y=y1, D=Xi, nu_over_L=nu_over_L, eps_corr=eps_RR, max_atoms=8
        )

        for xi, x in enumerate(env.codebook):
            yx = (Y.conj().T @ x.reshape(-1, 1) / env.L).reshape(-1)
            # Kronecker ALS
            gamma_RTR_hat, gamma_RT_hat, gamma_TR_hat, mse = kronecker_als(
                yx, H, env.Q_RT, env.Q_TR, env.eps_als, env.N_als
            )
            # Simple model order proxy
            n2 = int((torch.abs(gamma_RT_hat) > 1e-6).sum() * (torch.abs(gamma_TR_hat) > 1e-6).sum())

            total = env.L * mse + env.L * cost_RR + nu * (n1 + n2)
            if total < best[0]:
                best = (total, m, xi)

    _, m_hat, x_hat = best
    return m_hat, x_hat


# ----------------------------- Monte Carlo driver ---------------------------- #

def run_monte_carlo(
    env: Env,
    N_trials: int = 200,
    seed: int = 1
) -> None:
    """
    Run MC experiments:
      - Channels are FIXED across all trials (as requested).
      - Each trial draws a random (m*, x*) pair, synthesizes Y at target SNR,
        runs all decoders, and accumulates error rates.
      - Finally prints radar and tag detection error frequencies for every method.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    errs = {
        "approach_I": {"radar": 0, "tag": 0},
        "approach_II": {"radar": 0, "tag": 0},
        "approach_III": {"radar": 0, "tag": 0},
        "simplified_joint": {"radar": 0, "tag": 0},
        "original_joint": {"radar": 0, "tag": 0},
    }

    for t in range(N_trials):
        print(f"Trial {t+1}/{N_trials}", end='\r')
        # Draw a random true pair (m*, x*)
        m_true = random.randrange(env.Ma)
        x_true_idx = random.randrange(env.U)
        x_true = env.codebook[x_true_idx]

        # Synthesize observation
        Y, sigma_w2, _, _ = synthesize_Y(env, m_true, x_true)

        # --- Run all decoders ---
        m1, x1 = approach_I(env, Y, sigma_w2)
        m2, x2 = approach_II(env, Y, sigma_w2)
        m3, x3 = approach_III(env, Y, sigma_w2)
        ms, xs = simplified_joint_decoder(env, Y, sigma_w2)
        mo, xo = original_joint_decoder(env, Y, sigma_w2)

        # Accumulate errors
        for name, (mhat, xhat) in {
            "approach_I": (m1, x1),
            "approach_II": (m2, x2),
            "approach_III": (m3, x3),
            "simplified_joint": (ms, xs),
            "original_joint": (mo, xo),
        }.items():
            if mhat != m_true:
                errs[name]["radar"] += 1
            if xhat != x_true_idx:
                errs[name]["tag"] += 1

    # Print error frequencies
    print(f"\n=== Monte Carlo results (N={N_trials}, SNR={env.snr_db:.1f} dB) ===")
    for name in ["approach_I", "approach_II", "approach_III", "simplified_joint", "original_joint"]:
        er_r = errs[name]["radar"] / N_trials
        er_t = errs[name]["tag"] / N_trials
        print(f"{name:>18s} | radar err: {er_r:6.3f} | tag err: {er_t:6.3f}")


# ------------------------------- Main (demo) --------------------------------- #

if __name__ == "__main__":
    # Build the environment once; channels are fixed across MC as required.
    env = generate_environment(
        L=10, Ns=1, Ks=100, Qs=20,
        Q_xi=31, Q_RR=10, Q_RT=10, Q_TR=3,
        Ma=16, U=16, snr_db=1.0,
        true_sparsity=(5, 3, 1),
        P_RR=1.0, P_RT=1.0, P_TR=0.2,
        eps_als=1e-4, N_als=10, topK=10,
        seed=2025
    )

    # Run a short MC to verify the pipeline; increase N_trials for stable stats.
    run_monte_carlo(env, N_trials=50, seed=7)

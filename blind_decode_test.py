# blind_decode_test.py
# -*- coding: utf-8 -*-
"""
Comprehensive testing of system components:
1. Waveform auto/cross-correlation properties
2. Tag codebook orthogonality and energy constraints
3. Projector condition numbers with different regularization parameters
4. Performance comparison with different lambda values (0.001, 0.01, 0.1, 1.0)
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
# TEST 1: Waveform Properties
# -----------------------------
def test_waveform_properties(C: Tensor, Np: int, Ma: int):
    """Test auto-correlation and cross-correlation of waveforms."""
    print("\n" + "="*70)
    print("TEST 1: WAVEFORM AUTO/CROSS-CORRELATION PROPERTIES")
    print("="*70)
    
    C_cpu = C.cpu()
    
    # Auto-correlation (should be Np for all)
    auto_corrs = []
    for i in range(Ma):
        auto_corr = abs((C_cpu[i].conj() @ C_cpu[i]).item())
        auto_corrs.append(auto_corr)
    
    print(f"\n📊 Auto-correlation (should all equal Np={Np}):")
    print(f"  Min: {min(auto_corrs):.6f}, Max: {max(auto_corrs):.6f}")
    print(f"  Mean: {np.mean(auto_corrs):.6f}, Std: {np.std(auto_corrs):.10f}")
    print(f"  Max deviation from Np: {abs(np.array(auto_corrs) - Np).max():.2e}")
    
    # Cross-correlation (should be small)
    cross_corrs = []
    for i in range(Ma):
        for j in range(i+1, Ma):
            cross_corr = abs((C_cpu[i].conj() @ C_cpu[j]).item()) / Np
            cross_corrs.append(cross_corr)
    
    print(f"\n📊 Cross-correlation (normalized by Np, should be small):")
    print(f"  Min: {min(cross_corrs):.6f}, Max: {max(cross_corrs):.6f}")
    print(f"  Mean: {np.mean(cross_corrs):.6f}, Std: {np.std(cross_corrs):.6f}")
    
    # Peak-to-Average Ratio (PAR) in dB
    if len(cross_corrs) > 0:
        par_db = 20 * np.log10(1.0 / np.mean(cross_corrs))
        print(f"  Peak-to-Average Ratio (PAR): {par_db:.2f} dB")
    
    print(f"\n✓ Waveform test completed")


# -----------------------------
# TEST 2: Tag Codebook Properties
# -----------------------------
def test_tag_codebook_properties(X: Tensor, L: int, U: int):
    """Test energy and orthogonality constraints of tag codebook."""
    print("\n" + "="*70)
    print("TEST 2: TAG CODEBOOK PROPERTIES")
    print("="*70)
    
    X_cpu = X.cpu()
    ones = torch.ones(L, dtype=complex_dtype)
    
    # Energy: ||x||^2
    norms_squared = (X_cpu.conj() * X_cpu).real.sum(dim=0).numpy()
    
    print(f"\n📊 Energy ||x||^2 (should all equal L={L}):")
    print(f"  Min: {norms_squared.min():.10f}, Max: {norms_squared.max():.10f}")
    print(f"  Mean: {norms_squared.mean():.10f}, Std: {norms_squared.std():.10f}")
    print(f"  Max deviation from L: {abs(norms_squared - L).max():.2e}")
    
    # Orthogonality to 1_L
    inner_prods = (ones.conj() @ X_cpu).abs().numpy()
    
    print(f"\n📊 Orthogonality |1^H x| (should all be ~0):")
    print(f"  Min: {inner_prods.min():.2e}, Max: {inner_prods.max():.2e}")
    print(f"  Mean: {inner_prods.mean():.2e}, Std: {inner_prods.std():.2e}")
    
    # Check which codewords violate constraints
    violations_energy = np.where(abs(norms_squared - L) > 1e-5)[0]
    violations_ortho = np.where(inner_prods > 1e-5)[0]
    
    if len(violations_energy) > 0:
        print(f"  ⚠️  Codewords with ||x||^2 violation: {violations_energy.tolist()}")
    else:
        print(f"  ✓ All codewords satisfy energy constraint (||x||^2 = L)")
    
    if len(violations_ortho) > 0:
        print(f"  ⚠️  Codewords with |1^H x| > 1e-5: {violations_ortho.tolist()}")
    else:
        print(f"  ✓ All codewords satisfy orthogonality constraint (x ⊥ 1_L)")


# -----------------------------
# TEST 3: Projector Condition Numbers
# -----------------------------
def test_projector_condition_numbers(Xi_list: List[Tensor], K: int, Q: int, device: torch.device):
    """Test condition numbers with different regularization parameters."""
    print("\n" + "="*70)
    print("TEST 3: PROJECTOR CONDITION NUMBERS WITH REGULARIZATION")
    print("="*70)
    
    # Test different lambda multipliers
    lambda_multipliers = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    
    # Test first 3 waveforms
    for c in range(min(3, len(Xi_list))):
        Xi = Xi_list[c]
        
        # Compute spectral norm of Xi Xi^H
        A = Xi @ Xi.conj().T
        spectral_A = torch.linalg.norm(A, ord=2).item()
        
        # Original projector (no regularization)
        pinv = torch.linalg.pinv(Xi)
        P_orig = torch.eye(K, dtype=complex_dtype, device=device) - Xi @ pinv
        cond_orig = torch.linalg.cond(P_orig).item()
        
        print(f"\n📊 Waveform c={c}: ||Xi Xi^H||_2 = {spectral_A:.3f}")
        print(f"{'λ_mult':<10} {'λ':<12} {'cond(I-P_reg)':<18} {'Improvement':<15}")
        print("-" * 65)
        
        results = []
        for mult in lambda_multipliers:
            lam = mult * spectral_A
            
            if mult == 0.0:
                cond = cond_orig
                improvement = 1.0
            else:
                # Regularized projector: P_reg = Xi (Xi^H Xi + lambda I)^(-1) Xi^H
                XtX = Xi.conj().T @ Xi
                I_ch = torch.eye(Q + 1, dtype=complex_dtype, device=device)
                P_reg = Xi @ torch.linalg.inv(XtX + lam * I_ch) @ Xi.conj().T
                
                # Compute I - P_reg (projects to nullspace)
                I_K = torch.eye(K, dtype=complex_dtype, device=device)
                P = I_K - P_reg
                
                try:
                    cond = torch.linalg.cond(P).item()
                except:
                    cond = float('inf')
                
                improvement = cond_orig / cond if cond > 0 else float('inf')
            
            results.append((mult, lam, cond, improvement))
            
            # Format improvement string
            if improvement == 1.0:
                improv_str = "baseline"
            elif improvement > 1e6:
                improv_str = f"{improvement/1e6:.1f}M×"
            elif improvement > 1e3:
                improv_str = f"{improvement/1e3:.1f}k×"
            else:
                improv_str = f"{improvement:.1f}×"
            
            print(f"{mult:<10.4f} {lam:<12.6f} {cond:<18.2e} {improv_str:<15}")
        
        # Find optimal lambda (lowest condition number)
        optimal_idx = min(range(1, len(results)), key=lambda i: results[i][2])
        optimal_mult, optimal_lam, optimal_cond, optimal_improvement = results[optimal_idx]
        
        print(f"\n  ✓ Optimal: λ = {optimal_mult:.3f} × ||A||_2 = {optimal_lam:.6f}")
        print(f"    → cond(I-P_reg) = {optimal_cond:.2e}")
        print(f"    → Improvement: {optimal_improvement/1e6:.1f}M× better than baseline")


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
        print(f"⚠️  Warning: U={U} > L-1={L-1}, can only generate {L-1} orthogonal codewords")
        print(f"   Setting U = {L-1}")
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
                
                if (u_idx + 1) % 4 == 0 or u_idx == U - 1:
                    print(f"  ✓ Generated {u_idx + 1}/{U} codewords")
                break
        
        if not success:
            print(f"  ✗ Failed to generate codeword {u_idx + 1} after {max_attempts} attempts")
            print(f"  Space is likely full (dimension exhausted)")
            break
    
    if len(X_cols) < U:
        print(f"\n⚠️  Could only generate {len(X_cols)} orthogonal codewords (requested {U})")
        U = len(X_cols)
    
    X = torch.stack(X_cols, dim=1).to(device)  # (L, U)
    
    # ========================================================================
    # VERIFICATION: Check orthogonality properties
    # ========================================================================
    print(f"\n🔍 Verification:")
    
    ones_dev = torch.ones(L, dtype=complex_dtype, device=device).unsqueeze(1)
    
    # 1. Check orthogonality to 1_L
    inner_prods = (ones_dev.conj().T @ X).abs().squeeze()  # (U,)
    max_inner_ones = inner_prods.max().item()
    mean_inner_ones = inner_prods.mean().item()
    print(f"  Orthogonality to 1_L: max|1^H x| = {max_inner_ones:.2e}, mean = {mean_inner_ones:.2e}")
    
    # 2. Check energy constraint
    norms_squared = (X.conj() * X).real.sum(dim=0)  # (U,)
    energy_dev = (norms_squared - L).abs()
    max_energy_dev = energy_dev.max().item()
    mean_energy_dev = energy_dev.mean().item()
    print(f"  Energy constraint: max||x||²-L| = {max_energy_dev:.2e}, mean = {mean_energy_dev:.2e}")
    
    # 3. Check mutual orthogonality (Gram matrix should be diagonal)
    G = X.conj().T @ X  # (U, U) Gram matrix
    # Diagonal should be L
    diag_vals = torch.diag(G).real
    diag_dev = (diag_vals - L).abs()
    print(f"  Gram diagonal: max|G_ii - L| = {diag_dev.max().item():.2e}")
    
    # Off-diagonal should be 0
    mask = ~torch.eye(U, dtype=torch.bool, device=device)
    if mask.sum() > 0:
        off_diag_vals = G[mask].abs()
        max_cross = off_diag_vals.max().item()
        mean_cross = off_diag_vals.mean().item()
        print(f"  Mutual orthogonality: max|x_i^H x_j| (i≠j) = {max_cross:.2e}, mean = {mean_cross:.2e}")
        
        if max_cross > 0.1:
            print(f"  ⚠️  WARNING: Large cross-correlation detected!")
            # Find problematic pairs
            problem_pairs = torch.where(off_diag_vals > 0.1)
            if len(problem_pairs[0]) > 0:
                print(f"  First few problematic pairs: {problem_pairs[0][:5].tolist()}")
    
    # Summary
    if max_inner_ones < 1e-5 and max_energy_dev < 1e-3 and (U == 1 or max_cross < 1e-5):
        print(f"  ✅ All checks passed! Codebook is strictly orthogonal.")
    else:
        print(f"  ⚠️  Some constraints violated, but may still be acceptable")
    
    return X  # (L, U)


def visualize_tag_codebook(X: Tensor, L: int, save_path: str = "tag_codebook_check.png"):
    """
    Visualize tag codebook properties:
    1. ||x||^2 for each codeword (should all equal L)
    2. |1^H x| for each codeword (should all be close to 0)
    """
    U = X.shape[1]
    
    # Move to CPU for plotting
    X_cpu = X.cpu()
    ones = torch.ones(L, dtype=complex_dtype)
    
    # Compute norms squared: ||x||^2
    norms_squared = (X_cpu.conj() * X_cpu).real.sum(dim=0).numpy()  # (U,)
    
    # Compute inner products with 1_L: |1^H x|
    inner_prods = (ones.conj() @ X_cpu).abs().numpy()  # (U,)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Subplot 1: Norms squared
    ax1 = axes[0]
    codeword_indices = np.arange(U)
    ax1.bar(codeword_indices, norms_squared, alpha=0.7, color='steelblue', label='$||x||^2$')
    ax1.axhline(y=L, color='red', linestyle='--', linewidth=2, label=f'Target = L = {L}')
    ax1.set_xlabel('Codeword Index', fontsize=12)
    ax1.set_ylabel('$||x||^2$', fontsize=12)
    ax1.set_title('Tag Codeword Energy', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Inner products with 1_L
    ax2 = axes[1]
    ax2.bar(codeword_indices, inner_prods, alpha=0.7, color='coral', label='$|1^H x|$')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Target = 0')
    ax2.set_xlabel('Codeword Index', fontsize=12)
    ax2.set_ylabel('$|1^H x|$', fontsize=12)
    ax2.set_title('Orthogonality to All-Ones Vector', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Use log scale to see small values
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Tag codebook visualization saved to: {save_path}")
    plt.close()
    
    # Print statistics
    print(f"\n=== Tag Codebook Statistics ===")
    print(f"Energy ||x||^2:")
    print(f"  Target: L = {L}")
    print(f"  Min: {norms_squared.min():.6f}, Max: {norms_squared.max():.6f}")
    print(f"  Mean: {norms_squared.mean():.6f}, Std: {norms_squared.std():.6f}")
    print(f"Orthogonality |1^H x|:")
    print(f"  Target: 0")
    print(f"  Min: {inner_prods.min():.2e}, Max: {inner_prods.max():.2e}")
    print(f"  Mean: {inner_prods.mean():.2e}, Std: {inner_prods.std():.2e}")
    print("=" * 35 + "\n")



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
    
    # Visualize tag codebook properties (disabled for speed)
    # visualize_tag_codebook(X_mat, L, save_path="tag_codebook_check_regular.png")

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

        # Ridge projectors (Case II) - TEST MULTIPLE LAMBDA VALUES
        # Test lambda multipliers: 0.001, 0.01, 0.1, 1.0
        A = Xi @ Xi.conj().T  # (K,K)
        spectral_A = torch.linalg.norm(A, ord=2).real.item()
        
        # Store base lambda (will test multiple values)
        lam_base = 0.1 * spectral_A  # Original value
        lambdas_STR.append(lam_base)
        lambdas_SR.append(lam_base)

        # P_c = Xi (Xi^H Xi + lambda I)^(-1) Xi^H
        XtX = Xi.conj().T @ Xi   # (Q+1, Q+1)
        I_ch = torch.eye(Q + 1, dtype=complex_dtype, device=device)

        P_STR = Xi @ torch.linalg.inv(XtX + lam_base * I_ch) @ Xi.conj().T
        P_SR  = Xi @ torch.linalg.inv(XtX + lam_base * I_ch) @ Xi.conj().T
        ridge_proj_STR.append(P_STR)
        ridge_proj_SR.append(P_SR)

    # Diagnostic: Test different lambda values and their effect on condition number
    print("\n=== Testing Regularization Parameters ===")
    print("Format: lambda_mult * ||Xi Xi^H||_2")
    lambda_multipliers = [0.001, 0.01, 0.1, 1.0]  # Test range
    
    for c in range(min(2, Ma)):  # Test first 2 waveforms
        Xi = Xi_list[c]
        P_orig = proj_list[c]  # Original projector (no regularization)
        A = Xi @ Xi.conj().T
        spectral_A = torch.linalg.norm(A, ord=2).item()
        XtX = Xi.conj().T @ Xi
        I_ch = torch.eye(Q + 1, dtype=complex_dtype, device=device)
        I_K = torch.eye(K, dtype=complex_dtype, device=device)
        
        print(f"\nWaveform c={c}: ||Xi Xi^H||_2 = {spectral_A:.3f}")
        print(f"  Original P (no regularization): cond(P) = {torch.linalg.cond(P_orig).item():.2e}")
        
        for mult in lambda_multipliers:
            lam = mult * spectral_A
            # Regularized projector: P_reg = I - Xi (Xi^H Xi + lambda I)^(-1) Xi^H
            P_reg = I_K - Xi @ torch.linalg.inv(XtX + lam * I_ch) @ Xi.conj().T
            
            try:
                cond_Preg = torch.linalg.cond(P_reg).item()
            except:
                cond_Preg = float('inf')
            
            print(f"  λ = {mult:.3f} * ||A||_2 = {lam:.4f}: cond(P_reg) = {cond_Preg:.2e}")
    
    print("=" * 50 + "\n")

    return Env(
        device=device,
        L=L, Np=Np, Qmin=Qmin, Qmax=Qmax, Q=Q, K=K, Ma=Ma, U=U,
        C_mat=C_mat, X_mat=X_mat,
        gamma_SR=gamma_SR, gamma_STR=gamma_STR,
        Xi_list=Xi_list, pinv_list=pinv_list, proj_list=proj_list,
        ridge_proj_SR=ridge_proj_SR, ridge_proj_STR=ridge_proj_STR,
        lambdas_SR=lambdas_SR, lambdas_STR=lambdas_STR
    )


def generate_environment_with_lambda(
    L: int = 10,
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
    Y0 = x @ alpha_STR.conj().T + ones @ alpha_SR.conj().T  # (L,K)
    # Add complex Gaussian noise CN(0, sigma2):
    noise = (torch.randn(L, K, dtype=complex_dtype, device=device) +
             1j * torch.randn(L, K, dtype=complex_dtype, device=device)) * math.sqrt(sigma2 / 2.0)
    Y = Y0 + noise
    return Y, {"alpha_SR": alpha_SR, "alpha_STR": alpha_STR, "sigma2": torch.tensor(sigma2, device=device)}


# -----------------------------
# Case I: ML joint decoding (no penalty)
# -----------------------------
def decode_case_I(env: Env, Y: Tensor, debug: bool = False, use_projection_to_colspace: bool = False) -> Tuple[int, int, Tensor, Tensor]:
    """
    Joint ML decoding with two formulations:
    
    Method 1 (use_projection_to_colspace=False, original):
      minimize ||(I - Xi_c Xi_c^†) Y^H x||^2 + ||(I - Xi_c Xi_c^†) Y^H 1||^2
      Problem: Projects to orthogonal complement, signal is eliminated
    
    Method 2 (use_projection_to_colspace=True, NEW):
      maximize ||Xi_c Xi_c^† Y^H x||^2 + ||Xi_c Xi_c^† Y^H 1||^2
      Solution: Projects to column space, signal is preserved!

    Returns:
      (c_hat, x_hat, gamma_SR_hat, gamma_STR_hat)
    """
    device = env.device
    L, U, Ma = env.L, env.U, env.Ma
    ones = torch.ones((L, 1), dtype=complex_dtype, device=device)

    YH = Y.conj().T  # (K,L)
    YH1 = YH @ ones / env.L

    if use_projection_to_colspace:
        # NEW METHOD (图2): Maximize energy in column space
        best_val = float("-inf")  # We want to MAXIMIZE
        method_name = "Projection TO column space (NEW)"
    else:
        # ORIGINAL METHOD (图1): Minimize energy in orthogonal complement
        best_val = float("inf")  # We want to MINIMIZE
        method_name = "Projection to COMPLEMENT (Original)"
    
    best_c = 0
    best_x = 0
    
    # For debug: store all objective values
    if debug:
        all_objectives = torch.zeros((Ma, U), device=device)

    for c in range(Ma):
        if use_projection_to_colspace:
            # NEW: Project TO column space: Proj = Xi Xi^†
            Xi = env.Xi_list[c]  # (K, Q+1)
            Xi_dag = env.pinv_list[c]  # (Q+1, K)
            Proj = Xi @ Xi_dag  # (K, K) - projection TO column space
        else:
            # ORIGINAL: Project to complement: P = I - Xi Xi^†
            Proj = env.proj_list[c]  # (K, K)
        
        ProjYH = Proj @ YH  # (K, L)
        term_const = frob_norm2(ProjYH @ ones).item()

        # Compute ||Proj Y^H x||^2 for all x
        ProjX = ProjYH @ env.X_mat  # (K, U)
        vals = (ProjX.conj() * ProjX).real.sum(dim=0)  # (U,)
        vals = vals + term_const
        
        if debug:
            all_objectives[c, :] = vals

        if use_projection_to_colspace:
            # NEW: Find maximum (highest energy in column space)
            vmax, idx = torch.max(vals, dim=0)
            if vmax.item() > best_val:
                best_val = vmax.item()
                best_c = c
                best_x = int(idx.item())
        else:
            # ORIGINAL: Find minimum (lowest energy in complement)
            vmin, idx = torch.min(vals, dim=0)
            if vmin.item() < best_val:
                best_val = vmin.item()
                best_c = c
                best_x = int(idx.item())
    
    # Debug output
    if debug:
        print(f"\n  🔍 Case I Debug ({method_name}):")
        print(f"    Selected: c={best_c}, x={best_x}, objective={best_val:.3e}")
        # Show top 5 candidates
        flat_obj = all_objectives.view(-1)
        if use_projection_to_colspace:
            sorted_vals, sorted_idx = torch.sort(flat_obj, descending=True)  # Largest first
            print(f"    Top 5 candidates (highest energy):")
        else:
            sorted_vals, sorted_idx = torch.sort(flat_obj)  # Smallest first
            print(f"    Top 5 candidates (lowest residual):")
        for i in range(min(5, len(sorted_vals))):
            c_i = sorted_idx[i].item() // U
            x_i = sorted_idx[i].item() % U
            print(f"      #{i+1}: c={c_i}, x={x_i}, objective={sorted_vals[i].item():.3e}")

    # Channel estimates using closed forms
    Xi = env.Xi_list[best_c]
    Xi_dag = env.pinv_list[best_c]
    xhat = env.X_mat[:, best_x:best_x + 1]
    gamma_STR_hat = (Xi_dag @ (Y.conj().T @ xhat)) / env.L
    gamma_SR_hat = (Xi_dag @ (Y.conj().T @ ones)) / env.L
    return best_c, best_x, gamma_SR_hat, gamma_STR_hat


# -----------------------------
# Case II: ML with Tikhonov (ridge)
# -----------------------------
def decode_case_II(env: Env, Y: Tensor, debug: bool = False) -> Tuple[int, int, Tensor, Tensor]:
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
    
    if debug:
        all_objectives = torch.zeros((Ma, U), device=device)

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
        
        if debug:
            all_objectives[c, :] = vals

        vmin, idx = torch.min(vals, dim=0)
        if vmin.item() < best_val:
            best_val = vmin.item()
            best_c = c
            best_x = int(idx.item()) ## an index of the codebook, not the codeword itself
    
    # Debug output
    if debug:
        print(f"\n  🔍 Case II Debug (Ridge, λ={env.lambdas_STR[0]:.4f}):")
        print(f"    Selected: c={best_c}, x={best_x}, objective={best_val:.3e}")
        flat_obj = all_objectives.view(-1)
        sorted_vals, sorted_idx = torch.sort(flat_obj)
        print(f"    Top 5 candidates:")
        for i in range(min(5, len(sorted_vals))):
            c_i = sorted_idx[i].item() // U
            x_i = sorted_idx[i].item() % U
            print(f"      #{i+1}: c={c_i}, x={x_i}, objective={sorted_vals[i].item():.3e}")

    # Channel estimates via ridge closed-forms:
    Xi = env.Xi_list[best_c]
    XtX = Xi.conj().T @ Xi
    I_ch = torch.eye(env.Q + 1, dtype=complex_dtype, device=device)
    lam_str = env.lambdas_STR[best_c]
    lam_sr = env.lambdas_SR[best_c]

    xhat = env.X_mat[:, best_x:best_x + 1]
    ones = torch.ones((env.L, 1), dtype=complex_dtype, device=device)

    gamma_STR_hat = torch.linalg.solve(XtX + lam_str * I_ch, Xi.conj().T @ (Y.conj().T @ xhat)) / env.L
    gamma_SR_hat = torch.linalg.solve(XtX + lam_sr * I_ch, Xi.conj().T @ (Y.conj().T @ ones)) / env.L
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
            print(f"MC trial {_+1}/{N_mc} at SNR {snr_db:+.1f} dB", end="\r")
            # Sample a valid waveform & tag
            c_true = random.randrange(Ma)
            x_true = random.randrange(U)

            # Synthesize observation
            Y, aux = synthesize_observation(env, c_true, x_true, snr_db) ## c and x are randomly chosen in each trial

            # DEBUG: Detailed comparison for first trial
            if _ == 0:
                print(f"\n\n=== DEBUG: First Trial Details (SNR={snr_db}dB) ===")
                print(f"Ground Truth: c_true={c_true}, x_true={x_true}")

            # Decode: Case I
            cI, xI, gSR_I, gSTR_I = decode_case_I(env, Y, debug=(_ == 0))
            cnt_I["radar_err"] += int(cI != c_true)
            cnt_I["tag_err"] += int(xI != x_true)
            cnt_I["nmse_sr"] += nmse(gSR_I.squeeze(), env.gamma_SR.squeeze())
            cnt_I["nmse_str"] += nmse(gSTR_I.squeeze(), env.gamma_STR.squeeze())
            
            if _ == 0:
                print(f"Case I:  c_hat={cI}, x_hat={xI}, radar_correct={cI==c_true}, tag_correct={xI==x_true}")
                print(f"         NMSE_SR={nmse(gSR_I.squeeze(), env.gamma_SR.squeeze()):.3e}, NMSE_STR={nmse(gSTR_I.squeeze(), env.gamma_STR.squeeze()):.3e}")

            # Decode: Case II
            cII, xII, gSR_II, gSTR_II = decode_case_II(env, Y, debug=(_ == 0))
            cnt_II["radar_err"] += int(cII != c_true)
            cnt_II["tag_err"] += int(xII != x_true)
            cnt_II["nmse_sr"] += nmse(gSR_II.squeeze(), env.gamma_SR.squeeze())
            cnt_II["nmse_str"] += nmse(gSTR_II.squeeze(), env.gamma_STR.squeeze())
            
            if _ == 0:
                print(f"Case II: c_hat={cII}, x_hat={xII}, radar_correct={cII==c_true}, tag_correct={xII==x_true}")
                print(f"         NMSE_SR={nmse(gSR_II.squeeze(), env.gamma_SR.squeeze()):.3e}, NMSE_STR={nmse(gSTR_II.squeeze(), env.gamma_STR.squeeze()):.3e}")

            # Decode: Case III
            cIII, xIII, gSR_III, gSTR_III = decode_case_III(env, Y)
            cnt_III["radar_err"] += int(cIII != c_true)
            cnt_III["tag_err"] += int(xIII != x_true)
            cnt_III["nmse_sr"] += nmse(gSR_III.squeeze(), env.gamma_SR.squeeze())
            cnt_III["nmse_str"] += nmse(gSTR_III.squeeze(), env.gamma_STR.squeeze())
            
            if _ == 0:
                print(f"Case III: c_hat={cIII}, x_hat={xIII}, radar_correct={cIII==c_true}, tag_correct={xIII==x_true}")
                print(f"          NMSE_SR={nmse(gSR_III.squeeze(), env.gamma_SR.squeeze()):.3e}, NMSE_STR={nmse(gSTR_III.squeeze(), env.gamma_STR.squeeze()):.3e}")
                print("=" * 50 + "\n")

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
    """Run comprehensive tests."""
    print("\n" + "="*70)
    print("COMPREHENSIVE SYSTEM DIAGNOSTICS")
    print("="*70)
    
    # System parameters
    L = 20  # ✨ UPDATED: Increased from 10 to 20 for better codebook orthogonality
    Np = 31
    Qmin, Qmax = 10, 20
    Ma = 16
    U = 16
    Q = Qmax - Qmin
    K = Np + Q
    
    set_seed(2025)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n📱 Device: {device}")
    print(f"\n📐 System Parameters:")
    print(f"  L={L}, Np={Np}, Ma={Ma}, U={U}")
    print(f"  Q={Q}, K={K}")
    
    # Build waveforms
    print("\n" + "="*70)
    print("BUILDING WAVEFORMS...")
    print("="*70)
    C_mat = build_waveform_set(Np, Ma, device)
    
    # Build tag codebook
    print("\n" + "="*70)
    print("BUILDING TAG CODEBOOK...")
    print("="*70)
    X_mat = build_tag_codebook(L, U, device)
    
    # Build Xi matrices
    print("\n" + "="*70)
    print("BUILDING CONVOLUTION MATRICES...")
    print("="*70)
    Xi_list = []
    for i in range(Ma):
        Xi = build_Xi_linear(C_mat[i], K, Q)
        Xi_list.append(Xi)
    print(f"Generated {Ma} Xi matrices of shape ({K}, {Q+1})")
    
    # Run three core tests
    test_waveform_properties(C_mat, Np, Ma)
    test_tag_codebook_properties(X_mat, L, U)
    test_projector_condition_numbers(Xi_list, K, Q, device)
    
    print("\n" + "="*70)
    print("✅ ALL DIAGNOSTIC TESTS COMPLETED")
    print("="*70)
    
    # ========================================================================
    # NEW TEST: Compare two projection methods (图1 vs 图2)
    # ========================================================================
    print("\n" + "="*70)
    print("🆕 COMPARING TWO PROJECTION METHODS")
    print("="*70)
    print("\n📊 Method Comparison:")
    print("  图1 (Original): min ||(I - Xi Xi^†) Y^T x||^2  → Projects to COMPLEMENT")
    print("  图2 (New):      max ||Xi Xi^† Y^T x||^2        → Projects to COLUMN SPACE")
    print("\nHypothesis: 图2 should work better because signal is in column space!")
    
    # Generate a simple environment for quick test
    env_test = generate_environment_with_lambda(
        L=L, Np=Np, Qmin=Qmin, Qmax=Qmax, Ma=Ma, U=U,
        lambda_multiplier=0.0,  # No regularization for fair comparison
        use_gpu_if_available=(device.type == "cuda")
    )
    
    # Run a quick Monte Carlo test
    N_test = 50  # Small number for quick test
    snr_db_test = 5.0
    
    results_comparison = {
        "Method1_Original": {"radar_err": 0, "tag_err": 0},
        "Method2_New": {"radar_err": 0, "tag_err": 0},
        "Case_III": {"radar_err": 0, "tag_err": 0}
    }
    
    print(f"\nRunning {N_test} trials at SNR={snr_db_test}dB...")
    
    for trial in range(N_test):
        # Generate data
        c_true = random.randint(0, Ma - 1)
        x_true = random.randint(0, U - 1)
        
        # Generate random channels for this trial
        gamma_SR_trial = (torch.randn(Q + 1, 1, dtype=complex_dtype, device=device) +
                          1j * torch.randn(Q + 1, 1, dtype=complex_dtype, device=device)) / math.sqrt(2)
        gamma_STR_trial = (torch.randn(Q + 1, 1, dtype=complex_dtype, device=device) +
                           1j * torch.randn(Q + 1, 1, dtype=complex_dtype, device=device)) / math.sqrt(2)
        
        xvec = env_test.X_mat[:, x_true: x_true + 1]
        ones = torch.ones((L, 1), dtype=complex_dtype, device=device)
        Xi_true = env_test.Xi_list[c_true]
        
        a_SR = Xi_true @ gamma_SR_trial
        a_STR = Xi_true @ gamma_STR_trial
        
        Y_clean = xvec @ a_STR.T + ones @ a_SR.T
        
        snr_lin = 10 ** (snr_db_test / 10.0)
        sig_pow = (Y_clean.abs() ** 2).mean().item()
        noise_pow = sig_pow / snr_lin
        noise = (torch.randn_like(Y_clean) + 1j * torch.randn_like(Y_clean)) * math.sqrt(noise_pow / 2)
        Y = Y_clean + noise
        
        # Test Method 1 (Original - 图1)
        c1, x1, _, _ = decode_case_I(env_test, Y, debug=False, use_projection_to_colspace=False)
        results_comparison["Method1_Original"]["radar_err"] += int(c1 != c_true)
        results_comparison["Method1_Original"]["tag_err"] += int(x1 != x_true)
        
        # Test Method 2 (New - 图2)
        c2, x2, _, _ = decode_case_I(env_test, Y, debug=False, use_projection_to_colspace=True)
        results_comparison["Method2_New"]["radar_err"] += int(c2 != c_true)
        results_comparison["Method2_New"]["tag_err"] += int(x2 != x_true)
        
        # Test Case III (baseline)
        c3, x3, _, _ = decode_case_III(env_test, Y)
        results_comparison["Case_III"]["radar_err"] += int(c3 != c_true)
        results_comparison["Case_III"]["tag_err"] += int(x3 != x_true)
        
        if trial == 0:
            # Debug first trial
            print(f"\n  First Trial Debug (c_true={c_true}, x_true={x_true}):")
            print(f"    Method 1 (图1-Complement): c={c1}, x={x1}, correct={c1==c_true and x1==x_true}")
            print(f"    Method 2 (图2-ColSpace):   c={c2}, x={x2}, correct={c2==c_true and x2==x_true}")
            print(f"    Case III (baseline):       c={c3}, x={x3}, correct={c3==c_true and x3==x_true}")
    
    # Print comparison results
    print(f"\n" + "="*70)
    print("📊 COMPARISON RESULTS")
    print("="*70)
    print(f"{'Method':<30} {'Radar MER':<15} {'Tag MER':<15} {'Overall':<15}")
    print("-" * 70)
    
    for name, res in results_comparison.items():
        radar_mer = res["radar_err"] / N_test
        tag_mer = res["tag_err"] / N_test
        overall_mer = (res["radar_err"] + res["tag_err"]) / (2 * N_test)
        
        marker = ""
        if name == "Method2_New" and tag_mer < 0.1:
            marker = "  ✅ SUCCESS!"
        elif name == "Method1_Original" and tag_mer > 0.5:
            marker = "  ❌ FAILED"
        
        print(f"{name:<30} {radar_mer:<15.3f} {tag_mer:<15.3f} {overall_mer:<15.3f}{marker}")
    
    print("\n💡 Interpretation:")
    if results_comparison["Method2_New"]["tag_err"] < results_comparison["Method1_Original"]["tag_err"]:
        improvement = (results_comparison["Method1_Original"]["tag_err"] - 
                      results_comparison["Method2_New"]["tag_err"]) / N_test
        print(f"  ✅ Method 2 (图2) is BETTER by {improvement:.1%} tag error reduction!")
        print(f"  ✅ Hypothesis CONFIRMED: Projecting to column space works!")
    else:
        print(f"  ❌ Method 2 (图2) did not improve performance")
        print(f"  ❌ Need to investigate further...")
    
    print("\n" + "="*70)
    
    # Now test performance with different lambda values
    print("\n" + "="*70)
    print("PERFORMANCE TESTING WITH DIFFERENT λ VALUES")
    print("="*70)
    
    # Test smaller lambda values based on user feedback
    lambda_multipliers_to_test = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    
    for lam_mult in lambda_multipliers_to_test:
        print(f"\n{'='*70}")
        print(f"Testing with λ = {lam_mult} × ||Xi Xi^H||_2")
        print(f"{'='*70}")
        
        # Generate environment with this lambda
        env = generate_environment_with_lambda(
            L=L, Np=Np, Qmin=Qmin, Qmax=Qmax, Ma=Ma, U=U,
            P_SR=1.0, P_STR=1.0, seed=2025, 
            lambda_multiplier=lam_mult,
            use_gpu_if_available=True
        )
        
        # Run small MC simulation
        snr_grid_db = [5.0]
        N_mc = 100
        results = run_mc(env, snr_grid_db, N_mc=N_mc)
        
        # Print summary
        print(f"\n📊 Results with λ_mult={lam_mult}:")
        print(f"  Case I:   TagMER={results['CaseI']['TagMER'][0]:.3f}, NMSE_STR={results['CaseI']['NMSE_STR'][0]:.3e}")
        print(f"  Case II:  TagMER={results['CaseII']['TagMER'][0]:.3f}, NMSE_STR={results['CaseII']['NMSE_STR'][0]:.3e}")
        print(f"  Case III: TagMER={results['CaseIII']['TagMER'][0]:.3f}, NMSE_STR={results['CaseIII']['NMSE_STR'][0]:.3e}")
    
    print("\n" + "="*70)
    print("✅ ALL PERFORMANCE TESTS COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()


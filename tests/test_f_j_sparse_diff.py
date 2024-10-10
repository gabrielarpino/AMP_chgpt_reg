import pytest
import jax.numpy as jnp
import numpy as np
from jax import random
import sys
sys.path.append('/Users/xiaoqiliu/Desktop/3_Changepoints_detection/AMP/')
from amp.marginal_separable_jax import f_j, \
    f_j_sparse_diff, dfdB_j_sparse_diff, f_j_sparse_diff_old, dfdB_j_sparse_diff_old
from amp.signal_priors import SparseDiffSignal_old, SparseDiffSignal

# This script tests not only f_j_sparse_diff, but also SparseDiffSignal.
@pytest.mark.parametrize("δ", [0.5, 1, 2, 4])
@pytest.mark.parametrize("α", [0.1, 0.5]) 
@pytest.mark.parametrize("ρ_1", [0.1, 1, 2])
@pytest.mark.parametrize("σ_w", [1e-2, 0.2])
def test_f_j(δ, α, ρ_1, σ_w):
    L = 2 # atm only allow two signals
    key = random.PRNGKey(np.random.randint(0, 1000))
    signal_old = SparseDiffSignal_old(δ=δ, ρ_1=ρ_1, σ_w=σ_w, α=α)
    ρ_B_same = signal_old.ρ_B_same
    assert jnp.allclose(ρ_B_same, ρ_1)
    ρ_B_diff = signal_old.ρ_B_diff
    η_tmp = σ_w**2 / (δ * ρ_1)
    η = 1 / jnp.sqrt(1 + η_tmp)
    assert jnp.allclose(ρ_B_diff, ρ_1 * jnp.array([[1, η], [η, 1]]))

    # ================== Check signal samples ==================
    p = 10000
    B = signal_old.sample(p)
    assert B.shape == (p, 2)
    # Same versus different entries follow a binomial distribution:
    ϵ = 0.01
    assert jnp.abs(jnp.sum(B[:, 0] == B[:, 1]) - (1-α) * p) < p * ϵ # concentrate
    assert jnp.abs(jnp.sum(B[:, 0] != B[:, 1]) - α * p) < p * ϵ # concentrate
    # Zero mean and variance concentrates around δ * ρ_1 (2nd signal 
    # was scaled such that its variance is δ * ρ_1):
    ϵ = 0.05
    assert jnp.abs(np.mean(B[:, 0]) - 0) < ϵ
    assert jnp.abs(np.mean(B[:, 1]) - 0) < ϵ
    assert jnp.abs(jnp.var(B[:, 0]) - δ * ρ_1) < δ * ρ_1 * ϵ
    assert jnp.abs(jnp.var(B[:, 1]) - δ * ρ_1) < δ * ρ_1 * ϵ
    
    # ================== test f_j ==================
    s = random.normal(key, (1, L))
    ν̂ = random.normal(key, (L, L))
    ν̂ = ν̂ @ ν̂.T # ensure ν̂ is PSD
    κ_B = random.normal(key, (L, L))
    κ_B = κ_B @ κ_B.T # ensure κ_B is PSD

    res = f_j_sparse_diff_old(s, δ, signal_old, ν̂, κ_B)
    assert res.shape == (1, L)
    assert jnp.all(~jnp.isnan(res))
    assert jnp.all(~jnp.isinf(res))

@pytest.mark.parametrize("var_β_1", [0.5, 1, 2, 4])
@pytest.mark.parametrize("α", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("σ_w", [0.8, 1, 8])
def test_L_2(var_β_1, α, σ_w):
    """New version of f (for general L>=2) should match old version when L=2"""
    L = 2
    δ = 1.5
    ρ_1 = var_β_1 / δ
    signal_old = SparseDiffSignal_old(δ, ρ_1, σ_w, α)
    s = np.random.randn(1, 2)
    ν̂ = np.random.randn(2, 2)
    ν̂ = ν̂ @ ν̂.T
    κ_B = np.random.randn(2, 2)
    κ_B = κ_B @ κ_B.T
    res_old = f_j_sparse_diff_old(s, δ, signal_old, ν̂, κ_B)

    signal = SparseDiffSignal(var_β_1, σ_w, α, L)
    res = f_j_sparse_diff(s, signal, ν̂, κ_B)
    assert jnp.allclose(res, res_old)

    der_old = dfdB_j_sparse_diff_old(s, δ, signal_old, ν̂, κ_B)
    der = dfdB_j_sparse_diff(s, signal, ν̂, κ_B)
    assert jnp.allclose(der, der_old)


def test_α_extremes_L_2():
    """When α = 0 or 1, prior reduces from Gaussian mixture to Gaussian"""
    L = 2 # atm only allow two signals
    p = 100
    δ = 2
    ρ_1 = 1
    σ_w = 1e-2
    key = random.PRNGKey(np.random.randint(0, 1000))

    # ================== Check α=0 ==================
    α = 0 # two signals are always identical
    signal_old = SparseDiffSignal_old(δ, ρ_1, σ_w, α)
    ρ_B_same = signal_old.ρ_B_same
    ρ_B_diff = signal_old.ρ_B_diff
    ρ = signal_old.ρ_B
    assert jnp.allclose(ρ_B_same, ρ_1)
    assert jnp.allclose(ρ_B_same, ρ)
    B = signal_old.sample(p)
    assert B.shape == (p, 2)
    assert jnp.all(B[:, 0] == B[:, 1]) # all signal entries are identical

    s = random.normal(key, (1, L))
    ν̂ = random.normal(key, (L, L))
    ν̂ = ν̂ @ ν̂.T # ensure ν̂ is PSD
    κ_B = random.normal(key, (L, L))
    κ_B = κ_B @ κ_B.T # ensure κ_B is PSD
    
    res = f_j_sparse_diff_old(s, δ, signal_old, ν̂, κ_B)
    res_expected = f_j(s, δ, ρ, ν̂, κ_B)
    assert jnp.allclose(res, res_expected)

    # ================== Check α=1 ==================
    α = 1 # two signals are always different
    signal_old = SparseDiffSignal_old(δ=δ, ρ_1=ρ_1, σ_w=σ_w, α=α)
    ρ_B_same = signal_old.ρ_B_same
    ρ_B_diff = signal_old.ρ_B_diff
    ρ = signal_old.ρ_B
    η_tmp = σ_w**2/ (δ * ρ_1)
    η = 1 / jnp.sqrt(1 + η_tmp)
    assert jnp.allclose(ρ_B_diff, ρ_1 * jnp.array([[1, η], [η, 1]]))
    assert jnp.allclose(ρ_B_diff, ρ)
    B = signal_old.sample(p)
    assert B.shape == (p, 2)
    assert jnp.all(B[:, 0] != B[:, 1]) # all signal entries are different

    res = f_j_sparse_diff_old(s, δ, signal_old, ν̂, κ_B)
    res_expected = f_j(s, δ, ρ, ν̂, κ_B)
    assert jnp.allclose(res, res_expected)

@pytest.mark.parametrize("var_β_1", [0.5, 1, 2, 4])
@pytest.mark.parametrize("α", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("σ_w", [0.8, 1, 8])
@pytest.mark.parametrize("L", [3, 4])
def test_α_extremes(var_β_1, α, σ_w, L):
    """Test general version of f (for general L>=2) matches f for dense Gaussian"""
    # ================== Check α=0 ==================
    δ = 1.5
    α = 0 # signals are always identical
    s = np.random.randn(1, L)
    ν̂ = np.random.randn(L, L)
    ν̂ = ν̂ @ ν̂.T
    κ_B = np.random.randn(L, L)
    κ_B = κ_B @ κ_B.T
    signal = SparseDiffSignal(var_β_1, σ_w, α, L)
    res = f_j_sparse_diff(s, signal, ν̂, κ_B)
    res_expected = f_j(s, δ, signal.cov/δ, ν̂, κ_B)
    assert jnp.allclose(res, res_expected)

    # ================== Check α=1 ==================
    α = 1 # every entry in neighbouring signals are always different
    signal = SparseDiffSignal(var_β_1, σ_w, α, L)
    res = f_j_sparse_diff(s, signal, ν̂, κ_B)
    res_expected = f_j(s, δ, signal.cov/δ, ν̂, κ_B)
    assert jnp.allclose(res, res_expected)


def test_expected_B_fV_B():
    """TODO: E[B.T * f(V_B)] = E[f(V_B).T * f(V_B)]
    I suspect this does not hold, which is possibly why SE using this property 
    does not match AMP."""
    pass

@pytest.mark.parametrize("α", [0, 0.1, 0.5, 0.9, 1])
def test_derivative(α):
    """Test AD matches finite difference approximation of derivative"""
    L = 2 # atm only allow two signals
    p = 100
    δ = 2
    ρ_1 = 1
    σ_w = 1e-2
    key = random.PRNGKey(np.random.randint(0, 1000))
    signal_old = SparseDiffSignal_old(δ=δ, ρ_1=ρ_1, σ_w=σ_w, α=α)
    ρ_B_same = signal_old.ρ_B_same
    ρ_B_diff = signal_old.ρ_B_diff
    ρ = signal_old.ρ_B

    s = random.normal(key, (1, L))
    ν̂ = random.normal(key, (L, L))
    ν̂ = ν̂ @ ν̂.T # ensure ν̂ is PSD
    κ_B = random.normal(key, (L, L))
    κ_B = κ_B @ κ_B.T # ensure κ_B is PSD
    
    dfdB_j_AD = dfdB_j_sparse_diff_old(s, δ, signal_old, ν̂, κ_B) # LxL
    dfdB_j_finite_diff = np.zeros((L, L))
    EPS = 1e-6
    for j in range(L):
        s_perturbed = s.copy()
        s_perturbed = s_perturbed.at[0, j].set(s_perturbed[0, j] + EPS)
        # Fill in jth column which correspond to derivative wrt jth entry in input V:
        dfdB_j_finite_diff[:, j] = \
            (f_j_sparse_diff_old(s_perturbed, δ, signal_old, ν̂, κ_B) - \
            f_j_sparse_diff_old(s, δ, signal_old, ν̂, κ_B)) / EPS
    
    assert jnp.allclose(dfdB_j_AD, dfdB_j_finite_diff, rtol=1e-2)

    # ================== Check α=0 ==================
    if α == 0: # two signals are always identical
        Σ_V_same = ν̂ @ ρ_B_same @ ν̂ + 1/δ * κ_B
        χ_same = (ρ_B_same @ ν̂) @ jnp.linalg.pinv(Σ_V_same)
        expected_dfdB_j = χ_same
        assert jnp.allclose(dfdB_j_AD, expected_dfdB_j, rtol=1e-2)

    # ================== Check α=1 ==================
    if α == 1: # two signals are always different
        Σ_V_diff = ν̂ @ ρ_B_diff @ ν̂ + 1/δ * κ_B
        χ_diff = (ρ_B_diff @ ν̂) @ jnp.linalg.pinv(Σ_V_diff)
        expected_dfdB_j = χ_diff
        assert jnp.allclose(dfdB_j_AD, expected_dfdB_j, rtol=1e-2)

if __name__ == "__main__":
    var_β_1 = 2
    α = 0.1
    σ_w = 0.9
    L = 3
    print("Testing f_j")
    test_α_extremes(var_β_1, α, σ_w, L)
    print("Tested f_j")
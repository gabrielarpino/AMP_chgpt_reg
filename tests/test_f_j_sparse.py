import pytest
import jax.numpy as jnp
import numpy as np
from jax import random
import sys
sys.path.append('/Users/xiaoqiliu/Desktop/3_Changepoints_detection/AMP/')
from amp.marginal_separable_jax import f_j, SparseSignal, f_j_sparse, dfdB_j_sparse

@pytest.mark.parametrize("α", [0.3, 0.5, 0.8])
@pytest.mark.parametrize("δ", [0.5, 1, 2])
@pytest.mark.parametrize("L", [2, 4])
def test_sparse_signal(α, δ, L):
    """Test sparse signal generated follows β_l ~iid (1-α)δ_0 + α * N(0, σ_l^2)"""
    key = random.PRNGKey(np.random.randint(0, 1000))
    σ_l_arr = random.uniform(key=key, shape=(L,))
    signal = SparseSignal(α, δ, σ_l_arr)
    ρ_B = signal.ρ_B
    # zero off-diagonal:
    assert jnp.all(ρ_B[~jnp.eye(L, dtype=bool)] == 0)
    p = 10000
    B = signal.sample(p)
    assert B.shape == (p, L)
    ϵ = 0.05
    
    for l in range(L):
        is_nonzero = B[:, l] != 0
        num_nonzeros = jnp.sum(is_nonzero)
        # α fraction of entries are nonzero:
        assert jnp.abs(num_nonzeros / p - α) < ϵ
        # zero mean:
        assert jnp.abs(jnp.mean(B[is_nonzero, l])) < ϵ
        # variance:
        assert jnp.abs(1/δ * jnp.var(B[:, l]) - ρ_B[l, l]) < ϵ # all entries
        assert jnp.abs(jnp.var(B[is_nonzero, l]) - σ_l_arr[l]**2) < ϵ # nonzero entries

    
@pytest.mark.parametrize("δ", [0.5, 1, 2])
@pytest.mark.parametrize("L", [2, 3, 4])
def test_f_j_α1(δ, L):
    """
    When α = 1, all entries of B are simply Gaussian so f_j_sparse matches f_j.
    """
    α = 1
    key = random.PRNGKey(np.random.randint(0, 1000))
    σ_l_arr = random.uniform(key=key, shape=(L,))
    signal = SparseSignal(α, δ, σ_l_arr)
    ρ = signal.ρ_B
    p = 10000
    B = signal.sample(p)
    ϵ = 1e-3
    assert jnp.sum(B == 0) < p*L * ϵ

    s = random.normal(key, (1, L))
    ν̂ = random.normal(key, (L, L))
    ν̂ = ν̂ @ ν̂.T # ensure ν̂ is PSD
    κ_B = random.normal(key, (L, L))
    κ_B = κ_B @ κ_B.T # ensure κ_B is PSD
    res = f_j_sparse(s, δ, α, ρ, ν̂, κ_B)
    assert res.shape == (1, L)
    assert jnp.all(~jnp.isnan(res))
    assert jnp.all(~jnp.isinf(res))
    res_expected = f_j(s, δ, ρ, ν̂, κ_B)
    assert jnp.allclose(res, res_expected, rtol=1e-3)

    deriv = dfdB_j_sparse(s, δ, α, ρ, ν̂, κ_B)
    assert deriv.shape == (L, L)
    assert jnp.all(~jnp.isnan(deriv))
    assert jnp.all(~jnp.isinf(deriv))
    Σ_V = ν̂ @ ρ @ ν̂ + 1/δ * κ_B
    χ = (ρ @ ν̂) @ jnp.linalg.pinv(Σ_V)
    expected_deriv = χ
    assert jnp.allclose(deriv, expected_deriv, rtol=1e-3)

@pytest.mark.parametrize("α", [0.3, 0.5, 0.8])
@pytest.mark.parametrize("δ", [0.5, 1, 2])
@pytest.mark.parametrize("L", [2, 3, 4])
def test_f_j_α_not_0_or_1(α, δ, L):
    """Check f_j_sparse, dfdB_j_sparse dont return NaN or Inf."""
    key = random.PRNGKey(np.random.randint(0, 1000))
    σ_l_arr = random.uniform(key=key, shape=(L,))
    signal = SparseSignal(α, δ, σ_l_arr)
    ρ = signal.ρ_B
    p = 10000
    B = signal.sample(p)

    s = random.normal(key, (1, L))
    ν̂ = random.normal(key, (L, L))
    ν̂ = ν̂ @ ν̂.T # ensure ν̂ is PSD
    κ_B = random.normal(key, (L, L))
    κ_B = κ_B @ κ_B.T # ensure κ_B is PSD
    res = f_j_sparse(s, δ, α, ρ, ν̂, κ_B)
    assert res.shape == (1, L)
    assert jnp.all(~jnp.isnan(res))
    assert jnp.all(~jnp.isinf(res))

    deriv = dfdB_j_sparse(s, δ, α, ρ, ν̂, κ_B)
    assert deriv.shape == (L, L)
    assert jnp.all(~jnp.isnan(deriv))
    assert jnp.all(~jnp.isinf(deriv))

@pytest.mark.parametrize("L", [2, 3, 4])
@pytest.mark.parametrize("bern_p", [0.3, 0.5, 0.8])
def test_pinv_matrix_with_zero_submatrix(L, bern_p):
    """
    Test nonzero submatrix of pinv = inv of nonzero submatrix. 
    Using numpy instead of jax for convenience.
    This trick was used to invert inv_Σ̃_c in marginal_separable_jax.py.
    """
    is_nonzero = np.random.binomial(1, bern_p, size=(L,)).astype(bool)
    inv_Σ̃_c = np.random.randn(L, L)
    inv_Σ̃_c = inv_Σ̃_c @ inv_Σ̃_c.T # ensure inv_Σ̃_c is PSD
    inv_Σ̃_c = inv_Σ̃_c * is_nonzero[:, None] * is_nonzero[None, :]

    Σ̃_c = np.linalg.pinv(inv_Σ̃_c)

    sub_inv_Σ̃_c = inv_Σ̃_c[is_nonzero, :][:, is_nonzero]
    sub_Σ̃_c = np.linalg.inv(sub_inv_Σ̃_c)
    assert np.allclose(sub_Σ̃_c, Σ̃_c[is_nonzero, :][:, is_nonzero])



@pytest.mark.parametrize("L", [2, 3, 4])
@pytest.mark.parametrize("bern_p", [0.3, 0.5, 0.8])
def test_det_matrix_with_zero_submatrix(L, bern_p):
    """
    Test det of matrix with the zero block-diagonal submatrix 
    replaced with Identity = det of nonzero submatrix.
    Using numpy instead of jax for convenience.
    This trick was used to calculate det_Σ̃_c in marginal_separable_jax.py.
    """
    is_nonzero = np.random.binomial(1, bern_p, size=(L,)).astype(bool)
    Σ̃_c = np.random.randn(L, L)
    Σ̃_c = Σ̃_c @ Σ̃_c.T # ensure inv_Σ̃_c is PSD
    Σ̃_c = Σ̃_c * is_nonzero[:, None] * is_nonzero[None, :]
    semi_identity_mat = jnp.diag(~is_nonzero).astype(int)
    det_Σ̃_c = jnp.linalg.det(Σ̃_c + semi_identity_mat)
    
    sub_Σ̃_c = Σ̃_c[is_nonzero, :][:, is_nonzero]
    det_sub_Σ̃_c = jnp.linalg.det(sub_Σ̃_c)
    assert jnp.allclose(det_sub_Σ̃_c, det_Σ̃_c) # scalar
import sys
import jax.numpy as jnp
import jax
import numpy as np
import pytest
import numpy.random as nprandom

sys.path.append('/Users/xiaoqiliu/Desktop/3_Changepoints_detection/AMP/')
from amp.marginal_separable_jax import g_j, q, g, q_j_sample
from amp.signal_configuration import C_to_marginal, generate_C_distanced


def test_compare_two_estimates():
    n = 300
    L = 2
    σ = 0.2
    C_true = np.zeros(n, dtype=int)
    C_true[int(n/3):] = 1
    C_true = jnp.array(C_true)
    Δ = lambda n: int(n/5) 
    C_s = generate_C_distanced(n, L, Δ = Δ(n))
    ϕ = C_to_marginal(C_s) # Lxn
    ϕ = jnp.array(ϕ)

    num_samples = 1000
    ρ = nprandom.randn(L, L)
    ρ = ρ.T @ ρ + 1e-3 * jnp.eye(L)
    ρ = jnp.array(ρ)
    ρ_est = nprandom.randn(L, L)
    ρ_est = ρ_est.T @ ρ_est + 1e-3 * jnp.eye(L)
    ρ_est = jnp.array(ρ_est)

    ν = nprandom.randn(L, L)
    ν = jnp.array(ν)
    κ_T = nprandom.randn(L, L)
    κ_T = κ_T.T @ κ_T + 1e-3 * jnp.eye(L)
    κ_T = jnp.array(κ_T)

    ν_fixed = nprandom.randn(L, L)
    ν_fixed = jnp.array(ν_fixed)
    κ_T_fixed = nprandom.randn(L, L)
    κ_T_fixed = κ_T_fixed.T @ κ_T_fixed + 1e-3 * jnp.eye(L)
    κ_T_fixed = jnp.array(κ_T_fixed)

    Z_B_fixed = jnp.array(nprandom.multivariate_normal(
        jnp.zeros(L), ρ, size=(n, num_samples)))
    V_θ_fixed = Z_B_fixed @ jnp.linalg.inv(ρ) @ ν_fixed + \
        jnp.array(nprandom.multivariate_normal(
            jnp.zeros(L), κ_T_fixed, size=(n, num_samples)))
    # Note ϕ is the full marginal for defining g, rather than as an input argument to g.
    # All params defining g are the random-C version:
    assert Z_B_fixed.shape == (n, num_samples, L)
    assert V_θ_fixed.shape == (n, num_samples, L)
    assert C_true.shape == (n, )

    # Method 1 (the method we use inside SE-fixed-C): 
    # This method averages first over j, then over MC samples of each g_j.
    def estimate_κ_B(V, Z):
        g_fixed = g(V, q(Z, C_true, σ), ϕ, σ, ρ_est, ν, κ_T).reshape((n, L))
        return 1/n * g_fixed.T @ g_fixed
    κ_B_fixed = jnp.mean(jax.vmap(estimate_κ_B, in_axes=(1, 1), \
                                out_axes=0)(V_θ_fixed, Z_B_fixed), axis=0)
    

    # Method 2 (one element from κ_B_marginal):
    # This method averages first over MC samples of each g_j, then over j.
    def sm(ϕ_j, j, c_j):
        """Calculates one summand of E_{V, Z_B, Ψ}[g_j(V_j, Ȳ_j).T * g_j(V_j, Ȳ_j)]
        i.e. for one changepoint config c_j."""
        # Z_B_fixed[j, :, :] is num_samples x L
        u = q_j_sample(Z_B_fixed[j, :, :], c_j, σ, L)
        # g_j_ = g_j(V[j, :].flatten(), q_j(Z_B[j, :].reshape((1, L)), c_j, σ).sample(), ϕ_j, σ, ρ, ν, κ_T).reshape((1, L))
        # NOTE: input V, u to g are sampled according to the true signal prior, but g itself
        # is defined using the estimated signal prior ρ_est.
        g_j_full = jax.vmap(g_j, in_axes=(0, 0, None, None, None, None, None), 
                            out_axes=0)(
            V_θ_fixed[j, :, :], u, ϕ_j, σ, ρ_est, ν, κ_T).reshape((num_samples, L))
        return 1 / num_samples * g_j_full.T @ g_j_full # LxL
    κ_B_fixed_elements = jax.vmap(sm, in_axes=0, out_axes=0)(ϕ.T, jnp.arange(n), C_true)
    assert κ_B_fixed_elements.shape == (n, L, L)
    κ_B_fixed_1 = jnp.mean(κ_B_fixed_elements, axis=0)
    assert κ_B_fixed_1.shape == (L, L)
    assert jnp.allclose(κ_B_fixed, κ_B_fixed_1, rtol=1e-2*2)

if __name__ == "__main__":
    test_compare_two_estimates()
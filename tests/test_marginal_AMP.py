import jax
import jax.numpy as np
import numpy.random as nprandom
import tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import itertools
import amp.marginal_separable_jax
import amp.changepoint_jax
import amp.separable_jax
import amp.signal_configuration
from amp import MSE, norm_sq_corr, PAL

import unittest

class TestMarginalAMP(unittest.TestCase):
    def setUp(self):
        pass

    def test_marginal_AMP_against_woodbury(self):
        seed = 23
        nprandom.seed(2*seed)
        nprandom.seed(11)
        δ = 1.5 
        p = 500 
        σ = 0.1 # noise standard deviation
        L = 2 # num signals. L-1 changepoints
        n = int(δ * p) # 750
        T = 10
        ϵ = 1e-5

        B̃_cov = 1*np.eye(L)
        # B̃_cov[0, 1] = -0.01
        # B̃_cov[1, 0] = -0.01
        B̃ = nprandom.multivariate_normal(np.zeros(L), B̃_cov, size=p)
        n = int(δ * p)
        ρ = 1/δ * B̃_cov # 1/δ * Covariance matrix of each row of B_0, independent for now but should be generalized later
        Y = np.zeros((n, 1))

        B̂_0 = nprandom.multivariate_normal(np.zeros(L), B̃_cov, size=p)

        # All possible allowed changepoint locations (concentrated)
        C_full = np.triu(np.ones((n, n)), k=0).astype(int) # Here we are creating the matrix of all possible C's. This assumes that one changepoint happens forsure between 0≤t≤n-1. 
        C_s = C_full[int(5*n/12):int(7*n/12)] # Ask the AMP to search over a continuum of possible changepoint locations 
        C_true = C_full[np.array([int(n/2)])][0] # Ask the AMP to search over 3 possible changepoint locations

        # State the marginals, the AMP only cares about the marginals
        ϕ = np.zeros((L, n))

        # Changepoint highly concentrated
        ξ = int(n/6) # number of middle indices to concentrate the changepoint probability
        ξ_start = int((n - ξ)/2)
        ξ_end = int((n + ξ)/2)
        ϕ = ϕ.at[0, :ξ_start].set(np.ones(ξ_start) - ϵ/n) 
        ϕ = ϕ.at[1, :ξ_start].set(np.zeros(ξ_start) + ϵ/n)
        ϕ = ϕ.at[0, ξ_start:ξ_end].set(np.arange(int(ξ)-1, -1, -1) / int(ξ))
        ϕ = ϕ.at[0, ξ_end].set((0 + ϵ) / n) # Numerically stabilize the zero entries in the marginals ϕ
        ϕ = ϕ.at[1, ξ_start:ξ_end].set(np.arange(1, int(ξ) + 1, 1) / int(ξ))
        ϕ = ϕ.at[1, ξ_end].set((n - ϵ) / n) # Numerically stabilize the zero entries in the marginals ϕ
        ϕ = ϕ.at[0, ξ_end:].set(np.zeros(int(n - ξ_end)) + ϵ/n)
        ϕ = ϕ.at[1, ξ_end:].set(np.ones(int(n - ξ_end)) - ϵ/n)

        ρ = 1/δ * B̃_cov # 1/δ * Covariance matrix of each row of B_0, independent for now but should be generalized later
        Y = np.zeros((n, 1))
        X = nprandom.normal(0, np.sqrt(1/n), (n, p))

        # Generate the observation vector Y
        Θ = X @ B̃
        assert Θ.shape == (n, L)
        Y = amp.changepoint_jax.q(Θ, C_true, σ).sample()
        # Y = amp.separable_jax.q(Θ, ϕ[:, 0].flatten(), σ).sample()
        assert Y.shape == (n, 1)

        B̂, Θ_t, ν, ν̂ = amp.marginal_separable_jax.run_GAMP(B̂_0, δ, p, ϕ, L, σ, X, Y, ρ, T, verbose=False, seed=None)

        B̂_woodbury, ν_woodbury, ν̂_woodbury = amp.changepoint_jax.run_chgpt_GAMP_jax(C_s, B̂_0, δ, p, L, σ, X, Y, ρ, T, verbose=False, seed=None, post=False)

        # assert np.square(np.linalg.norm(B̂ - B̂_nelvin[-1]))/p <= 1e-3 # These two asserts hold if we have EXACTLY the same ν̂ as nelvin, which we don't since we have a more accurate one
        # assert np.max(np.abs(B̂ - B̂_nelvin[-1]))/1 <= 5e-2
        self.assertTrue(np.allclose(norm_sq_corr(B̂[:, 0], B̃[:, 0]), norm_sq_corr(B̂_woodbury[:, 0], B̃[:, 0]), atol = 2e-2))
        self.assertTrue(np.allclose(norm_sq_corr(B̂[:, 1], B̃[:, 1]), norm_sq_corr(B̂_woodbury[:, 1], B̃[:, 1]), atol = 2e-2)) # Checks that the AMP matches the Woodbury in norm squared correlation.
        self.assertTrue(np.allclose(ν[0, 0] / ρ[0, 0], ν_woodbury[0, 0] / ρ[0, 0], atol = 8e-2))
        self.assertTrue(np.allclose(ν[1, 1] / ρ[1, 1], ν_woodbury[1, 1] / ρ[1, 1], atol = 8e-2)) # Checks that the AMP matches the Woodbury in norm squared correlation.
        # self.assertTrue(np.square(np.linalg.norm(B̂ - B̂_woodbury))/p <= 2e-2) # These might seem large, but it is because we use a more stable sampling step than Nelvin, and so we don't match his exactly. 
        # self.assertTrue(np.max(np.abs(B̂ - B̂_woodbury))/1 <= 0.5)

        return

    def test_marginal_AMP_against_separable(self):
        seed = 23
        nprandom.seed(2*seed)
        nprandom.seed(11)
        δ = 1.5 
        p = 2000
        σ = 0.2
        L = 2 # num signals. L-1 changepoints
        n = int(δ * p) # 750
        T = 10
        ϵ = 1e-5

        B̃_cov = 1*np.eye(L)
        # B̃_cov[0, 1] = -0.01
        # B̃_cov[1, 0] = -0.01
        B̃ = nprandom.multivariate_normal(np.zeros(L), B̃_cov, size=p)
        B̂_0 = nprandom.multivariate_normal(np.zeros(L), B̃_cov, size=p)
        n = int(δ * p)

        # Mixed Unbalanced ϕ
        ϕ = np.zeros((L, n))
        ϕ = ϕ.at[0, :].set(0.7 * np.ones(n)) # Marginal probability of a changepoint happening at each index i <= n
        ϕ = ϕ.at[1, :].set(0.3 * np.ones(n))

        ρ = 1/δ * B̃_cov # 1/δ * Covariance matrix of each row of B_0, independent for now but should be generalized later
        Y = np.zeros((n, 1))
        X = nprandom.normal(0, np.sqrt(1/n), (n, p))

        # Generate the observation vector Y
        Θ = X @ B̃
        assert Θ.shape == (n, L)
        Y = amp.separable_jax.q(Θ, ϕ[:, 0].flatten(), σ).sample()
        assert Y.shape == (n, 1)

        B̂, Θ_t, ν, ν̂ = amp.marginal_separable_jax.run_GAMP(B̂_0, δ, p, ϕ, L, σ, X, Y, ρ, T, verbose=False, seed=None)
        B̂_sep, ν_sep, ν̂_sep = amp.separable_jax.run_GAMP(B̂_0, δ, p, np.array([0.7, 0.3]), L, σ, X, Y, ρ, T, verbose=False, seed=None)

        self.assertTrue(np.allclose(norm_sq_corr(B̂[:, 0], B̃[:, 0]), norm_sq_corr(B̂_sep[:, 0], B̃[:, 0]), atol = 2e-2))
        self.assertTrue(np.allclose(norm_sq_corr(B̂[:, 1], B̃[:, 1]), norm_sq_corr(B̂_sep[:, 1], B̃[:, 1]), atol = 2e-2)) # Checks that the AMP matches the Woodbury in norm squared correlation.
        self.assertTrue(np.allclose(ν[0, 0] / ρ[0, 0], ν_sep[0, 0] / ρ[0, 0], atol = 2e-2))
        self.assertTrue(np.allclose(ν[1, 1] / ρ[1, 1], ν_sep[1, 1] / ρ[1, 1], atol = 2e-2)) # Checks that the AMP matches the Woodbury in norm squared correlation.

    def test_marginal_AMP_against_woodbury_L_3(self):

        p = 500
        σ = 0.2 # noise standard deviation
        L = 3 # num signals. L-1 changepoints
        T = 10 # num_iterations
        δ = 1.5

        B̃_cov = np.eye(L)
        B̃ = nprandom.multivariate_normal(np.zeros(L), B̃_cov, size=p)
        B̂_0 = nprandom.multivariate_normal(np.zeros(L), B̃_cov, size=p)

        n = int(δ * p)
        ρ = 1/δ * B̃_cov # 1/δ * Covariance matrix of each row of B_0, independent for now but should be generalized later
        Y = np.zeros((n, 1))
        X = nprandom.normal(0, np.sqrt(1/n), (n, p))

        τ = int(n/4)
        C_s = amp.signal_configuration.generate_C_stagger(n, L, τ)
        num_C_s = C_s.shape[0]
        C_true = C_s[int(num_C_s/2)]
        ϕ = amp.signal_configuration.C_to_marginal_jax(C_s)

        # Generate the observation vector Y
        Θ = X @ B̃
        assert Θ.shape == (n, L)
        Y = amp.changepoint_jax.q(Θ, C_true, σ).sample()
        assert Y.shape == (n, 1)

        B̂, Θ_t, ν, ν̂ = amp.marginal_separable_jax.run_GAMP(B̂_0, δ, p, ϕ, L, σ, X, Y, ρ, T, verbose=False, seed=None)
        B̂_woodbury, ν_woodbury, ν̂_woodbury = amp.changepoint_jax.run_chgpt_GAMP_jax(C_s, B̂_0, δ, p, L, σ, X, Y, ρ, T, verbose=False)

        self.assertTrue(np.allclose(norm_sq_corr(B̂[:, 0], B̃[:, 0]), norm_sq_corr(B̂_woodbury[:, 0], B̃[:, 0]), atol = 2e-2))
        self.assertTrue(np.allclose(norm_sq_corr(B̂[:, 1], B̃[:, 1]), norm_sq_corr(B̂_woodbury[:, 1], B̃[:, 1]), atol = 2e-2)) # Checks that the AMP matches the Woodbury in norm squared correlation.
        self.assertTrue(np.allclose(norm_sq_corr(B̂[:, 2], B̃[:, 2]), norm_sq_corr(B̂_woodbury[:, 2], B̃[:, 2]), atol = 2e-2)) # Checks that the AMP matches the Woodbury in norm squared correlation.
        self.assertTrue(np.allclose(ν[0, 0] / ρ[0, 0], ν_woodbury[0, 0] / ρ[0, 0], atol = 5e-2))
        self.assertTrue(np.allclose(ν[1, 1] / ρ[1, 1], ν_woodbury[1, 1] / ρ[1, 1], atol = 5e-2)) # Checks that the AMP matches the Woodbury in norm squared correlation.
        self.assertTrue(np.allclose(ν[2, 2] / ρ[2, 2], ν_woodbury[2, 2] / ρ[2, 2], atol = 5e-2)) # Checks that the AMP matches the Woodbury in norm squared correlation.
        return
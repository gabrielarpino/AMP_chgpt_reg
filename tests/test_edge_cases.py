## Test that non-separable matches separable when there is only one signal
import numpy as np
import matplotlib.pyplot as plt
import amp.fully_separable as ours
import amp.sep_GAMP_template as nelvin
import unittest
import jax.numpy as jnp
import numpy.random as nprandom
import matplotlib.pyplot as plt
from amp import q, MSE, norm_sq_corr, run_chgpt_GAMP_jax, run_GAMP, PAL, signal_configuration
from jax import config
config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)

class TestEdgeCases(unittest.TestCase):
    def test_single_signal(self):
        """ We use L = 2 here instead of L = 1 because that is the only way to get the separable code working (without further modifications to the separable code). """
        seed = 11
        np.random.seed(seed)

        p = 500
        σ = 0.0 # noise standard deviation
        L = 2 # num signals. L-1 changepoints
        ϕ = 0.0
        T = 5 # num_iterations
        # run_chgpt_GAMP_jit = jax.jit(run_chgpt_GAMP_jax) # For later

        δ = 0.5
        B̃_cov = jnp.eye(L)
        B̃ = nprandom.multivariate_normal(jnp.zeros(L), B̃_cov, size=p)
        B̂_0 = nprandom.multivariate_normal(jnp.zeros(L), B̃_cov, size=p)

        n = int(δ * p)
        ρ = 1/δ * B̃_cov # 1/δ * Covariance matrix of each row of B_0, independent for now but should be generalized later
        Y = jnp.zeros((n, 1))
        B_bar_mean = jnp.array([0, 0])
        η = nprandom.normal(0.0, σ, (n, 1)) # noise
        X = nprandom.normal(0, jnp.sqrt(1/n), (n, p))

        # Select where the true changepoint will be, as that will highly affect the result. 
        C_full = jnp.triu(jnp.ones((n, n)), k=0).astype(int) # Here we are creating the matrix of all possible C's. This assumes that one changepoint happens forsure between 0≤t≤n-1. 
        C_s = C_full[jnp.array([0])] # Ask the AMP to search over 1 possible changepoint locations
        C_true = C_s[0] # Select the true change point to lie in the middle

        # Generate the observation vector Y
        Θ = X @ B̃
        assert Θ.shape == (n, L)
        Y = q(Θ, C_true, η).sample()
        assert Y.shape == (n, 1)

        print("Non-separable: ")
        B̂, ν, ν̂ = run_chgpt_GAMP_jax(C_s, B̂_0, δ, p, L, σ, X, Y, ρ, T, verbose=False, seed = seed)
        # ν, ν̂ = run_gaussian_SE(C_s, δ, p, L, σ, ρ, T, verbose=False) # Run the State Evolution separately just so that it is not contaminated by AMP (THIS IS EVEN WORSE!)
        print("Separable:" )
        B̂_sep, ν_sep, ν̂_sep = run_GAMP(B̂_0, δ, p, ϕ, L, σ, X, Y, ρ, T, verbose=False, seed = seed)
        
        # print("Norm sq diff 1: ", norm_sq_corr(B̂[:, 0], B̃[:, 0]) -  norm_sq_corr(B̂_sep[:, 0], B̃[:, 0]))
        # print("Norm sq diff 2: ", norm_sq_corr(B̂[:, 1], B̃[:, 1]) -  norm_sq_corr(B̂_sep[:, 1], B̃[:, 1]))
        # print("SE diff 0: ", ν[0, 0] / ρ[0, 0] - ν_sep[0, 0] / ρ[0, 0])
        # print("SE diff 1: ", ν[1, 1] / ρ[1, 1] - ν_sep[1, 1] / ρ[1, 1])
        
        self.assertTrue(jnp.allclose(ν, ν_sep, atol=1e-1))
        self.assertTrue(jnp.allclose(norm_sq_corr(B̂[:, 0], B̃[:, 0]), norm_sq_corr(B̂_sep[:, 0], B̃[:, 0]), atol=1e-4))
        self.assertTrue(jnp.allclose(norm_sq_corr(B̂[:, 1], B̃[:, 1]), norm_sq_corr(B̂_sep[:, 1], B̃[:, 1]), atol=5e-4))
        self.assertTrue(np.allclose(ν[0, 0] / ρ[0, 0], ν_sep[0, 0] / ρ[0, 0], atol=5e-2))
        self.assertTrue(np.allclose(ν[1, 1] / ρ[1, 1], ν_sep[1, 1] / ρ[1, 1], atol=5e-2))

    def test_L_4(self):
        p = 500
        σ = 0.0 # noise standard deviation
        L = 4 # num signals. L-1 changepoints

        T = 10 # num_iterations

        seed = 35
        nprandom.seed(2*seed)

        δ_list = [1.0]
        for δ in δ_list:
            print("--- δ: ", δ, " ---")

            B̃_cov = jnp.eye(L)
            B̃ = nprandom.multivariate_normal(jnp.zeros(L), B̃_cov, size=p)
            B̂_0 = nprandom.multivariate_normal(jnp.zeros(L), B̃_cov, size=p)

            n = int(δ * p)
            ρ = 1/δ * B̃_cov # 1/δ * Covariance matrix of each row of B_0, independent for now but should be generalized later
            Y = jnp.zeros((n, 1))
            X = nprandom.normal(0, jnp.sqrt(1/n), (n, p))

            τ = int(n/5)
            C_s = signal_configuration.generate_C_stagger(n, L, τ)
            num_C_s = C_s.shape[0]
            C_true_idx = int(num_C_s/2)
            C_true = C_s[C_true_idx]
            C_s = C_s[jnp.array([C_true_idx])]

            # Generate the observation vector Y
            Θ = X @ B̃
            Y = q(Θ, C_true, σ).sample()

            B̂, ν, ν̂= run_chgpt_GAMP_jax(C_s, B̂_0, δ, p, L, σ, X, Y, ρ, T, verbose=False)

            self.assertTrue(jnp.allclose(norm_sq_corr(B̂[:, 0], B̃[:, 0]), ν[0, 0] / ρ[0, 0], atol = 5e-2))
            self.assertTrue(jnp.allclose(norm_sq_corr(B̂[:, 1], B̃[:, 1]), ν[1, 1] / ρ[1, 1], atol = 5e-2))
            self.assertTrue(jnp.allclose(norm_sq_corr(B̂[:, 2], B̃[:, 2]), ν[2, 2] / ρ[2, 2], atol = 5e-2))
            self.assertTrue(jnp.allclose(norm_sq_corr(B̂[:, 3], B̃[:, 3]), ν[3, 3] / ρ[3, 3], atol = 5e-2))

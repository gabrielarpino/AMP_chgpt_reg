import jax
import jax.numpy as jnp
import numpy.random as nprandom
import tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from amp import q, MSE, norm_sq_corr, run_chgpt_GAMP_jax, run_GAMP, PAL

import unittest

class TestNonsepSEMatch(unittest.TestCase):
    def setUp(self):
        pass

    def test_nonsep_SE_match(self):
        seed = 17
        nprandom.seed(2*seed)
        p = 500
        σ = 0.0 # noise standard deviation
        L = 2 # num signals. L-1 changepoints

        T = 8 # num_iterations

        norm_sq_corr_1_list = []
        norm_sq_corr_2_list = []
        norm_sq_corr_1_SE_list = []
        norm_sq_corr_2_SE_list = []

        δ_list = [0.5, 1.0, 1.5] # Nelvin's list
        for δ in δ_list: 
            print("--- δ: ", δ, " ---")

            B̃_cov = jnp.eye(L)
            B̃ = nprandom.multivariate_normal(jnp.zeros(L), B̃_cov, size=p)
            B̂_0 = nprandom.multivariate_normal(jnp.zeros(L), B̃_cov, size=p)

            n = int(δ * p)
            ρ = 1/δ * B̃_cov # 1/δ * Covariance matrix of each row of B_0, independent for now but should be generalized later
            Y = jnp.zeros((n, 1))
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
            # print("Separable:" )
            # B̂_sep, ν_sep, ν̂_sep = run_GAMP(B̂_0, δ, p, ϕ, L, σ, X, Y, ρ, T, verbose=False, seed = seed)

            norm_sq_corr_1_list.append(norm_sq_corr(B̂[:, 0], B̃[:, 0]))
            norm_sq_corr_2_list.append(norm_sq_corr(B̂[:, 1], B̃[:, 1]))
            norm_sq_corr_1_SE_list.append(ν[0, 0] / ρ[0, 0])
            norm_sq_corr_2_SE_list.append(ν[1, 1] / ρ[1, 1])

            norm_sq_corr_1_list_arr = jnp.array(norm_sq_corr_1_list)
            norm_sq_corr_2_list_arr = jnp.array(norm_sq_corr_2_list)
            norm_sq_corr_1_SE_list_arr = jnp.array(norm_sq_corr_1_SE_list)
            norm_sq_corr_2_SE_list_arr = jnp.array(norm_sq_corr_2_SE_list)

            self.assertTrue(jnp.allclose(norm_sq_corr_1_list_arr, norm_sq_corr_1_SE_list_arr, atol = 2e-2))
            self.assertTrue(jnp.allclose(norm_sq_corr_2_list_arr, norm_sq_corr_2_SE_list_arr, atol = 2e-2)) # Checks that the AMP matches the SE. 


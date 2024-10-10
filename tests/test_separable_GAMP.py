## Test that ours matches Nelvin's in the separable case by running one AMP iteration and checking if those are equivalent
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import amp.fully_separable as ours
import amp.sep_GAMP_template as nelvin
from amp import norm_sq_corr
import unittest

class TestSepGAMP(unittest.TestCase):
    def test_sep_GAMP(self):
        np.random.seed(11)
        # np.random.seed(np.random.randint(1, 100))
        δ = 1.5
        p = 500 
        σ = 0.1 # noise standard deviation
        L = 2 # num signals. L-1 changepoints
        ϕ = 1 - 1/4
        n = int(δ * p) # 750

        B̃_cov = 1*np.eye(L)
        # B̃_cov[0, 1] = -0.01
        # B̃_cov[1, 0] = -0.01
        B̃ = np.random.multivariate_normal(np.zeros(2), B̃_cov, size=p)
        n = int(δ * p)
        ρ = 1/δ * B̃_cov # 1/δ * Covariance matrix of each row of B_0, independent for now but should be generalized later
        Y = np.zeros((n, 1))
        B_bar_mean = np.array([0, 0])
        Ψ = np.random.binomial(1, 1 - ϕ, size=(n, 1))
        η = np.random.normal(0.0, σ, (n, 1)) # noise
        X = np.random.normal(0, np.sqrt(1/n), (n, p))

        # Test that we generate Y the same way
        θ = X @ B̃
        self.assertTrue(θ.shape == (n, L))
        Y_ours = ours.q_sep(θ, Ψ, η)
        Y = Y_ours
        c = (1 - Ψ).flatten()[:, None]
        Y_nelvin = (θ * np.c_[c, 1-c]).sum(1) + η.flatten()
        self.assertTrue(np.allclose(Y_ours.flatten(), Y_nelvin, rtol=1e-16))

        B̂_0 = np.random.multivariate_normal(np.zeros(2), δ * ρ, size=p)

        # See the GAMPs match after T iterations: 
        T = 5
        B̂, ν, ν̂ = ours.run_GAMP(B̂_0, δ, p, ϕ, L, σ, X, Y, ρ, T, verbose=False)

        B̂_nelvin, ν̂_nelvin = nelvin.run_matrix_GAMP(n, p, ϕ, σ, X, Y.flatten(), B̃, B_bar_mean, B̃_cov, 
                                            B̂_0, B_bar_mean, B̃_cov, T)


        # print("B̂ norm difference: ", np.square(np.linalg.norm(B̂ - B̂_nelvin[-1]))/p)
        # assert np.square(np.linalg.norm(B̂ - B̂_nelvin[-1]))/p <= 1e-3 # These two asserts hold if we have EXACTLY the same ν̂ as nelvin, which we don't since we have a more accurate one
        # assert np.max(np.abs(B̂ - B̂_nelvin[-1]))/1 <= 5e-2
        self.assertTrue(np.square(np.linalg.norm(B̂ - B̂_nelvin[-1]))/p <= 2e-2) # These might seem large, but it is because we use a more stable sampling step than Nelvin, and so we don't match his exactly. 
        self.assertTrue(np.max(np.abs(B̂ - B̂_nelvin[-1]))/1 <= 0.5)

import numpy as np
from amp.fully_separable import *
from amp.sep_GAMP_template import *
import unittest

class TestSeparableFns(unittest.TestCase):
    def setUp(self):
        np.random.seed(10)
        # Definitions for f
        self.δ = 1.2
        self.p = 500 
        self.n = int(self.p * self.δ)
        self.L = 2
        self.ρ = 1/self.δ * np.eye(self.L)
        self.ν̂ = psd_mat(self.L)
        self.κ_B = self.ν̂
        self.test_dim = 10

    def test_f(self):
        np.random.seed(10)
        f = define_f(self.δ, self.L, self.ν̂, self.κ_B, self.ρ)
        x = np.random.rand(self.test_dim, self.L)
        f_f = f_full(x, self.δ, self.L, self.ν̂, self.κ_B, self.ρ) 

        self.assertTrue(f_f.shape == (self.test_dim, self.L))
        self.assertTrue(np.all(f_f[1, :] == f(x[1, :].reshape(1, self.L))))

        # Check that the f function matches that from Nelvin's
        B_bar_mean = np.array([0, 0])
        B_t_plus_1 = np.random.multivariate_normal(np.zeros(2), self.δ * self.ρ, size=self.p)
        B̂_ours = f_full(B_t_plus_1, self.δ, self.L, self.ν̂, self.κ_B, self.ρ)
        B̂_nelvin = np.apply_along_axis(f_k_bayes, 1, B_t_plus_1, self.ν̂, self.κ_B, B_bar_mean, self.δ * self.ρ) # Calling Nelvin's, no change
        self.assertTrue(np.allclose(B̂_ours, B̂_nelvin, rtol=1e-15))

    def test_F(self):
        np.random.seed(10)
        ν̂ = psd_mat(self.L)
        κ_B = psd_mat(self.L)
        χ = (self.ρ @ ν̂) @ np.linalg.pinv(ν̂.T @ self.ρ @ ν̂ + 1/self.δ * κ_B)
        F_ours = 1/self.δ * χ.T
        F_nelvin = (self.p/self.n) * f_k_prime(ν̂, κ_B, self.δ * self.ρ)
        self.assertTrue(np.allclose(F_ours, F_nelvin, rtol=1e-14))

    def test_g_and_C(self):
        np.random.seed(10)
        ν = psd_mat(self.L)
        κ_T = psd_mat(self.L)
        Θ = np.random.normal(size=(self.test_dim, self.L))
        Y = np.random.normal(size=(self.test_dim, 1))
        ϕ = 0.3
        σ = 0.1

        # Test that the g function extends well
        g = define_g(ϕ, self.L, σ, self.ρ, ν, κ_T)
        x = np.random.rand(self.test_dim, self.L)
        g_f = g_full(Θ, Y, ϕ, self.L, σ, self.ρ, ν, κ_T) 

        self.assertTrue(g_f.shape == (self.test_dim, self.L))
        self.assertTrue(np.all(g_f[1, :] == g(Θ[1, :].reshape(1, self.L), Y[1, :][0])))

        # Test the inner g
        X = np.random.normal(0.0, np.sqrt(1.0/self.n), (self.n, self.p))
        Y = np.zeros((self.n, 1))
        B̃ = np.random.multivariate_normal(np.zeros(2), self.δ * self.ρ, size=self.p)
        ϕ = 0.7
        σ = 0.0
        Ψ = np.random.binomial(1, ϕ , size=(self.n, 1)) # Stores signal index for each row of X
        η = np.random.normal(0.0, σ, (self.n, 1)) # noise
        θ = X @ B̃
        Y = q_sep(θ, Ψ, η)

        Θ_t = np.random.multivariate_normal(np.zeros(2), self.δ * self.ρ, size=self.n)

        ν = psd_mat(self.L)
        κ_T = psd_mat(self.L)
        # κ_T = ν - ν.T @ np.linalg.inv(ρ) @ ν

        # Applying our g
        R̂_ours = g_full(Θ_t, Y, ϕ, self.L, σ, self.ρ, ν, κ_T)

        # Applying Nelvin's g
        Σ_t = np.zeros((2*self.L, 2*self.L))
        Σ_t[0:self.L, 0:self.L] = self.ρ
        Σ_t[0:self.L, self.L:2*self.L] = ν
        Σ_t[self.L:2*self.L, 0:self.L] = ν
        Σ_t[self.L:2*self.L, self.L:2*self.L] = ν.T @ np.linalg.inv(self.ρ) @ ν + κ_T
        Θ_t_and_Y = np.concatenate((Θ_t,Y), axis=1) 
        R̂_nelvin = np.apply_along_axis(g_k_bayes_wrapper, 1, Θ_t_and_Y, Σ_t, ϕ, σ)

        norm_sq_corr = lambda x1, x2: np.trace(x1.T @ x2) /  (np.linalg.norm(x1, 'fro') * np.linalg.norm(x2, 'fro'))
        self.assertTrue(np.max(np.abs(R̂_ours - R̂_nelvin)) < 4e-5)

        # Test C: 
        C_nelvin = compute_C_k(Θ_t, R̂_ours, Σ_t) 
        Ω = ν.T @ np.linalg.inv(self.ρ) @ ν + κ_T
        C_ours = (np.linalg.pinv(Ω) @ (1/self.n * Θ_t.T @ R̂_ours - ν.T @ (1/self.n * R̂_ours.T @ R̂_ours))).T
        self.assertTrue(np.allclose(C_ours, C_nelvin, rtol=1e-15))
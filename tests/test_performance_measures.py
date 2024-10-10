import numpy as np
from amp.fully_separable import *
import amp.sep_GAMP_template as nelvin
import unittest

class TestPerformanceMeasures(unittest.TestCase):
    def test_performance_measures(self):
        # Test Norm Squared Correlation in the fully gaussian case
        # β̂_1 = np.random.normal(0, 1, size=p)
        # β̂_2 = np.random.normal(0, 1, size=p)
        L = 2
        B_bar_mean = np.array([0, 0])

        δ = 1.2
        ρ = 1/δ * np.eye(L)
        ν̂ = psd_mat(L)
        κ_B = ν̂
        χ = (ρ @ ν̂) @ np.linalg.pinv(ν̂.T @ ρ @ ν̂ + 1/δ * κ_B) 
        ν = ρ @ ν̂ @ χ.T

        norm_sq_1_ours = ν[0, 0] / ρ[0, 0]
        norm_sq_2_ours = ν[1, 1] / ρ[1, 1]
        norm_sq_1_nelvin = nelvin.norm_sq_corr1_SE(ν̂, B_bar_mean, δ * ρ)
        norm_sq_2_nelvin = nelvin.norm_sq_corr2_SE(ν̂, B_bar_mean, δ * ρ)
        self.assertTrue(np.allclose(norm_sq_1_ours, norm_sq_1_nelvin, rtol=1e-15))
        self.assertTrue(np.allclose(norm_sq_2_ours, norm_sq_2_nelvin, rtol=1e-15))

        # Test MSE in the fully gaussian case
        L = 2
        B_bar_mean = np.array([0, 0])
        δ = 1.2
        ρ = 1/δ * np.eye(L)
        ν̂ = psd_mat(L)
        κ_B = ν̂
        χ = (ρ @ ν̂) @ np.linalg.pinv(ν̂.T @ ρ @ ν̂ + 1/δ * κ_B) 
        ν = ρ @ ν̂ @ χ.T

        MSE_1_ours = (δ * ρ - δ * ν)[0, 0]
        MSE_2_ours = (δ * ρ - δ * ν)[1, 1]
        MSE_1_nelvin = nelvin.MSE_beta1_SE(ν̂, B_bar_mean, δ * ρ)
        MSE_2_nelvin = nelvin.MSE_beta2_SE(ν̂, B_bar_mean, δ * ρ)
        self.assertTrue(np.allclose(MSE_1_ours, MSE_1_nelvin, rtol=1e-15))
        self.assertTrue(np.allclose(MSE_2_ours, MSE_2_nelvin, rtol=1e-15))

import jax
import jax.numpy as jnp
import numpy.random as nprandom
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm
from amp import ϵ_0
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from amp.changepoint_jax import *
from amp.fully_separable import psd_mat
import random
import matplotlib.pyplot as plt
import amp.covariances
import unittest

class TestOptimizedWoodbury(unittest.TestCase):
    def setUp(self):
        self.n = 10
        self.L = 2
        self.σ = 0.1
        self.ρ = 1.0 * jnp.eye(self.L)
        self.ρ = self.ρ.at[0, 1].set(-0.2)
        self.ρ = self.ρ.at[1, 0].set(-0.2)
        self.ν = psd_mat(self.L)
        self.κ_T = psd_mat(self.L)
        self.j = 4
        self.indx = jnp.arange(0,self.n).reshape((self.n, ))
        self.V = nprandom.normal(size=(1, self.L))
        self.u = nprandom.normal(size=(self.n, 1))
        self.C_full = jnp.triu(jnp.ones((self.n, self.n)), k=0).astype(int)

    def test_woodbury_x_optimized(self):
        C_s = self.C_full[jnp.array([int(self.n/3), int(2*self.n/3)])]
        C_0 = C_s[0]
        x = jnp.concatenate((self.V.T, self.u))
        A_inv_x_ = amp.covariances.A_inv_x(x, C_0, self.n, self.ρ, self.σ, self.ν, self.κ_T)

        Σⱼ = form_Σ(C_0, self.j, self.L, self.σ, self.ρ, self.ν, self.κ_T)
        B = Σⱼ[2:, 2:]
        A_inv = jnp.block([
            [jnp.linalg.inv(B[:self.L, :self.L]), jnp.zeros((self.L, self.n))],
            [jnp.zeros((self.n, self.L)), jnp.linalg.inv(B[self.L:, self.L:])]
        ])

        U = jnp.zeros((self.L + self.n, self.L))
        U = U.at[:self.L, :self.L].set(psd_mat(self.L))
        U = U.at[self.L + self.j, :].set(jnp.ones((1, self.L)).flatten())

        V = jnp.zeros((self.L + self.n, self.L)).T
        V = V.at[:self.L, :self.L].set(psd_mat(self.L))
        V = V.at[:, self.L + self.j].set(jnp.ones((1, self.L)).flatten())

        C_diag = jnp.sqrt(2)/2

        res_regular = amp.covariances.woodbury_C_diag(A_inv, U, C_diag, V) @ x
        res_x = amp.covariances.woodbury_x(A_inv_x_, U, V, C_diag, C_0, self.n, self.ρ, self.σ, self.ν, self.κ_T)
        ν_κ_inv = jnp.linalg.pinv(self.ν.T @ jnp.linalg.inv(self.ρ) @ self.ν + self.κ_T)
        A_inv_x_update_L, A_inv_x_update_L_plus_j = amp.covariances.woodbury_x_optimized(self.j, A_inv_x_, U[:self.L, :self.L], U[self.L + self.j, :], V[:self.L, :self.L], V[:, self.L + self.j], C_diag, self.ρ, self.σ, ν_κ_inv)

        Σ_res_optimized = A_inv_x_
        Σ_res_optimized = Σ_res_optimized.at[:self.L].set(Σ_res_optimized[:self.L] - A_inv_x_update_L)
        Σ_res_optimized = Σ_res_optimized.at[self.L + self.j].set(Σ_res_optimized[self.L + self.j] - A_inv_x_update_L_plus_j)
        self.assertTrue(jnp.allclose(res_regular, res_x, atol = 1e-5))
        self.assertTrue(jnp.allclose(res_regular, Σ_res_optimized, atol = 1e-5))

    def test_A_inv_x(self):
        C_s = self.C_full[jnp.array([int(self.n/3), int(2*self.n/3)])]
        C_0 = C_s[0]
        x = jnp.concatenate((self.V.T, self.u))
        A_inv_x_ = amp.covariances.A_inv_x(x, C_0, self.n, self.ρ, self.σ, self.ν, self.κ_T)

        Σⱼ = form_Σ(C_0, self.j, self.L, self.σ, self.ρ, self.ν, self.κ_T)
        B = Σⱼ[2:, 2:]
        A_inv = jnp.block([
            [jnp.linalg.inv(B[:self.L, :self.L]), jnp.zeros((self.L, self.n))],
            [jnp.zeros((self.n, self.L)), jnp.linalg.inv(B[self.L:, self.L:])]
        ])
        self.assertTrue(jnp.allclose(A_inv @ x, A_inv_x_, atol = 1e-5))
    
    def test_Σ_V_Y_inv_x(self):
        self.assertTrue(jnp.all(jnp.linalg.eigh(self.ν.T @ jnp.linalg.inv(self.ρ) @ self.ν + self.κ_T)[0] > 0))
        C_s = self.C_full[jnp.array([int(self.n/3), int(2*self.n/3)])]
        C_0 = C_s[0]

        ν_κ_inv = jnp.linalg.pinv(self.ν.T @ jnp.linalg.inv(self.ρ) @ self.ν + self.κ_T)
        x = jnp.concatenate((self.V.T, self.u))
        A_inv_x_ = amp.covariances.A_inv_x_no_C(x, self.n, self.ρ, self.σ, self.ν, self.κ_T)

        A_inv_x_update_L, A_inv_x_update_L_plus_j, log_det_res_optimized = amp.covariances.Σ_V_Y_inv_x(self.j, self.V, self.u, C_0, self.n, self.ρ, self.σ, self.ν, self.κ_T, ν_κ_inv, A_inv_x_)
        # Σ_res_optimized, log_det_res_optimized = amp.covariances.Σ_V_Y_inv_x(self.j, self.V, self.u, C_0, self.n, self.ρ, self.σ, self.ν, self.κ_T, ν_κ_inv, A_inv_x_)

        Σ_res_optimized = A_inv_x_
        Σ_res_optimized = Σ_res_optimized.at[:self.L].set(Σ_res_optimized[:self.L] - A_inv_x_update_L)
        Σ_res_optimized = Σ_res_optimized.at[self.L + self.j].set(Σ_res_optimized[self.L + self.j] - A_inv_x_update_L_plus_j)
        # A_inv_x_updated = A_inv_x_updated.at[L+j].set(A_inv_x_updated[L + j] - (Σ_Y_inv_diag_[j] * U_L_j @ X2 @ V_L @ A_inv_x_[:L] + jnp.dot((Σ_Y_inv_diag_[j] * U_L_j).flatten(), (X2 @ V_L_j).flatten()) * A_inv_x_[L+j]))
        Σⱼ = form_Σ(C_0, self.j, self.L, self.σ, self.ρ, self.ν, self.κ_T)
        Σ_res_regular = jnp.linalg.solve(Σⱼ[2:, 2:], jnp.concatenate((self.V.T, self.u))) 
        log_det_res_regular = jnp.linalg.slogdet(Σⱼ[2:, 2:])[1]
        self.assertTrue(jnp.allclose(Σ_res_optimized, Σ_res_regular, atol = 1e-7))
        self.assertTrue(jnp.allclose(log_det_res_optimized, log_det_res_regular, atol = 1e-7))




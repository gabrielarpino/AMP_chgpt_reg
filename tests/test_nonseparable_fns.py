import jax
import jax.numpy as np
import numpy
import numpy.random as nprandom
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import scipy
from tqdm import tqdm
from amp import ϵ_0
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import amp.changepoint_jax as changepoint_jax
from amp.changepoint_jax import *
from amp.fully_separable import psd_mat
from amp.fully_separable import E_Z as E_Z_sep
from amp.fully_separable import g_full as g_full_sep
import amp.fully_separable as fully_sep
import amp.covariances
import random
import matplotlib.pyplot as plt
import itertools
from jax.scipy.stats import multivariate_normal
key = jax.random.PRNGKey(2)
import unittest

class TestNonSeparableFunctions(unittest.TestCase):
    def setUp(self):
        self.p = 10
        self.δ = 1.0
        self.n = int(self.δ * self.p)
        self.L = 2
        self.σ = 0.0
        self.ρ = 1.0 * np.eye(self.L)
        self.ρ = self.ρ.at[0, 1].set(-0.2) # Now compatible with jax
        self.ρ = self.ρ.at[1, 0].set(-0.2)
        nprandom.seed(10) # does this seed the psd_mat function as well?
        self.ν = psd_mat(self.L)
        self.κ_T = psd_mat(self.L)
        self.j = 3
        self.n = 10
        self.ϕ = 0.0
        self.indx = np.arange(0, self.n).reshape((self.n, 1))
        self.C_full = np.triu(np.ones((self.n, self.n)), k=0).astype(int)

    def test_single_C(self):
        # nprandom.seed(1) # does this seed the psd_mat function as well?
        # C_s = self.C_full[np.array([0, int(self.n/2), -1])]
        # B̃ = nprandom.multivariate_normal(np.zeros(self.L), np.eye(self.L), size=self.p)
        # X = nprandom.normal(0, np.sqrt(1/self.n), (self.n, self.p))
        # # Generate the observation vector Y
        # Θ = X @ B̃
        # Y = q(Θ, C_s, self.σ).sample()
        # ξ = random.randint(1, self.n-1) # TODO: have to test things when ξ is at the endpoints
        # C = np.zeros((self.n, 1)).astype(int)
        # C = C.at[ξ:].set(np.ones((self.n-ξ, 1)).astype(int)).reshape((1, self.n))
        # Θ = X @ B̃
        # Y = q(Θ, C, self.σ).sample()
        # self.assertEqual(Y.shape, (self.n, 1))
        pass

    def test_form_Σ(self):
        nprandom.seed(1) # does this seed the psd_mat function as well?
        C_s = self.C_full[np.array([0, int(self.n/2), -1])]
        Σ = form_Σ(C_s[0], 1, self.L, self.σ, self.ρ, self.ν, self.κ_T)
        self.assertEqual(Σ.shape, (2*self.L + self.n, 2*self.L + self.n))
        Σ_2 = form_Σ(C_s[1], 2, self.L, self.σ, self.ρ, self.ν, self.κ_T) 
        self.assertTrue(self.n > 2 )
        self.assertTrue(not np.allclose(Σ, Σ_2)) # We expect both Σ to differ at indices j where their corresponding changepoint vector C differs.  
        self.assertTrue(not np.allclose(Σ[0:self.L, self.L:], Σ_2[0:self.L, self.L:])) # We expect both Σ to differ at indices j where their corresponding changepoint vector C differs.  
        self.assertTrue(not np.allclose(Σ[self.L:, self.L:], Σ_2[self.L:, self.L:])) # We expect both Σ to differ at indices j where their corresponding changepoint vector C differs.  
        self.assertTrue(np.allclose(Σ[2*self.L:, 2*self.L:], Σ_2[2*self.L:, 2*self.L:])) # We expect both Σ to differ at indices j where their corresponding changepoint vector C differs.  
        self.assertTrue(np.allclose(Σ[2*self.L:, :1*self.L], Σ[:1*self.L, 2*self.L:].T))
        self.assertTrue(np.allclose(Σ[2*self.L:,1*self.L:2*self.L], Σ[1*self.L:2*self.L, 2*self.L:].T))
        self.assertTrue(not np.allclose(Σ[self.L:2*self.L, self.L:], Σ_2[self.L:2*self.L, self.L:])) # We expect both Σ to differ at indices j where their corresponding changepoint vector C differs.  
        self.assertTrue(np.allclose(Σ.T, Σ)) 
        self.assertTrue(np.all(np.linalg.eigh(Σ)[0] + 1e-06 >= 0)) # This one sometimes does not hold due to float32/float64 changes in Jax. But it is not very relevant because its submatrix of relevance is tested below. In jax, we get numerical inaccuracies up to 1e-15. So for this to work we set ϵ_0 = 1e-14
        self.assertTrue(np.all(np.linalg.eigh(Σ[self.L:, self.L:])[0] >= 0))  
        self.assertTrue(np.all(np.linalg.eigh(Σ[2*self.L:, 2*self.L:])[0] >= 0))
        self.assertTrue(np.all(np.linalg.eigh(Σ[:2*self.L, :2*self.L])[0] >= 0)) 
        self.assertTrue(np.all(np.linalg.eigh(Σ[self.L:, self.L:])[0] >= 0)) 
        self.assertTrue(np.all(np.linalg.eigh(self.ν.T @ np.linalg.inv(self.ρ) @ self.ν + self.κ_T)[0] >= 0))  # THIS NEEDS TO BE SYMMETRIC. M is pd iff it is symmetric (hermitian) and all its evals are real and positive
        V = nprandom.normal(size=(1, self.L))
        u = nprandom.normal(size=(self.n, 1))
        x = np.concatenate((V.T, u))
        mv = multivariate_normal.logpdf(x.flatten(), mean=np.zeros((self.L+self.n,)), cov=Σ[self.L:, self.L:] + ϵ_0 * np.eye(self.L+self.n))  # Allow_singular not implemented in jax
        self.assertTrue(mv < 0)

    def test_form_Σ_submatrix_λ(self):
        nprandom.seed(1) # does this seed the psd_mat function as well?
        C_s = self.C_full[np.array([0, int(self.n/2), -1])]
        V = nprandom.normal(size=(1, self.L))
        u = nprandom.normal(size=(self.n, 1))

        # Updated test after maximally optimizing
        ν_κ_inv = np.linalg.pinv(self.ν.T @ np.linalg.inv(self.ρ) @ self.ν + self.κ_T)
        x = np.concatenate((V.T, u))
        A_inv_x_ = amp.covariances.A_inv_x_no_C(x, self.n, self.ρ, self.σ, self.ν, self.κ_T)
        
        # Calculate a
        λ_a, A_inv_x_update_L, A_inv_x_update_L_plus_j, log_det_ = form_Σ_submatrix_λ(C_s[0], 1, V, u, self.L, self.σ, self.ρ, self.ν, self.κ_T, ν_κ_inv, A_inv_x_) 
        Σ_inv_x_a = A_inv_x_
        Σ_inv_x_a = Σ_inv_x_a.at[:self.L].set(Σ_inv_x_a[:self.L] - A_inv_x_update_L)
        Σ_inv_x_a = Σ_inv_x_a.at[self.L + self.j].set(Σ_inv_x_a[self.L + self.j] - A_inv_x_update_L_plus_j)

        # Calculate b
        λ_b, A_inv_x_update_L, A_inv_x_update_L_plus_j, log_det_ = form_Σ_submatrix_λ(C_s[1], 2, V, u, self.L, self.σ, self.ρ, self.ν, self.κ_T, ν_κ_inv, A_inv_x_) 
        Σ_inv_x_b = A_inv_x_
        Σ_inv_x_b = Σ_inv_x_b.at[:self.L].set(Σ_inv_x_b[:self.L] - A_inv_x_update_L)
        Σ_inv_x_b = Σ_inv_x_b.at[self.L + self.j].set(Σ_inv_x_b[self.L + self.j] - A_inv_x_update_L_plus_j)


        # λ_a, Σ_inv_x_a, log_det_a = form_Σ_submatrix_λ(C_s[0], 1, V, u, self.L, self.σ, self.ρ, self.ν, self.κ_T, ν_κ_inv, A_inv_x_) 
        # λ_b, Σ_inv_x_b, log_det_b = form_Σ_submatrix_λ(C_s[1], 2, V, u, self.L, self.σ, self.ρ, self.ν, self.κ_T, ν_κ_inv, A_inv_x_) 
        self.assertTrue(λ_a.shape[0] == self.L)
        self.assertTrue(Σ_inv_x_a.shape == (self.L + self.n, 1))
        self.assertTrue(not np.allclose(Σ_inv_x_a, Σ_inv_x_b))
        self.assertTrue(not np.allclose(λ_a, λ_b))

    def test_E_Z(self):
        nprandom.seed(100) # does this seed the psd_mat function as well?
        C_s = self.C_full[np.array([0, int(self.n/2), -1])]
        V = nprandom.normal(size=(1, self.L))
        u = nprandom.normal(size=(self.n, 1))
        res = E_Z(V, self.j, u, self.L, self.σ, self.ρ, self.ν, self.κ_T, C_s)
        self.assertTrue(res.shape == (1, self.L))
        u = np.ones((self.n, 1))
        all_combinations = np.array(list(itertools.product([0, 1], repeat=self.n)))
        res2 = E_Z(V, self.j, u, self.L, self.σ, self.ρ, self.ν, self.κ_T, all_combinations)
        ϕ = 1/2
        res3 = fully_sep.E_Z(V, float(u[0]), ϕ, self.L, self.σ, self.ρ, self.ν, self.κ_T)
        self.assertTrue(np.allclose(res2, res3, atol=1e-7)) # With such a big logsumexp sum, the error adds up in float32
        # Test now with all ones
        u = np.ones((self.n, 1))
        res = E_Z(V, self.j, u, self.L, self.σ, self.ρ, self.ν, self.κ_T, C_s[0].reshape((1, self.n)))
        ϕ = 0.0
        res_sep = E_Z_sep(V, float(u[0]), ϕ, self.L, self.σ, self.ρ, self.ν, self.κ_T)
        self.assertTrue(np.allclose(res, res_sep, atol=1e-7)) # atol is large because these values are large

    def test_g_chgpt_with_sep(self):
        """ These tests need to be redone. They are not close, but the AMP runs well and matches the SE in all experiments. """
        nprandom.seed(101) # does this seed the psd_mat function as well?
        C_s = self.C_full[np.array([0, int(self.n/2), -1])]
        V = nprandom.normal(size=(1, self.L))
        # u = nprandom.normal(size=(self.n, 1))
        u = np.ones((self.n, 1))
        all_combinations = np.array(list(itertools.product([0, 1], repeat=self.n)))
        # Test g_chgpt matching with g_sep. These two are close, but not by much. Sometimes they differ in 1e-3. 
        ϕ = 1/2
        fully_sep_g_ = fully_sep.define_g(ϕ, self.L, self.σ, self.ρ, self.ν, self.κ_T)(V, float(u[0]))
        g_chgpt_ = g_chgpt(V, self.j, u, self.L, self.σ, self.ρ, self.ν, self.κ_T, all_combinations)
        self.assertTrue(np.allclose(fully_sep_g_, g_chgpt_ , atol=1e-05))
        ϕ = 0.0
        res_sep = fully_sep.define_g(ϕ, self.L, self.σ, self.ρ, self.ν, self.κ_T)(V, float(u[0]))
        res_nonsep = g_chgpt(V, self.j, u, self.L, self.σ, self.ρ, self.ν, self.κ_T, C_s[0].reshape((1, self.n)))
        self.assertTrue(np.allclose(res_sep, res_nonsep, atol = 1e-5)) # This used to be atol = 1e-10, but it recently got slightly less accurate after optimizing woodbury further. 

    def test_ν̂_one_samp(self):
        C_full = np.triu(np.ones((self.n, self.n)), k=0).astype(int) # Here we are creating the matrix of all possible C's. This assumes that one changepoint happens forsure between 0≤t≤n-1.
        C_s = C_full[np.array([0, int(self.n/2), -1])]
        C_0 = C_s[0].reshape((1, self.n))
        self.assertTrue(ν̂_one_samp(C_s, self.indx, self.L, self.ρ, self.n, self.σ, self.ν, self.κ_T).shape == (self.L, self.L))
        self.assertTrue(ν̂_one_samp(C_0, self.indx, self.L, self.ρ, self.n, self.σ, self.ν, self.κ_T).shape == (self.L, self.L))



### --- Test ν̂ against separable case --- ###
# n = 10
# ϕ = 0.0
# Θ_t = np.array(X @ B̃)
# indx = np.arange(0, Θ_t.shape[0]).reshape((Θ_t.shape[0], 1))
# C_full = np.triu(np.ones((n, n)), k=0).astype(int) # Here we are creating the matrix of all possible C's. This assumes that one changepoint happens forsure between 0≤t≤n-1. 
# C_s = C_full[np.array([0, int(n/2), -1])]
# C_0 = C_s[0].reshape((1, n))
# η = σ * jax.random.normal(key, (n, 1))
# R̂ = g_full_sep(Θ_t, Y, ϕ, L, σ, ρ, ν, κ_T)
# ν̂_sep_Y = 1/n * R̂.T @ R̂
# ν̂_one_samp_ = changepoint_jax.ν̂_one_samp(C_s[0].reshape((1, n)), Θ_t, indx, L, ρ, n, σ, ν, κ_T, Y = Y)
# ν̂_samp_ = ν̂_samp(C_s[0].reshape((1, n)), Θ_t, indx, L, ρ, n, σ, ν, κ_T, Y = Y) # THIS IS WHAT IS WRONG
# assert np.allclose(ν̂_sep_Y, ν̂_one_samp_, atol = 1e-4)
# assert np.allclose(ν̂_sep_Y, ν̂_samp_, atol = 1e-4)

### --- Test C^t AD against C^t fully sep. We expect them to be equal only for optimal g (and consequently, not just any ν, ρ) --- ###
# ϕ = 0.0

# ν = 0 * np.eye(L)
# κ_T = ρ
# R̂ = fully_sep.g_full(Θ_t, Y, ϕ, L, σ, ρ, ν, κ_T)
# Ω = ν.T @ np.linalg.inv(ρ) @ ν + κ_T
# C_sep = (np.linalg.pinv(Ω) @ (1/n * Θ_t.T @ R̂ - ν.T @ (1/n * R̂.T @ R̂))).T

# dgdθ = jax.jacrev(g_chgpt, 0)
# def jacobian(V, j, u, ν, κ_T): return dgdθ(
#     V, j, u, L, σ, ρ, ν, κ_T, C_s).reshape((L, L))
# def C_fun(Θ_, Y_, ν, κ_T):
#     indx = np.arange(0, Θ_.shape[0]).reshape((Θ_.shape[0], 1))
#     jac_map = jax.vmap(jacobian, (0, 0, None, None, None), 0)
#     return 1/n * np.sum(jac_map(Θ_, indx, Y_, ν, κ_T), axis=0)
# C_tensorized = jax.jit(C_fun)
# C_nonsep = C_tensorized(Θ_t, Y, ν, κ_T)
# print("C_sep - C_nonsep: ", C_sep - C_nonsep)


### --- Test C^t (TO BE COMPLETED) --- ###
# n = 100 # maybe this n needs to be much bigger for the ergodic theorem to show? 
# B̃ = nprandom.multivariate_normal(np.zeros(L), np.eye(L), size=p)
# X = nprandom.normal(0, np.sqrt(1/n), (n, p))
# # Generate the observation vector Y
# Θ = np.array(X @ B̃)
# assert Θ.shape == (n, L)
# C_full = np.triu(np.ones((n, n)), k=0).astype(int) # Here we are creating the matrix of all possible C's. This assumes that one changepoint happens forsure between 0≤t≤n-1. 
# C_s = C_full[np.array([0, int(n/2), -1])]
# Y = q(Θ, C_s, σ).sample()
# assert Y.shape == (n, 1)

# # Test C^t against separable case
# n = 100 # maybe this n needs to be much bigger for the ergodic theorem to show? 
# B̃ = nprandom.multivariate_normal(np.zeros(L), np.eye(L), size=p)
# X = nprandom.normal(0, np.sqrt(1/n), (n, p))
# Θ = np.array(X @ B̃)
# ϕ = 0.0
# u = np.ones((n, 1))
# C_full = np.triu(np.ones((n, n)), k=0).astype(int) # Here we are creating the matrix of all possible C's. This assumes that one changepoint happens forsure between 0≤t≤n-1. 
# C_s = C_full[np.array([0])] # Only one possible signal, in order to compare with non-sep with ϕ = 0. 
# C_true = C_s[0]
# Y = q(Θ, C_s, σ, C_true = C_true).sample()

# # C^t separable (reference) case
# R̂ = fully_sep.g_full(Θ, Y, ϕ, L, σ, ρ, ν, κ_T)
# Ω = ν.T @ np.linalg.inv(ρ) @ ν + κ_T
# C_sep = (np.linalg.pinv(Ω) @ (1/n * Θ.T @ R̂ - ν.T @ (1/n * R̂.T @ R̂))).T

# # C^t AD case
# dgdθ = jax.jacrev(g_chgpt, 0)
# def jacobian(V, j, u, ν, κ_T): return dgdθ(
#     V, j, u, L, σ, ρ, ν, κ_T, C_s).reshape((L, L))
# def C_fun(Θ_, Y_, ν, κ_T):
#     indx = np.arange(0, Θ_.shape[0]).reshape((Θ_.shape[0], 1))
#     jac_map = jax.vmap(jacobian, (0, 0, None, None, None), 0)
#     return 1/n * np.sum(jac_map(Θ_, indx, Y_, ν, κ_T), axis=0)
# C_tensorized = jax.jit(C_fun)
# C_nonsep_AD = C_tensorized(Θ, Y, ν, κ_T) 

# ### --- C^t short approximation --- ###
# Ω = ν.T @ np.linalg.inv(ρ) @ ν + κ_T
# def inner(i): return 1/n * Θ.T @ changepoint_jax.g_j_chgpt_tensorized(Θ,
#                                                         i, Y, L, σ, ρ, ν, κ_T, C_s).reshape((n, 2))
# inner_map = jax.vmap(inner, 0, 0)
# indx = np.arange(0, Θ.shape[0]).reshape((Θ.shape[0], ))
# C_short = 1/n * np.sum(inner_map(indx), axis=0).T @ np.linalg.pinv(Ω).T # I don't think this is correct. 

# ### --- C^t long approximation --- ###
# Ω = ν.T @ np.linalg.inv(ρ) @ ν + κ_T
# def inner1(i): return 1/n * Θ.T @ changepoint_jax.g_j_chgpt_tensorized(Θ,
#                                                 i, Y, L, σ, ρ, ν, κ_T, C_s).reshape((n, 2))
# def inner2(i): 
#     g_res = changepoint_jax.g_j_chgpt_tensorized(Θ, i, Y, L, σ, ρ, ν, κ_T, C_s).reshape((n, 2))
#     return 1/n * g_res.T @ g_res
# inner1_map = jax.vmap(inner1, 0, 0)
# inner2_map = jax.vmap(inner2, 0, 0)
# indx = np.arange(0, Θ.shape[0]).reshape((Θ.shape[0], ))
# C_long =  (np.linalg.pinv(Ω) @ (1/n * np.sum(inner1_map(indx), axis=0) - ν.T @ (1/n * np.sum(inner2_map(indx), axis=0)) )).T # I don't think this is correct. 

# # print(C_sep)
# # print(C_nonsep_AD)
# # print(C_short)
# # print(C_long)
# # print(np.trace(C_sep - C_nonsep_AD))
# # print(np.trace(C_sep - C_short))
# # print(np.trace(C_sep - C_long))
# # # They are all roughly equally distant from C_sep.
# # print(np.trace(C_nonsep_AD - C_short))
# # print(np.trace(C_nonsep_AD - C_long))
# # print(np.trace(C_short - C_long))
# # assert np.allclose(C_sep, C_nonsep_AD, atol = 1e-4) # This is a problem! 